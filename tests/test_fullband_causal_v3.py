import inspect
import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from deep_anc.dsp.fullband_causal_v3 import (
    CLOCK_ANCHOR_FRAMES, CLOCK_ANCHOR_RESPONSE_GUARD, CLOCK_COMBINED_MAX,
    CLOCK_CUBIC_MAX, CLOCK_HARD_MAX, CLOCK_LEAVEOUT_MAX, LIVE_AUTHORITY,
    MARKER_GUARD, MAX_CONDITION, PILOT_BAND,
    absolute_dac_q_timewarp_v3, build_plan, generate_candidate, freeze_fit_candidates,
)
from deep_anc.dsp.fullband_causal_aperiodic import linear_fft_convolve


def _callbacks(frames):
    count=np.full(int(np.ceil(frames/256)),256,dtype=np.int64)
    count[-1]=frames-256*(len(count)-1)
    index=np.r_[0,np.cumsum(count[:-1])]
    return {
        "frame_index":index,
        "frame_count":count,
        "input_adc_time":index/48000.0,
        "output_dac_time":index/48000.0,
    }


def _add_sparse_path(output,source,*,delay,support,gain,highband_mutation=False):
    taps=[(delay,gain),(delay+17,-0.12*gain),(delay+support-1,0.025*gain)]
    if highband_mutation:
        # A first-difference pair materially changes the high-band response but
        # is still one fixed causal plant.  Clock fitting must not inspect it.
        taps.extend(((delay+1,0.30*gain),(delay+2,-0.30*gain)))
    for offset,value in taps:
        output[offset:offset+len(source)]+=value*source


def _plant_output(plan,pcm,*,profile="ordinary",highband_mutation=False):
    signal=pcm.astype(np.float64)/32767.0
    if profile=="ordinary":
        specifications=(
            ((1700,1024,0.80),(1300,2048,0.60)),
            ((1100,4096,0.50),(900,1024,0.40)),
        )
    elif profile=="maximum_delay_and_support":
        specifications=(
            ((4800,4096,0.80),(4700,4096,0.60)),
            ((4600,4096,0.50),(4500,4096,0.40)),
        )
    else:
        raise ValueError(profile)
    length=len(pcm)+10000
    result=[]
    for microphone in specifications:
        response=np.zeros(length,dtype=np.float64)
        for channel,(delay,support,gain) in enumerate(microphone):
            _add_sparse_path(
                response,signal[:,channel],delay=delay,support=support,gain=gain,
                highband_mutation=highband_mutation,
            )
        result.append(response)
    return result


def _warp_raw(responses,pcm_frames,*,drift_ppm,step_samples=0.0,step_adc=None):
    ratio=1.0+float(drift_ppm)*1e-6
    minimum=int(np.ceil((pcm_frames-1)/ratio))+4
    frames=max(pcm_frames,minimum)
    adc=np.arange(frames,dtype=np.float64)
    q=ratio*adc
    if step_adc is not None:
        q=q+(adc>=float(step_adc))*float(step_samples)
    raw=[]
    for response in responses:
        spline=CubicSpline(np.arange(len(response),dtype=np.float64),response,extrapolate=False)
        raw.append(np.nan_to_num(spline(q),nan=0.0,posinf=0.0,neginf=0.0))
    return raw,_callbacks(frames)


def _warp_piecewise_raw(responses,pcm_frames,*,break_adc,first_ppm,second_ppm):
    first=1.0+float(first_ppm)*1e-6
    second=1.0+float(second_ppm)*1e-6
    frames=pcm_frames+2048
    adc=np.arange(frames,dtype=np.float64)
    q=np.where(
        adc<=float(break_adc),
        first*adc,
        first*float(break_adc)+second*(adc-float(break_adc)),
    )
    raw=[]
    for response in responses:
        spline=CubicSpline(np.arange(len(response),dtype=np.float64),response,extrapolate=False)
        raw.append(np.nan_to_num(spline(q),nan=0.0,posinf=0.0,neginf=0.0))
    return raw,_callbacks(frames)


@pytest.fixture(scope="module")
def v3_signal():
    return build_plan()

def test_v3_signal_order_duration_clock_and_thermal(v3_signal):
    p,x=v3_signal; assert p["output"]["duration_seconds"]<50; assert p["output"]["peak_pcm"]<=98
    kinds=[r["kind"] for r in p["layout"]]
    assert kinds.index("primary_fit_a_burst") < kinds.index("secondary_fit_a_burst") < kinds.index("primary_fit_b_burst") < kinds.index("secondary_fit_b_burst") < kinds.index("primary_holdout_burst") < kinds.index("secondary_holdout_burst")
    for m in p["bursts"].values():
        assert m["rms"]<=.0015 and m["pilot_complete_windows"]==10
        assert m["causal_reserved_band_max_abs_dft"] < 1e-8
        assert m["pilot_quantisation_snr_proxy_db"] > 30
        assert set(m["toeplitz_condition_by_support"])=={"1024","2048","4096"}
        assert max(m["toeplitz_condition_by_support"].values()) <= MAX_CONDITION
    assert p["clock_overlay"]["highband_clock_fit_forbidden"] is True
    assert p["memory"]["burst_guard_samples"]==262144
    assert p["memory"]["marker_guard_samples"]==MARKER_GUARD
    assert p["memory"]["infinite_tail_claim"] is False
    assert p["memory"]["full_exact_zero_guard_precedes_clock_anchor"] is True
    assert p["memory"]["clock_anchor_excluded_from_candidate_data"] is True
    anchors=[row for row in p["layout"] if row["kind"].endswith("_clock_anchor")]
    assert len(anchors)==6 and all(row["frames"]==CLOCK_ANCHOR_FRAMES for row in anchors)
    assert all(row["candidate_data"] is False for row in anchors)
    response_guards=[row for row in p["layout"] if row["kind"].endswith("_clock_anchor_response_guard")]
    assert len(response_guards)==6 and all(row["frames"]==CLOCK_ANCHOR_RESPONSE_GUARD for row in response_guards)
    assert p["clock_overlay"]["pilot_band_hz"]==list(PILOT_BAND)
    assert p["clock_overlay"]["hard_max_samples"]==CLOCK_HARD_MAX
    assert p["canonical_training_eligible"] is False
    assert p["canonical_blocker"]=="continuous_electrical_clock_witness_absent_during_exact_zero_tail"
    assert LIVE_AUTHORITY is None and p["live_capture_enabled"] is False

def test_scalar_receipt_cannot_spoof_raw_clock_core(v3_signal):
    assert 413.931e-6*65536 > 27
    parameters=inspect.signature(absolute_dac_q_timewarp_v3).parameters
    assert set(parameters)=={"plan","submitted_pcm","raw_err","raw_ref","callback_time_info"}
    p,x=v3_signal
    try:
        absolute_dac_q_timewarp_v3(plan=p,submitted_pcm=x,
            raw_err=np.zeros(len(x)),raw_ref=np.zeros(len(x)),callback_time_info={})
    except ValueError as exc:
        assert "callback" in str(exc)
    else: raise AssertionError("scalar/callback-free spoof accepted")
    with pytest.raises(ValueError,match="exact.*int16"):
        absolute_dac_q_timewarp_v3(
            plan=p,submitted_pcm=x.astype(np.float64),
            raw_err=np.zeros(len(x)),raw_ref=np.zeros(len(x)),
            callback_time_info=_callbacks(len(x)),
        )

def test_candidate_api_is_fit_only_and_freeze_excludes_holdout():
    assert "holdout" not in inspect.signature(generate_candidate).parameters
    plan,pcm=build_plan(); row=next(r for r in plan["layout"] if r["kind"]=="primary_fit_a_burst"); x=pcm[row["start_frame"]:row["stop_frame"],0]/32767.; taps=np.zeros(1024); taps[:3]=[.8,-.1,.03]; delay=31
    y=linear_fft_convolve(x,np.r_[np.zeros(delay),taps])
    a=generate_candidate(x=x,response=y,delay=delay,support=1024,source_role="fit_a")
    b=generate_candidate(x=x,response=y,delay=delay,support=1024,source_role="fit_b")
    frozen=freeze_fit_candidates(a,b); assert frozen["holdout_used_for_generation_or_selection"] is False and frozen["canonical_training_eligible"] is False
    try: generate_candidate(x=x,response=y,delay=delay,support=1024,source_role="holdout")
    except ValueError: pass
    else: raise AssertionError("holdout leakage")


def test_4096_candidate_uses_toeplitz_operator_without_dense_normal_matrix(v3_signal):
    source=inspect.getsource(generate_candidate)
    assert "eigh(" not in source and "G=toeplitz" not in source
    plan,pcm=v3_signal
    row=next(value for value in plan["layout"] if value["kind"]=="primary_fit_a_burst")
    x=pcm[row["start_frame"]:row["stop_frame"],0]/32767.0
    taps=np.zeros(4096,dtype=np.float64)
    taps[[0,17,4095]]=[0.8,-0.1,0.02]
    delay=31
    response=linear_fft_convolve(x,np.r_[np.zeros(delay),taps])
    candidate=generate_candidate(
        x=x,response=response,delay=delay,support=4096,source_role="fit_a",
    )
    recovered=np.asarray(candidate["post_onset_fir"])
    assert candidate["rank"]==4096
    assert candidate["condition_number"] <= MAX_CONDITION
    assert np.max(np.abs(recovered-taps)) < 1e-10


@pytest.mark.parametrize(
    ("drift_ppm","profile"),
    [
        (413.931,"ordinary"),
        (-413.931,"maximum_delay_and_support"),
        (0.0,"ordinary"),
    ],
)
def test_raw_derived_common_map_recovers_signed_drift_and_known_causal_plants(
    v3_signal,drift_ppm,profile,
):
    plan,pcm=v3_signal
    responses=_plant_output(plan,pcm,profile=profile)
    raw,callbacks=_warp_raw(responses,len(pcm),drift_ppm=drift_ppm)
    receipt=absolute_dac_q_timewarp_v3(
        plan=plan,submitted_pcm=pcm,raw_err=raw[0],raw_ref=raw[1],
        callback_time_info=callbacks,
    )
    end_to_end_error=abs(receipt["slope"]-drift_ppm*1e-6)*len(pcm)
    assert end_to_end_error <= CLOCK_LEAVEOUT_MAX
    assert receipt["leaveout_max_samples"] <= CLOCK_LEAVEOUT_MAX
    assert receipt["cubic_max_samples"] <= CLOCK_CUBIC_MAX
    assert receipt["combined_max_samples"] <= CLOCK_COMBINED_MAX
    assert receipt["highband_used_for_clock_fit"] is False
    assert receipt["clock_fit_receipt"]["holdout_used_for_fit_or_selection"] is False
    assert set(receipt["clock_fit_receipt"]["view_rate_ratios"]) == {
        "err_primary","err_secondary","ref_primary","ref_secondary",
    }
    if profile=="maximum_delay_and_support":
        assert receipt["marker_delay_err"]["primary"]==4800


@pytest.mark.parametrize("trajectory_step",[0.1,1.0])
def test_fractional_offset_and_one_sample_raw_slip_are_rejected(
    v3_signal,trajectory_step,
):
    plan,pcm=v3_signal
    responses=_plant_output(plan,pcm)
    first_guard=next(row for row in plan["layout"] if row["kind"]=="primary_fit_a_guard")
    raw,callbacks=_warp_raw(
        responses,len(pcm),drift_ppm=413.931,step_samples=trajectory_step,
        step_adc=int(first_guard["start_frame"]+10000),
    )
    with pytest.raises(ValueError,match="leaveout|different clock maps|disagreement"):
        absolute_dac_q_timewarp_v3(
            plan=plan,submitted_pcm=pcm,raw_err=raw[0],raw_ref=raw[1],
            callback_time_info=callbacks,
        )


def test_callback_sample_slip_is_rejected_before_clock_fit(v3_signal):
    plan,pcm=v3_signal
    callbacks=_callbacks(len(pcm))
    callbacks["frame_index"]=callbacks["frame_index"].copy()
    callbacks["frame_index"][5:]+=1
    with pytest.raises(ValueError,match="slip"):
        absolute_dac_q_timewarp_v3(
            plan=plan,submitted_pcm=pcm,
            raw_err=np.zeros(len(pcm)),raw_ref=np.zeros(len(pcm)),
            callback_time_info=callbacks,
        )


def test_non_affine_clock_change_across_zero_guard_is_fail_closed(v3_signal):
    plan,pcm=v3_signal
    responses=_plant_output(plan,pcm)
    first_guard=next(row for row in plan["layout"] if row["kind"]=="primary_fit_a_guard")
    raw,callbacks=_warp_piecewise_raw(
        responses,len(pcm),break_adc=int(first_guard["start_frame"]+10000),
        first_ppm=413.931,second_ppm=-300.0,
    )
    with pytest.raises(ValueError,match="different clock maps|leaveout|disagreement"):
        absolute_dac_q_timewarp_v3(
            plan=plan,submitted_pcm=pcm,raw_err=raw[0],raw_ref=raw[1],
            callback_time_info=callbacks,
        )


def test_low_snr_pilot_is_rejected(v3_signal):
    plan,pcm=v3_signal
    rng=np.random.default_rng(20260828)
    raw=[rng.normal(0.0,1e-6,len(pcm)),rng.normal(0.0,1e-6,len(pcm))]
    with pytest.raises(ValueError,match="clock|pilot|boundary"):
        absolute_dac_q_timewarp_v3(
            plan=plan,submitted_pcm=pcm,raw_err=raw[0],raw_ref=raw[1],
            callback_time_info=_callbacks(len(pcm)),
        )


def test_clock_map_is_invariant_to_highband_plant_mutation(v3_signal):
    plan,pcm=v3_signal
    baseline=_plant_output(plan,pcm,highband_mutation=False)
    mutated=_plant_output(plan,pcm,highband_mutation=True)
    raw_a,callbacks_a=_warp_raw(baseline,len(pcm),drift_ppm=413.931)
    raw_b,callbacks_b=_warp_raw(mutated,len(pcm),drift_ppm=413.931)
    first=absolute_dac_q_timewarp_v3(
        plan=plan,submitted_pcm=pcm,raw_err=raw_a[0],raw_ref=raw_a[1],
        callback_time_info=callbacks_a,
    )
    second=absolute_dac_q_timewarp_v3(
        plan=plan,submitted_pcm=pcm,raw_err=raw_b[0],raw_ref=raw_b[1],
        callback_time_info=callbacks_b,
    )
    assert abs(first["slope"]-second["slope"])*len(pcm) <= CLOCK_CUBIC_MAX
    assert first["highband_used_for_clock_fit"] is False
    assert second["highband_used_for_clock_fit"] is False


def test_marker_delay_alias_is_rejected(v3_signal):
    plan,pcm=v3_signal
    responses=_plant_output(plan,pcm)
    signal=pcm.astype(np.float64)/32767.0
    # Add a second equal marker-only arrival.  Burst/anchor pilot evidence stays
    # valid, so rejection must come from the coarse branch gate itself.
    for row in plan["layout"]:
        if not row["kind"].endswith("_4hz_marker"):
            continue
        source=signal[row["start_frame"]:row["stop_frame"],row["output_channel"]]
        path=row["path"]
        for mic,response in enumerate(responses):
            gain={
                (0,"primary"):0.80,(0,"secondary"):0.60,
                (1,"primary"):0.50,(1,"secondary"):0.40,
            }[(mic,path)]
            start=int(row["start_frame"]+({"primary":1700,"secondary":1300}[path])+3000)
            response[start:start+len(source)]+=gain*source
    raw,callbacks=_warp_raw(responses,len(pcm),drift_ppm=413.931)
    with pytest.raises(ValueError,match="marker coarse-delay branch"):
        absolute_dac_q_timewarp_v3(
            plan=plan,submitted_pcm=pcm,raw_err=raw[0],raw_ref=raw[1],
            callback_time_info=callbacks,
        )
