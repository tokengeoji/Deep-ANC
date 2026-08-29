"""v3 full-band causal stimulus and raw-derived common-clock evidence."""
from __future__ import annotations
from functools import lru_cache
import hashlib, json, math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import matmul_toeplitz, solve_toeplitz
from scipy.optimize import minimize_scalar
from scipy.signal import butter, fftconvolve, sosfiltfilt
from scipy.sparse.linalg import LinearOperator, eigsh
from .interleaved_probe import schroeder_phases

FS=48000; BLOCK=256; N=65536; GUARD=262144; LEAD=24000
ROLES=(("fit_a",710001),("fit_b",810013),("holdout",910019))
SUPPORTS=(1024,2048,4096); MAX_DELAY=4800; MAX_CONDITION=20.0
PILOT_PERIOD=6000; PILOT_BAND=(152.,600.); PILOT_PEAK_PCM=20; CAUSAL_PEAK_PCM=78
CLOCK_ANCHOR_PERIODS=3
CLOCK_ANCHOR_FRAMES=CLOCK_ANCHOR_PERIODS*PILOT_PERIOD
CLOCK_ANCHOR_RESPONSE_GUARD=MAX_DELAY+max(SUPPORTS)
MARKER_FRAMES=12000; MARKER_TAIL_ONSET=32768
MARKER_GUARD=MAX_DELAY+max(SUPPORTS)+MARKER_TAIL_ONSET
CLOCK_LEAVEOUT_MAX=.050; CLOCK_CUBIC_MAX=.006; CLOCK_COMBINED_MAX=.056
CLOCK_HARD_MAX=.06755189029558946
CLOCK_MAX_ABS_PPM=1000.0
CLOCK_MIN_TRANSFER_COHERENCE=.995
CLOCK_FIT_CYCLES=(2,4,6,8)
CLOCK_VALIDATION_CYCLES=(3,5,7)
LIVE_AUTHORITY=None

def _sha(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()
def _jsha(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def _pilot():
    f=np.fft.rfftfreq(PILOT_PERIOD,1/FS); b=np.flatnonzero((f>=PILOT_BAND[0])&(f<=PILOT_BAND[1]))
    X=np.zeros(PILOT_PERIOD//2+1,complex); X[b]=np.exp(1j*schroeder_phases(len(b)))
    x=np.fft.irfft(X,PILOT_PERIOD); return x/abs(x).max()*PILOT_PEAK_PCM

def _toeplitz_extremal(ac,support):
    ac=np.asarray(ac,dtype=np.float64).reshape(-1)
    if ac.size < support:
        raise ValueError("Toeplitz autocorrelation is shorter than support")
    # Exact symmetric Toeplitz operator; extremal iterative eigensolve avoids allocating
    # and repeatedly diagonalising a 4096x4096 dense matrix during plan validation.
    op=LinearOperator((support,support),matvec=lambda v:matmul_toeplitz((ac,ac),v),dtype=np.float64)
    v0=np.full(support,1.0/math.sqrt(support),dtype=np.float64)
    hi=float(eigsh(op,k=1,which="LA",return_eigenvectors=False,tol=1e-10,v0=v0)[0])
    lo=float(eigsh(op,k=1,which="SA",return_eigenvectors=False,tol=1e-10,v0=v0)[0])
    return lo,hi

def _condition(x,support):
    ac=fftconvolve(x,x[::-1],mode="full")[len(x)-1:len(x)-1+support]
    lo,hi=_toeplitz_extremal(ac,support)
    return math.inf if lo<=0 else hi/lo

@lru_cache(None)
def _burst_cached(seed):
    # Both components live on the same 6000-sample (8 Hz) grid.  Consequently every
    # complete analysis window has an exact spectral partition; merely notching one
    # 65536-point FFT would leak causal energy into every 6000-point pilot estimate.
    m=PILOT_PERIOD//2-1; base=schroeder_phases(m); rnd=np.random.default_rng(seed).uniform(0,2*np.pi,m)
    f=np.fft.rfftfreq(PILOT_PERIOD,1/FS); reserved=(f>=PILOT_BAND[0])&(f<=PILOT_BAND[1]); best=None
    for a in np.linspace(0,.3,61):
        X=np.zeros(PILOT_PERIOD//2+1,complex); X[1:-1]=np.exp(1j*((1-a)*base+a*rnd)); X[reserved]=0
        x=np.fft.irfft(X,PILOT_PERIOD); cf=20*np.log10(abs(x).max()/np.sqrt(np.mean(x*x)))
        if best is None or abs(cf-6.5)<best[0]: best=(abs(cf-6.5),x)
    causal_period=best[1]/abs(best[1]).max()*CAUSAL_PEAK_PCM; pilot=_pilot(); overlay=np.resize(pilot,N)
    causal=np.resize(causal_period,N); windows=N//PILOT_PERIOD
    q=np.rint(causal+overlay).astype(np.int16); z=q/32767.; cond={str(s):_condition(z,s) for s in SUPPORTS}
    if max(cond.values())>MAX_CONDITION: raise RuntimeError(f"Toeplitz condition gate: {cond}")
    meta={"frames":N,"seed":seed,"peak_pcm":int(abs(q.astype(np.int32)).max()),"rms":float(np.sqrt(np.mean(z*z))),
          "causal_component_peak_pcm":CAUSAL_PEAK_PCM,"pilot_component_peak_pcm":PILOT_PEAK_PCM,
          "pilot_period_samples":PILOT_PERIOD,"pilot_complete_windows":windows,"pilot_band_hz":list(PILOT_BAND),
          "causal_reserved_band_max_abs_dft":float(abs(np.fft.rfft(causal_period)[reserved]).max()),
          "pilot_quantisation_snr_proxy_db":float(20*np.log10(np.sqrt(np.mean(pilot*pilot))/(1/math.sqrt(12)))),
          "toeplitz_condition_by_support":cond,"pilot_snr_requires_live_validation":True,"pcm_sha256":_sha(q)}
    q.setflags(write=False); return q,meta

def _burst(seed):
    q,m=_burst_cached(seed); return q.copy(),dict(m)

def build_plan():
    probes={r:_burst(s) for r,s in ROLES}; parts=[]; layout=[]; cur=0
    def add(kind,a,**kw):
        nonlocal cur; parts.append(a); layout.append({"kind":kind,"start_frame":cur,"stop_frame":cur+len(a),"frames":len(a),**kw}); cur+=len(a)
    add("lead_silence",np.zeros((LEAD,2),np.int16))
    for path,ch,seed in (("primary",0,111),("secondary",1,211)):
        rng=np.random.default_rng(seed); X=np.zeros(MARKER_FRAMES//2+1,complex); X[25:2829]=np.exp(1j*rng.uniform(0,2*np.pi,2804))
        marker=np.fft.irfft(X,MARKER_FRAMES); marker=np.rint(marker/abs(marker).max()*98).astype(np.int16); slot=np.zeros((MARKER_FRAMES,2),np.int16); slot[:,ch]=marker
        add(f"{path}_4hz_marker",slot,path=path,output_channel=ch,delay_search_samples=[0,MAX_DELAY],maximum_branch_width_samples=2999,marker_pcm_sha256=_sha(marker))
        add(f"{path}_marker_guard",np.zeros((MARKER_GUARD,2),np.int16),path=path,minimum_tail_free_samples=MARKER_TAIL_ONSET,guard_derivation="max_delay+max_support+tail_onset")
    for role,_ in ROLES:
        for path,ch in (("primary",0),("secondary",1)):
            a=np.zeros((N,2),np.int16); a[:,ch]=probes[role][0]
            add(f"{path}_{role}_burst",a,path=path,role=role,output_channel=ch)
            add(
                f"{path}_{role}_guard",np.zeros((GUARD,2),np.int16),
                path=path,role=role,exact_zero_candidate_observation=True,
            )
            # The 5.46 s exact-zero tail must remain untouched.  A separate
            # low-band anchor follows it so a slip or integrated drift hidden
            # in that silent interval cannot silently move the next burst.
            anchor=np.zeros((CLOCK_ANCHOR_FRAMES,2),np.int16)
            anchor[:,ch]=np.resize(np.rint(_pilot()).astype(np.int16),CLOCK_ANCHOR_FRAMES)
            add(
                f"{path}_{role}_clock_anchor",anchor,path=path,role=role,
                output_channel=ch,candidate_data=False,
                analysis_cycle=CLOCK_ANCHOR_PERIODS-1,
            )
            add(
                f"{path}_{role}_clock_anchor_response_guard",
                np.zeros((CLOCK_ANCHOR_RESPONSE_GUARD,2),np.int16),
                path=path,role=role,candidate_data=False,
            )
    out=np.concatenate(parts); pad=(-len(out))%BLOCK
    if pad: add("padding",np.zeros((pad,2),np.int16)); out=np.concatenate(parts)
    plan={"schema":"fullband_causal_reserved_pilot_v3","role":"signal_only_dry_run_no_audio","live_capture_enabled":False,"canonical_training_eligible":False,"canonical_blocker":"continuous_electrical_clock_witness_absent_during_exact_zero_tail","bursts":{r:m for r,(_,m) in probes.items()},"layout":layout,
      "clock_overlay":{"pilot_period_samples":PILOT_PERIOD,"pilot_band_hz":list(PILOT_BAND),"causal_component_exactly_zero_in_pilot_band":True,"highband_clock_fit_forbidden":True,"fit_windows":"fit-role even; odd, holdout, and post-zero-tail anchors validation-only","shared_ps_err_ref_map_required":True,"leaveout_max_samples":CLOCK_LEAVEOUT_MAX,"cubic_max_samples":CLOCK_CUBIC_MAX,"combined_max_samples":CLOCK_COMBINED_MAX,"hard_max_samples":CLOCK_HARD_MAX,"callback_role":"monotonic_and_slip_witness_only","actual_total_int16_is_denominator":True,"post_zero_tail_anchor_periods":CLOCK_ANCHOR_PERIODS,"post_zero_tail_anchor_frames":CLOCK_ANCHOR_FRAMES,"post_anchor_response_guard_samples":CLOCK_ANCHOR_RESPONSE_GUARD,"silent_gap_non_affine_clock_authority":"blocked_without_electrical_loopback"},
      "memory":{"burst_guard_samples":GUARD,"marker_guard_samples":MARKER_GUARD,"maximum_physical_memory_samples":GUARD-MAX_DELAY,"finite_memory_assumption_receipt_required":True,"infinite_tail_claim":False,"clock_anchor_excluded_from_candidate_data":True,"full_exact_zero_guard_precedes_clock_anchor":True},
      "identifiability":{"actual_linear_toeplitz_condition_receipt_required_for_each_support":list(SUPPORTS),"hard_max_condition":MAX_CONDITION,"circulant_condition_is_not_toeplitz_proof":True},
      "output":{"frames":len(out),"duration_seconds":len(out)/FS,"peak_pcm":int(abs(out.astype(np.int32)).max()),"pcm_sha256":_sha(out)}}
    return plan,out

def _delay(ratio,f,expected,width):
    ok=np.isfinite(ratio)&(abs(ratio)>0); ratio=ratio[ok]/abs(ratio[ok]); f=f[ok]
    if len(f)<8: raise ValueError("pilot SNR/bin count insufficient")
    def loss(t): return -abs(np.sum(ratio*np.exp(2j*np.pi*f*t/FS)))/len(f)
    z=minimize_scalar(loss,bounds=(expected-width,expected+width),method="bounded",options={"xatol":1e-9})
    return float(z.x),float(-z.fun)

def _callbacks(c,n):
    keys=("frame_index","input_adc_time","output_dac_time","frame_count")
    if any(k not in c for k in keys): raise ValueError("callback time_info raw arrays missing")
    frame=np.asarray(c[keys[0]],np.int64); adc=np.asarray(c[keys[1]],float); dac=np.asarray(c[keys[2]],float); count=np.asarray(c[keys[3]],np.int64)
    if not(len(frame)==len(adc)==len(dac)==len(count) and len(frame)>=2): raise ValueError("callback witness shape")
    if frame[0]!=0 or np.any(np.diff(frame)!=count[:-1]) or np.any(count<=0) or np.any(np.diff(adc)<=0) or np.any(np.diff(dac)<=0) or frame[-1]+count[-1]<n: raise ValueError("callback slip/non-monotonic/coverage")
    p={"frame_index_sha256":_sha(frame),"input_adc_time_sha256":_sha(adc),"output_dac_time_sha256":_sha(dac),"frame_count_sha256":_sha(count),"role":"monotonic_and_slip_witness_only"}; p["sha256"]=_jsha(p); return p

def _markers(plan,submitted,raw):
    ans={}; baseline=np.median(abs(raw[:LEAD]))+1e-9
    for row in plan["layout"]:
        if not row["kind"].endswith("_4hz_marker"): continue
        marker=submitted[row["start_frame"]:row["stop_frame"],row["output_channel"]].astype(float); search=raw[row["start_frame"]:row["stop_frame"]+MAX_DELAY]
        corr=fftconvolve(search,marker[::-1],mode="valid"); order=np.argsort(abs(corr))[::-1]; peak=int(order[0]); aliases=[int(i) for i in order[1:] if abs(int(i)-peak)>=32 and abs(corr[i])>=.98*abs(corr[peak])]
        if aliases: raise ValueError("marker coarse-delay branch is not unique")
        ans[row["path"]]=peak; guard=next(x for x in plan["layout"] if x["kind"]==f"{row['path']}_marker_guard"); tail=raw[guard["stop_frame"]-MARKER_TAIL_ONSET:guard["stop_frame"]]
        if np.sqrt(np.mean(tail*tail))>max(8*baseline,.001*abs(search).max()): raise ValueError("marker P/S tail cross-contamination")
    if set(ans)!={"primary","secondary"}: raise ValueError("both markers required")
    return ans

def _clock_rows(plan):
    rows=[row for row in plan["layout"] if row["kind"].endswith("_burst")]
    expected={(path,role) for path in ("primary","secondary") for role,_ in ROLES}
    if {(row.get("path"),row.get("role")) for row in rows} != expected:
        raise ValueError("P/S fit_a/fit_b/holdout burst layout is incomplete")
    return rows


def _clock_anchors(plan):
    rows=[row for row in plan["layout"] if row["kind"].endswith("_clock_anchor")]
    expected={(path,role) for path in ("primary","secondary") for role,_ in ROLES}
    if {(row.get("path"),row.get("role")) for row in rows} != expected:
        raise ValueError("post-zero-tail clock anchor layout is incomplete")
    if any(bool(row.get("candidate_data",True)) for row in rows):
        raise ValueError("clock anchor must be excluded from causal candidate data")
    return rows


def _clock_isolate(raw):
    """Pilot-only view fixed before seeing a measured high-band response."""

    signal=np.asarray(raw,dtype=np.float64).reshape(-1)
    if signal.size < N or not np.all(np.isfinite(signal)):
        raise ValueError("clock raw is short or non-finite")
    sos=butter(12,(120.,680.),btype="bandpass",fs=FS,output="sos")
    return sosfiltfilt(sos,signal)


def _interpolate(signal,coordinates,method,spline=None):
    coordinates=np.asarray(coordinates,dtype=np.float64)
    if coordinates.size == 0 or coordinates[0] < 0.0 or coordinates[-1] > len(signal)-1:
        raise ValueError("clock interpolation support missing")
    if method == "linear":
        left=np.floor(coordinates).astype(np.int64)
        right=np.minimum(left+1,len(signal)-1)
        fraction=coordinates-left
        return signal[left]*(1.0-fraction)+signal[right]*fraction
    if method == "cubic":
        if spline is None:
            spline=CubicSpline(np.arange(len(signal),dtype=np.float64),signal,extrapolate=False)
        values=np.asarray(spline(coordinates),dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("cubic clock interpolation produced non-finite values")
        return values
    raise ValueError(f"unknown interpolation method: {method}")


def _periodicity_objective(
    rate_ratio,
    *,
    records,
    method,
    splines=None,
):
    """Profile out the unknown path response and score only pilot periodicity.

    ``rate_ratio`` is DAC-q samples per ADC sample.  Every comparison is made
    after mapping the same *submitted DAC coordinates* into ADC coordinates;
    therefore the 2.48-sample dilation inside a 6000-sample window is not
    approximated as a rigid delay.
    """

    a=float(rate_ratio)
    if not 1.0-CLOCK_MAX_ABS_PPM*1e-6 <= a <= 1.0+CLOCK_MAX_ABS_PPM*1e-6:
        return math.inf
    # Every eighth sample is sufficient for a <=600 Hz waveform at 48 kHz and
    # keeps the repeated independent view fits cheap enough for a live-offline
    # publisher.  The final transfer validation still uses all 6000 samples.
    phase=np.arange(256,PILOT_PERIOD-256,8,dtype=np.float64)
    numerator=0.0; denominator=0.0
    for record_index,(signal,row) in enumerate(records):
        cycles=[]
        spline=None if splines is None else splines[record_index]
        for cycle in CLOCK_FIT_CYCLES:
            q=float(row["start_frame"]+cycle*PILOT_PERIOD)+phase
            cycles.append(_interpolate(signal,q/a,method,spline))
        values=np.stack(cycles)
        values-=np.mean(values,axis=1,keepdims=True)
        template=np.mean(values,axis=0)
        numerator+=float(np.sum((values-template[None,:])**2))
        denominator+=float(np.sum(values**2))
    if denominator <= 1e-30:
        return math.inf
    return numerator/denominator


def _estimate_rate_ratio(records,*,method):
    if not records:
        raise ValueError("clock fit records are empty")
    splines=None
    if method == "cubic":
        cache={}
        splines=[]
        for signal,_ in records:
            key=id(signal)
            if key not in cache:
                cache[key]=CubicSpline(
                    np.arange(len(signal),dtype=np.float64),signal,extrapolate=False,
                )
            splines.append(cache[key])
    lo=1.0-CLOCK_MAX_ABS_PPM*1e-6; hi=1.0+CLOCK_MAX_ABS_PPM*1e-6
    grid=np.linspace(lo,hi,9)
    scores=np.asarray([_periodicity_objective(x,records=records,method=method,splines=splines) for x in grid])
    if not np.all(np.isfinite(scores)):
        raise ValueError("pilot periodicity objective is non-finite")
    best=int(np.argmin(scores)); left=grid[max(0,best-1)]; right=grid[min(len(grid)-1,best+1)]
    if best in (0,len(grid)-1):
        raise ValueError("clock rate hit the search boundary")
    result=minimize_scalar(
        lambda value:_periodicity_objective(value,records=records,method=method,splines=splines),
        bounds=(float(left),float(right)),method="bounded",
        options={"xatol":1e-13,"maxiter":100},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise ValueError("pilot periodicity optimization failed")
    return float(result.x),float(result.fun)


def _pilot_bins():
    frequency=np.fft.rfftfreq(PILOT_PERIOD,1/FS)
    bins=np.flatnonzero((frequency>=PILOT_BAND[0])&(frequency<=PILOT_BAND[1]))
    if bins.size < 8:
        raise AssertionError("reserved pilot has fewer than eight bins")
    return frequency[bins],bins


def _cycle_transfer(*,signal,submitted,row,cycle,rate_ratio,method,spline=None):
    start=int(row["start_frame"]+cycle*PILOT_PERIOD); stop=start+PILOT_PERIOD
    if stop > int(row["stop_frame"]):
        raise ValueError("pilot cycle exceeds its burst")
    frequency,bins=_pilot_bins()
    q=np.arange(start,stop,dtype=np.float64)
    response=_interpolate(signal,q/float(rate_ratio),method,spline)
    channel=int(row["output_channel"])
    active=np.asarray(submitted[start:stop,channel],dtype=np.float64)
    idle=np.asarray(submitted[start:stop,1-channel],dtype=np.float64)
    denominator=np.fft.rfft(active)[bins]
    opposite=np.fft.rfft(idle)[bins]
    if np.min(np.abs(denominator)) < 1e-8:
        raise ValueError("actual submitted int16 pilot denominator missing")
    if float(np.max(np.abs(opposite))) > 1e-12:
        raise ValueError("actual submitted pilot is present on the opposite DAC channel")
    return np.fft.rfft(response)[bins]/denominator,frequency,_sha(denominator)


def _compare_transfer(reference,candidate,frequency):
    reference=np.asarray(reference,dtype=np.complex128); candidate=np.asarray(candidate,dtype=np.complex128)
    floor=max(float(np.max(np.abs(reference))),float(np.max(np.abs(candidate))))*1e-8
    mask=(np.abs(reference)>floor)&(np.abs(candidate)>floor)
    if int(mask.sum()) < 8:
        raise ValueError("pilot low SNR/bin count insufficient")
    ratio=candidate[mask]/reference[mask]
    delay,phase_score=_delay(ratio,frequency[mask],0.0,1.0)
    corrected=candidate[mask]*np.exp(2j*np.pi*frequency[mask]*delay/FS)
    coherence=float(abs(np.vdot(reference[mask],corrected))/(np.linalg.norm(reference[mask])*np.linalg.norm(corrected)+1e-30))
    if phase_score < CLOCK_MIN_TRANSFER_COHERENCE or coherence < CLOCK_MIN_TRANSFER_COHERENCE:
        raise ValueError(
            "pilot low SNR/coherence: "
            f"phase={phase_score:.9f}, complex={coherence:.9f}"
        )
    return abs(float(delay)),float(min(phase_score,coherence))


def _validate_map(*,plan,submitted,isolated,rate_ratio,method):
    """Even fit cycles are nuisance templates; odd and holdout stay validation-only."""

    rows=_clock_rows(plan); anchors=_clock_anchors(plan)
    residuals=[]; coherences=[]; spectra=[]; means={}
    for mic_name,signal in isolated.items():
        spline=CubicSpline(np.arange(len(signal),dtype=np.float64),signal,extrapolate=False) if method=="cubic" else None
        for row in rows:
            even=[]
            for cycle in CLOCK_FIT_CYCLES:
                transfer,frequency,digest=_cycle_transfer(
                    signal=signal,submitted=submitted,row=row,cycle=cycle,
                    rate_ratio=rate_ratio,method=method,spline=spline,
                )
                even.append(transfer); spectra.append((row["path"],row["role"],cycle,digest))
            reference=np.mean(np.stack(even),axis=0)
            means[(mic_name,row["path"],row["role"])]=reference
            for cycle in CLOCK_VALIDATION_CYCLES:
                transfer,_,digest=_cycle_transfer(
                    signal=signal,submitted=submitted,row=row,cycle=cycle,
                    rate_ratio=rate_ratio,method=method,spline=spline,
                )
                residual,coherence=_compare_transfer(reference,transfer,frequency)
                residuals.append(residual); coherences.append(coherence)
                spectra.append((row["path"],row["role"],cycle,digest))
            anchor_row=next(
                value for value in anchors
                if value["path"]==row["path"] and value["role"]==row["role"]
            )
            anchor_cycle=int(anchor_row["analysis_cycle"])
            anchor_transfer,_,anchor_digest=_cycle_transfer(
                signal=signal,submitted=submitted,row=anchor_row,cycle=anchor_cycle,
                rate_ratio=rate_ratio,method=method,spline=spline,
            )
            residual,coherence=_compare_transfer(reference,anchor_transfer,frequency)
            residuals.append(residual); coherences.append(coherence)
            spectra.append((row["path"],f"{row['role']}_post_guard_anchor",anchor_cycle,anchor_digest))
    # A role-dependent 0.1-sample jump or a sample slip is invisible to local
    # periodicity, but not to the time-invariant plant transfer.  Only fit_a is
    # the anchor; fit_b and holdout are never allowed to shift the frozen map.
    for mic_name in isolated:
        for path in ("primary","secondary"):
            anchor=means[(mic_name,path,"fit_a")]
            for role in ("fit_b","holdout"):
                residual,coherence=_compare_transfer(
                    anchor,means[(mic_name,path,role)],frequency,
                )
                residuals.append(residual); coherences.append(coherence)
    maximum=float(max(residuals)); minimum=float(min(coherences))
    return {
        "maximum_residual_samples":maximum,
        "minimum_transfer_coherence":minimum,
        "submitted_pilot_spectra_sha256":_jsha(spectra),
        "holdout_used_for_fit_or_selection":False,
    }


def _fit_common(plan,submitted,err,ref):
    rows=_clock_rows(plan)
    isolated={"err":_clock_isolate(err),"ref":_clock_isolate(ref)}
    # Four independently fitted views must agree before they may be collapsed
    # into the one common ADC->DAC-q map.
    view_ratios={}; view_objectives={}
    for mic_name,signal in isolated.items():
        for path in ("primary","secondary"):
            records=[(signal,row) for row in rows if row["path"]==path and row["role"] in ("fit_a","fit_b")]
            ratio,objective=_estimate_rate_ratio(records,method="linear")
            view_ratios[f"{mic_name}_{path}"]=ratio; view_objectives[f"{mic_name}_{path}"]=objective
    disagreement=(max(view_ratios.values())-min(view_ratios.values()))*len(submitted)
    if disagreement > CLOCK_LEAVEOUT_MAX:
        raise ValueError(f"ERR/REF/P/S different clock maps: {disagreement}")
    records=[
        (signal,row)
        for signal in isolated.values()
        for row in rows
        if row["role"] in ("fit_a","fit_b")
    ]
    linear_ratio,linear_objective=_estimate_rate_ratio(records,method="linear")
    cubic_ratio,cubic_objective=_estimate_rate_ratio(records,method="cubic")
    cubic_difference=abs(linear_ratio-cubic_ratio)*len(submitted)
    if cubic_difference > CLOCK_CUBIC_MAX:
        raise ValueError(f"linear/cubic clock-map disagreement {cubic_difference}")
    linear_validation=_validate_map(
        plan=plan,submitted=submitted,isolated=isolated,
        rate_ratio=linear_ratio,method="linear",
    )
    cubic_validation=_validate_map(
        plan=plan,submitted=submitted,isolated=isolated,
        rate_ratio=linear_ratio,method="cubic",
    )
    leaveout=float(linear_validation["maximum_residual_samples"])
    cubic=max(
        cubic_difference,
        abs(
            float(cubic_validation["maximum_residual_samples"])
            -float(linear_validation["maximum_residual_samples"])
        ),
    )
    combined=leaveout+cubic
    if leaveout>CLOCK_LEAVEOUT_MAX:
        raise ValueError(f"pilot leaveout residual {leaveout}")
    if cubic>CLOCK_CUBIC_MAX or combined>CLOCK_COMBINED_MAX or combined>CLOCK_HARD_MAX:
        raise ValueError(
            f"20dB raw-derived timewarp budget cubic={cubic} combined={combined}"
        )
    receipt={
        "rate_ratio_dac_q_per_adc_sample":linear_ratio,
        "linear_objective":linear_objective,
        "cubic_objective":cubic_objective,
        "view_rate_ratios":view_ratios,
        "view_objectives":view_objectives,
        "view_end_to_end_disagreement_samples":disagreement,
        "leaveout_max_samples":leaveout,
        "cubic_max_samples":cubic,
        "combined_max_samples":combined,
        "minimum_transfer_coherence":min(
            float(linear_validation["minimum_transfer_coherence"]),
            float(cubic_validation["minimum_transfer_coherence"]),
        ),
        "submitted_pilot_spectra_sha256":linear_validation["submitted_pilot_spectra_sha256"],
        "fit_cycles":list(CLOCK_FIT_CYCLES),
        "validation_cycles":list(CLOCK_VALIDATION_CYCLES),
        "holdout_used_for_fit_or_selection":False,
    }
    receipt["sha256"]=_jsha(receipt)
    return linear_ratio,isolated,receipt

def absolute_dac_q_timewarp_v3(*,plan,submitted_pcm,raw_err,raw_ref,callback_time_info):
    submitted=np.asarray(submitted_pcm)
    if submitted.dtype != np.int16 or submitted.ndim != 2 or submitted.shape[1] != 2:
        raise ValueError("actual submitted PCM must be exact [frames,2] int16")
    submitted=np.ascontiguousarray(submitted)
    err=np.asarray(raw_err,float).reshape(-1); ref=np.asarray(raw_ref,float).reshape(-1)
    if plan.get("schema")!="fullband_causal_reserved_pilot_v3" or _sha(submitted)!=plan["output"]["pcm_sha256"]: raise ValueError("plan/PCM lineage mismatch")
    if min(len(err),len(ref))<len(submitted): raise ValueError("raw shorter than plan")
    cb=_callbacks(callback_time_info,max(len(err),len(ref)))
    rate_ratio,_,clock_receipt=_fit_common(plan,submitted,err,ref)
    slope=rate_ratio-1.0
    adc_at_dac=np.arange(len(submitted),dtype=float)/rate_ratio
    if adc_at_dac[-1]>min(len(err),len(ref))-1: raise ValueError("raw interpolation support missing")
    er=CubicSpline(np.arange(len(err)),err)(adc_at_dac); rf=CubicSpline(np.arange(len(ref)),ref)(adc_at_dac)
    # Marker delays are evaluated only after the pilot-derived map is frozen.
    # This keeps a +/- clock drift from moving a true 0..4800 branch outside the
    # nominal correlation window.
    me=_markers(plan,submitted,er); mr=_markers(plan,submitted,rf)
    adc_knots=np.array([0.0,(len(submitted)-1)/rate_ratio],dtype=np.float64)
    dac_knots=np.array([0.0,len(submitted)-1],dtype=np.float64)
    payload={
        "schema":"absolute_dac_q_timewarp_v3",
        "adc_knots_sha256":_sha(adc_knots),
        "dac_knots_sha256":_sha(dac_knots),
        "slope":slope,
        "rate_ratio_dac_q_per_adc_sample":rate_ratio,
        "intercept":0.0,
        "intercept_semantics":"duplex_callback_origin_q0_equals_adc0",
        "fit_band_hz":list(PILOT_BAND),
        "highband_used_for_clock_fit":False,
        "clock_fit_receipt_sha256":clock_receipt["sha256"],
    }
    payload["map_sha256"]=_jsha(payload)
    return {
        **payload,
        "adc_knots":adc_knots,
        "dac_knots":dac_knots,
        "leaveout_max_samples":clock_receipt["leaveout_max_samples"],
        "cubic_max_samples":clock_receipt["cubic_max_samples"],
        "combined_max_samples":clock_receipt["combined_max_samples"],
        "clock_fit_receipt":clock_receipt,
        "marker_delay_err":me,
        "marker_delay_ref":mr,
        "callback_witness":cb,
        "submitted_pcm_sha256":_sha(submitted),
        "raw_err_sha256":_sha(err),
        "raw_ref_sha256":_sha(ref),
        "resampled_err":er,
        "resampled_ref":rf,
        "resampled_err_sha256":_sha(er),
        "resampled_ref_sha256":_sha(rf),
        "passed":True,
    }

def generate_candidate(*,x,response,delay:int,support:int,source_role:str):
    x=np.asarray(x,float).reshape(-1); y=np.asarray(response,float).reshape(-1)
    if source_role not in ("fit_a","fit_b") or support not in SUPPORTS or not 0<=delay<=MAX_DELAY: raise ValueError("fit-only role/delay/support 위반")
    if y.size<x.size+delay+support-1: raise ValueError("linear response tail가 부족합니다")
    ac=fftconvolve(x,x[::-1],mode="full")[x.size-1:x.size-1+support]
    # valid[k] == dot(x, y[k:k+len(x)]), without the former O(len(x)*support)
    # Python loop.  Only the Toeplitz vectors are materialized; never a dense
    # 4096x4096 normal matrix.
    correlation=fftconvolve(y,x[::-1],mode="valid")
    rhs=np.asarray(correlation[delay:delay+support],dtype=np.float64)
    if rhs.size != support:
        raise ValueError("linear response correlation tail가 부족합니다")
    lo,hi=_toeplitz_extremal(ac,support); cond=math.inf if lo<=0 else hi/lo
    if lo<=0 or cond>MAX_CONDITION: raise ValueError(f"linear Toeplitz condition {cond} > {MAX_CONDITION}")
    taps=solve_toeplitz((ac,ac),rhs); p={"schema":"fit_only_integer_delay_post_onset_fir_v3","source_role":source_role,"input_sha256":_sha(x),"response_sha256":_sha(y),"solver":"exact_unregularized_toeplitz_normal_equations","ridge":0.,"rank":support,"condition_number":cond,"integer_delay_samples":delay,"post_onset_support_samples":support,"coefficient_l2_norm":float(np.linalg.norm(taps)),"post_onset_fir":taps.tolist()}; p["candidate_sha256"]=_jsha(p); return p

def freeze_fit_candidates(a,b):
    if {a.get("source_role"),b.get("source_role")}!={"fit_a","fit_b"}: raise ValueError("fit A/B만 freeze 가능")
    if (a["integer_delay_samples"],a["post_onset_support_samples"])!=(b["integer_delay_samples"],b["post_onset_support_samples"]): raise ValueError("fit timing/support mismatch")
    ta=np.asarray(a["post_onset_fir"]); tb=np.asarray(b["post_onset_fir"]); rel=float(np.linalg.norm(ta-tb)/max(np.linalg.norm(ta),1e-30))
    if rel>.1: raise ValueError("fit A/B disagreement")
    p={"schema":"frozen_fit_only_causal_candidate_v3","fit_candidate_sha256":[a["candidate_sha256"],b["candidate_sha256"]],"integer_delay_samples":a["integer_delay_samples"],"post_onset_support_samples":a["post_onset_support_samples"],"post_onset_fir":((ta+tb)/2).tolist(),"fit_relative_disagreement":rel,"holdout_used_for_generation_or_selection":False,"canonical_training_eligible":False}; p["freeze_sha256"]=_jsha(p); return p
