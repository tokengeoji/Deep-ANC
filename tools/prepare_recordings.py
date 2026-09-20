#!/usr/bin/env python3
"""Split synchronized ANC-OFF stereo PCM16 recordings without changing samples."""

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import wave


SAMPLE_RATE = 16000
READ_FRAMES = 65536


def _stereo_info(path, destinations=None):
    """Inspect every PCM frame, optionally copying exact channel bytes to WAVs."""
    hashes = [hashlib.sha256(), hashlib.sha256(), hashlib.sha256()]
    frames = 0
    try:
        with wave.open(str(path), "rb") as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate(),
                    source.getcomptype()) != (2, 2, SAMPLE_RATE, "NONE"):
                raise ValueError(f"Expected stereo PCM16 at {SAMPLE_RATE} Hz: {path}")
            declared_frames = source.getnframes()
            if not declared_frames:
                raise ValueError(f"Empty recording: {path}")
            while True:
                payload = source.readframes(READ_FRAMES)
                if not payload:
                    break
                if len(payload) % 4:
                    raise ValueError(f"Incomplete stereo PCM16 frame: {path}")
                # No conversion or scaling: slicing preserves each sample's bytes.
                samples = array("h")
                samples.frombytes(payload)
                channels = (samples[0::2].tobytes(), samples[1::2].tobytes())
                hashes[0].update(payload)
                for index, channel in enumerate(channels):
                    hashes[index + 1].update(channel)
                    if destinations is not None:
                        destinations[index].writeframesraw(channel)
                frames += len(payload) // 4
            if frames != declared_frames:
                raise ValueError(f"Truncated recording: {path}")
    except (wave.Error, EOFError) as error:
        raise ValueError(f"Invalid WAV {path}: {error}") from error
    return {"frames": frames, "stereo_pcm_sha256": hashes[0].hexdigest(),
            "reference_pcm_sha256": hashes[1].hexdigest(),
            "disturbance_pcm_sha256": hashes[2].hexdigest()}


def prepare_recordings(input_dir, output_dir, *, anc_off=False, raw_pcm=False):
    """Validate the full input before creating a new, never-overwritten output."""
    if not anc_off or not raw_pcm:
        raise ValueError("Explicit --anc-off and --raw-pcm attestations are required")
    input_dir = Path(input_dir).resolve()
    requested_output = Path(output_dir)
    if requested_output.exists() or requested_output.is_symlink():
        raise ValueError(f"Output already exists; refusing overwrite: {requested_output}")
    output_dir = requested_output.resolve()
    if (input_dir == output_dir or input_dir in output_dir.parents
            or output_dir in input_dir.parents):
        raise ValueError("Input and output directories must be disjoint")
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    plan, session_splits, audio_splits, output_names, capture_metadata = [], {}, {}, set(), {}
    for split in ("train", "valid"):
        split_dir = input_dir / split
        if not split_dir.is_dir() or split_dir.is_symlink():
            raise ValueError(f"Both train and valid session directories are required: {split_dir}")
        split_count = 0
        for session_dir in sorted(split_dir.iterdir()):
            if session_dir.is_symlink():
                raise ValueError(f"Symlinked input is not supported: {session_dir}")
            if not session_dir.is_dir():
                if session_dir.suffix.lower() == ".wav":
                    raise ValueError(f"WAV must be inside a session directory: {session_dir}")
                continue
            session = session_dir.name
            session_key = session.casefold()
            if session_key in session_splits and session_splits[session_key] != split:
                raise ValueError(f"Session leakage across train/valid: {session}")
            session_splits[session_key] = split
            sidecar = session_dir / "capture.json"
            if sidecar.exists():
                if sidecar.is_symlink():
                    raise ValueError(f"Symlinked metadata is not supported: {sidecar}")
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict):
                    raise ValueError(f"capture.json must contain a JSON object: {sidecar}")
                capture_metadata[f"{split}/{session}"] = metadata
            for path in sorted(session_dir.iterdir()):
                if path.is_symlink():
                    raise ValueError(f"Symlinked input is not supported: {path}")
                if path.is_dir():
                    raise ValueError(f"Nested session directories are not supported: {path}")
                if path.suffix.lower() != ".wav":
                    continue
                info = _stereo_info(path)
                for key in ("reference_pcm_sha256", "disturbance_pcm_sha256"):
                    digest = info[key]
                    if digest in audio_splits and audio_splits[digest] != split:
                        raise ValueError(f"Identical PCM audio across train/valid: {path}")
                    audio_splits[digest] = split
                relative = Path(split) / session
                reference = relative / f"{path.stem}_reference.wav"
                disturbance = relative / f"{path.stem}_disturbance.wav"
                for destination in (reference, disturbance):
                    name = destination.as_posix().casefold()
                    if name in output_names:
                        raise ValueError(f"Output filename collision: {destination}")
                    output_names.add(name)
                plan.append({"source": path, "reference": reference,
                             "disturbance": disturbance, "split": split,
                             "session": session, "info": info})
                split_count += 1
        if not split_count:
            raise ValueError(f"No WAV recordings in {split_dir}")

    # This exclusive mkdir also refuses an output created during validation.
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest, provenance = [], []
    for item in plan:
        reference = output_dir / item["reference"]
        disturbance = output_dir / item["disturbance"]
        reference.parent.mkdir(parents=True, exist_ok=True)
        with reference.open("xb") as ref_file, disturbance.open("xb") as dist_file:
            with wave.open(ref_file, "wb") as ref_wave, wave.open(dist_file, "wb") as dist_wave:
                for destination in (ref_wave, dist_wave):
                    destination.setnchannels(1)
                    destination.setsampwidth(2)
                    destination.setframerate(SAMPLE_RATE)
                observed = _stereo_info(item["source"], (ref_wave, dist_wave))
        if observed != item["info"]:
            raise ValueError(f"Source changed while preparing: {item['source']}; output is incomplete")
        manifest.append({"reference": item["reference"].as_posix(),
                         "disturbance": item["disturbance"].as_posix(),
                         "split": item["split"], "session": item["session"],
                         "anc_enabled": False, "signal_domain": "raw_pcm"})
        provenance.append({"source": item["source"].relative_to(input_dir).as_posix(),
                           **item["info"]})
    report = {"format_version": 1, "sample_rate": SAMPLE_RATE, "pcm_bits": 16,
              "reference_channel": "left", "disturbance_channel": "right",
              "attested_anc_off": True, "attested_raw_pcm": True,
              "capture_metadata": capture_metadata, "recordings": provenance}
    with (output_dir / "preparation.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    # Write the consumable manifest last; a failed conversion has no ready manifest.
    manifest_path = output_dir / "manifest.jsonl"
    pending_manifest = output_dir / "manifest.jsonl.partial"
    with pending_manifest.open("x", encoding="utf-8") as stream:
        for record in manifest:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    pending_manifest.rename(manifest_path)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/anc/raw"))
    parser.add_argument("--output", type=Path, default=Path("datasets/anc/prepared"))
    parser.add_argument("--anc-off", action="store_true", required=True,
                        help="Attest all recordings were captured with ANC speaker muted")
    parser.add_argument("--raw-pcm", action="store_true", required=True,
                        help="Attest raw ADC PCM, without DC filtering, AGC or normalization")
    args = parser.parse_args()
    try:
        manifest = prepare_recordings(args.input, args.output, anc_off=args.anc_off,
                                      raw_pcm=args.raw_pcm)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Recording preparation failed: {error}\n")
    print(f"Prepared {manifest}; samples, channel gain and synchronization preserved.")


if __name__ == "__main__":
    main()
