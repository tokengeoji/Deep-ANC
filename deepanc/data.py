"""Paired ANC recordings in fixed ADC/DAC digital units, without normalization."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Recording:
    reference: Path
    disturbance: Path
    split: str
    session: str
    frames: int


def _wav_info(path):
    try:
        with wave.open(str(path), "rb") as stream:
            if stream.getnchannels() != 1:
                raise ValueError(f"Expected mono WAV: {path}")
            if stream.getsampwidth() not in (2, 3, 4) or stream.getcomptype() != "NONE":
                raise ValueError(f"Expected uncompressed PCM16/24/32 WAV: {path}")
            if stream.getnframes() == 0:
                raise ValueError(f"Empty recording: {path}")
            return stream.getframerate(), stream.getnframes()
    except (wave.Error, EOFError) as error:
        raise ValueError(f"Invalid PCM WAV {path}: {error}") from error


def read_pcm(path, start, count):
    """Read a segment; divide by PCM full scale only, preserving relative gain."""
    with wave.open(str(path), "rb") as stream:
        width = stream.getsampwidth()
        stream.setpos(start)
        raw = stream.readframes(count)
    if len(raw) != count * width:
        raise ValueError(f"Truncated recording or requested range exceeds WAV frames: {path}")
    if width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        values = (values ^ 0x800000) - 0x800000
    else:
        values = np.frombuffer(raw, dtype="<i2" if width == 2 else "<i4")
    return values.astype(np.float32) / float(1 << (8 * width - 1))


def pcm_sha256(path):
    """Fingerprint sample payload, independent of filename and WAV metadata."""
    digest = hashlib.sha256()
    with wave.open(str(path), "rb") as stream:
        digest.update(str((stream.getframerate(), stream.getsampwidth())).encode())
        expected_bytes = stream.getnframes() * stream.getsampwidth() * stream.getnchannels()
        observed_bytes = 0
        for payload in iter(lambda: stream.readframes(65536), b""):
            digest.update(payload)
            observed_bytes += len(payload)
        if observed_bytes != expected_bytes:
            raise ValueError(f"Truncated recording: {path}")
    return digest.hexdigest()


def load_manifest(path, sample_rate):
    """Validate all sessions before constructing either split.

    Each line has reference, disturbance, split (train/valid), session,
    anc_enabled=false, and signal_domain="raw_pcm".
    Both channels must be simultaneously recorded, with ANC OFF. Filenames
    cannot prove these conditions; the acquisition procedure must ensure them.
    """
    path = Path(path).resolve()
    records, sessions, file_splits, pairs = [], {}, {}, set()
    sample_hashes, hash_splits = {}, {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            reference = (path.parent / item["reference"]).resolve()
            disturbance = (path.parent / item["disturbance"]).resolve()
            split, session = item["split"], item["session"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid manifest line {line_number}: {error}") from error
        if split not in ("train", "valid"):
            raise ValueError(f"Line {line_number}: split must be train or valid")
        if not isinstance(session, str) or not session.strip():
            raise ValueError(f"Line {line_number}: session must be a nonempty string")
        if item.get("anc_enabled") is not False:
            raise ValueError(f"Line {line_number}: explicitly declare anc_enabled=false (ANC OFF)")
        if item.get("signal_domain") != "raw_pcm":
            raise ValueError(f"Line {line_number}: signal_domain must be raw_pcm; no filtering/normalization")
        if session in sessions and sessions[session] != split:
            raise ValueError(f"Session leakage across train/valid: {session}")
        sessions[session] = split
        if reference == disturbance:
            raise ValueError(f"Reference and disturbance must be separate WAVs: {reference}")
        if (reference, disturbance) in pairs:
            raise ValueError(f"Duplicate recording pair on line {line_number}")
        pairs.add((reference, disturbance))
        for recording_path in (reference, disturbance):
            if recording_path in file_splits and file_splits[recording_path] != split:
                raise ValueError(f"Recording leakage across train/valid: {recording_path}")
            file_splits[recording_path] = split
        ref_rate, ref_frames = _wav_info(reference)
        dist_rate, dist_frames = _wav_info(disturbance)
        if ref_rate != sample_rate or dist_rate != sample_rate:
            raise ValueError(f"Line {line_number}: both WAVs must be {sample_rate} Hz; resampling is not implicit")
        if ref_frames != dist_frames:
            raise ValueError(f"Line {line_number}: paired WAV lengths differ; preserve synchronization")
        for recording_path in (reference, disturbance):
            if recording_path not in sample_hashes:
                sample_hashes[recording_path] = pcm_sha256(recording_path)
            digest = sample_hashes[recording_path]
            if digest in hash_splits and hash_splits[digest] != split:
                raise ValueError(f"Audio-content leakage across train/valid: {recording_path}")
            hash_splits[digest] = split
        records.append(Recording(reference, disturbance, split, session, ref_frames))
    if {record.split for record in records} != {"train", "valid"}:
        raise ValueError("Manifest needs separate train and valid sessions")
    return records


class PairedWaveDataset(Dataset):
    """Read bounded chunks with enough true history for both controller and FIR."""

    def __init__(self, records, split, chunk_samples, context_samples):
        if chunk_samples < 1 or context_samples < 0:
            raise ValueError("chunk_samples must be positive and context_samples nonnegative")
        self.records = [record for record in records if record.split == split]
        self.chunk_samples = chunk_samples
        self.context_samples = context_samples
        self.indices = [(i, start) for i, record in enumerate(self.records)
                        for start in range(0, record.frames, chunk_samples)]
        if not self.indices:
            raise ValueError(f"No recordings in split {split}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        record_index, start = self.indices[index]
        record = self.records[record_index]
        first = max(0, start - self.context_samples)
        valid = min(self.chunk_samples, record.frames - start)
        left_padding = max(0, self.context_samples - start)
        count = start + valid - first
        size = self.context_samples + self.chunk_samples
        channels = []
        for path in (record.reference, record.disturbance):
            values = np.zeros(size, dtype=np.float32)
            values[left_padding:left_padding + count] = read_pcm(path, first, count)
            channels.append(torch.from_numpy(values))
        mask = torch.arange(self.chunk_samples) < valid
        return {"reference": channels[0], "disturbance": channels[1], "mask": mask}


class SyntheticDataset(Dataset):
    """Deterministic plumbing-only data, never evidence of measured ANC quality."""

    def __init__(self, chunk_samples, context_samples, seed, count=4):
        self.chunk_samples, self.context_samples = chunk_samples, context_samples
        self.seed, self.count = seed, count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)
        size = self.context_samples + self.chunk_samples
        reference = torch.randn(size, generator=generator) * 0.05
        disturbance = torch.zeros_like(reference)
        # Deliberate preview exists in this toy pair, not asserted for hardware.
        disturbance[24:] = 0.6 * reference[:-24]
        return {"reference": reference, "disturbance": disturbance,
                "mask": torch.ones(self.chunk_samples, dtype=torch.bool)}
