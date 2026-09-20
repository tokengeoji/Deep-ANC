# DeepANC continuation instructions

## Goal and established evidence

The user has already succeeded with real OMAP-L138 FxNLMS; Jetson was NOT involved.
Prepare and continue an offline deep ANC workflow on Jetson Orin without disturbing
the working hardware baseline. Respond to the user in Korean.

- Canonical remote: `https://github.com/tokengeoji/DeepANC.git` (renamed from Roka-jsj).
- `rir.txt` is the authoritative, measured secondary path: 16,000 Hz, 500 taps.
- Its coefficients exactly match `firmware/omap_l138/FxNLMS0/ISR.c` `S_hat`.
- Metadata and original SHA-256 are in `calibration/secondary_path.json`.
- Preserve its gain, sign, all taps and original leading delay. Do not normalize,
  truncate, peak-align, invert, resample, or add a guessed codec delay.
- Python neural output is the actual normalized DAC command `u`: `e=d+S*u`.
  Existing DSP uses `u=-y`; do not introduce a second inversion in Python.
- Original DSP sources must remain the known working baseline. Do not flash a
  device or enable speaker output as part of an offline training setup.

## When the user says "이어서 해줘" on Jetson

1. Read `docs/JETSON_HANDOFF.md`, `docs/JETSON_SETUP.md`, `docs/DATASET.md`.
2. Inspect `git status` and the installed platform before making changes; preserve
   any user recordings, calibration, checkpoints and existing CUDA PyTorch.
3. Run `bash tools/prepare_jetson.sh` in the existing working Python environment.
   Use `--install` to create `.venv` and install non-Torch dependencies when needed.
   Never replace Jetson's NVIDIA wheel with an ordinary CPU PyPI torch wheel.
4. Run `python -m pytest -q` and inspect `artifacts/jetson_preflight.json` and
   the calibration report. CUDA execution must really succeed on the board.
5. Inspect available recordings. Training requires synchronous raw reference and
   ANC-OFF disturbance at 16 kHz with original common digital scaling, plus
   sessions separated between train and valid. Never use ANC-ON residual as d,
   independent normalization, future alignment, or arbitrary synthetic P as
   measured primary-path data.
6. If real recordings exist, prepare the manifest and run a short real-data train,
   validate finite loss/gradient/output, save a checkpoint, then scale the run to
   the board's available memory. If missing, report precisely what must be
   recorded; synthetic smoke is a software test, not acoustic validation.
7. Real-time integration is a later task requiring measured end-to-end latency,
   input/output scaling, feedback handling and the actual OMAP/Jetson transport.
   A causal offline model and a 3.0625 ms RIR peak do not prove real-time feasibility.

## Development and validation

- `deepanc/` is the new ANC path; `scripts/` is retained legacy GCRN speech
  enhancement, not an ANC model. Avoid mixing the old mix/sph training objective.
- Use `python tools/prepare_secondary_path.py --verify-only` to verify the source.
- Test meaningful DSP properties: impulse gain/sign/delay, causal convolution,
  no FFT wraparound, block history, finite gradients, split leakage and resume.
- Run `python -m pytest -q`; `python -m deepanc.train --config
  configs/anc_train.json --smoke-test --device cpu --output runs/smoke` is a local
  CPU smoke command. Use CUDA on Jetson and label the environment actually tested.
- Large datasets, generated FIR files, build caches, PDFs and checkpoints stay
  local. Track source, configs, calibration provenance and documentation.
