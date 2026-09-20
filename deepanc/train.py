"""Offline ANC baseline training: e = d + measured_secondary_path(u)."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader

from .data import PairedWaveDataset, SyntheticDataset, load_manifest
from .model import CausalController
from .secondary_path import CausalSecondaryPath, load_secondary_path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(records):
    items, hashes = [], {}
    for record in records:
        for path in (record.reference, record.disturbance):
            if path not in hashes:
                hashes[path] = file_sha256(path)
        items.append([record.session, record.split,
                      hashes[record.reference], hashes[record.disturbance]])
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def validate_config(config):
    for name in ("sample_rate", "chunk_samples", "batch_size", "epochs"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("secondary_delay_samples", "num_workers", "seed"):
        if type(config.get(name)) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("learning_rate", "control_penalty", "slew_penalty"):
        value = config.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if config["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    for name in ("secondary_path", "manifest"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError(f"{name} must be a nonempty path")
    if not isinstance(config.get("model"), dict):
        raise ValueError("model must be an object of controller arguments")


def compatibility_signature(config, rir_hash, data_hash, synthetic):
    # Longer training, moving files, and changing workers are compatible.
    parameters = {key: value for key, value in config.items()
                  if key not in ("epochs", "num_workers", "secondary_path", "manifest")}
    return {"config": parameters, "rir_sha256": rir_hash,
            "data_sha256": data_hash, "synthetic": synthetic, "format_version": 1}


def save_checkpoint(path, payload):
    """Atomic replacement leaves the previous checkpoint intact on failed save."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_checkpoint(path, model, optimizer, signature, device):
    # Plain tensor/dict checkpoints; never load arbitrary pickle objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("signature") != signature:
        raise ValueError("Incompatible resume: model, optimizer settings, data, RIR, rate, or delay changed")
    model.load_state_dict(checkpoint["model"])
    # Optimizer loading follows each parameter's device and keeps Adam's step
    # counter on CPU when capturable=False. Moving every state tensor to CUDA
    # breaks some supported builds and adds synchronization on others.
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if device.type == "cuda" and checkpoint.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return checkpoint["epoch"], checkpoint["best_validation_loss"]


def run_epoch(model, secondary, loader, context_samples, device, config, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = {"disturbance_energy": 0.0, "residual_energy": 0.0,
            "control_energy": 0.0, "slew_energy": 0.0, "samples": 0, "output_peak": 0.0}
    with torch.set_grad_enabled(training):
        for batch in loader:
            reference = batch["reference"].to(device)
            disturbance = batch["disturbance"].to(device)
            mask = batch["mask"].to(device)
            command = model(reference)
            residual = disturbance + secondary(command)
            command_target = command[:, context_samples:]
            residual_target = residual[:, context_samples:]
            disturbance_target = disturbance[:, context_samples:]
            previous_command = torch.nn.functional.pad(command, (1, 0))[:, :-1]
            slew = (command - previous_command)[:, context_samples:]
            count = mask.sum()
            error_energy = residual_target.square().masked_select(mask).sum()
            control_energy = command_target.square().masked_select(mask).sum()
            slew_energy = slew.square().masked_select(mask).sum()
            loss = (error_energy + config["control_penalty"] * control_energy
                    + config["slew_penalty"] * slew_energy) / count
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; check recording units, measured RIR, and learning rate")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            sums["disturbance_energy"] += disturbance_target.square().masked_select(mask).sum().item()
            sums["residual_energy"] += error_energy.item()
            sums["control_energy"] += control_energy.item()
            sums["slew_energy"] += slew_energy.item()
            sums["samples"] += count.item()
            sums["output_peak"] = max(sums["output_peak"], command_target.abs().masked_select(mask).max().item())
    if sums["disturbance_energy"] <= 0:
        raise ValueError("All disturbance samples are silent; noise reduction is undefined")
    samples = sums["samples"]
    sums["loss"] = (sums["residual_energy"] + config["control_penalty"] * sums["control_energy"]
                    + config["slew_penalty"] * sums["slew_energy"]) / samples
    sums["residual_mse"] = sums["residual_energy"] / samples
    sums["noise_reduction_db"] = 10 * math.log10(sums["disturbance_energy"] / max(sums["residual_energy"], 1e-30))
    return sums


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/anc_train.json"))
    parser.add_argument("--manifest", type=Path, help="Real ANC-OFF paired manifest; path relative to cwd")
    parser.add_argument("--output", type=Path, default=Path("runs/anc_baseline"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, help="Total epochs, including already completed epochs")
    parser.add_argument("--smoke-test", action="store_true", help="Synthetic forward/backward/checkpoint test only")
    args = parser.parse_args(argv)
    if args.smoke_test and args.manifest:
        parser.error("--smoke-test and --manifest cannot be used together")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.smoke_test:
        config["chunk_samples"] = min(config["chunk_samples"], 512)
        config["epochs"] = 1
        config["num_workers"] = 0
        # Avoid oversized thread pools on developer CPUs for this tiny smoke run.
        if args.device == "cpu" or not torch.cuda.is_available():
            torch.set_num_threads(min(torch.get_num_threads(), 2))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    validate_config(config)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; install JetPack-compatible PyTorch or use --device cpu explicitly")
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(config["seed"])
    rir_path = (config_path.parent / config["secondary_path"]).resolve()
    coefficients = load_secondary_path(rir_path, sample_rate=config["sample_rate"])
    rir_hash = file_sha256(rir_path)
    model = CausalController(**config["model"]).to(device)
    secondary = CausalSecondaryPath(coefficients, delay_samples=config["secondary_delay_samples"]).to(device)
    context = model.receptive_field - 1 + len(coefficients) - 1 + config["secondary_delay_samples"]
    # One extra history sample makes the command-slew penalty correct at boundaries.
    context += 1
    if args.smoke_test:
        train_data = SyntheticDataset(config["chunk_samples"], context, config["seed"], count=4)
        valid_data = SyntheticDataset(config["chunk_samples"], context, config["seed"] + 10000, count=2)
        data_hash = "synthetic-plumbing-v1"
        data_source = "SYNTHETIC SMOKE TEST: metrics are NOT measured hardware ANC performance"
    else:
        manifest = args.manifest.resolve() if args.manifest else (config_path.parent / config["manifest"]).resolve()
        records = load_manifest(manifest, config["sample_rate"])
        train_data = PairedWaveDataset(records, "train", config["chunk_samples"], context)
        valid_data = PairedWaveDataset(records, "valid", config["chunk_samples"], context)
        data_hash = dataset_fingerprint(records)
        data_source = str(manifest)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    signature = compatibility_signature(config, rir_hash, data_hash, args.smoke_test)
    start_epoch, best = 0, float("inf")
    if args.resume:
        start_epoch, best = restore_checkpoint(args.resume, model, optimizer, signature, device)
    if start_epoch >= config["epochs"]:
        raise ValueError(f"Checkpoint already completed {start_epoch} epochs; set --epochs greater than {start_epoch}")
    args.output.mkdir(parents=True, exist_ok=True)
    has_artifacts = any((args.output / name).exists() for name in ("last.pt", "best.pt", "metrics.jsonl", "run.json"))
    if has_artifacts:
        if not args.resume:
            raise FileExistsError("Output already contains a run; choose a new --output or use --resume")
        if not (args.output / "last.pt").is_file() or (args.output / "last.pt").resolve() != args.resume.resolve():
            raise FileExistsError("Output contains a different run; resume its last.pt or use a new output directory")
    run_info = {"config": config, "rir_sha256": rir_hash, "data_sha256": data_hash,
                "data_source": data_source, "synthetic": args.smoke_test,
                "context_samples": context, "device": str(device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "scope": "offline causal baseline; hardware latency and measured cancellation unverified"}
    (args.output / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_info), flush=True)
    loader_options = {"batch_size": config["batch_size"], "num_workers": config["num_workers"],
                      "pin_memory": use_cuda}
    valid_loader = DataLoader(valid_data, shuffle=False, **loader_options)
    for epoch in range(start_epoch, config["epochs"]):
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
        train_metrics = run_epoch(model, secondary, train_loader, context, device, config, optimizer)
        valid_metrics = run_epoch(model, secondary, valid_loader, context, device, config)
        improved = valid_metrics["loss"] < best
        best = min(best, valid_metrics["loss"])
        payload = {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "epoch": epoch + 1, "config": config, "rir_sha256": rir_hash,
                   "signature": signature, "synthetic": args.smoke_test,
                   "best_validation_loss": best, "validation": valid_metrics,
                   "torch_rng_state": torch.get_rng_state(),
                   "cuda_rng_state": torch.cuda.get_rng_state_all() if use_cuda else []}
        save_checkpoint(args.output / "last.pt", payload)
        if improved:
            save_checkpoint(args.output / "best.pt", payload)
        metrics = {"epoch": epoch + 1, "synthetic": args.smoke_test,
                   "train": train_metrics, "valid": valid_metrics}
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
    print(f"Saved {args.output / 'last.pt'}; offline validation only.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
