"""alloybench command line: generate, train, splitgap, suite, demo.

Run from the repository root; paths in configs are repo-relative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def _load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def cmd_generate(args: argparse.Namespace) -> int:
    from alloybench.calculators import get_calculator
    from alloybench.dataset import build_dataset
    from alloybench.frames import write_frames

    cfg = _load_yaml(args.config)
    calc, info = get_calculator(cfg["teacher"])
    frames, provenance = build_dataset(cfg, calc, info)
    write_frames(args.out, frames)
    _write_json(Path(args.out).parent.parent / "provenance.json", provenance)
    print(f"wrote {len(frames)} labeled frames to {args.out}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from alloybench.checkpoints import save_checkpoint
    from alloybench.evaluate import evaluate_model
    from alloybench.frames import read_frames
    from alloybench.splits import group_split
    from alloybench.train import fit_and_select, prepare_frames

    cfg = _load_yaml(args.config)
    frames = read_frames(cfg["dataset"])
    split = group_split(
        frames,
        holdout_compositions=tuple(cfg["holdout_compositions"]),
        seed=int(cfg["split_seed"]),
    )
    print(
        f"split: {len(split.train)} train / {len(split.val)} val / "
        f"{len(split.test)} test / {len(split.transfer)} transfer",
        flush=True,
    )
    training: dict = {
        "split": {k: len(getattr(split, k)) for k in ("train", "val", "test", "transfer")}
    }
    accuracy: dict = {}
    for kind in cfg["models"]:
        model, info = fit_and_select(
            kind,
            [frames[k] for k in split.train],
            [frames[k] for k in split.val],
            grid=cfg["grid"],
            screen_epochs=int(cfg["screen_epochs"]),
            final_epochs=int(cfg["final_epochs"]),
            batch_size=int(cfg.get("batch_size", 8)),
            seed=int(cfg.get("train_seed", 0)),
        )
        cutoff = model.cutoff
        angular = getattr(model, "angular_cutoff", cutoff)
        acc = {}
        for part in ("test", "transfer"):
            fts = prepare_frames([frames[k] for k in getattr(split, part)], cutoff, angular)
            acc[part] = evaluate_model(model, fts, collect_parity=(part == "test"))
        save_checkpoint(Path(cfg["model_dir"]) / f"{kind}.pt", model, train_info=info)
        training[kind] = info
        accuracy[kind] = acc
        print(
            f"{kind}: test {acc['test']['e_mae_mev_atom']:.2f} meV/atom / "
            f"{acc['test']['f_mae_mev_a']:.1f} meV/A ({info['n_parameters']} params, "
            f"{info['train_seconds']:.0f} s)",
            flush=True,
        )
    _write_json(cfg["training_out"], training)
    _write_json(cfg["accuracy_out"], accuracy)
    return 0


def cmd_splitgap(args: argparse.Namespace) -> int:
    from alloybench.evaluate import evaluate_model
    from alloybench.frames import read_frames
    from alloybench.models import fit_element_refs, set_element_refs
    from alloybench.splits import group_split, random_frame_split
    from alloybench.train import build_model, prepare_frames, train_potential

    cfg = _load_yaml(args.config)
    frames = read_frames(cfg["dataset"])
    holdout = tuple(cfg["holdout_compositions"])
    kind = cfg.get("model", "compact")
    runs: list[dict] = []
    for seed in cfg["seeds"]:
        for protocol, splitter in (("group", group_split), ("random_frame", random_frame_split)):
            split = splitter(frames, holdout_compositions=holdout, seed=int(seed))
            probe = build_model(kind)
            cutoff, angular = probe.cutoff, getattr(probe, "angular_cutoff", probe.cutoff)
            train_fts = prepare_frames([frames[k] for k in split.train], cutoff, angular)
            val_fts = prepare_frames([frames[k] for k in split.val], cutoff, angular)
            test_fts = prepare_frames([frames[k] for k in split.test], cutoff, angular)
            model = build_model(kind, seed=int(seed))
            set_element_refs(model, fit_element_refs([frames[k] for k in split.train]))
            if hasattr(model, "fit_scaler"):
                model.fit_scaler(train_fts)
            train_potential(
                model,
                train_fts,
                val_fts,
                epochs=int(cfg["epochs"]),
                lr=float(cfg["lr"]),
                force_weight=float(cfg["force_weight"]),
                batch_size=int(cfg.get("batch_size", 8)),
                seed=int(seed),
            )
            metrics = evaluate_model(model, test_fts)
            runs.append({"seed": int(seed), "protocol": protocol, **metrics})
            print(
                f"seed {seed} {protocol}: {metrics['e_mae_mev_atom']:.2f} meV/atom / "
                f"{metrics['f_mae_mev_a']:.1f} meV/A",
                flush=True,
            )
    summary = {}
    for protocol in ("group", "random_frame"):
        sel = [r for r in runs if r["protocol"] == protocol]
        summary[protocol] = {
            key: {
                "mean": float(np.mean([r[key] for r in sel])),
                "std": float(np.std([r[key] for r in sel])),
            }
            for key in ("e_mae_mev_atom", "f_mae_mev_a")
        }
    _write_json(cfg["out"], {"model": kind, "config": cfg, "runs": runs, "summary": summary})
    return 0


def cmd_suite(args: argparse.Namespace) -> int:
    from alloybench.suite import run_model_suite

    cfg = _load_yaml(args.config)
    out_root = Path(cfg.get("out_dir", "results/suite"))
    for name in args.models:
        run_model_suite(name, cfg, out_root / name, checkpoint=args.checkpoint)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from alloybench.calculators import get_calculator
    from alloybench.evaluate import evaluate_model
    from alloybench.frames import read_frames
    from alloybench.properties import eos_scan, speed_benchmark
    from alloybench.train import prepare_frames

    calc, info = get_calculator("compact", checkpoint=args.checkpoint)
    print(f"compact potential: {info['n_parameters']} parameters, {info['checkpoint']}")
    frames = read_frames(args.sample)
    model = calc.model
    fts = prepare_frames(frames, model.cutoff, model.angular_cutoff)
    m = evaluate_model(model, fts)
    print(
        f"committed sample ({m['n_frames']} frames, teacher labels): "
        f"energy MAE {m['e_mae_mev_atom']:.2f} meV/atom, "
        f"force MAE {m['f_mae_mev_a']:.1f} meV/A"
    )
    eos = eos_scan(calc, "Nb", reps=3, seed=20260807)
    print(f"Nb EOS: a0 {eos['a0_angstrom']:.4f} A, B0 {eos['b0_gpa']:.1f} GPa")
    spd = speed_benchmark(calc, "TiZrNb", a0=eos["a0_angstrom"], seed=20260807, timed_steps=50)
    print(
        f"MD speed: {spd['ms_per_step']:.1f} ms/step "
        f"({spd['ms_per_atom_step']:.3f} ms/atom/step, {spd['n_atoms']} atoms)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alloybench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate and teacher-label the dataset")
    p.add_argument("--config", default="configs/dataset.yaml")
    p.add_argument("--out", default="data/full/dataset.extxyz")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("train", help="train the compact and pair potentials")
    p.add_argument("--config", default="configs/train.yaml")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("splitgap", help="group vs random-frame split control")
    p.add_argument("--config", default="configs/split_gap.yaml")
    p.set_defaults(fn=cmd_splitgap)

    p = sub.add_parser("suite", help="run the property benchmark suite")
    p.add_argument("--config", default="configs/suite.yaml")
    p.add_argument("--models", nargs="+", default=["compact"])
    p.add_argument("--checkpoint", default=None, help="checkpoint override for trained models")
    p.set_defaults(fn=cmd_suite)

    p = sub.add_parser("demo", help="offline demo on committed checkpoints and sample")
    p.add_argument("--checkpoint", default="models/compact.pt")
    p.add_argument("--sample", default="data/sample_frames.extxyz")
    p.set_defaults(fn=cmd_demo)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
