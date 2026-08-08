"""Dataset assembly: batches per composition, teacher labeling, provenance.

The generation recipe (compositions, batch layout, master seed) is kept
identical across this TiZrNb MLIP cluster so cross-repo numbers stay
internally consistent; the implementation here is fresh. All labels are
surrogate model labels from the teacher, never DFT.
"""

from __future__ import annotations

import time

from alloybench.frames import Frame
from alloybench.labeling import label_frames, teacher_md_batch
from alloybench.lattice import rattle_batch, strain_batch


def build_dataset(cfg: dict, teacher_calc, teacher_info: dict) -> tuple[list[Frame], dict]:
    """Generate and teacher-label the full dataset described by cfg.

    Args:
        cfg: Parsed configs/dataset.yaml.
        teacher_calc: The teacher ASE calculator (labels and drives MD).
        teacher_info: Provenance of the teacher (package, version, license).

    Returns:
        (frames, provenance).
    """
    frames: list[Frame] = []
    md_seconds = 0.0
    for ci, comp in enumerate(cfg["compositions"] + cfg["holdout_compositions"]):
        seed_base = int(cfg["seed"]) + 1000 * ci
        for b, amp in enumerate(cfg["rattle_amplitudes"]):
            frames.extend(
                rattle_batch(
                    comp,
                    cfg["reps_small"],
                    amp,
                    cfg["rattle_frames_per_batch"],
                    seed=seed_base + b,
                    batch=b,
                )
            )
        # One rattle batch on the larger cell for size diversity.
        frames.extend(
            rattle_batch(
                comp,
                cfg["reps_large"],
                cfg["rattle_amplitudes"][1],
                cfg["rattle_frames_large"],
                seed=seed_base + 50,
                batch=9,
            )
        )
        frames.extend(
            strain_batch(
                comp,
                cfg["reps_small"],
                cfg["strain_frames_per_batch"],
                seed=seed_base + 100,
                batch=0,
            )
        )
        t0 = time.perf_counter()
        for b, temp in enumerate(cfg["md_temperatures_k"]):
            frames.extend(
                teacher_md_batch(
                    comp,
                    cfg["reps_small"],
                    teacher_calc,
                    temperature_k=temp,
                    n_steps=cfg["md_steps"],
                    sample_every=cfg["md_sample_every"],
                    seed=seed_base + 200 + b,
                    batch=b,
                )
            )
        md_seconds += time.perf_counter() - t0
        print(f"generated {comp}: {len(frames)} frames so far", flush=True)

    sec_per_frame = label_frames(frames, teacher_calc)
    provenance = {
        "label_source": "surrogate model labels from a pretrained universal potential, NOT DFT",
        "teacher": teacher_info,
        "n_frames": len(frames),
        "n_groups": len({fr.group for fr in frames}),
        "compositions": sorted({fr.composition for fr in frames}),
        "atoms_min": min(fr.n_atoms for fr in frames),
        "atoms_max": max(fr.n_atoms for fr in frames),
        "labeling_seconds_per_frame": round(sec_per_frame, 4),
        "md_generation_seconds": round(md_seconds, 1),
        "config": cfg,
    }
    return frames, provenance
