#!/usr/bin/env python3
"""Make the checkpoint of a trainViT_sz.py run its saved state again, for when the final state
written after that checkpoint turned out bad. For each (shape, G, seed), in --root/<shape>/:

    <name>_symm.mpack, _metadata.json, _energy_history.json  ->  <name>_symm_discarded*
    <name>_ckpt.mpack, .json                                 ->  new <name>_symm* (ckpt kept)

Nothing is deleted. The new state holds the parameters and the equilibrated chains of the
checkpoint, so `trainViT_sz.py --continue` trains further from it, on a machine that has no
checkpoint of the run (one would be resumed instead). No energy is evaluated here: final_results
holds the last training steps, marked as such. Only needs the standard library.

    python promote_ckpt.py --shapes honeycomb --gens 5 --seeds 1 2 3 4 5
"""
import argparse
import json
import os
import shutil
import statistics

SUFFIXES = (".mpack", "_metadata.json", "_energy_history.json")


def read(path):
    with open(path) as f:
        return json.load(f)


def promote(base, ckpt):
    """Replace the state at base (.../<name>_symm) by the checkpoint ckpt (.../<name>_ckpt)."""
    if os.path.exists(f"{base}_discarded.mpack"):
        print(f"{base}: already promoted, leaving it alone")
        return
    ck, meta, old = read(f"{ckpt}.json"), read(f"{base}_metadata.json"), read(f"{base}_energy_history.json")
    main, c3v = ck["history"]["main"], ck["history"]["c3v"]
    # A checkpoint of another run would silently swap the state
    if (old["energies"][:len(main["energies"])] != main["energies"]
            or old["symm_energies"][:len(c3v["energies"])] != c3v["energies"]):
        raise SystemExit(f"{ckpt}.json does not match the history of {base}: not promoting")

    energies = main["energies"] + c3v["energies"]
    variances = main["variances"] + c3v["variances"]
    meta["optimizer"].update(steps_done=len(main["energies"]), c3v_steps_done=len(c3v["energies"]))
    meta["final_results"] = {
        "exact": False,
        "final_energy": statistics.fmean(energies[-10:]),
        "final_energy_error": statistics.stdev(energies[-10:]),
        "final_variance": statistics.median(variances[-10:]),
        "measured": "last 10 training steps of the checkpoint (mean, stdev, median variance), no evaluation",
        "train_energy_last50": statistics.fmean(energies[-50:]),
        "ed_sector_energy": None,
        "before_c3v": ck["milestones"].get("before_c3v"),
        "milestones": ck["milestones"],
        "training_time": ck["elapsed"],
    }
    meta["promoted_from_checkpoint"] = {"checkpoint": f"{os.path.basename(ckpt)}.mpack",
                                        "discarded": f"{os.path.basename(base)}_discarded*"}

    # Everything is written before anything is moved. copyfile, not copy2: the new state must
    # count as newer than the bad one for rsync -u.
    shutil.copyfile(f"{ckpt}.mpack", f"{base}.mpack.tmp")
    with open(f"{base}_metadata.json.tmp", "w") as f:
        json.dump(meta, f, indent=2)
    with open(f"{base}_energy_history.json.tmp", "w") as f:
        json.dump({"energies": main["energies"], "variances": main["variances"],
                   "symm_start_iter": len(main["energies"]),
                   "symm_energies": c3v["energies"], "symm_variances": c3v["variances"]}, f, indent=2)
    for suffix in SUFFIXES:
        os.replace(f"{base}{suffix}", f"{base}_discarded{suffix}")
        os.replace(f"{base}{suffix}.tmp", f"{base}{suffix}")
    print(f"{os.path.basename(base)}: main {len(main['energies'])} + C3v {len(c3v['energies'])} steps, "
          f"E = {meta['final_results']['final_energy']:.4f}, var = {meta['final_results']['final_variance']:.3g}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shapes", nargs="+", required=True)
    p.add_argument("--gens", type=int, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--root", default="data/ViT_sz")
    args = p.parse_args()
    runs = [f"{args.root}/{shape}/vit_{shape}_G={G}_seed={seed}"
            for shape in args.shapes for G in args.gens for seed in args.seeds]
    # All or nothing as far as missing files go
    for name in runs:
        if os.path.exists(f"{name}_symm_discarded.mpack"):
            continue
        for path in [f"{name}_ckpt.mpack", f"{name}_ckpt.json"] + [f"{name}_symm{s}" for s in SUFFIXES]:
            if not os.path.exists(path):
                raise SystemExit(f"{path} is missing: nothing was changed")
    for name in runs:
        promote(f"{name}_symm", f"{name}_ckpt")


if __name__ == "__main__":
    main()
