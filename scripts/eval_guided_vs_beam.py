"""
Compare guided sampling (QED surrogate) vs beam search (actual QED) vs unconditional.

Generates molecules with each method, evaluates all with actual RDKit QED,
and reports validity, uniqueness, QED stats, and timing.

Usage:
    python scripts/eval_guided_vs_beam.py \
        --model_path model_v2.ckpt \
        --surrogate_path outputs/qed_surrogate/surrogate_best.pt \
        --num_samples 100 \
        --output_dir outputs/guided_vs_beam
"""

import argparse
import json
import os
import sys
import time

import torch
from rdkit import Chem
from rdkit.Chem import QED as QED_module

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")


def compute_qed(smiles):
    """Compute QED for a SMILES string. Returns 0.0 if invalid."""
    if not smiles:
        return 0.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return float(QED_module.qed(mol))


def eval_samples(samples, method_name):
    """Compute metrics for a list of SMILES."""
    valid = [s for s in samples if s and Chem.MolFromSmiles(s) is not None]
    qeds = sorted([compute_qed(s) for s in valid], reverse=True)

    n_total = len(samples)
    n_valid = len(valid)
    n_unique = len(set(valid))

    qed_mean = sum(qeds) / len(qeds) if qeds else 0.0
    top10_n = max(1, len(qeds) // 10)
    qed_top10 = sum(qeds[:top10_n]) / top10_n if qeds else 0.0
    qed_max = qeds[0] if qeds else 0.0

    print(f"\n  [{method_name}]")
    print(f"    Samples:    {n_total}")
    print(f"    Valid:      {n_valid}/{n_total} ({n_valid/max(n_total,1)*100:.1f}%)")
    print(f"    Unique:     {n_unique}/{n_valid} ({n_unique/max(n_valid,1)*100:.1f}%)")
    print(f"    QED mean:   {qed_mean:.4f}")
    print(f"    QED top10:  {qed_top10:.4f}")
    print(f"    QED max:    {qed_max:.4f}")

    return {
        "method": method_name,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_unique": n_unique,
        "validity": n_valid / max(n_total, 1),
        "uniqueness": n_unique / max(n_valid, 1),
        "qed_mean": qed_mean,
        "qed_top10": qed_top10,
        "qed_max": qed_max,
        "all_qeds": qeds,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare guided vs beam search vs unconditional")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--surrogate_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str,
                        default="outputs/guided_vs_beam")

    # Guided sampler params
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--guidance_schedule", type=str, default="constant")

    # Beam search params
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--beam_length", type=int, default=4)

    # Shared sampling params
    parser.add_argument("--softmax_temp", type=float, default=0.8)
    parser.add_argument("--randomness", type=float, default=0.5)
    parser.add_argument("--min_add_len", type=int, default=40)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    # -----------------------------------------------------------------------
    # 1. Unconditional sampling
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("1. Unconditional sampling")
    print("=" * 60)

    from genmol.sampler import Sampler

    sampler = Sampler(path=args.model_path)
    t0 = time.time()
    uncond_samples = sampler.de_novo_generation(
        num_samples=args.num_samples,
        softmax_temp=args.softmax_temp,
        randomness=args.randomness,
        min_add_len=args.min_add_len,
    )
    uncond_time = time.time() - t0
    print(f"    Time: {uncond_time:.1f}s")

    results["unconditional"] = eval_samples(uncond_samples, "Unconditional")
    results["unconditional"]["time_sec"] = uncond_time

    # -----------------------------------------------------------------------
    # 2. Guided sampling (QED surrogate)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. Guided sampling (QED surrogate)")
    print("=" * 60)

    from genmol.guided_sampler import GuidedSampler

    guided_sampler = GuidedSampler(
        path=args.model_path,
        surrogate_path=args.surrogate_path,
        guidance_scale=args.guidance_scale,
        guidance_schedule=args.guidance_schedule,
    )
    t0 = time.time()
    guided_samples = guided_sampler.de_novo_generation(
        num_samples=args.num_samples,
        softmax_temp=args.softmax_temp,
        randomness=args.randomness,
        min_add_len=args.min_add_len,
    )
    guided_time = time.time() - t0
    print(f"    Time: {guided_time:.1f}s")

    results["guided"] = eval_samples(guided_samples, "Guided (surrogate)")
    results["guided"]["time_sec"] = guided_time

    # Free guided sampler memory
    del guided_sampler
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 3. Beam search (actual QED)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. Beam search (actual QED)")
    print("=" * 60)

    from genmol.beam_search_sampler import BeamSearchSampler

    beam_sampler = BeamSearchSampler(
        path=args.model_path,
        beam_width=args.beam_width,
        beam_length=args.beam_length,
    )
    t0 = time.time()
    beam_samples = beam_sampler.de_novo_generation(
        num_samples=args.num_samples,
        softmax_temp=args.softmax_temp,
        randomness=args.randomness,
        min_add_len=args.min_add_len,
    )
    beam_time = time.time() - t0
    print(f"    Time: {beam_time:.1f}s")

    results["beam_search"] = eval_samples(beam_samples, "Beam search (QED)")
    results["beam_search"]["time_sec"] = beam_time

    # -----------------------------------------------------------------------
    # Summary comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'Valid%':>7} {'Uniq%':>7} {'QED_mean':>9} "
          f"{'QED_top10':>10} {'QED_max':>8} {'Time(s)':>8}")
    print("-" * 80)
    for key in ["unconditional", "guided", "beam_search"]:
        r = results[key]
        print(f"{r['method']:<25} {r['validity']*100:>6.1f}% "
              f"{r['uniqueness']*100:>6.1f}% {r['qed_mean']:>9.4f} "
              f"{r['qed_top10']:>10.4f} {r['qed_max']:>8.4f} "
              f"{r['time_sec']:>8.1f}")

    # Save results
    # Remove non-serializable fields
    save_results = {}
    for k, v in results.items():
        save_results[k] = {kk: vv for kk, vv in v.items() if kk != "all_qeds"}

    with open(os.path.join(args.output_dir, "comparison.json"), "w") as f:
        json.dump(save_results, f, indent=2)

    # Save all samples
    import csv
    for key, samples in [("unconditional", uncond_samples),
                         ("guided", guided_samples),
                         ("beam_search", beam_samples)]:
        csv_path = os.path.join(args.output_dir, f"{key}_samples.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "qed"])
            for s in samples:
                writer.writerow([s, compute_qed(s)])

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
