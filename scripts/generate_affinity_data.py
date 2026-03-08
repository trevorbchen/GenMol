"""
Generate (token_ids, affinity_score) training pairs for an affinity surrogate.

Step 1: Sample molecules from GenMol -> decode to SMILES
Step 2: Write SMILES to CSV
Step 3: Run FlashAffinity pipeline (external, via predict_value.sh)
Step 4: Parse affinity predictions + tokenize -> save .pt dataset

Steps 1-2 and 4 are handled here. Step 3 is run externally because
FlashAffinity requires its own conda env (fabind_h100) and checkpoints.

Usage:
    # Step 1-2: Generate SMILES from GenMol
    python scripts/generate_affinity_data.py generate \
        --model_path model_v2.ckpt \
        --num_molecules 10000 \
        --output_dir outputs/affinity_data

    # Step 3: Run FlashAffinity (external, in fabind_h100 env)
    cd FlashAffinity
    SMILES_CSV=../outputs/affinity_data/smiles.csv \
    PROT_ID=2VT4 \
    DATA_DIR=./data/test \
    OUT_DIR=../outputs/affinity_data/flash_out \
    bash predict_value.sh

    # Step 4: Parse FlashAffinity results -> .pt dataset
    python scripts/generate_affinity_data.py package \
        --model_path model_v2.ckpt \
        --smiles_csv outputs/affinity_data/smiles.csv \
        --affinity_json outputs/affinity_data/flash_out/affinity_predictions_id.json \
        --output_dir outputs/affinity_data
"""

import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Step 1-2: Generate molecules with GenMol and save SMILES
# ---------------------------------------------------------------------------

def generate_smiles(args):
    from genmol.sampler import Sampler
    from genmol.utils.utils_chem import safe_to_smiles
    from genmol.utils.bracket_safe_converter import bracketsafe2safe

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading GenMol from {args.model_path}...")
    sampler = Sampler(path=args.model_path)
    model = sampler.model
    use_bracket = model.config.training.get("use_bracket_safe")

    os.makedirs(args.output_dir, exist_ok=True)

    all_smiles = []
    all_token_ids = []
    remaining = args.num_molecules
    batch_num = 0

    print(f"Generating {args.num_molecules} molecules...")
    while remaining > 0:
        bs = min(remaining, args.gen_batch_size)
        batch_num += 1

        # Build fully masked input
        x = torch.hstack([
            torch.full((1, 1), model.bos_index),
            torch.full((1, 1), model.eos_index),
        ])
        x = sampler._insert_mask(x, num_samples=bs,
                                 min_add_len=args.min_add_len)
        x = x.to(device)

        # Denoise
        with torch.no_grad():
            num_steps = max(sampler.mdlm.get_num_steps_confidence(x), 2)
            attention_mask = x != sampler.pad_index
            for i in range(num_steps):
                logits = model(x, attention_mask)
                x = sampler.mdlm.step_confidence(
                    logits, x, i, num_steps,
                    args.softmax_temp, args.randomness
                )

        # Decode and collect
        strings = model.tokenizer.batch_decode(x, skip_special_tokens=True)
        for j, s in enumerate(strings):
            try:
                if use_bracket:
                    smi = safe_to_smiles(bracketsafe2safe(s), fix=True)
                else:
                    smi = safe_to_smiles(s, fix=True)
            except Exception:
                smi = None

            if smi:
                smi = sorted(smi.split("."), key=len)[-1]

            if smi:
                # Pad/truncate token ids to max_len
                ids = x[j].cpu()
                max_len = model.config.model.max_position_embeddings
                if len(ids) < max_len:
                    ids = torch.cat([ids, torch.full(
                        (max_len - len(ids),), sampler.pad_index,
                        dtype=ids.dtype)])
                else:
                    ids = ids[:max_len]
                all_smiles.append(smi)
                all_token_ids.append(ids)

        remaining -= bs
        n_valid = len(all_smiles)
        print(f"  batch {batch_num}: {n_valid} valid molecules so far")

    # Save SMILES CSV (for FlashAffinity input)
    csv_path = os.path.join(args.output_dir, "smiles.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        for smi in all_smiles:
            writer.writerow([smi])

    # Save token IDs (for later pairing with affinity scores)
    token_ids_tensor = torch.stack(all_token_ids)
    torch.save(token_ids_tensor, os.path.join(args.output_dir, "token_ids.pt"))

    print(f"\nGenerated {len(all_smiles)} valid molecules")
    print(f"  SMILES CSV: {csv_path}")
    print(f"  Token IDs:  {os.path.join(args.output_dir, 'token_ids.pt')}")
    print(f"\nNext: run FlashAffinity predict_value.sh with SMILES_CSV={csv_path}")


# ---------------------------------------------------------------------------
# Step 4: Parse FlashAffinity results and package into .pt dataset
# ---------------------------------------------------------------------------

def package_dataset(args):
    # Load token IDs
    token_ids = torch.load(
        os.path.join(args.output_dir, "token_ids.pt"),
        weights_only=True)
    print(f"Loaded {len(token_ids)} token ID sequences")

    # Load SMILES (to match ordering)
    smiles_list = []
    with open(args.smiles_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            smiles_list.append(row[0])
    print(f"Loaded {len(smiles_list)} SMILES")

    assert len(smiles_list) == len(token_ids), \
        f"Mismatch: {len(smiles_list)} SMILES vs {len(token_ids)} token_ids"

    # Load FlashAffinity predictions
    with open(args.affinity_json, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} FlashAffinity predictions")

    # Match predictions to SMILES by index
    # FlashAffinity uses {prot_id}_L{idx:06d} as keys
    affinity_scores = []
    valid_mask = []
    for idx in range(len(smiles_list)):
        ligand_id = f"L{idx:06d}"
        key = f"{args.prot_id}_{ligand_id}"
        pred = predictions.get(key, {})

        if pred.get("status") == "success" and pred.get("pred_value") is not None:
            affinity_scores.append(float(pred["pred_value"]))
            valid_mask.append(True)
        else:
            affinity_scores.append(0.0)
            valid_mask.append(False)

    valid_mask = torch.tensor(valid_mask)
    affinity_scores = torch.tensor(affinity_scores, dtype=torch.float32)

    n_valid = valid_mask.sum().item()
    print(f"\nMatched {n_valid}/{len(smiles_list)} predictions")

    if n_valid > 0:
        valid_affinities = affinity_scores[valid_mask]
        print(f"  Affinity range: [{valid_affinities.min():.3f}, "
              f"{valid_affinities.max():.3f}]")
        print(f"  Affinity mean:  {valid_affinities.mean():.3f}")

    # Filter to valid only
    token_ids_valid = token_ids[valid_mask]
    affinity_valid = affinity_scores[valid_mask]

    # Save dataset
    dataset_path = os.path.join(args.output_dir, "affinity_dataset.pt")
    torch.save({
        "token_ids": token_ids_valid,
        "affinity_scores": affinity_valid,
        "prot_id": args.prot_id,
        "n_samples": n_valid,
    }, dataset_path)

    print(f"\nSaved dataset: {dataset_path}")
    print(f"  {n_valid} samples, ready for training")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate affinity surrogate training data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen_parser = subparsers.add_parser(
        "generate", help="Generate molecules with GenMol")
    gen_parser.add_argument("--model_path", type=str, required=True)
    gen_parser.add_argument("--num_molecules", type=int, default=10000)
    gen_parser.add_argument("--gen_batch_size", type=int, default=128)
    gen_parser.add_argument("--softmax_temp", type=float, default=1.0)
    gen_parser.add_argument("--randomness", type=float, default=0.3)
    gen_parser.add_argument("--min_add_len", type=int, default=40)
    gen_parser.add_argument("--output_dir", type=str,
                            default="outputs/affinity_data")

    # --- package ---
    pkg_parser = subparsers.add_parser(
        "package", help="Package FlashAffinity results into .pt dataset")
    pkg_parser.add_argument("--smiles_csv", type=str, required=True)
    pkg_parser.add_argument("--affinity_json", type=str, required=True)
    pkg_parser.add_argument("--prot_id", type=str, default="2VT4")
    pkg_parser.add_argument("--output_dir", type=str,
                            default="outputs/affinity_data")

    args = parser.parse_args()
    if args.command == "generate":
        generate_smiles(args)
    elif args.command == "package":
        package_dataset(args)


if __name__ == "__main__":
    main()
