"""
Train a QED surrogate for GenMol with online data generation.

Data is generated on-the-fly (not stored), mirroring TTT's
DiffusionPriorBuffer pattern: sample x0 from the model, compute QED
with RDKit, train the surrogate, repeat.

Usage:
    python scripts/train_qed_surrogate.py \
        --model_path model_v2.ckpt \
        --num_steps 10000 \
        --batch_size 128 \
        --lr 1e-4
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from genmol.sampler import Sampler
from genmol.qed_surrogate import QEDSurrogate, save_surrogate


# ---------------------------------------------------------------------------
# Online data generator (modeled after TTT's DiffusionPriorBuffer)
# ---------------------------------------------------------------------------

class MoleculeBuffer:
    """On-the-fly molecule generation buffer.

    Generates a batch of molecules from GenMol, decodes to SMILES, computes
    QED. Refreshes when exhausted. Mirrors TTT's DiffusionPriorBuffer which
    samples x0 from the diffusion prior and computes A(x0) as the target.

    No data is stored to disk — everything is generated online.
    """

    def __init__(self, sampler, buffer_size=512, gen_batch_size=128,
                 softmax_temp=1.0, randomness=0.3, min_add_len=40):
        self.sampler = sampler
        self.model = sampler.model
        self.buffer_size = buffer_size
        self.gen_batch_size = gen_batch_size
        self.softmax_temp = softmax_temp
        self.randomness = randomness
        self.min_add_len = min_add_len
        self.pad_id = sampler.pad_index
        self.max_len = self.model.config.model.max_position_embeddings

        # Lazy imports (only needed on GPU machine)
        from rdkit import Chem
        from rdkit.Chem import QED as QED_module
        from genmol.utils.utils_chem import safe_to_smiles
        from genmol.utils.bracket_safe_converter import bracketsafe2safe
        self._Chem = Chem
        self._QED = QED_module
        self._safe_to_smiles = safe_to_smiles
        self._bracketsafe2safe = bracketsafe2safe

        self._token_ids = None  # [buffer_size, max_len]
        self._qed_scores = None  # [buffer_size]
        self._perm = None
        self._cursor = 0

    @torch.no_grad()
    def _generate_batch(self, bs):
        """Generate bs molecules, return (token_ids [bs, max_len], qed [bs])."""
        model = self.model
        sampler = self.sampler

        # Build fully masked input
        x = torch.hstack([
            torch.full((1, 1), model.bos_index),
            torch.full((1, 1), model.eos_index),
        ])
        x = sampler._insert_mask(x, num_samples=bs, min_add_len=self.min_add_len)
        x = x.to(model.device)

        # Denoise
        num_steps = max(sampler.mdlm.get_num_steps_confidence(x), 2)
        attention_mask = x != self.pad_id
        for i in range(num_steps):
            logits = model(x, attention_mask)
            x = sampler.mdlm.step_confidence(
                logits, x, i, num_steps, self.softmax_temp, self.randomness
            )

        # Decode and score
        strings = model.tokenizer.batch_decode(x, skip_special_tokens=True)
        use_bracket = model.config.training.get("use_bracket_safe")

        token_ids_list = []
        qed_list = []

        for j, s in enumerate(strings):
            # Decode SAFE -> SMILES
            try:
                if use_bracket:
                    smi = self._safe_to_smiles(self._bracketsafe2safe(s), fix=True)
                else:
                    smi = self._safe_to_smiles(s, fix=True)
            except Exception:
                smi = None

            if smi:
                smi = sorted(smi.split("."), key=len)[-1]

            # Compute QED
            mol = self._Chem.MolFromSmiles(smi) if smi else None
            qed = float(self._QED.qed(mol)) if mol is not None else 0.0

            # Pad/truncate token_ids
            ids = x[j].cpu()
            if len(ids) < self.max_len:
                ids = torch.cat([ids, torch.full((self.max_len - len(ids),),
                                                  self.pad_id, dtype=ids.dtype)])
            else:
                ids = ids[:self.max_len]

            token_ids_list.append(ids)
            qed_list.append(qed)

        return (torch.stack(token_ids_list),
                torch.tensor(qed_list, dtype=torch.float32))

    def refresh(self):
        """Regenerate the entire buffer (like TTT's buffer.refresh())."""
        all_ids = []
        all_qed = []
        remaining = self.buffer_size
        while remaining > 0:
            bs = min(remaining, self.gen_batch_size)
            ids, qed = self._generate_batch(bs)
            all_ids.append(ids)
            all_qed.append(qed)
            remaining -= bs

        self._token_ids = torch.cat(all_ids)[:self.buffer_size]
        self._qed_scores = torch.cat(all_qed)[:self.buffer_size]
        self._perm = torch.randperm(self.buffer_size)
        self._cursor = 0

        n_valid = (self._qed_scores > 0).sum().item()
        mean_qed = self._qed_scores[self._qed_scores > 0].mean().item() if n_valid > 0 else 0
        return n_valid, mean_qed

    def sample(self, batch_size):
        """Get next batch. Auto-refreshes when exhausted."""
        if self._token_ids is None or self._cursor + batch_size > self.buffer_size:
            self.refresh()

        idx = self._perm[self._cursor:self._cursor + batch_size]
        self._cursor += batch_size
        return self._token_ids[idx], self._qed_scores[idx]


# ---------------------------------------------------------------------------
# Training loop (modeled after TTT's train_cbg.py)
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build sampler (loads GenMol) ---
    print(f"Loading GenMol from {args.model_path}...")
    sampler = Sampler(path=args.model_path)

    # --- Build buffer ---
    buffer = MoleculeBuffer(
        sampler,
        buffer_size=args.buffer_size,
        gen_batch_size=args.gen_batch_size,
        softmax_temp=args.softmax_temp,
        randomness=args.randomness,
    )

    # --- Build surrogate ---
    surrogate = QEDSurrogate(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        embed_dim=args.embed_dim,
        base_channels=args.base_channels,
        channel_mult=tuple(args.channel_mult),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        pad_token_id=args.pad_token_id,
    ).to(device)

    n_params = sum(p.numel() for p in surrogate.parameters())
    print(f"QEDSurrogate: {n_params:,} parameters")

    # --- Optimizer (AdamW, same as TTT) ---
    optimizer = torch.optim.AdamW(
        surrogate.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # --- Output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Initial buffer fill ---
    print("Filling initial buffer...")
    n_valid, mean_qed = buffer.refresh()
    print(f"  {n_valid}/{args.buffer_size} valid, mean QED={mean_qed:.3f}")

    # --- Training loop ---
    step_losses = []
    val_losses = []
    best_val_loss = float("inf")
    t_start = time.time()

    print(f"\nTraining for {args.num_steps} steps...")
    for step in range(1, args.num_steps + 1):
        surrogate.train()

        # Sample batch from buffer (auto-refreshes when exhausted)
        token_ids, qed_true = buffer.sample(args.batch_size)
        token_ids = token_ids.to(device)
        qed_true = qed_true.to(device)

        # Forward + loss (MSE, same as TTT)
        qed_pred = surrogate(token_ids)
        loss = F.mse_loss(qed_pred, qed_true)

        # Backward + step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(surrogate.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        step_losses.append(loss.item())

        # --- Logging ---
        if step % args.log_every == 0:
            recent = step_losses[-args.log_every:]
            avg = sum(recent) / len(recent)
            elapsed = time.time() - t_start
            print(f"  step {step}/{args.num_steps}  loss={avg:.6f}  "
                  f"[{elapsed:.0f}s]")

        # --- Validation (generate fresh batch, don't use buffer) ---
        if step % args.val_every == 0:
            surrogate.eval()
            with torch.no_grad():
                val_ids, val_qed = buffer._generate_batch(args.val_size)
                val_ids = val_ids.to(device)
                val_qed = val_qed.to(device)
                val_pred = surrogate(val_ids)
                val_loss = F.mse_loss(val_pred, val_qed).item()

            val_losses.append((step, val_loss))
            print(f"    val_loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_surrogate(
                    surrogate,
                    os.path.join(args.output_dir, "surrogate_best.pt"),
                    step=step, val_loss=best_val_loss,
                )

    # --- Save final ---
    save_surrogate(
        surrogate,
        os.path.join(args.output_dir, "surrogate_final.pt"),
        step=args.num_steps,
        val_loss=val_losses[-1][1] if val_losses else None,
    )

    # --- Loss curves (same as TTT's save_loss_curves) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(step_losses, alpha=0.3, linewidth=0.5, label="per step")
    if len(step_losses) > 10:
        window = max(len(step_losses) // 20, 5)
        smoothed = np.convolve(step_losses, np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
                 linewidth=1.5, label=f"smoothed (w={window})")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if val_losses:
        val_steps, val_vals = zip(*val_losses)
        ax2.plot(val_steps, val_vals, "o-", label="val")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Validation loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # --- Save metrics ---
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "n_params": n_params,
            "num_steps": args.num_steps,
            "best_val_loss": best_val_loss,
            "final_train_loss": step_losses[-1] if step_losses else None,
            "elapsed_sec": time.time() - t_start,
        }, f, indent=2)

    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Saved to {args.output_dir}/")
    print(f"Total time: {time.time() - t_start:.0f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train QED surrogate for GenMol")

    # GenMol model
    parser.add_argument("--model_path", type=str, required=True)

    # Online generation
    parser.add_argument("--buffer_size", type=int, default=1024,
                        help="Molecules per buffer refresh")
    parser.add_argument("--gen_batch_size", type=int, default=128,
                        help="Batch size for molecule generation")
    parser.add_argument("--softmax_temp", type=float, default=1.0)
    parser.add_argument("--randomness", type=float, default=0.3)

    # Architecture
    parser.add_argument("--vocab_size", type=int, default=1880)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 4, 4])
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pad_token_id", type=int, default=3)

    # Training
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--val_size", type=int, default=256)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/qed_surrogate")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
