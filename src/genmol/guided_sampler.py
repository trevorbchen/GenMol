"""
DPS-style guided sampler for GenMol using a differentiable QED surrogate.

At each denoising step, the surrogate's gradient w.r.t. the BERT logits
biases token selection toward higher-reward molecules. This replaces
beam search's expensive rollout+decode+score loop with a single cheap
backward pass through the small surrogate.

Analogous to TTT's CBGDPS: gradient flows through the surrogate (~1-3M
params) instead of through the non-differentiable decode+RDKit pipeline.
"""

import os
import warnings
import torch
import torch.nn.functional as F

from genmol.sampler import Sampler
from genmol.qed_surrogate import load_surrogate
from genmol.utils.utils_chem import safe_to_smiles
from genmol.utils.bracket_safe_converter import bracketsafe2safe

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


class GuidedSampler(Sampler):
    """
    MDLM sampler with DPS-style reward guidance via a QED surrogate.

    At each denoising step:
      1. BERT predicts logits for all positions
      2. Soft token probs: p = softmax(logits / temp)
      3. Surrogate predicts QED from soft embeddings (differentiable)
      4. Backprop: grad_logits = d(QED_pred) / d(logits)
      5. Shift logits: logits += guidance_scale * grad_logits
      6. MDLM samples tokens from shifted logits as usual

    The gradient flows through:
      logits -> softmax -> matmul with embedding weights -> surrogate -> QED
    All differentiable. No Gumbel-Softmax or straight-through needed.
    """

    def __init__(self, path, surrogate_path, guidance_scale=1.0,
                 guidance_schedule="constant", forward_op=None, **kwargs):
        """
        Args:
            path:             GenMol checkpoint path
            surrogate_path:   Trained QED surrogate checkpoint
            guidance_scale:   Strength of reward guidance (like DPS step size)
            guidance_schedule: "constant", "linear_decay", or "linear_warmup"
        """
        super().__init__(path, **kwargs)
        self.surrogate = load_surrogate(surrogate_path, device=self.model.device)
        self.surrogate.eval()
        self.guidance_scale = guidance_scale
        self.guidance_schedule = guidance_schedule

    def _get_guidance_scale(self, step, total_steps):
        """Guidance strength schedule (like DPS step size scheduling)."""
        frac = step / max(total_steps - 1, 1)
        if self.guidance_schedule == "linear_decay":
            # High early (explore), low late (exploit)
            return self.guidance_scale * (1.0 - frac)
        elif self.guidance_schedule == "linear_warmup":
            # Low early, high late
            return self.guidance_scale * frac
        else:  # constant
            return self.guidance_scale

    def _compute_guidance(self, logits, x, attention_mask):
        """
        Compute reward gradient w.r.t. logits.

        The key differentiable path:
          logits -> softmax(logits) -> soft_emb = probs @ emb_weight -> surrogate -> QED

        Returns gradient of QED w.r.t. logits at masked positions.
        """
        # Only guide masked positions (unmasked tokens are already decided)
        mask_positions = (x == self.model.mask_index)
        if not mask_positions.any():
            return torch.zeros_like(logits)

        # Enable gradient tracking on logits
        logits_for_grad = logits.detach().requires_grad_(True)

        # Soft token probabilities (differentiable)
        token_probs = F.softmax(logits_for_grad, dim=-1)  # [B, L, V]

        # Build attention mask for surrogate (non-pad positions)
        surr_mask = attention_mask.float()

        # Forward through surrogate
        qed_pred = self.surrogate.forward_soft(token_probs, surr_mask)  # [B]

        # Backward: gradient of total predicted QED w.r.t. logits
        qed_pred.sum().backward()
        grad = logits_for_grad.grad  # [B, L, V]

        # Normalize gradient per-sample (like TTT's per-sample normalization)
        grad_norm = grad.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-8)
        grad = grad / grad_norm.unsqueeze(-1)

        # Only apply gradient at masked positions
        grad = grad * mask_positions.unsqueeze(-1).float()

        return grad

    @torch.no_grad()
    def generate(self, x, softmax_temp=1.2, randomness=2, fix=True,
                 gamma=0, w=2, **kwargs):
        """
        Guided generation: MDLM denoising with DPS-style reward guidance.
        """
        x = x.to(self.model.device)
        num_steps = max(self.mdlm.get_num_steps_confidence(x), 2)
        attention_mask = x != self.pad_index

        for i in range(num_steps):
            logits = self.model(x, attention_mask)

            # MCG (classifier-free guidance) if configured
            if gamma and w:
                import random as _random
                x_poor = x.clone()
                context_tokens = (
                    (x_poor[0] != self.model.bos_index).to(int)
                    * (x_poor[0] != self.model.eos_index).to(int)
                    * (x_poor[0] != self.model.mask_index).to(int)
                    * (x_poor[0] != self.pad_index).to(int)
                )
                context_token_ids = context_tokens.nonzero(as_tuple=True)[0].tolist()
                num_mask_poor = int(context_tokens.sum() * gamma)
                mask_idx_poor = _random.sample(context_token_ids, num_mask_poor)
                x_poor[:, mask_idx_poor] = self.model.mask_index
                logits_poor = self.model(x_poor, attention_mask=attention_mask)
                logits = w * logits + (1 - w) * logits_poor

            # DPS-style reward guidance (the key addition)
            scale = self._get_guidance_scale(i, num_steps)
            if scale > 0 and (x == self.model.mask_index).any():
                # Temporarily enable grad for surrogate backward
                with torch.enable_grad():
                    grad = self._compute_guidance(logits, x, attention_mask)
                logits = logits + scale * grad

            # Standard MDLM confidence-based unmasking
            x = self.mdlm.step_confidence(
                logits, x, i, num_steps, softmax_temp, randomness
            )

        # Decode to SMILES (same as base Sampler)
        samples = self.model.tokenizer.batch_decode(x, skip_special_tokens=True)
        if self.model.config.training.get("use_bracket_safe"):
            samples = [safe_to_smiles(bracketsafe2safe(s), fix=fix) for s in samples]
        else:
            samples = [safe_to_smiles(s, fix=fix) for s in samples]
        samples = [sorted(s.split("."), key=len)[-1] for s in samples if s]
        return samples

    @torch.no_grad()
    def de_novo_generation(self, num_samples=1, softmax_temp=0.8,
                           randomness=0.5, min_add_len=40, **kwargs):
        """Generate molecules with reward guidance."""
        x = torch.hstack([
            torch.full((1, 1), self.model.bos_index),
            torch.full((1, 1), self.model.eos_index),
        ])
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        x = x.to(self.model.device)
        return self.generate(x, softmax_temp, randomness, **kwargs)
