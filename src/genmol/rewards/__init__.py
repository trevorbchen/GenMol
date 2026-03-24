"""Axis 3: Reward / forward operator functions for guided generation.

Each reward is a callable: ``List[str] -> torch.Tensor`` of scores.

Available rewards (increasing compute cost):
    - QEDReward, LogPReward, MolecularWeightReward, TPSAReward  (~free, RDKit)
    - FlashAffinityForwardOp  (~100-500ms/mol, neural binding prediction)
    - BoltzAffinityReward     (~3-10s/mol, structure-based oracle)

Registry:
    >>> from genmol.rewards import get_reward
    >>> reward = get_reward("qed")
    >>> reward = get_reward("flash_affinity", protein_id="2VT4")
"""

from .threshold import ThresholdReward
from .properties import (
    RewardFunction,
    MolecularWeightReward,
    QEDReward,
    LogPReward,
    TPSAReward,
    MolecularWeightForwardOp,  # backward-compat alias
    QEDForwardOp,              # backward-compat alias
    _safe_mol,
)

# ── Registry ──────────────────────────────────────────────────────────

REWARD_REGISTRY = {
    "mw": MolecularWeightReward,
    "qed": QEDReward,
    "logp": LogPReward,
    "tpsa": TPSAReward,
    # Expensive rewards — lazy-loaded to avoid heavy imports when not needed.
    # String values are "module.ClassName" resolved on first use.
    "flash_affinity": "genmol.rewards.flash_affinity.FlashAffinityForwardOp",
    "boltz": "genmol.rewards.boltz.BoltzAffinityReward",
}


def get_reward(name: str, **kwargs):
    """Look up a reward by short name.  Returns an *instance*.

    For parameterised rewards (e.g. flash_affinity), pass constructor
    kwargs directly::

        get_reward("flash_affinity", protein_id="2VT4", task="binary")
    """
    key = name.lower().strip()
    if key in ("none", ""):
        return None
    entry = REWARD_REGISTRY.get(key)
    if entry is None:
        raise ValueError(
            f"Unknown reward {name!r}. "
            f"Available: {', '.join(REWARD_REGISTRY)}"
        )
    if isinstance(entry, str):
        import importlib
        module_path, class_name = entry.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    else:
        cls = entry
    return cls(**kwargs)



# ── Cached reward wrapper ──────────────────────────────────────────────

class CachedReward:
    """Wraps any reward fn and persists (SMILES -> score) pairs to a JSONL file."""

    def __init__(self, base_reward_fn, cache_path):
        import json, os, torch
        self._fn = base_reward_fn
        self._path = cache_path
        self._cache = {}

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self._cache[entry["smiles"]] = entry["score"]
            print("[CachedReward] Loaded %d cached FA scores from %s" % (len(self._cache), cache_path), flush=True)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            print("[CachedReward] Starting fresh cache at %s" % cache_path, flush=True)

    def __call__(self, smiles_list):
        import json, torch
        results = [None] * len(smiles_list)
        uncached_idx = []
        uncached_smi = []

        for i, smi in enumerate(smiles_list):
            if smi in self._cache:
                results[i] = self._cache[smi]
            else:
                uncached_idx.append(i)
                uncached_smi.append(smi)

        if uncached_smi:
            new_scores = self._fn(uncached_smi)
            newline = chr(10)
            with open(self._path, "a") as fh:
                for smi, score_t in zip(uncached_smi, new_scores):
                    score = float(score_t)
                    self._cache[smi] = score
                    fh.write(json.dumps({"smiles": smi, "score": score}) + newline)
            for i, score_t in zip(uncached_idx, new_scores):
                results[i] = float(score_t)

        return torch.tensor(results, dtype=torch.float32)

    def __getattr__(self, name):
        return getattr(self._fn, name)

    @property
    def cache_size(self):
        return len(self._cache)


__all__ = [
    "RewardFunction",
    "MolecularWeightReward",
    "QEDReward",
    "LogPReward",
    "TPSAReward",
    "MolecularWeightForwardOp",
    "QEDForwardOp",
    "REWARD_REGISTRY",
    "get_reward",
    "ThresholdReward",
    "CachedReward",
]
