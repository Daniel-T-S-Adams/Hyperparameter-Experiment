import math
import random
from typing import Dict


def _sample_log_uniform(rng: random.Random, min_val: float, max_val: float) -> float:
    log_min = math.log(min_val)
    log_max = math.log(max_val)
    return math.exp(rng.uniform(log_min, log_max))


def _sample_uniform(rng: random.Random, min_val: float, max_val: float) -> float:
    return rng.uniform(min_val, max_val)


def sample_adamw(search_space: Dict, rng: random.Random) -> Dict[str, float]:
    lr_cfg = search_space["lr"]
    beta1_cfg = search_space["beta1"]
    beta2_cfg = search_space["beta2"]
    eps_cfg = search_space["eps"]
    wd_cfg = search_space["weight_decay"]

    lr = _sample_log_uniform(rng, lr_cfg["min"], lr_cfg["max"])
    beta1 = _sample_uniform(rng, beta1_cfg["min"], beta1_cfg["max"])
    beta2 = _sample_uniform(rng, beta2_cfg["min"], beta2_cfg["max"])
    eps = _sample_log_uniform(rng, eps_cfg["min"], eps_cfg["max"])
    weight_decay = _sample_log_uniform(rng, wd_cfg["min"], wd_cfg["max"])

    return {
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "weight_decay": weight_decay,
    }
