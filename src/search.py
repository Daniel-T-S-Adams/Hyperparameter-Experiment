import math
import random
from typing import Dict


def _sample_log_uniform(rng: random.Random, min_val: float, max_val: float) -> float:
    log_min = math.log(min_val)
    log_max = math.log(max_val)
    return math.exp(rng.uniform(log_min, log_max))


def _sample_uniform(rng: random.Random, min_val: float, max_val: float) -> float:
    return rng.uniform(min_val, max_val)


def _sample_parameter(rng: random.Random, cfg: Dict) -> float:
    if "choices" in cfg:
        return float(rng.choice(cfg["choices"]))

    scale = cfg.get("scale", "uniform")
    if scale == "log":
        return _sample_log_uniform(rng, cfg["min"], cfg["max"])
    if scale == "uniform":
        return _sample_uniform(rng, cfg["min"], cfg["max"])

    raise ValueError(f"Unsupported scale '{scale}' in search space config")


def sample_adamw(search_space: Dict, rng: random.Random) -> Dict[str, float]:
    return {
        "lr": _sample_parameter(rng, search_space["lr"]),
        "beta1": _sample_parameter(rng, search_space["beta1"]),
        "beta2": _sample_parameter(rng, search_space["beta2"]),
        "eps": _sample_parameter(rng, search_space["eps"]),
        "weight_decay": _sample_parameter(rng, search_space["weight_decay"]),
    }


def sample_model_hyperparameters(search_space: Dict, rng: random.Random) -> Dict[str, float]:
    return {
        "dropout": _sample_parameter(rng, search_space["dropout"]),
        "leaky_relu_alpha": _sample_parameter(rng, search_space["leaky_relu_alpha"]),
    }


def sample_hyperparameters(search_space: Dict, rng: random.Random) -> Dict[str, float]:
    return {
        **sample_adamw(search_space["adamw"], rng),
        **sample_model_hyperparameters(search_space["model"], rng),
    }
