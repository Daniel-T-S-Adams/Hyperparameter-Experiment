import argparse

from src.experiments import run_search, run_transfer
from src.utils import deep_update, load_yaml, resolve_device


def load_config(defaults_path: str, phase_path: str):
    defaults = load_yaml(defaults_path)
    phase_cfg = load_yaml(phase_path)
    return deep_update(defaults, phase_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST MLP Hyperparameter Experiment")
    parser.add_argument("--phase", choices=["search", "transfer"], required=True)
    parser.add_argument("--defaults", default="configs/defaults.yaml")
    parser.add_argument("--search-space", default="configs/search_space.yaml")
    parser.add_argument("--architectures", default="configs/architectures.yaml")
    parser.add_argument("--config", required=True, help="Phase-specific config YAML")

    args = parser.parse_args()

    cfg = load_config(args.defaults, args.config)
    search_space = load_yaml(args.search_space)
    arch_cfg = load_yaml(args.architectures)

    device = resolve_device(cfg.get("device", "auto"))

    if args.phase == "search":
        run_search(cfg, arch_cfg, search_space, device)
    else:
        run_transfer(cfg, arch_cfg, device)


if __name__ == "__main__":
    main()
