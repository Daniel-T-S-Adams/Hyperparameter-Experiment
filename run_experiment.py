import argparse
import os
import re
from datetime import datetime

from src.experiments import run_search, run_search_report, run_transfer
from src.utils import deep_update, ensure_dir, load_yaml, resolve_device, save_yaml


def load_config(defaults_path: str, phase_path: str):
    defaults = load_yaml(defaults_path)
    phase_cfg = load_yaml(phase_path)
    return deep_update(defaults, phase_cfg)


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("._-")
    return cleaned or "experiment"


def _unique_path(base_path: str) -> str:
    if not os.path.exists(base_path):
        return base_path

    suffix = 2
    while True:
        candidate = f"{base_path}_{suffix}"
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def _build_run_dirs(runs_root: str, project_name: str, run_name: str | None) -> tuple[str, str, str]:
    if run_name:
        stem = _sanitize_name(run_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{_sanitize_name(project_name)}_{timestamp}"

    run_dir = _unique_path(os.path.join(runs_root, stem))
    search_dir = os.path.join(run_dir, "search")
    transfer_dir = os.path.join(run_dir, "transfer")
    return run_dir, search_dir, transfer_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST MLP Hyperparameter Experiment")
    parser.add_argument(
        "--phase",
        choices=["search", "search_report", "transfer", "run"],
        required=True,
    )
    parser.add_argument("--defaults", default="configs/defaults.yaml")
    parser.add_argument("--search-space", default="configs/search_space.yaml")
    parser.add_argument("--architectures", default="configs/architectures.yaml")
    parser.add_argument("--config", help="Phase-specific config YAML")
    parser.add_argument(
        "--search-config",
        default="configs/search.yaml",
        help="Search config for --phase run",
    )
    parser.add_argument(
        "--transfer-config",
        default="configs/transfer.yaml",
        help="Transfer config for --phase run",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory used by --phase run for auto-named outputs",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name used by --phase run (otherwise timestamped)",
    )

    args = parser.parse_args()

    search_space = load_yaml(args.search_space)
    arch_cfg = load_yaml(args.architectures)

    if args.phase == "run":
        search_cfg = load_config(args.defaults, args.search_config)
        transfer_cfg = load_config(args.defaults, args.transfer_config)

        project_name = search_cfg.get("project_name", "experiment")
        run_dir, search_dir, transfer_dir = _build_run_dirs(
            runs_root=args.runs_root,
            project_name=project_name,
            run_name=args.run_name,
        )
        ensure_dir(run_dir)

        search_cfg["output_dir"] = search_dir
        transfer_cfg["output_dir"] = transfer_dir
        transfer_cfg.setdefault("transfer", {})
        transfer_cfg["transfer"]["search_dir"] = search_dir

        save_yaml(
            os.path.join(run_dir, "run_manifest.yaml"),
            {
                "run_dir": run_dir,
                "search_output_dir": search_dir,
                "transfer_output_dir": transfer_dir,
                "search_config_path": args.search_config,
                "transfer_config_path": args.transfer_config,
            },
        )

        search_device = resolve_device(search_cfg.get("device", "auto"))
        transfer_device = resolve_device(
            transfer_cfg.get("device", search_cfg.get("device", "auto"))
        )
        print(f"[run] run_dir={run_dir}", flush=True)
        print(f"[run] search_output_dir={search_dir}", flush=True)
        print(f"[run] transfer_output_dir={transfer_dir}", flush=True)
        run_search(search_cfg, arch_cfg, search_space, search_device)
        run_transfer(transfer_cfg, arch_cfg, transfer_device)
        print("[run] complete", flush=True)
        return

    if not args.config:
        parser.error("--config is required for phases: search, search_report, transfer")

    cfg = load_config(args.defaults, args.config)
    device = resolve_device(cfg.get("device", "auto"))

    if args.phase == "search":
        run_search(cfg, arch_cfg, search_space, device)
    elif args.phase == "search_report":
        run_search_report(cfg, arch_cfg)
    else:
        run_transfer(cfg, arch_cfg, device)


if __name__ == "__main__":
    main()
