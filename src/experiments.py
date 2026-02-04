import os
import random
from typing import Dict, List

import torch
import yaml

from .data import create_dataloaders, get_datasets
from .model import MLP
from .reporting import (
    compute_stats,
    running_best,
    save_dicts_csv,
    save_history_csv,
    save_plot,
)
from .search import sample_adamw
from .train import evaluate, train_with_early_stopping
from .utils import ensure_dir, save_yaml, set_seed


def _architecture_name(layers: List[int], dropout: float, alpha: float) -> str:
    layer_str = "x".join(str(l) for l in layers)
    return f"mlp_{len(layers)}x_{layer_str}_do{dropout}_a{alpha}"


def generate_architectures(arch_cfg: Dict) -> List[Dict]:
    architectures: List[Dict] = []

    for layers in arch_cfg["two_layer_widths"]:
        for dropout in arch_cfg["dropouts"]:
            for alpha in arch_cfg["leaky_relu_alphas"]:
                architectures.append(
                    {
                        "name": _architecture_name(layers, dropout, alpha),
                        "layers": layers,
                        "dropout": dropout,
                        "leaky_relu_alpha": alpha,
                    }
                )

    for layers in arch_cfg["three_layer_widths"]:
        for dropout in arch_cfg["dropouts"]:
            for alpha in arch_cfg["leaky_relu_alphas"]:
                architectures.append(
                    {
                        "name": _architecture_name(layers, dropout, alpha),
                        "layers": layers,
                        "dropout": dropout,
                        "leaky_relu_alpha": alpha,
                    }
                )

    return architectures


def _build_model(arch: Dict, device: torch.device) -> torch.nn.Module:
    model = MLP(
        layers=arch["layers"],
        dropout=arch["dropout"],
        leaky_relu_alpha=arch["leaky_relu_alpha"],
    )
    return model.to(device)


def _create_optimizer(model, hparams: Dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=hparams["lr"],
        betas=(hparams["beta1"], hparams["beta2"]),
        eps=hparams["eps"],
        weight_decay=hparams["weight_decay"],
    )


def _run_trial(
    arch: Dict,
    hparams: Dict,
    datasets,
    cfg: Dict,
    device: torch.device,
    seed: int,
    output_dir: str | None,
    compute_test: bool,
    log_interval: int = 0,
    log_prefix: str | None = None,
    save_history: bool = True,
    save_plot_flag: bool = True,
    save_config: bool = True,
):
    set_seed(seed, cfg["deterministic"])

    train_dataset, val_dataset, test_dataset = datasets
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        cfg["training"],
        cfg["evaluation"],
        cfg["data"],
        seed,
    )

    model = _build_model(arch, device)
    optimizer = _create_optimizer(model, hparams)

    if log_prefix:
        print(f"[{log_prefix}] starting", flush=True)

    history, best_state, best_epoch, best_val_loss, terminating_epoch = (
        train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            max_epochs=cfg["training"]["max_epochs"],
            patience=cfg["training"]["patience"],
            min_delta=cfg["training"]["min_delta"],
            pin_memory=cfg["data"]["pin_memory"],
            val_interval=cfg["training"].get("val_interval", 1),
            val_schedule=cfg["training"].get("val_schedule"),
            log_interval=log_interval,
            log_prefix=log_prefix,
        )
    )

    model.load_state_dict(best_state)
    test_loss = None
    if compute_test:
        test_loss = evaluate(
            model,
            test_loader,
            torch.nn.CrossEntropyLoss(reduction="mean"),
            device,
            cfg["data"]["pin_memory"],
        )

    if output_dir and save_config:
        ensure_dir(output_dir)
        if save_history:
            save_history_csv(os.path.join(output_dir, "metrics.csv"), history)
            if cfg["artifacts"]["save_plots"] and save_plot_flag:
                save_plot(os.path.join(output_dir, "loss_curve.png"), history)

        trial_config = {
            "architecture": arch,
            "hyperparameters": hparams,
            "seed": seed,
            "best_validation_loss": best_val_loss,
            "best_epoch": best_epoch,
            "terminating_epoch": terminating_epoch,
            "test_loss": test_loss,
        }
        save_yaml(os.path.join(output_dir, "config.yaml"), trial_config)

    if log_prefix:
        test_str = f"{test_loss:.4f}" if test_loss is not None else "n/a"
        print(
            f"[{log_prefix}] done "
            f"best_val={best_val_loss:.4f} "
            f"best_epoch={best_epoch} "
            f"terminating_epoch={terminating_epoch} "
            f"test_loss={test_str}",
            flush=True,
        )

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "terminating_epoch": terminating_epoch,
        "test_loss": test_loss,
    }


def run_phase_a(cfg: Dict, search_space: Dict, device: torch.device) -> None:
    output_dir = cfg["output_dir"]
    ensure_dir(output_dir)

    datasets = get_datasets(cfg["data"], cfg["seed"], device)
    log_interval = cfg["training"].get("log_interval", 0)

    arch = cfg["architecture"]
    if "name" not in arch:
        arch = {
            **arch,
            "name": _architecture_name(
                arch["layers"], arch["dropout"], arch["leaky_relu_alpha"]
            ),
        }

    rng = random.Random(cfg["search"]["seed"])

    best_val_loss = float("inf")
    best_hparams = None
    trial_summaries = []
    val_losses = []

    for trial_idx in range(cfg["search"]["num_trials"]):
        trial_label = f"phase_a trial {trial_idx + 1}/{cfg['search']['num_trials']}"
        hparams = sample_adamw(search_space["adamw"], rng)
        trial_seed = cfg["seed"]
        trial_dir = os.path.join(output_dir, f"trial_{trial_idx:03d}")

        result = _run_trial(
            arch=arch,
            hparams=hparams,
            datasets=datasets,
            cfg=cfg,
            device=device,
            seed=trial_seed,
            output_dir=trial_dir,
            compute_test=False,
            log_interval=log_interval,
            log_prefix=trial_label,
        )

        val_losses.append(result["best_val_loss"])
        trial_summaries.append(
            {
                "trial": trial_idx,
                "best_val_loss": result["best_val_loss"],
                "best_epoch": result["best_epoch"],
                "terminating_epoch": result["terminating_epoch"],
                **hparams,
            }
        )

        if result["best_val_loss"] < best_val_loss:
            best_val_loss = result["best_val_loss"]
            best_hparams = hparams

    save_dicts_csv(os.path.join(output_dir, "trial_summaries.csv"), trial_summaries)

    running_best_vals = running_best(val_losses)
    budget_curve = [
        {"trial": i, "best_val_loss": running_best_vals[i]} for i in range(len(val_losses))
    ]
    save_dicts_csv(os.path.join(output_dir, "budget_curve.csv"), budget_curve)

    if cfg["artifacts"]["save_plots"]:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            plt.plot([b["trial"] + 1 for b in budget_curve], [b["best_val_loss"] for b in budget_curve])
            plt.xlabel("Trial")
            plt.ylabel("Running Best Validation Loss")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "budget_curve.png"))
            plt.close()
        except Exception:
            pass

    summary = {
        "architecture": arch,
        "best_hyperparameters": best_hparams,
        "best_validation_loss": best_val_loss,
        "num_trials": cfg["search"]["num_trials"],
    }
    save_yaml(os.path.join(output_dir, "summary.yaml"), summary)


def run_phase_b(cfg: Dict, arch_cfg: Dict, search_space: Dict, device: torch.device) -> None:
    output_dir = cfg["output_dir"]
    ensure_dir(output_dir)

    datasets = get_datasets(cfg["data"], cfg["seed"], device)
    architectures = generate_architectures(arch_cfg)
    log_interval = cfg["training"].get("log_interval", 0)

    overall_summary = []

    for arch in architectures:
        arch_dir = os.path.join(output_dir, arch["name"])
        ensure_dir(arch_dir)

        rng = random.Random(cfg["search"]["seed"])
        best_val_loss = float("inf")
        best_hparams = None
        best_history = None
        best_epoch = None
        best_terminating_epoch = None

        search_results = []

        for trial_idx in range(cfg["search"]["num_trials"]):
            trial_label = (
                f"phase_b {arch['name']} trial {trial_idx + 1}/{cfg['search']['num_trials']}"
            )
            hparams = sample_adamw(search_space["adamw"], rng)
            trial_seed = cfg["seed"]

            result = _run_trial(
                arch=arch,
                hparams=hparams,
                datasets=datasets,
                cfg=cfg,
                device=device,
                seed=trial_seed,
                output_dir=None,
                compute_test=False,
                log_interval=log_interval,
                log_prefix=trial_label,
                save_history=False,
                save_plot_flag=False,
                save_config=False,
            )

            search_results.append(
                {
                    "trial": trial_idx,
                    "best_val_loss": result["best_val_loss"],
                    "best_epoch": result["best_epoch"],
                    "terminating_epoch": result["terminating_epoch"],
                    **hparams,
                }
            )

            if result["best_val_loss"] < best_val_loss:
                best_val_loss = result["best_val_loss"]
                best_hparams = hparams
                best_history = result["history"]
                best_epoch = result["best_epoch"]
                best_terminating_epoch = result["terminating_epoch"]

        save_dicts_csv(os.path.join(arch_dir, "search_results.csv"), search_results)

        best_trial_dir = os.path.join(arch_dir, "best_trial")
        ensure_dir(best_trial_dir)
        save_history_csv(os.path.join(best_trial_dir, "metrics.csv"), best_history or [])
        if cfg["artifacts"]["save_plots"] and best_history:
            save_plot(os.path.join(best_trial_dir, "loss_curve.png"), best_history)

        save_yaml(
            os.path.join(arch_dir, "best_hyperparameters.yaml"),
            {
                "architecture": arch,
                "hyperparameters": best_hparams,
                "best_validation_loss": best_val_loss,
                "best_epoch": best_epoch,
                "terminating_epoch": best_terminating_epoch,
            },
        )

        retrain_dir = os.path.join(arch_dir, "retrain")
        ensure_dir(retrain_dir)
        test_losses = []
        seed_rows = []
        for seed in cfg["retrain"]["seeds"]:
            seed_dir = os.path.join(retrain_dir, f"seed_{seed}")
            result = _run_trial(
                arch=arch,
                hparams=best_hparams,
                datasets=datasets,
                cfg=cfg,
                device=device,
                seed=seed,
                output_dir=seed_dir,
                compute_test=True,
                log_interval=log_interval,
                log_prefix=f"phase_b {arch['name']} retrain seed {seed}",
                save_history=False,
                save_plot_flag=False,
                save_config=True,
            )
            test_losses.append(result["test_loss"])
            seed_rows.append({"seed": seed, "test_loss": result["test_loss"]})

        save_dicts_csv(os.path.join(retrain_dir, "seed_losses.csv"), seed_rows)

        test_stats = compute_stats(test_losses)
        summary = {
            "architecture": arch,
            "best_validation_loss": best_val_loss,
            "test_loss_stats": test_stats,
        }
        save_yaml(os.path.join(arch_dir, "summary.yaml"), summary)
        overall_summary.append(summary)

    save_yaml(os.path.join(output_dir, "summary.yaml"), {"architectures": overall_summary})


def run_transfer(cfg: Dict, arch_cfg: Dict, device: torch.device) -> None:
    output_dir = cfg["output_dir"]
    ensure_dir(output_dir)

    datasets = get_datasets(cfg["data"], cfg["seed"], device)
    architectures = generate_architectures(arch_cfg)
    log_interval = cfg["training"].get("log_interval", 0)

    # Load best hyperparameters from phase B
    phase_b_dir = cfg["transfer"]["phase_b_dir"]
    best_hparams = {}
    phase_b_test_means = {}

    for arch in architectures:
        arch_dir = os.path.join(phase_b_dir, arch["name"])
        hparams_path = os.path.join(arch_dir, "best_hyperparameters.yaml")
        summary_path = os.path.join(arch_dir, "summary.yaml")
        if not os.path.exists(hparams_path) or not os.path.exists(summary_path):
            raise FileNotFoundError(
                f"Missing phase B artifacts for {arch['name']} in {arch_dir}"
            )
        with open(hparams_path, "r", encoding="utf-8") as f:
            hparams_data = yaml.safe_load(f)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = yaml.safe_load(f)

        best_hparams[arch["name"]] = hparams_data["hyperparameters"]
        phase_b_test_means[arch["name"]] = summary_data["test_loss_stats"]["mean"]

    transfer_results = []
    transfer_seed_rows = []

    for target_arch in architectures:
        for source_arch in architectures:
            source_name = source_arch["name"]
            hparams = best_hparams[source_name]

            test_losses = []
            for seed in cfg["transfer"]["seeds"]:
                result = _run_trial(
                    arch=target_arch,
                    hparams=hparams,
                    datasets=datasets,
                    cfg=cfg,
                    device=device,
                    seed=seed,
                    output_dir=None,
                    compute_test=True,
                    log_interval=log_interval,
                    log_prefix=(
                        f"transfer target {target_arch['name']} "
                        f"source {source_name} seed {seed}"
                    ),
                    save_history=False,
                    save_plot_flag=False,
                    save_config=False,
                )
                test_losses.append(result["test_loss"])
                transfer_seed_rows.append(
                    {
                        "target_arch": target_arch["name"],
                        "source_arch": source_name,
                        "seed": seed,
                        "test_loss": result["test_loss"],
                    }
                )

            stats = compute_stats(test_losses)
            transfer_results.append(
                {
                    "target_arch": target_arch["name"],
                    "source_arch": source_name,
                    **stats,
                }
            )

    save_dicts_csv(os.path.join(output_dir, "transfer_results.csv"), transfer_results)
    save_dicts_csv(os.path.join(output_dir, "transfer_seed_losses.csv"), transfer_seed_rows)

    # Compute regret matrix
    regret_matrix = []
    for result in transfer_results:
        target = result["target_arch"]
        source = result["source_arch"]
        baseline = phase_b_test_means[target]
        regret_matrix.append(
            {
                "target_arch": target,
                "source_arch": source,
                "mean_regret": result["mean"] - baseline,
                "var_regret": result["var"],
                "min_regret": result["min"] - baseline,
                "max_regret": result["max"] - baseline,
            }
        )

    save_dicts_csv(os.path.join(output_dir, "regret_matrix.csv"), regret_matrix)
