import os
import random
from typing import Dict, List

import torch
import yaml

from .data import create_dataloaders, get_datasets
from .model import MLP
from .reporting import (
    compute_stats,
    load_dicts_csv,
    running_best,
    save_dicts_csv,
    save_history_csv,
    save_plot,
)
from .search import sample_hyperparameters
from .train import evaluate, train_with_early_stopping
from .utils import ensure_dir, save_yaml, set_seed


def _architecture_name(layers: List[int]) -> str:
    layer_str = "x".join(str(l) for l in layers)
    return f"mlp_{len(layers)}x_{layer_str}"


def generate_architectures(arch_cfg: Dict) -> List[Dict]:
    architectures: List[Dict] = []

    layer_keys = [key for key in arch_cfg.keys() if key.endswith("_layer_widths")]
    for key in layer_keys:
        for layers in arch_cfg[key]:
            architectures.append(
                {
                    "name": _architecture_name(layers),
                    "layers": layers,
                }
            )

    return architectures


def _build_model(arch: Dict, hparams: Dict, device: torch.device) -> torch.nn.Module:
    model = MLP(
        layers=arch["layers"],
        dropout=hparams["dropout"],
        leaky_relu_alpha=hparams["leaky_relu_alpha"],
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

    model = _build_model(arch, hparams, device)
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


def run_search(cfg: Dict, arch_cfg: Dict, search_space: Dict, device: torch.device) -> None:
    output_dir = cfg["output_dir"]
    ensure_dir(output_dir)

    datasets = get_datasets(cfg["data"], cfg["seed"], device)
    architectures = generate_architectures(arch_cfg)
    log_interval = cfg["training"].get("log_interval", 0)

    overall_summary = []

    rng = random.Random(cfg["search"]["seed"])
    hparam_trials = [
        sample_hyperparameters(search_space, rng) for _ in range(cfg["search"]["num_trials"])
    ]

    for arch in architectures:
        arch_dir = os.path.join(output_dir, arch["name"])
        ensure_dir(arch_dir)

        best_val_loss = float("inf")
        best_hparams = None
        best_history = None
        best_epoch = None
        best_terminating_epoch = None
        best_trial_idx = 0

        search_results = []

        for trial_idx in range(cfg["search"]["num_trials"]):
            trial_label = (
                f"search {arch['name']} trial {trial_idx + 1}/{cfg['search']['num_trials']}"
            )
            hparams = dict(hparam_trials[trial_idx])
            trial_seed = cfg["seed"]

            result = _run_trial(
                arch=arch,
                hparams=hparams,
                datasets=datasets,
                cfg=cfg,
                device=device,
                seed=trial_seed,
                output_dir=None,
                compute_test=True,
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
                    "test_loss": result["test_loss"],
                    "best_epoch": result["best_epoch"],
                    "terminating_epoch": result["terminating_epoch"],
                    **hparams,
                }
            )

            if result["best_val_loss"] < best_val_loss:
                best_val_loss = result["best_val_loss"]
                best_hparams = hparams
                best_trial_idx = trial_idx
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
                "best_trial_idx": best_trial_idx,
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
                log_prefix=f"search {arch['name']} retrain seed {seed}",
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
    save_yaml(
        os.path.join(output_dir, "search_meta.yaml"),
        {"search_training_seed": cfg["seed"]},
    )


def run_transfer(cfg: Dict, arch_cfg: Dict, device: torch.device) -> None:
    output_dir = cfg["output_dir"]
    ensure_dir(output_dir)

    datasets = get_datasets(cfg["data"], cfg["seed"], device)
    architectures = generate_architectures(arch_cfg)
    log_interval = cfg["training"].get("log_interval", 0)

    # Load best hyperparameters from search phase
    search_dir = cfg["transfer"]["search_dir"]
    best_hparams = {}
    search_test_means = {}
    best_trial_indices = {}

    # Retrain results: {arch_name: {seed: test_loss}}
    retrain_test_losses = {}
    # Search results: {arch_name: {trial_idx: test_loss}}
    search_test_losses = {}

    for arch in architectures:
        arch_dir = os.path.join(search_dir, arch["name"])
        hparams_path = os.path.join(arch_dir, "best_hyperparameters.yaml")
        summary_path = os.path.join(arch_dir, "summary.yaml")
        if not os.path.exists(hparams_path) or not os.path.exists(summary_path):
            raise FileNotFoundError(
                f"Missing search phase artifacts for {arch['name']} in {arch_dir}"
            )
        with open(hparams_path, "r", encoding="utf-8") as f:
            hparams_data = yaml.safe_load(f)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = yaml.safe_load(f)

        best_hparams[arch["name"]] = hparams_data["hyperparameters"]
        search_test_means[arch["name"]] = summary_data["test_loss_stats"]["mean"]

        # Load best trial index (may be absent in older search outputs)
        if "best_trial_idx" in hparams_data:
            best_trial_indices[arch["name"]] = hparams_data["best_trial_idx"]

        # Load retrain seed losses
        retrain_csv_path = os.path.join(arch_dir, "retrain", "seed_losses.csv")
        if os.path.exists(retrain_csv_path):
            rows = load_dicts_csv(retrain_csv_path)
            retrain_test_losses[arch["name"]] = {
                int(row["seed"]): float(row["test_loss"]) for row in rows
            }

        # Load search results with test losses
        search_csv_path = os.path.join(arch_dir, "search_results.csv")
        if os.path.exists(search_csv_path):
            rows = load_dicts_csv(search_csv_path)
            if rows and "test_loss" in rows[0]:
                search_test_losses[arch["name"]] = {
                    int(row["trial"]): float(row["test_loss"]) for row in rows
                }

    # Load search training seed (may be absent in older search outputs)
    search_training_seed = None
    search_meta_path = os.path.join(search_dir, "search_meta.yaml")
    if os.path.exists(search_meta_path):
        with open(search_meta_path, "r", encoding="utf-8") as f:
            search_meta = yaml.safe_load(f)
        search_training_seed = search_meta["search_training_seed"]

    has_search_cache = bool(search_test_losses and best_trial_indices and search_training_seed is not None)
    has_retrain_cache = bool(retrain_test_losses)
    if has_retrain_cache or has_search_cache:
        cache_parts = []
        if has_retrain_cache:
            cache_parts.append("retrain results (diagonal)")
        if has_search_cache:
            cache_parts.append(f"search results (seed={search_training_seed})")
        print(f"[transfer] reusing search phase {', '.join(cache_parts)}", flush=True)
    else:
        print("[transfer] no search phase cache available, training all combinations", flush=True)

    def _hparam_signature(hparams: Dict[str, float]) -> tuple:
        return tuple(sorted((name, float(value)) for name, value in hparams.items()))

    source_signature = {
        arch["name"]: _hparam_signature(best_hparams[arch["name"]]) for arch in architectures
    }
    sources_by_signature = {}
    for source_name, signature in source_signature.items():
        sources_by_signature.setdefault(signature, []).append(source_name)

    unique_hparam_sets = len(sources_by_signature)
    total_sources = len(architectures)
    if unique_hparam_sets < total_sources:
        print(
            f"[transfer] found {unique_hparam_sets} unique hyperparameter sets "
            f"across {total_sources} source architectures; reusing equivalent columns",
            flush=True,
        )

    transfer_results = []
    transfer_seed_rows = []
    reused_count = 0
    trained_count = 0
    eval_cache = {}

    for target_arch in architectures:
        for source_arch in architectures:
            source_name = source_arch["name"]
            target_name = target_arch["name"]
            hparams = best_hparams[source_name]
            signature = source_signature[source_name]

            test_losses = []
            for seed in cfg["transfer"]["seeds"]:
                log_prefix = (
                    f"transfer target {target_name} "
                    f"source {source_name} seed {seed}"
                )
                test_loss = None
                cache_key = (target_name, seed, signature)

                if cache_key in eval_cache:
                    test_loss = eval_cache[cache_key]
                    reused_count += 1
                    print(
                        f"[{log_prefix}] reusing equivalent hyperparameters: "
                        f"test_loss={test_loss:.4f}",
                        flush=True,
                    )

                # Check if we can reuse a search phase result.
                # If the target's own optimum is equivalent to this source optimum,
                # prefer the target retrain cache and share it across equivalent sources.
                if (
                    test_loss is None
                    and has_retrain_cache
                    and target_name in sources_by_signature.get(signature, [])
                ):
                    cached = retrain_test_losses.get(target_name, {}).get(seed)
                    if cached is not None:
                        test_loss = cached
                        reused_count += 1
                        print(
                            f"[{log_prefix}] reusing retrain result "
                            f"(target-equivalent): test_loss={test_loss:.4f}",
                            flush=True,
                        )

                if test_loss is None and has_search_cache and seed == search_training_seed:
                    for equivalent_source in sources_by_signature.get(signature, []):
                        trial_idx = best_trial_indices.get(equivalent_source)
                        if trial_idx is None:
                            continue
                        cached = search_test_losses.get(target_name, {}).get(trial_idx)
                        if cached is None:
                            continue
                        test_loss = cached
                        reused_count += 1
                        print(
                            f"[{log_prefix}] reusing search result "
                            f"(source {equivalent_source}, trial {trial_idx}): "
                            f"test_loss={test_loss:.4f}",
                            flush=True,
                        )
                        break

                if test_loss is None:
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
                        log_prefix=log_prefix,
                        save_history=False,
                        save_plot_flag=False,
                        save_config=False,
                    )
                    test_loss = result["test_loss"]
                    trained_count += 1
                eval_cache[cache_key] = test_loss

                test_losses.append(test_loss)
                transfer_seed_rows.append(
                    {
                        "target_arch": target_name,
                        "source_arch": source_name,
                        "seed": seed,
                        "test_loss": test_loss,
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

    total = reused_count + trained_count
    print(
        f"[transfer] done: {trained_count} trained, {reused_count} reused, "
        f"{total} total",
        flush=True,
    )

    save_dicts_csv(os.path.join(output_dir, "transfer_results.csv"), transfer_results)
    save_dicts_csv(os.path.join(output_dir, "transfer_seed_losses.csv"), transfer_seed_rows)

    # Compute regret matrix
    regret_matrix = []
    relative_regret_matrix = []
    for result in transfer_results:
        target = result["target_arch"]
        source = result["source_arch"]
        baseline = search_test_means[target]
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
        relative_regret_matrix.append(
            {
                "target_arch": target,
                "source_arch": source,
                "mean_relative_regret": result["mean"] / baseline if baseline else float("nan"),
            }
        )

    save_dicts_csv(os.path.join(output_dir, "regret_matrix.csv"), regret_matrix)
    save_dicts_csv(os.path.join(output_dir, "relative_regret_matrix.csv"), relative_regret_matrix)

    if cfg["artifacts"]["save_plots"]:
        try:
            import matplotlib.pyplot as plt

            arch_names = [arch["name"] for arch in architectures]
            index = {name: idx for idx, name in enumerate(arch_names)}
            size = len(arch_names)

            # --- Transfer test loss heatmap ---
            loss_matrix = [[float("nan") for _ in range(size)] for _ in range(size)]
            for result in transfer_results:
                i = index[result["target_arch"]]
                j = index[result["source_arch"]]
                loss_matrix[i][j] = result["mean"]

            fig_w = max(6.0, size * 0.7)
            fig_h = max(5.0, size * 0.6)
            plt.figure(figsize=(fig_w, fig_h))
            plt.imshow(loss_matrix, cmap="viridis")
            plt.colorbar(label="Mean Test Loss")
            plt.xticks(range(size), arch_names, rotation=45, ha="right")
            plt.yticks(range(size), arch_names)
            plt.xlabel("Source Architecture (Hyperparameters)")
            plt.ylabel("Target Architecture")
            for i in range(size):
                for j in range(size):
                    val = loss_matrix[i][j]
                    if val == val:
                        plt.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "transfer_test_loss.png"))
            plt.close()

            # --- Regret heatmap ---
            regret_vals = [[float("nan") for _ in range(size)] for _ in range(size)]
            for row in regret_matrix:
                i = index[row["target_arch"]]
                j = index[row["source_arch"]]
                regret_vals[i][j] = row["mean_regret"]

            vals = [v for row in regret_vals for v in row if v == v]
            vmin = vmax = None
            if vals:
                max_abs = max(abs(min(vals)), abs(max(vals)))
                vmin, vmax = -max_abs, max_abs

            plt.figure(figsize=(fig_w, fig_h))
            plt.imshow(regret_vals, cmap="coolwarm", vmin=vmin, vmax=vmax)
            plt.colorbar(label="Mean Regret (Test Loss)")
            plt.xticks(range(size), arch_names, rotation=45, ha="right")
            plt.yticks(range(size), arch_names)
            plt.xlabel("Source Architecture (Hyperparameters)")
            plt.ylabel("Target Architecture")
            for i in range(size):
                for j in range(size):
                    val = regret_vals[i][j]
                    if val == val:
                        plt.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "regret_matrix_mean.png"))
            plt.close()

            # --- Relative regret heatmap ---
            rel_matrix = [[float("nan") for _ in range(size)] for _ in range(size)]
            for row in relative_regret_matrix:
                i = index[row["target_arch"]]
                j = index[row["source_arch"]]
                rel_matrix[i][j] = row["mean_relative_regret"]

            rel_vals = [v for row in rel_matrix for v in row if v == v]
            rel_vmin = rel_vmax = None
            if rel_vals:
                max_dev = max(abs(v - 1.0) for v in rel_vals)
                rel_vmin, rel_vmax = 1.0 - max_dev, 1.0 + max_dev

            plt.figure(figsize=(fig_w, fig_h))
            plt.imshow(rel_matrix, cmap="coolwarm", vmin=rel_vmin, vmax=rel_vmax)
            plt.colorbar(label="Relative Test Loss (transferred / own)")
            plt.xticks(range(size), arch_names, rotation=45, ha="right")
            plt.yticks(range(size), arch_names)
            plt.xlabel("Source Architecture (Hyperparameters)")
            plt.ylabel("Target Architecture")
            for i in range(size):
                for j in range(size):
                    val = rel_matrix[i][j]
                    if val == val:
                        plt.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "relative_regret_matrix.png"))
            plt.close()
        except Exception:
            pass
