from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from torch import nn
import time


def _to_device(tensor: torch.Tensor, device: torch.device, pin_memory: bool) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    if device.type == "cuda" and pin_memory:
        return tensor.to(device, non_blocking=True)
    return tensor.to(device)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    pin_memory: bool,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in loader:
        batch_x = _to_device(batch_x, device, pin_memory)
        batch_y = _to_device(batch_y, device, pin_memory)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        batch_size = batch_x.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(1, total_samples)


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    pin_memory: bool,
) -> float:
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = _to_device(batch_x, device, pin_memory)
            batch_y = _to_device(batch_y, device, pin_memory)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            batch_size = batch_x.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    return running_loss / max(1, total_samples)


def train_with_early_stopping(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_epochs: int,
    patience: int,
    min_delta: float,
    pin_memory: bool,
    val_interval: int,
    val_schedule: list | None,
    log_interval: int = 0,
    log_prefix: str | None = None,
) -> Tuple[List[Dict[str, float]], Dict[str, torch.Tensor], int, float, int]:
    history: List[Dict[str, float]] = []
    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_improvement = 0

    train_criterion = nn.CrossEntropyLoss(reduction="mean")
    eval_criterion = nn.CrossEntropyLoss(reduction="mean")

    terminating_epoch = 0
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, train_criterion, device, pin_memory
        )
        train_time = time.perf_counter() - train_start
        val_loss = None
        val_time = 0.0
        current_interval = max(1, val_interval)
        if val_schedule:
            for stage in val_schedule:
                interval = int(stage.get("interval", current_interval))
                until_epoch = stage.get("until_epoch")
                current_interval = max(1, interval)
                if until_epoch is None or epoch <= int(until_epoch):
                    break
        should_validate = (epoch % current_interval == 0) or (epoch == max_epochs)
        patience_checks = max(1, patience)
        if should_validate:
            val_start = time.perf_counter()
            val_loss = evaluate(model, val_loader, eval_criterion, device, pin_memory)
            val_time = time.perf_counter() - val_start
        epoch_time = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "training_loss": train_loss,
                "validation_loss": val_loss if val_loss is not None else float("nan"),
            }
        )

        if val_loss is not None:
            improved = val_loss < (best_val_loss - min_delta)
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

        terminating_epoch = epoch

        if log_interval and (epoch == 1 or epoch % log_interval == 0 or epoch == max_epochs):
            elapsed = time.time() - start_time
            prefix = f"[{log_prefix}] " if log_prefix else ""
            val_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            best_val_str = (
                f"{best_val_loss:.4f}" if best_val_loss != float("inf") else "n/a"
            )
            print(
                f"{prefix}epoch {epoch}/{max_epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_str} "
                f"best_val={best_val_str} "
                f"patience_checks={patience_checks} "
                f"train_s={train_time:.2f} "
                f"val_s={val_time:.2f} "
                f"epoch_s={epoch_time:.2f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        if val_loss is not None and epochs_since_improvement >= patience_checks:
            if log_interval:
                prefix = f"[{log_prefix}] " if log_prefix else ""
                print(
                    f"{prefix}early stopping triggered at epoch {epoch} "
                    f"(best_epoch={best_epoch}, best_val={best_val_loss:.4f}, "
                    f"patience_checks={patience_checks})",
                    flush=True,
                )
            break

    return history, best_state, best_epoch, best_val_loss, terminating_epoch
