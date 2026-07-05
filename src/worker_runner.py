from gradient_sync import parameter_server, ring, tree
from models import ann_model, cnn_model, rnn_model

import torch
import time
from pathlib import Path

from metrics.rank_metrics import RankMetrics
from metrics.step_metrics import StepMetrics
from data.fashion_mnist import get_dataloader


def get_algo_module(algo_tag: str):
    return {
        "ring": ring,
        "tree": tree,
        "parameter_server": parameter_server,
    }[algo_tag]


def get_model_module(model_tag: str):
    return {
        "ann": ann_model,
        "cnn": cnn_model,
        "rnn": rnn_model,
    }[model_tag]


def _compute_grad_norm(grad_tensor: torch.Tensor) -> float:
    """Compute L2 norm of gradient tensor."""
    if isinstance(grad_tensor, torch.Tensor):
        return torch.norm(grad_tensor).item()
    return 0.0


def run_worker(config):
    algo_module = get_algo_module(config["algo"])
    model_module = get_model_module(config["model"])

    rank = config["rank"]
    world_size = config["world_size"]

    epochs = int(config.get("epochs", 1))
    steps_per_epoch = int(config.get("steps_per_epoch", 1))

    output_dir = Path(
        config.get("benchmark_results_dir", "benchmark_results")
    )

    metrics = RankMetrics(rank, world_size, output_dir)

    print(
        f"[rank {rank}] start "
        f"algo={config['algo']} "
        f"model={config['model']} "
        f"epochs={epochs} "
        f"steps={steps_per_epoch}",
        flush=True,
    )

    comm_ctx = None

    setup_start = time.perf_counter()

    try:
        # ---------------- Setup ----------------
        comm_ctx = algo_module.setup(config)
        state = model_module.build_model(config)

        train_loader = get_dataloader(
            batch_size=config["batch_size"],
            rank=config["rank"],
            world_size=config["world_size"]
        )

        train_iterator = iter(train_loader)

        setup_time = time.perf_counter() - setup_start

        print(
            f"[rank {rank}] setup complete "
            f"({setup_time:.4f}s)",
            flush=True,
        )

        # ---------------- Training Loop ----------------
        for epoch in range(epochs):
            train_loader.sampler.set_epoch(epoch)
            for step in range(steps_per_epoch):

                step_config = {
                    **config,
                    "current_epoch": epoch,
                    "step": step,
                }
                # ---------- Get next batch ----------
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)

                # ---------- Compute ----------
                start = time.perf_counter()

                local_grad = model_module.train_step(
                    state,
                    batch,
                    step_config,
                )

                compute_time = time.perf_counter() - start
                loss = local_grad.get("loss", 0.0)

                # ---------- Synchronization ----------
                start = time.perf_counter()

                synced_grad = algo_module.average(
                    local_grad,
                    comm_ctx,
                    config,
                )

                sync_time = time.perf_counter() - start

                if isinstance(synced_grad, dict):
                    grad_tensor = synced_grad["gradients"]
                else:
                    grad_tensor = synced_grad

                grad_norm = _compute_grad_norm(grad_tensor)

                grad_size_bytes = (
                    grad_tensor.numel()
                    * grad_tensor.element_size()
                )

                # ---------- Optimizer ----------
                start = time.perf_counter()

                model_module.apply_synced_gradients(
                    state,
                    grad_tensor,
                )

                optim_time = time.perf_counter() - start

                step_metrics = StepMetrics(
                epoch=epoch,
                step=step,
                is_warmup=(epoch == 0 and step == 0),

                compute_time=compute_time,
                sync_time=sync_time,
                optim_time=optim_time,

                loss=float(loss)
                if isinstance(loss, torch.Tensor)
                else loss,

                grad_norm=grad_norm,

                bytes_sent=grad_size_bytes,
                bytes_received=grad_size_bytes,
            )

                metrics.add_step(step_metrics)
            print(
                    f"[rank {rank}] "
                    f"epoch={epoch} "
                    f"step={step} "
                    f"compute={compute_time:.4f}s "
                    f"sync={sync_time:.4f}s "
                    f"optim={optim_time:.4f}s "
                    f"total={(compute_time + sync_time + optim_time):.4f}s "
                    f"loss={loss:.6f} "
                    f"grad_norm={grad_norm:.6f}",
                    flush=True,
                )

    finally:
        if comm_ctx is not None:
            algo_module.teardown(comm_ctx)

        metrics_path = metrics.save()
        print(
            f"[rank {rank}] metrics saved to {metrics_path}",
            flush=True,
        )

        print(
            f"[rank {rank}] teardown complete",
            flush=True,
        )

        print(
            f"[rank {rank}] metrics: {metrics_path}",
            flush=True,
        )