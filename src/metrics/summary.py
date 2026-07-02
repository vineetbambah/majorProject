from statistics import mean, median, stdev
from pathlib import Path
import json
import numpy as np

from .rank_metrics import RankMetrics

class SummaryGenerator:
    """
    Computes experiment-level statistics from one or more RankMetrics objects.
    """

    def __init__(
        self,
        rank_metrics: list[RankMetrics],
    ) -> None:

        if not rank_metrics:
            raise ValueError(
                "SummaryGenerator requires at least one RankMetrics object."
            )

        self.rank_metrics = rank_metrics

    def _collect(
        self,
        attribute: str,
    ) -> list[float]:

        """
        Collect one attribute from every recorded step.
        """

        values = []

        for rank in self.rank_metrics:
            for step in rank.steps:
                values.append(
                    getattr(
                        step,
                        attribute,
                    )
                )

        return values
    
    def _compute_statistics(
        self,
        values: list[float],
    ) -> dict:

        """
        Compute descriptive statistics for a list of values.
        """

        if not values:
            return {}

        values = sorted(values)

        return {
            "mean": mean(values),
            "median": median(values),
            "std": (
                stdev(values)
                if len(values) > 1
                else 0.0
            ),
            "min": values[0],
            "max": values[-1],
            "p95": float(np.percentile(values, 95)),
        }
    
    def _compute_totals(
        self,
        values: list[float],
    ) -> dict:

        """
        Compute totals for cumulative metrics.
        """

        if not values:
            return {}

        return {
            "total": sum(values),
            "mean": mean(values),
        }
    def compute(self) -> dict:
        """
        Compute experiment-level statistics across all ranks.
        """

        # --------------------------------------------------
        # Timing Metrics
        # --------------------------------------------------

        compute = self._compute_statistics(
            self._collect("compute_time")
        )

        sync = self._compute_statistics(
            self._collect("sync_time")
        )

        optimizer = self._compute_statistics(
            self._collect("optim_time")
        )

        iteration = self._compute_statistics(
            self._collect("iteration_time")
        )

        # --------------------------------------------------
        # Communication Metrics
        # --------------------------------------------------

        bytes_sent = self._compute_totals(
            self._collect("bytes_sent")
        )

        bytes_received = self._compute_totals(
            self._collect("bytes_received")
        )

        # --------------------------------------------------
        # Model Metrics
        # --------------------------------------------------
        reference_rank = self.rank_metrics[0]

        losses = [
            step.loss
            for step in reference_rank.steps
        ]

        loss = {
            "initial": losses[0],
            "final": losses[-1],
            "mean": mean(losses),
        }

        grad_norm = self._compute_statistics(
            self._collect("grad_norm")
        )

        # --------------------------------------------------
        # Derived Metrics
        # --------------------------------------------------

        mean_iteration = iteration["mean"]
        mean_compute = compute["mean"]
        mean_sync = sync["mean"]
        mean_optimizer = optimizer["mean"]

        fractions = {
            "compute_fraction": mean_compute / mean_iteration,
            "sync_fraction": mean_sync / mean_iteration,
            "optimizer_fraction": mean_optimizer / mean_iteration,
        }

        # --------------------------------------------------
        # Final Summary
        # --------------------------------------------------

        return {
            "timing": {
                "compute": compute,
                "sync": sync,
                "optimizer": optimizer,
                "iteration": iteration,
            },
            "communication": {
                "bytes_sent": bytes_sent,
                "bytes_received": bytes_received,
            },
            "model": {
                "loss": loss,
                "grad_norm": grad_norm,
            },
            "derived": fractions,
        }
    def save_summary(
        self,
        output_path: Path,
    ) -> Path:
        """
        Save the experiment summary to disk.
        """

        output_path = Path(output_path)

        summary = self.compute()

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file_handle:

            json.dump(
                summary,
                file_handle,
                indent=4,
            )

        return output_path 