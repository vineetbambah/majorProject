import json
from pathlib import Path

from .step_metrics import StepMetrics


class RankMetrics:
    """
    Stores all metrics collected for a single worker (rank)
    during one benchmark experiment.
    """
    def __init__(
        self,
        rank: int,
        world_size: int,
        output_dir: Path,
    ) -> None:

        self.rank = rank
        self.world_size = world_size

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.steps: list[StepMetrics] = []
    
    def add_step(
        self,
        step: StepMetrics,
    ) -> None:
        """
        Add one completed training step.
        """

        if not isinstance(step, StepMetrics):
            raise TypeError(
                "Expected StepMetrics."
            )

        self.steps.append(step)

    def save(self) -> Path:
        """
        Save this rank's metrics to disk.
        """

        output_path = (
            self.output_dir
            / f"rank_{self.rank}_metrics.json"
        )

        data = {
            "rank": self.rank,
            "world_size": self.world_size,
            "steps": [
                step.to_dict()
                for step in self.steps
            ],
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file_handle:

            json.dump(
                data,
                file_handle,
                indent=4,
            )

        return output_path
    
    @classmethod
    def load(
        cls,
        metrics_path: Path,
    ) -> "RankMetrics":
        """
        Load a RankMetrics object from disk.
        """

        metrics_path = Path(metrics_path)

        with metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file_handle:

            data = json.load(file_handle)

        metrics = cls(
            rank=data["rank"],
            world_size=data["world_size"],
            output_dir=metrics_path.parent,
        )

        for step_data in data["steps"]:

            metrics.add_step(
                StepMetrics.from_dict(step_data)
            )

        return metrics