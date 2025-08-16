import os
from dataclasses import asdict
from typing import Any

import wandb
import pickle

from craftax.craftax_classic.envs.craftax_state import EnvParams, StaticEnvParams


class TrainLogger:
    train_params: dict[str, Any]
    env_params: EnvParams
    static_env_params: StaticEnvParams
    model_snapshots: list[tuple[int, Any]]
    stats: dict[str, list[tuple[int, Any]]]
    wandb_project: str
    run_name: str

    def __init__(
        self,
        train_params: dict[str, Any],
        env_params: EnvParams,
        static_env_params: StaticEnvParams,
        wandb_project: str = "",
        run_name: str = "",
    ) -> None:
        self.train_params = train_params
        self.env_params = env_params
        self.static_env_params = static_env_params
        self.model_snapshots = []
        self.stats = {}
        self.wandb_project = wandb_project
        self.run_name = run_name
        if wandb_project:
            os.environ["WANDB_MODE"] = "online"
            os.environ["WANDB_PROJECT"] = wandb_project

            config = {**train_params, **asdict(env_params), **asdict(static_env_params)}
            wandb.init(
                project=self.wandb_project,
                entity="maytusp",
                config=config,
                name=self.run_name,
            )

            print("CONFIG")
            print(wandb.config)

            wandb.define_metric("train/episode")
            wandb.define_metric("train/*", step_metric="train/episode")

    def insert_model_snapshot(self, iteration: int, model: Any) -> None:
        if self.wandb_project:
            file_name = f"{self.run_name}_model_iter_{iteration}.pickle"
            with open(os.path.join(wandb.run.dir, file_name), "wb") as f:
                pickle.dump(model, f)
        else:
            self.model_snapshots.append((iteration, model))

    def insert_stat(self, iteration: int, key: str, data: Any) -> None:
        """
        Inserts statistic, using wandb if possible.
        Note that the data should be an array containing data for all agents.
        """
        if self.wandb_project:
            # TODO: double check if wandb will plot the data properly
            wandb.log(
                {
                    "train/episode": iteration,
                    **{f"train/{key}/agent_{i}": value for i, value in enumerate(data)},
                },
                step=iteration,
            )
        elif key in self.stats:
            self.stats[key].append((iteration, data))
        else:
            self.stats[key] = [(iteration, data)]

    def __repr__(self) -> str:
        return f"Train Log with {len(self.model_snapshots)} snapshots and {len(next(iter(self.stats.values())))} items of {len(self.stats)} different statistics"
