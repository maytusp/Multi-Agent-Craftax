from typing import Any
from dataclasses import asdict
import wandb
import os
from craftax.craftax_classic.envs.craftax_state import EnvParams, StaticEnvParams


class TrainLogger:
    env_params: EnvParams
    static_env_params: StaticEnvParams
    model_snapshots: list[tuple[int, Any]]
    stats: dict[str, list[tuple[int, Any]]]
    wandb_project: str

    def __init__(
        self, env_params: EnvParams, static_env_params: StaticEnvParams, wandb_project: str = ''
    ) -> None:
        self.env_params = env_params
        self.static_env_params = static_env_params
        self.model_snapshots = []
        self.stats = {}
        self.wandb_project = wandb_project
        if wandb_project:
            os.environ["WANDB_MODE"] = "online"
            os.environ["WANDB_PROJECT"] = wandb_project

            config = {**asdict(env_params), **asdict(static_env_params)}
            wandb.init(project=wandb_project, config=config)
            
            print("CONFIG")
            print(wandb.config)

            wandb.define_metric("train/episode")
            wandb.define_metric("train/*", step_metric="train/episode")


    def insert_model_snapshot(self, iteration: int, model: Any) -> None:
        self.model_snapshots.append((iteration, model))

    def insert_stat(self, iteration: int, key: str, data: Any) -> None:
        """
        Inserts statistic, using wandb if possible.
        Note that the data should be an array containing data for all agents.
        """
        if self.wandb_project:
            # TODO: double check if wandb will plot the data properly
            wandb.log({
                'train/episode': iteration, 
                **{f'train/{key}/agent_{i}': value for i, value in enumerate(data)}
                }, step=iteration)
        elif key in self.stats:
            self.stats[key].append((iteration, data))
        else:
            self.stats[key] = [(iteration, data)]

    def __repr__(self) -> str:
        return f"Train Log with {len(self.model_snapshots)} snapshots and {len(next(iter(self.stats.values())))} items of {len(self.stats)} different statistics"
