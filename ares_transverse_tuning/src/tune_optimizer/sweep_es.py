import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import tqdm
from gymnasium.wrappers import (
    FlattenObservation,
    FrameStack,
    RecordVideo,
    RescaleAction,
    TimeLimit,
)
from stable_baselines3.common.monitor import Monitor
from xopt import VOCS

import wandb
from src.trial import Trial, load_trials

from ..environments import ea
from ..utils import save_config
from ..wrappers import LogTaskStatistics, PlotEpisode
from ..xopt_helper import (
    BOXoptCompatibleWrapper,
    XoptAgent,
    prepare_ARES_EA_data,
    rescale_magnet_values,
)


def main() -> None:
    config = {
        # Environment
        "action_mode": "direct",
        "max_quad_delta": 72,
        "max_steerer_delta": 6.1782e-3,
        "magnet_init_mode": "random",
        "incoming_mode": "random",
        "misalignment_mode": "random",
        "max_misalignment": 5e-4,
        "target_beam_mode": "random",
        "threshold_hold": 1,
        "clip_magnets": True,
        # Wrappers
        "frame_stack": 1,  # 1 means no frame stacking
        "rescale_action": True,
        "target_threshold": None,  # 2e-5 m is estimated screen resolution
        "max_episode_steps": 150,
        "action_order": ["q1", "q2", "cv", "q3", "ch"],
        # Optimizer settings
        "objective": "logmae",  # mae, logmae
        "objective_mode": "MINIMIZE",  # MINIMIZE, MAXIMIZE
        "method": "ES",
        "optimizer_kwargs": {
            "k": 2.0,
            "oscillation_size": 0.1,
            "decay_rate": 0.99,
        },
        # For evaluation of the performance
        # "n_testepisodes": 100,
        "convergence_threshold": 4e-5,  # 2 times the screen resolution
    }

    train(config)


def train(config: dict) -> None:
    # Setup wandb
    wandb.init(
        project="ares-opttune-es",
        entity="msk-ipc",
        sync_tensorboard=True,
        monitor_gym=True,
        config=config,
        dir=".wandb",
        # mode="disabled",  # for local testing
    )
    config = dict(wandb.config)
    config["run_name"] = wandb.run.name

    # Setup environments
    # env = make_env(config)
    # record_env = make_env(config, record_video=True, plot_episode=True, log_frequency=1)

    # Setup agent
    if config["rescale_action"]:
        vocs_variables = {
            "q1": [-1, 1],
            "q2": [-1, 1],
            "cv": [-1, 1],
            "q3": [-1, 1],
            "ch": [-1, 1],
        }
    else:
        vocs_variables = {
            "q1": [-config["max_quad_delta"], config["max_quad_delta"]],
            "q2": [-config["max_quad_delta"], config["max_quad_delta"]],
            "cv": [-config["max_steerer_delta"], config["max_steerer_delta"]],
            "q3": [-config["max_quad_delta"], config["max_quad_delta"]],
            "ch": [-config["max_steerer_delta"], config["max_steerer_delta"]],
        }

    vocs = VOCS(
        variables=vocs_variables,
        objectives={config["objective"]: config["objective_mode"]},
    )

    dfs = []

    trials = load_trials(Path("data/trials.yaml"))

    for i in tqdm.trange(300):
        trial = trials[i]
        env = ea.TransverseTuning(
            backend="cheetah",
            backend_args={
                "incoming_mode": trial.incoming_beam,
                "misalignment_mode": trial.misalignments,
            },
            action_mode="direct",
            magnet_init_mode=np.array([10, -10, 0, 10, 0]),
            target_beam_mode=trial.target_beam,
        )
        env = TimeLimit(env, max_episode_steps=config["max_episode_steps"])

        if config["rescale_action"]:
            env = RescaleAction(env, -1, 1)
        env = BOXoptCompatibleWrapper(env, prepare_ARES_EA_data)

        # Reset Agent
        xopt_agent = XoptAgent(
            env,
            vocs,
            method=config["method"],
            action_order=config["action_order"],
            **config["optimizer_kwargs"],
        )
        output, info = env.reset()
        if config["rescale_action"]:
            init_magnet_values = rescale_magnet_values(
                env.unwrapped.backend.get_magnets(), env.env
            )
        else:
            init_magnet_values = env.unwrapped.backend.get_magnets()
        init_sample = [
            {k: v for k, v in zip(["q1", "q2", "cv", "q3", "ch"], init_magnet_values)}
        ]
        xopt_agent.add_data(pd.DataFrame(init_sample), output)
        done = False
        # Actual optimisation
        while not done:
            action = xopt_agent.predict(n_samples=1)
            output, reward, done, truncated, info = env.step(action)
            xopt_agent.add_data(xopt_agent.last_sample, output)

        df = xopt_agent.xopt.data
        df.index.name = "step"
        df["episode"] = i
        df["best_mae"] = df["mae"].cummin()
        dfs.append(df)
    # Concatenate all dataframes
    df_combined = pd.concat(dfs)
    # For each episode, find the first step where the best_mae is below the threshold
    df_combined["convergence_step"] = df_combined.groupby("episode")[
        "best_mae"
    ].transform(
        lambda x: x[x < config["convergence_threshold"]].index[0]
        if len(x[x < config["convergence_threshold"]]) > 0
        else config["max_episode_steps"]
    )
    # Get the convergence step of each episode as a list
    convergence_steps = df_combined.groupby("episode")["convergence_step"].first()

    # Mean over all episodes
    df_mean = df_combined.groupby("step").mean()
    mean_table = wandb.Table(dataframe=df_mean[["mae", "best_mae", "logmae"]])

    for step, row in df_mean.iterrows():
        wandb.log(
            {
                "step": step,
                "mean_mae": row["mae"],
                "mean_best_mae": row["best_mae"],
                "logmae": row["logmae"],
            }
        )

    # Write summary
    wandb.run.summary["mean_table"] = mean_table
    wandb.run.summary["convergence_steps"] = wandb.Table(
        dataframe=convergence_steps.to_frame(name="convergence_step")
    )
    wandb.run.summary["final_best_mae"] = df_mean["best_mae"].iloc[-1]
    wandb.run.summary["mean_convergence_step"] = np.mean(convergence_steps)

    # Save data
    save_dir = f"models/tune_optimizer/es/{wandb.run.name}"
    os.makedirs(save_dir, exist_ok=True)
    df_combined.to_csv(f"{save_dir}/data.csv")
    save_config(config, f"{save_dir}/config")

    env.close()
    # record_env.close()


def make_env(
    config: dict,
    record_video: bool = False,
    plot_episode: bool = False,
    log_task_statistics: bool = False,
    log_frequency: int = 5,
) -> gym.Env:
    env = ea.TransverseTuning(
        backend="cheetah",
        backend_args={
            "incoming_mode": config["incoming_mode"],
            "misalignment_mode": config["misalignment_mode"],
            "max_misalignment": config["max_misalignment"],
            "generate_screen_images": plot_episode,
        },
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        max_quad_delta=config["max_quad_delta"],
        max_steerer_delta=config["max_steerer_delta"],
        target_beam_mode=config["target_beam_mode"],
        target_threshold=config["target_threshold"],
        threshold_hold=config["threshold_hold"],
        clip_magnets=config["clip_magnets"],
        render_mode="rgb_array",
    )
    env = TimeLimit(env, config["max_episode_steps"])
    if plot_episode:
        env = PlotEpisode(
            env,
            save_dir=f"plots/{config['run_name']}",
            episode_trigger=lambda x: x % log_frequency
            == 0,  # Once per (5x) evaluation
            log_to_wandb=True,
        )
    if log_task_statistics:
        env = LogTaskStatistics(env)
    if config["rescale_action"]:
        env = RescaleAction(env, -1, 1)
    # env = FlattenObservation(env)
    if config["frame_stack"] > 1:
        env = FrameStack(env, config["frame_stack"])
    env = Monitor(env)
    if record_video:
        env = RecordVideo(
            env,
            video_folder=f"recordings/{config['run_name']}",
            episode_trigger=lambda x: x % log_frequency
            == 0,  # Once per (5x) evaluation
        )

    env = BOXoptCompatibleWrapper(env, prepare_ARES_EA_data)

    return env


if __name__ == "__main__":
    main()
