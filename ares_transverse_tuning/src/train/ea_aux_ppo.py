from functools import partial

import gymnasium as gym
import wandb
from gymnasium.wrappers import (
    FlattenObservation,
    FrameStack,
    RecordVideo,
    RescaleAction,
    TimeLimit,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from ..environments import ea_auxiliary
from ..utils import save_config
from ..wrappers import FlattenAction, PlotEpisode, RescaleObservation


def main() -> None:
    config = {
        # Environment
        "action_mode": "delta",
        "max_quad_delta": 72,
        "max_steerer_delta": 6.1782e-3,
        "magnet_init_mode": "random",
        "incoming_mode": "random",
        "misalignment_mode": "random",
        "max_misalignment": 5e-4,
        "target_beam_mode": "random",
        "threshold_hold": 1,
        "clip_magnets": True,
        # Reward (also environment)
        "beam_param_transform": "SoftPlus",
        "beam_param_combiner": "SmoothMax",
        "beam_param_combiner_args": {"alpha": -5},
        "beam_param_combiner_weights": [1, 1, 1, 1],
        "magnet_change_transform": "Sigmoid",
        "magnet_change_combiner": "Mean",
        "magnet_change_combiner_args": {},
        "magnet_change_combiner_weights": [1, 1, 1, 1, 1],
        "final_combiner": "SmoothMax",
        "final_combiner_args": {"alpha": -5},
        "final_combiner_weights": [1, 1, 0.5],
        # Auxiliary reward
        "incoming_transform": "Sigmoid",
        "incoming_combiner": "GeometricMean",
        "incoming_combiner_args": {},
        "incoming_combiner_weights": [1] * 11,
        "misalignment_transform": "Sigmoid",
        "misalignment_combiner": "GeometricMean",
        "misalignment_combiner_args": {},
        "misalignment_combiner_weights": [1] * 8,
        "aux_combiner": "SmoothMax",
        "aux_combiner_args": {"alpha": -5},
        "aux_combiner_weights": [1, 1],
        "combined_combiner": "SmoothMax",
        "combined_combiner_args": {"alpha": -5},
        "combined_combiner_weights": [2, 1],
        # Wrappers
        "frame_stack": 1,  # 1 means no frame stacking
        "normalize_observation": True,
        "running_obs_norm": False,
        "normalize_reward": False,
        "rescale_action": True,
        "target_threshold": None,  # 2e-5 m is estimated screen resolution
        "max_episode_steps": 50,
        # RL algorithm
        "batch_size": 100,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "n_envs": 40,
        "n_steps": 100,
        "ent_coef": 0.01,
        "total_timesteps": 10_000_000,
        # SB3 config
        "sb3_device": "auto",
        "vec_env": "subproc",
    }

    train(config)


def train(config: dict) -> None:
    # Setup wandb
    wandb.init(
        project="ares-ea-v3",
        entity="msk-ipc",
        sync_tensorboard=True,
        monitor_gym=True,
        config=config,
        dir=".wandb",
    )
    config = dict(wandb.config)
    config["run_name"] = wandb.run.name

    # Setup environments
    if config["vec_env"] == "dummy":
        vec_env = DummyVecEnv(
            [partial(make_env, config) for _ in range(config["n_envs"])]
        )
    elif config["vec_env"] == "subproc":
        vec_env = SubprocVecEnv(
            [partial(make_env, config) for _ in range(config["n_envs"])]
        )
    else:
        raise ValueError(f"Invalid value \"{config['vec_env']}\" for dummy")
    eval_vec_env = DummyVecEnv(
        [partial(make_env, config, record_video=True, plot_episode=True)]
    )

    if (config["normalize_observation"] and config["running_obs_norm"]) or config[
        "normalize_reward"
    ]:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=config["normalize_observation"] and config["running_obs_norm"],
            norm_reward=config["normalize_reward"],
            gamma=config["gamma"],
        )
        eval_vec_env = VecNormalize(
            eval_vec_env,
            norm_obs=config["normalize_observation"] and config["running_obs_norm"],
            norm_reward=config["normalize_reward"],
            gamma=config["gamma"],
            training=False,
        )

    # Train
    model = PPO(
        "MlpPolicy",
        vec_env,
        device=config["sb3_device"],
        gamma=config["gamma"],
        tensorboard_log=f"log/{config['run_name']}",
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        ent_coef=config["ent_coef"],
        verbose=1,
    )

    eval_callback = EvalCallback(eval_vec_env, eval_freq=1_000, n_eval_episodes=5)
    wandb_callback = WandbCallback()

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[eval_callback, wandb_callback],
    )

    model.save(f"models/{wandb.run.name}/model")
    if (config["normalize_observation"] and config["running_obs_norm"]) or config[
        "normalize_reward"
    ]:
        vec_env.save(f"models/{wandb.run.name}/vec_normalize.pkl")
    save_config(config, f"models/{wandb.run.name}/config")


def make_env(
    config: dict, record_video: bool = False, plot_episode: bool = False
) -> gym.Env:
    env = ea_auxiliary.TransverseTuning(
        backend="cheetah",
        incoming_transform=config["incoming_transform"],
        incoming_combiner=config["incoming_combiner"],
        incoming_combiner_args=config["incoming_combiner_args"],
        incoming_combiner_weights=config["incoming_combiner_weights"],
        misalignment_transform=config["misalignment_transform"],
        misalignment_combiner=config["misalignment_combiner"],
        misalignment_combiner_args=config["misalignment_combiner_args"],
        misalignment_combiner_weights=config["misalignment_combiner_weights"],
        aux_combiner=config["aux_combiner"],
        aux_combiner_args=config["aux_combiner_args"],
        aux_combiner_weights=config["aux_combiner_weights"],
        combined_combiner=config["combined_combiner"],
        combined_combiner_args=config["combined_combiner_args"],
        combined_combiner_weights=config["combined_combiner_weights"],
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
        beam_param_transform=config["beam_param_transform"],
        beam_param_combiner=config["beam_param_combiner"],
        beam_param_combiner_args=config["beam_param_combiner_args"],
        beam_param_combiner_weights=config["beam_param_combiner_weights"],
        final_combiner=config["final_combiner"],
        final_combiner_args=config["final_combiner_args"],
        final_combiner_weights=config["final_combiner_weights"],
        render_mode="rgb_array",
    )
    env = TimeLimit(env, config["max_episode_steps"])
    if plot_episode:
        env = PlotEpisode(
            env,
            save_dir=f"plots/{config['run_name']}",
            episode_trigger=lambda x: x % 5 == 0,  # Once per (5x) evaluation
            log_to_wandb=True,
        )
    env = FlattenAction(env)
    if config["normalize_observation"] and not config["running_obs_norm"]:
        env = RescaleObservation(env, -1, 1)
    if config["rescale_action"]:
        env = RescaleAction(env, -1, 1)
    env = FlattenObservation(env)
    if config["frame_stack"] > 1:
        env = FrameStack(env, config["frame_stack"])
    env = Monitor(env)
    if record_video:
        env = RecordVideo(
            env,
            video_folder=f"recordings/{config['run_name']}",
            episode_trigger=lambda x: x % 5 == 0,  # Once per (5x) evaluation
        )
    return env


if __name__ == "__main__":
    main()
