# import dummypydoocs as pydoocs
import numpy as np
from gymnasium.wrappers import FlattenObservation, FrameStack, RescaleAction, TimeLimit
from stable_baselines3 import PPO

from src.ea_optimize import (  # noqa: F401
    BaseCallback,
    CallbackList,
    OptimizeFunctionCallback,
    TestCallback,
    setup_callback,
)
from src.environments import bc
from src.utils import load_config
from src.wrappers import ARESeLog, RecordEpisode, RescaleObservation, TQDMWrapper


def optimize(
    target_mu_x,
    target_sigma_x,
    target_mu_y,
    target_sigma_y,
    target_mu_x_threshold=4e-5,
    target_mu_y_threshold=4e-5,
    target_sigma_x_threshold=4e-5,
    target_sigma_y_threshold=4e-5,
    max_steps=50,
    model_name="chocolate-totem-247",
    logbook=False,
    data_log_dir=None,
    progress_bar=False,
    callback=None,
    backend="cheetah",
    initial_magnets=None,
):
    """
    Optimise beam in ARES BC using a reinforcement learning agent.
    """
    config = load_config(f"../models/bc/ppo/{model_name}/config")

    # Load the model
    model = PPO.load(f"../models/bc/ppo/{model_name}/model")

    callback = setup_callback(callback)

    # Create the environment
    env = bc.TransverseTuning(
        backend=backend,
        backend_args={"generate_screen_images": True} if backend == "cheetah" else {},
        action_mode=config["action_mode"],
        magnet_init_mode=(
            config["magnet_init_mode"] if initial_magnets is None else initial_magnets
        ),
        max_quad_setting=config["max_quad_setting"],
        max_quad_delta=config["max_quad_delta"],
        max_steerer_delta=config["max_steerer_delta"],
        target_beam_mode=np.array(
            [target_mu_x, target_sigma_x, target_mu_y, target_sigma_y]
        ),
        target_threshold=np.array(
            [
                target_mu_x_threshold,
                target_sigma_x_threshold,
                target_mu_y_threshold,
                target_sigma_y_threshold,
            ]
        ),
        clip_magnets=False,  # TODO: Clip in future? Should've happened by itself on the machine but limits might be different
        beam_param_transform=config["beam_param_transform"],
        beam_param_combiner=config["beam_param_combiner"],
        beam_param_combiner_args=config["beam_param_combiner_args"],
        beam_param_combiner_weights=config["beam_param_combiner_weights"],
        final_combiner=config["final_combiner"],
        final_combiner_args=config["final_combiner_args"],
        final_combiner_weights=config["final_combiner_weights"],
    )
    if max_steps is not None:
        env = TimeLimit(env, max_steps)
    if progress_bar:
        env = TQDMWrapper(env)
    if callback is not None:
        env = OptimizeFunctionCallback(env, callback)
    if data_log_dir is not None:
        env = RecordEpisode(env, save_dir=data_log_dir)
    if logbook:
        env = ARESeLog(env, agent_name=model_name)
    if config["normalize_observation"] and not config["running_obs_norm"]:
        env = RescaleObservation(env, -1, 1)
    if config["rescale_action"]:
        env = RescaleAction(env, -1, 1)
    env = FlattenObservation(env)
    if config["frame_stack"] > 1:
        env = FrameStack(env, config["frame_stack"])

    callback.env = env

    # Actual optimisation
    observation, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()
