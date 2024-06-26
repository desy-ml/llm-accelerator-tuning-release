import pickle
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
from gymnasium.wrappers import (
    FilterObservation,
    FlattenObservation,
    RescaleAction,
    TimeLimit,
)
from stable_baselines3.common.env_util import unwrap_wrapper
from tqdm import tqdm

from src.environments import ea
from src.trial import Trial, load_trials
from src.utils import RecordEpisode


def try_problem(trial_index: int, trial: Trial) -> None:
    # Create the environment
    env = ea.TransverseTuning(
        backend="cheetah",
        backend_args={
            "incoming_mode": "constant",
            "incoming_values": trial.incoming_beam,
            "misalignment_mode": "constant",
            "misalignment_values": trial.misalignments,
        },
        action_mode="direct",
        magnet_init_mode="constant",
        magnet_init_values=np.array([10, -10, 0, 10, 0]),
        reward_mode="differential",
        target_beam_mode="constant",
        target_beam_values=trial.target_beam,
        target_mu_x_threshold=None,
        target_mu_y_threshold=None,
        target_sigma_x_threshold=None,
        target_sigma_y_threshold=None,
        threshold_hold=5,
        w_beam=1.0,
        w_mu_x=1.0,
        w_mu_y=1.0,
        w_on_screen=10.0,
        w_sigma_x=1.0,
        w_sigma_y=1.0,
    )
    env = TimeLimit(env, 150)
    env = RecordEpisode(env)
    env = FilterObservation(env, ["beam", "magnets", "target"])
    env = FlattenObservation(env)
    env = RescaleAction(env, -1, 1)

    # Actual optimisation
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

    final_beam = unwrap_wrapper(env, RecordEpisode).observations[-1]["beam"]
    env.close()
    return final_beam


def main():
    original_trials = load_trials(Path("data/trials.yaml"))

    base_trial_index = 0

    base_trial = original_trials[base_trial_index]

    target_mu_xs = np.linspace(-2e-3, 2e-3, num=20)
    target_sigma_xs = np.geomspace(2e-5, 2e-3, num=20)
    target_mu_ys = np.linspace(-2e-3, 2e-3, num=20)
    target_sigma_ys = np.geomspace(2e-5, 2e-3, num=20)

    # Create trials
    modified_trials = []
    for base_trial_index, (mu_x, sigma_x, mu_y, sigma_y) in enumerate(
        product(target_mu_xs, target_sigma_xs, target_mu_ys, target_sigma_ys)
    ):
        trial = deepcopy(base_trial)
        trial.target_beam = np.array([mu_x, sigma_x, mu_y, sigma_y])
        modified_trials.append(trial)

    # Run trials
    with ProcessPoolExecutor() as executor:
        futures = tqdm(
            executor.map(try_problem, range(len(modified_trials)), modified_trials),
            total=len(modified_trials),
        )
        results = [
            {"target_beam": trial.target_beam, "final_beam": future}
            for trial, future in zip(modified_trials, futures)
        ]

    # Save data
    Path("data/bo_vs_rl/simulation/random_grid/").mkdir(parents=True, exist_ok=True)
    with open("data/bo_vs_rl/simulation/random_grid/random.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
