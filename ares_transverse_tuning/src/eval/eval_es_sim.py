from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
from gymnasium.wrappers import RescaleAction, TimeLimit
from tqdm import tqdm
from xopt import VOCS

from src.environments import ea
from src.trial import Trial, load_trials
from src.wrappers.record_episode import RecordEpisode
from src.xopt_helper import (
    BOXoptCompatibleWrapper,
    XoptAgent,
    prepare_ARES_EA_data,
    rescale_magnet_values,
)


def try_problem(
    trial_index: int,
    trial: Trial,
    version="tuned",
    rescale_action: bool = True,
):
    if version == "tuned":
        # Tuned hyperparameters
        optimizer_kwargs = {
            "k": 3.7,
            "oscillation_size": 0.11,
            "decay_rate": 0.987,
        }
        save_dir = f"data/bo_vs_rl/simulation/es_tuned/problem_{trial_index:03d}"
    elif version == "reviewer_1":
        # Tuned hyperparameters + gains changed following reviewer's suggestion 1
        optimizer_kwargs = {
            "k": 5.0,
            "oscillation_size": 0.11,
            "decay_rate": 0.987,
        }
        save_dir = f"data/bo_vs_rl/simulation/es_reviewer_1/problem_{trial_index:03d}"
    elif version == "reviewer_2":
        # Tuned hyperparameters + gains changed following reviewer's suggestion 2
        optimizer_kwargs = {
            "k": 7.4,
            "oscillation_size": 0.11,
            "decay_rate": 0.987,
        }
        save_dir = f"data/bo_vs_rl/simulation/es_reviewer_2/problem_{trial_index:03d}"
    elif version == "default":
        # Default hyperparameters
        optimizer_kwargs = {
            "k": 2.0,
            "oscillation_size": 0.1,
            "decay_rate": 1.0,
        }
        save_dir = f"data/bo_vs_rl/simulation/es_default/problem_{trial_index:03d}"
    elif version == "with_decay":
        # Default hyperparameters
        optimizer_kwargs = {
            "k": 2.0,
            "oscillation_size": 0.1,
            "decay_rate": 0.99,
        }
        save_dir = f"data/bo_vs_rl/simulation/es_with_decay/problem_{trial_index:03d}"
    else:
        raise ValueError(f"Unknown version: {version}")

    # Create the environment
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
    env = TimeLimit(env, 150)
    env = RecordEpisode(
        env,
        save_dir=save_dir,
        # save_dir=f"data/bo_vs_rl/simulation/es/problem_{trial_index:03d}",
    )

    if rescale_action:
        env = RescaleAction(env, -1, 1)

    env = BOXoptCompatibleWrapper(env, prepare_ARES_EA_data)

    if rescale_action is not None:
        vocs_variables = {
            "q1": [-1, 1],
            "q2": [-1, 1],
            "cv": [-1, 1],
            "q3": [-1, 1],
            "ch": [-1, 1],
        }
    else:
        vocs_variables = {
            "q1": [-72, 72],
            "q2": [-72, 72],
            "cv": [-6.1782e-3, 6.1782e-3],
            "q3": [-72, 72],
            "ch": [-6.1782e-3, 6.1782e-3],
        }

    vocs = VOCS(
        variables=vocs_variables,
        # objectives={"mae": "MINIMIZE"},
        objectives={"logmae": "MINIMIZE"},
        # constraints={"max_beamparam": ["LESS_THAN", 2e-3]},
    )

    xopt_agent = XoptAgent(
        env,
        vocs,
        method="ES",
        action_order=["q1", "q2", "cv", "q3", "ch"],
        **optimizer_kwargs,
    )

    # Actual optimisation
    output, info = env.reset()
    if rescale_action is not None:
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
    while not done:
        action = xopt_agent.predict(n_samples=1)
        output, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        xopt_agent.add_data(xopt_agent.last_sample, output)

    env.close()


def main():
    trials = load_trials(Path("data/trials.yaml"))

    with ProcessPoolExecutor() as executor:
        futures = tqdm(
            executor.map(
                try_problem,
                range(len(trials)),
                trials,
                # repeat("with_decay"),
                # repeat("reviewer_1"),
                repeat("reviewer_2"),
            ),
            total=300,
        )

    for future in futures:
        pass


if __name__ == "__main__":
    main()
