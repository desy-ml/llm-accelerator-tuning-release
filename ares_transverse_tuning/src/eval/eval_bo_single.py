import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from gymnasium.wrappers import RescaleAction, TimeLimit
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
    trial_index: int, trial: Trial, rescale_action: bool = False, mode: str = "hard"
):
    # BO hyperparameters
    if mode == "hard":
        optimizer_kwargs = {
            "beta": 2.0,
            "max_travel_distances": [0.1] * 5,
            "proximal_weights": None,
        }
        save_dir = f"data/bo_vs_rl/simulation/bo_hard/problem_{trial_index:03d}"
    else:
        optimizer_kwargs = {
            "beta": 2.0,
            "proximal_weights": 0.5,
            "max_travel_distances": 1,
        }
        save_dir = f"data/bo_vs_rl/simulation/bo_proximal/problem_{trial_index:03d}"
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
        save_dir=f"data/bo_vs_rl/simulation/bo_hard/problem_{trial_index:03d}",
        # save_dir=f"data/bo_vs_rl/simulation/es/problem_{trial_index:03d}",
    )

    if rescale_action:
        print("Rescaling action")
        env = RescaleAction(env, -1, 1)
        vocs = VOCS(
            variables={
                "q1": [-1, 1],
                "q2": [-1, 1],
                "cv": [-1, 1],
                "q3": [-1, 1],
                "ch": [-1, 1],
            },
            objectives={"logmae": "MINIMIZE"},
        )
    else:
        vocs = VOCS(
            variables={
                "q1": [-72, 72],
                "q2": [-72, 72],
                "cv": [-6.1782e-3, 6.1782e-3],
                "q3": [-72, 72],
                "ch": [-6.1782e-3, 6.1782e-3],
            },
            # objectives={"mae": "MINIMIZE"},
            objectives={"logmae": "MINIMIZE"},
            # constraints={"max_beamparam": ["LESS_THAN", 2e-3]},
        )

    env = BOXoptCompatibleWrapper(env, prepare_ARES_EA_data)

    vocs = VOCS(
        variables={
            "q1": [-72, 72],
            "q2": [-72, 72],
            "cv": [-6.1782e-3, 6.1782e-3],
            "q3": [-72, 72],
            "ch": [-6.1782e-3, 6.1782e-3],
        },
        # objectives={"mae": "MINIMIZE"},
        objectives={"logmae": "MINIMIZE"},
        # constraints={"max_beamparam": ["LESS_THAN", 2e-3]},
    )

    xopt_agent = XoptAgent(
        env,
        vocs,
        method="UCB",
        action_order=["q1", "q2", "cv", "q3", "ch"],
        **optimizer_kwargs,
    )

    # Actual optimisation
    output, info = env.reset()
    if rescale_action:
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
        output, reward, done, truncated, info = env.step(action)
        xopt_agent.add_data(xopt_agent.last_sample, output)

    env.close()


def main(start_index: int, end_index: int, mode: str = "hard"):
    trials = load_trials(Path("data/trials.yaml"))
    for trial_index in range(start_index, end_index + 1):
        try_problem(trial_index, trials[trial_index], rescale_action=False, mode=mode)
    print(f"Evaluation {trial_index} finished ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start_index", type=int)
    parser.add_argument("end_index", type=int)
    parser.add_argument("--mode", type=str, default="hard")
    args = parser.parse_args()

    main(args.start_index, args.end_index, args.mode)
