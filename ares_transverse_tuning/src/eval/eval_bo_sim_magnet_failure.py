from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gymnasium.wrappers import RescaleAction, TimeLimit
from tqdm import tqdm

from src.bayesopt import BayesianOptimizationAgent
from src.environments import ea
from src.trial import Trial, load_trials
from src.utils import FailQ3, RecordEpisode


def try_problem(trial_index: int, trial: Trial, next_trial: Trial) -> None:
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
        reward_mode="feedback",
        target_beam_mode="constant",
        target_beam_values=trial.target_beam,
        target_mu_x_threshold=None,
        target_mu_y_threshold=None,
        target_sigma_x_threshold=None,
        target_sigma_y_threshold=None,
        threshold_hold=5,
        unidirectional_quads=True,
        w_beam=1.0,
        w_mu_x=1.0,
        w_mu_y=1.0,
        w_on_screen=10.0,
        w_sigma_x=1.0,
        w_sigma_y=1.0,
        logarithmic_beam_distance=True,
        normalize_beam_distance=False,
    )
    env = TimeLimit(env, 80)
    env = FailQ3(env, at_step=0)
    env = RecordEpisode(
        env,
        save_dir=f"data/bo_vs_rl/simulation/bo_magnet_failed/problem_{trial_index:03d}",
    )
    env = RescaleAction(env, -3, 3)

    model = BayesianOptimizationAgent(
        env=env,
        stepsize=0.1,
        init_samples=5,
        acquisition="EI",
        mean_module=None,
    )

    # Actual optimisation
    observation = env.reset()
    reward = None
    done = False
    while not done:
        action = model.predict(observation, reward)
        observation, reward, done, info = env.step(action)

    # Set back to best
    action = model.X[model.Y.argmax()].detach().numpy()
    env.step(action)

    env.close()


def main():
    trials = load_trials(Path("data/trials.yaml"))
    next_trials = trials[1:] + [trials[0]]

    with ProcessPoolExecutor() as executor:
        _ = tqdm(
            executor.map(try_problem, range(len(trials)), trials, next_trials),
            total=300,
        )


if __name__ == "__main__":
    main()
