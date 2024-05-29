import sys
from pathlib import Path

import pandas as pd

sys.path.append("../")
import numpy as np
from gymnasium.wrappers import (
    FilterObservation,
    FlattenObservation,
    RescaleAction,
    TimeLimit,
)
from stable_baselines3 import TD3
from xopt import VOCS

from src.environments import ea
from src.trial import load_trials
from src.wrappers import (
    AnimateIncomingBeam,
    NotVecNormalize,
    PolishedDonkeyCompatibility,
    RecordEpisode,
)
from src.xopt_helper import (
    BOXoptCompatibleWrapper,
    XoptAgent,
    prepare_ARES_EA_data,
    rescale_magnet_values,
)

rl_model_name = "polished-donkey-996"
rl_model_path = f"models/ea/legacy/{rl_model_name}/"


def make_continuous_env(config: dict):
    if config["algorithm"] == "xopt":
        env_kwargs = {
            "action_mode": "direct",
        }
    elif config["algorithm"] == "rl":
        env_kwargs = {
            "action_mode": "delta",
            "max_quad_delta": 30 * 0.1,
            "max_steerer_delta": 6e-3 * 0.1,
        }

    env = ea.TransverseTuning(
        backend="cheetah",
        backend_args={
            "incoming_mode": config["incoming_beam"],
            "misalignment_mode": config["misalignments"],
        },
        magnet_init_mode=np.array([10, -10, 0, 10, 0]),
        target_beam_mode=trial.target_beam,
        **env_kwargs,
    )
    env = TimeLimit(env, config["time_step"])

    env = AnimateIncomingBeam(
        env,
        over_n_steps=config["over_n_steps"],
        to_beam_parameters=config["to_beam_parameters"],
        parameter_indices=config["parameter_indices"],
        n_start_steps=config["n_start_steps"],
        param_mode=config["param_mode"],
        animate_mode=config["animate_mode"],
    )

    if config["save_dir"] is not None:
        env = RecordEpisode(
            env,
            save_dir=config["save_dir"],
        )

    if config["algorithm"] == "rl":
        env = FilterObservation(env, ["beam", "magnets", "target"])
        env = FlattenObservation(env)
        env = PolishedDonkeyCompatibility(env)
        env = NotVecNormalize(env, f"{rl_model_path}/vec_normalize.pkl")
        env = RescaleAction(env, -1, 1)
    elif config["algorithm"] == "xopt":
        env = RescaleAction(env, -1, 1)
        env = BOXoptCompatibleWrapper(env, prepare_ARES_EA_data)

    return env


trials = load_trials(Path("data/trials.yaml"))

trial_index = 33
trial = trials[trial_index]


BASE_CONFIG = {
    "algorithm": "xopt",
    "incoming_beam": trial.incoming_beam,
    "misalignments": trial.misalignments,
    "time_step": 300,
    "over_n_steps": 50,
    "to_beam_parameters": [1e-3, 1e-3],
    "parameter_indices": [1, 3],
    "n_start_steps": 25,
    "param_mode": "delta",
    "animate_mode": "sinusoidal",
    "save_dir": None,
}

## Sinusoidal modulation
eval_config = BASE_CONFIG.copy()
eval_config["root_dir"] = f"data/bo_vs_rl/simulation/continuous_sin/"

### RL
print("Running RL")
config = eval_config.copy()
config["algorithm"] = "rl"
config["save_dir"] = f"{config['root_dir']}/rl/"

env = make_continuous_env(config)
model = TD3.load(f"{rl_model_path}/model")
observation, info = env.reset()
done = False
while not done:
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()

### RL with stopped action
print("Running RL with stopped action")
config = eval_config.copy()
config["algorithm"] = "rl"
config["save_dir"] = f"{config['root_dir']}/rl_stopped/"

env = make_continuous_env(config)
model = TD3.load(f"{rl_model_path}/model")
observation, info = env.reset()
done = False
while not done:
    action, _ = model.predict(observation, deterministic=True)
    if env.steps_taken >= config["n_start_steps"]:
        action = np.zeros(5)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()

### Do Nothing
print("Running Do Nothing")
config = eval_config.copy()
config["algorithm"] = "rl"
config["save_dir"] = f"{config['root_dir']}/do_nothing/"

env = make_continuous_env(config)
# model = TD3.load(f"{rl_model_path}/model")
observation, info = env.reset()
done = False
while not done:
    action = np.zeros(5)
    # action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()


### Normal BO
print("Running normal BO")
config = eval_config.copy()
config["algorithm"] = "xopt"
config["save_dir"] = f"{config['root_dir']}/bo_normal/"

env = make_continuous_env(config)

optimizer_kwargs = {
    "beta": 2.0,
    "max_travel_distances": [0.1] * 5,
    # "proximal_weights": 0.5,
}

vocs_variables = {
    "q1": [-1, 1],
    "q2": [-1, 1],
    "cv": [-1, 1],
    "q3": [-1, 1],
    "ch": [-1, 1],
}
vocs = VOCS(
    variables=vocs_variables,
    objectives={"logmae": "MINIMIZE"},
)

xopt_BO = XoptAgent(
    env,
    vocs,
    method="UCB",
    action_order=["q1", "q2", "cv", "q3", "ch"],
    **optimizer_kwargs,
)

output, info = env.reset()

init_magnet_values = rescale_magnet_values(env.unwrapped.backend.get_magnets(), env.env)
init_sample = [
    {k: v for k, v in zip(["q1", "q2", "cv", "q3", "ch"], init_magnet_values)}
]
xopt_BO.add_data(pd.DataFrame(init_sample), output)
done = False
while not done:
    action = xopt_BO.predict(n_samples=1)
    output, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    xopt_BO.add_data(xopt_BO.last_sample, output)
env.close()


### Contextual BO
print("Running contextual BO")
config = eval_config.copy()
config["algorithm"] = "xopt"
config["save_dir"] = f"{config['root_dir']}/bo_contextual/"

env = make_continuous_env(config)

optimizer_kwargs = {
    "beta": 2.0,
    "max_travel_distances": [0.1] * 6,
    # "proximal_weights": 0.5,
}

vocs_variables = {
    "q1": [-1, 1],
    "q2": [-1, 1],
    "cv": [-1, 1],
    "q3": [-1, 1],
    "ch": [-1, 1],
    "steps": [0, 1000],
}
vocs = VOCS(
    variables=vocs_variables,
    objectives={"logmae": "MINIMIZE"},
)

xopt_contextualBO = XoptAgent(
    env,
    vocs,
    method="UCB",
    action_order=["q1", "q2", "cv", "q3", "ch"],
    **optimizer_kwargs,
)

output, info = env.reset()

init_magnet_values = rescale_magnet_values(env.unwrapped.backend.get_magnets(), env.env)
init_sample = [
    {k: v for k, v in zip(["q1", "q2", "cv", "q3", "ch"], init_magnet_values)}
]
init_sample[0]["steps"] = 0
xopt_contextualBO.add_data(pd.DataFrame(init_sample), output)
done = False
while not done:
    xopt_contextualBO.xopt.generator.fixed_features = {"steps": env.steps_taken + 1}
    action = xopt_contextualBO.predict(n_samples=1)
    output, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    xopt_contextualBO.add_data(xopt_contextualBO.last_sample, output)
env.close()


### Contextual BO with reduced beta
print("Running contextual BO with reduced beta")
config = eval_config.copy()
config["algorithm"] = "xopt"
config["save_dir"] = f"{config['root_dir']}/bo_contextual_reduce_beta/"

env = make_continuous_env(config)

optimizer_kwargs = {
    "beta": 2.0,
    "max_travel_distances": [0.1] * 6,
    # "proximal_weights": 0.5,
}

vocs_variables = {
    "q1": [-1, 1],
    "q2": [-1, 1],
    "cv": [-1, 1],
    "q3": [-1, 1],
    "ch": [-1, 1],
    "steps": [0, 1000],
}
vocs = VOCS(
    variables=vocs_variables,
    objectives={"logmae": "MINIMIZE"},
)

xopt_contextualBO = XoptAgent(
    env,
    vocs,
    method="UCB",
    action_order=["q1", "q2", "cv", "q3", "ch"],
    **optimizer_kwargs,
)

output, info = env.reset()

init_magnet_values = rescale_magnet_values(env.unwrapped.backend.get_magnets(), env.env)
init_sample = [
    {k: v for k, v in zip(["q1", "q2", "cv", "q3", "ch"], init_magnet_values)}
]
init_sample[0]["steps"] = 0
xopt_contextualBO.add_data(pd.DataFrame(init_sample), output)
done = False
while not done:
    xopt_contextualBO.xopt.generator.fixed_features = {"steps": env.steps_taken + 1}
    if env.steps_taken >= 50:
        xopt_contextualBO.xopt.generator.beta = 0.2
    action = xopt_contextualBO.predict(n_samples=1)
    output, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    xopt_contextualBO.add_data(xopt_contextualBO.last_sample, output)
env.close()


## Extremum Seeking
print("Running ES")
config = eval_config.copy()
config["algorithm"] = "xopt"
config["save_dir"] = f"{config['root_dir']}/es/"

env = make_continuous_env(config)

optimizer_kwargs = {
    "k": 2.0,
    "oscillation_size": 0.1,
    "decay_rate": 1.0,
    # "k": 3.7,
    # "oscillation_size": 0.11,
    # "decay_rate": 0.987,
    # "k": 2.0,
    # "oscillation_size": 0.1,
    # "decay_rate": 0.99,
}

vocs_variables = {
    "q1": [-1, 1],
    "q2": [-1, 1],
    "cv": [-1, 1],
    "q3": [-1, 1],
    "ch": [-1, 1],
}
vocs = VOCS(
    variables=vocs_variables,
    objectives={"logmae": "MINIMIZE"},
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

init_magnet_values = rescale_magnet_values(env.unwrapped.backend.get_magnets(), env.env)
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


## Extremum Seeking with Tuned Parameters
print("Running ES with tuned parameters")
config = eval_config.copy()
config["algorithm"] = "xopt"
config["save_dir"] = f"{config['root_dir']}/es_tuned/"

env = make_continuous_env(config)

optimizer_kwargs = {
    # "k": 2.0,
    # "oscillation_size": 0.1,
    # "decay_rate": 1.0,
    "k": 3.7,
    "oscillation_size": 0.11,
    "decay_rate": 0.987,
    # "k": 2.0,
    # "oscillation_size": 0.1,
    # "decay_rate": 0.99,
}

vocs_variables = {
    "q1": [-1, 1],
    "q2": [-1, 1],
    "cv": [-1, 1],
    "q3": [-1, 1],
    "ch": [-1, 1],
}
vocs = VOCS(
    variables=vocs_variables,
    objectives={"logmae": "MINIMIZE"},
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

init_magnet_values = rescale_magnet_values(env.unwrapped.backend.get_magnets(), env.env)
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


### Extremum Seeking with more time steps
print("Running ES with longer time steps")
config = eval_config.copy()
config["n_start_steps"] = 300
config["time_step"] = 600
config["algorithm"] = "xopt"
config["save_dir"] = f"{config['root_dir']}/es_trackoptimum/"

env = make_continuous_env(config)

optimizer_kwargs = {
    "k": 2.0,
    "oscillation_size": 0.1,
    "decay_rate": 1.0,
    # "k": 3.7,
    # "oscillation_size": 0.11,
    # "decay_rate": 0.987,
    # "k": 2.0,
    # "oscillation_size": 0.1,
    # "decay_rate": 0.99,
}

vocs_variables = {
    "q1": [-1, 1],
    "q2": [-1, 1],
    "cv": [-1, 1],
    "q3": [-1, 1],
    "ch": [-1, 1],
}
vocs = VOCS(
    variables=vocs_variables,
    objectives={"logmae": "MINIMIZE"},
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

init_magnet_values = rescale_magnet_values(env.unwrapped.backend.get_magnets(), env.env)
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
