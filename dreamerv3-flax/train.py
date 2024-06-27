import subprocess
import re
import os
import argparse
# Set the environment variable early, even before importing JAX
parser = argparse.ArgumentParser(description='Set GPU device for training.')
parser.add_argument('--gpu', type=str, help='GPU index to use', default=None)
args = parser.parse_args()

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPU: {args.gpu}")
else:
    # Function to get least used GPU if none specified
    def get_least_used_gpu():
        smi_output = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'], capture_output=True, text=True)
        gpu_memory = [re.split(r',\s*', line.strip()) for line in smi_output.stdout.strip().split('\n')]
        least_used_gpu = sorted(gpu_memory, key=lambda x: int(x[1]), reverse=True)[0][0]
        return least_used_gpu

    least_used_gpu = get_least_used_gpu()
    print(f"Didn't specify devices, using least used GPU: {least_used_gpu}")
    os.environ['CUDA_VISIBLE_DEVICES'] = least_used_gpu
    
import jax
import jax.numpy as jnp
import chex
import flax
import wandb
import optax
from functools import partial
from typing import Dict, Sequence
import wandb
from craftax.craftax_env import make_craftax_env_from_name
import flashbax as fbx
from typing import Sequence, NamedTuple, Any

from dreamerv3_flax.async_vector_env import AsyncVectorEnv
from dreamerv3_flax.buffer import ReplayBuffer
from dreamerv3_flax.env import CrafterEnv, VecCrafterEnv, TASKS
from dreamerv3_flax.jax_agent import JAXAgent
from wrappers import LogWrapper, FlattenObservationWrapper


# def get_eval_metric(achievements: Sequence[Dict]) -> float:
#     achievements = [list(achievement.values()) for achievement in achievements]
#     success_rate = 100 * (np.array(achievements) > 0).mean(axis=0)
#     score = np.exp(np.mean(np.log(1 + success_rate))) - 1
#     eval_metric = {
#         "success_rate": {k: v for k, v in zip(TASKS, success_rate)},
#         "score": score,
#     }
#     return eval_metric

@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]
    
    # Environment
    basic_env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=True)
    env_params = basic_env.default_params
    # env = FlattenObservationWrapper(basic_env)
    breakpoint()
    env = LogWrapper(basic_env)
    
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def train(rng):
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=config["BUFFER_SIZE"],
            min_length=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_sequences=False,
            add_batch_size=config["NUM_ENVS"],
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        rng = jax.random.PRNGKey(0)  # use a dummy rng here
        _action = basic_env.action_space().sample(rng)
        _, _env_state = env.reset(rng, env_params)
        _obs, _, _reward, _done, _ = env.step(rng, _env_state, _action, env_params)
        _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done)
        # breakpoint()
        buffer_state = buffer.init(_timestep)

        # Agent
        agent = JAXAgent(env, env_params, seed=config["SEED"])
        state = agent.initial_state(1)

        # Reset
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obs, env_state = vmap_reset(reset_rng, env_params)

        # Train
        for step in range(args.steps):
            actions, state = agent.act(obs, firsts, state)
            
            # buffer.add(obs, actions, rewards, dones, firsts)

            # actions = np.argmax(actions, axis=-1)
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obs, rewards, dones, firsts, infos = vmap_step(rng_step, env_state, actions)
            
            # BUFFER UPDATE
            timestep = TimeStep(obs=obs, action=actions, reward=rewards, done=done)
            buffer_state = buffer.add(buffer_state, timestep)
            
            for done, info in zip(dones, infos):
                if done:
                    rollout_metric = {
                        "episode_return": info["episode_return"],
                        "episode_length": info["episode_length"],
                    }
                    wandb.log(rollout_metric)

            if step >= 1024 and step % 2 == 0:
                data = buffer.sample(buffer_state, rng).experience
                _, train_metric = agent.train(data)
                if step % 100 == 0:
                    wandb.log(train_metric, step)
    return train


if __name__ == "__main__":
    print("jax is using devices:", jax.devices())
    config = {
        "NUM_ENVS": int(10),
        "BUFFER_SIZE": int(1e6),
        "BUFFER_BATCH_SIZE": int(128),
        "TOTAL_TIMESTEPS": 5e5,
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.05,
        "EPSILON_ANNEAL_TIME": 25e4,
        "TARGET_UPDATE_INTERVAL": 500,
        "LR": 2.5e-4,
        "LEARNING_STARTS": 10000,
        "TRAINING_INTERVAL": 10,
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "TAU": 1.0,
        "ENV_NAME": "Craftax-Classic-Pixels-v1",
        # "ENV_NAME": "Craftax-Classic-Symbolic-v1",
        "SEED": 0,
        "NUM_SEEDS": 1,
        "WANDB_MODE": "online",  # set to online to activate wandb
        "ENTITY": "",
        "PROJECT": "dreamerv3_flax_craftax",
    }
    
    wandb.init(
        # entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["DreamerV3", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
        name=f'purejaxrl_ppo_{config["ENV_NAME"]}',
        config=config,
        mode="online",
    )

    rng = jax.random.PRNGKey(config["SEED"])
    # rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
