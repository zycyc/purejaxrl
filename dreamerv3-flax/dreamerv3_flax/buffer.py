from chex import Array, ArrayTree
import numpy as np

from gym import spaces

from dreamerv3_flax.env import VecCrafterEnv
from wrappers import LogWrapper, FlattenObservationWrapper
from craftax.craftax_classic.envs.craftax_state import EnvParams

class ReplayBuffer:
    """Replay buffer."""

    def __init__(
        self,
        env: LogWrapper,
        env_params: EnvParams,
        batch_size: int = 16,
        num_steps: int = 64,
        buffer_size: int = int(1e6),
    ):
        """Initializes a replay buffer."""
        # Environment
        obs_space = env.observation_space(env_params)
        action_space = env.action_space(env_params)
        assert isinstance(obs_space, spaces.Box)
        assert isinstance(action_space, spaces.Discrete)
        obs_shape = obs_space.shape
        num_actions = action_space.n

        # Buffer
        self.obs = np.zeros((buffer_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((buffer_size, num_actions), dtype=np.float32)
        self.rewards = np.zeros((buffer_size), dtype=np.float32)
        self.dones = np.zeros((buffer_size), dtype=np.float32)
        self.firsts = np.zeros((buffer_size), dtype=np.float32)

        # Status
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: ArrayTree,
        actions: Array,
        rewards: Array,
        dones: Array,
        firsts: Array,
    ):
        """Adds the data to the buffer."""
        # Update the buffer.
        self.obs[self.pos] = obs.copy()
        self.actions[self.pos] = actions.copy()
        self.rewards[self.pos] = rewards.copy()
        self.dones[self.pos] = dones.copy()
        self.firsts[self.pos] = firsts.copy()

        # Update the status.
        self.pos += 1
        if self.pos == self.buffer_size:
            self.pos = 0
            self.full = True

    def sample(self) -> ArrayTree:
        """Samples data from the buffer."""
        # Define the indices.
        low = self.pos if self.full else 0
        high = self.pos - self.num_steps + 1
        if low >= high:
            high += self.buffer_size
        pos_indices = np.random.randint(low, high=high, size=self.batch_size)
        env_indices = np.random.randint(0, high=self.num_envs, size=self.batch_size)

        # Sample the batch.
        obs_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        first_batch = []

        for pos_idx, env_idx in zip(pos_indices, env_indices):
            step = np.arange(pos_idx, pos_idx + self.num_steps)
            step %= self.buffer_size
            obs_batch.append(self.obs[step, env_idx])
            action_batch.append(self.actions[step, env_idx])
            reward_batch.append(self.rewards[step, env_idx])
            done_batch.append(self.dones[step, env_idx])
            first_batch.append(self.firsts[step, env_idx])

        # Stack the batch.
        obs_batch = np.stack(obs_batch, axis=0)
        action_batch = np.stack(action_batch, axis=0)
        reward_batch = np.stack(reward_batch, axis=0)
        done_batch = np.stack(done_batch, axis=0)
        first_batch = np.stack(first_batch, axis=0)
        cont_batch = 1.0 - done_batch

        # Define the data.
        data = {
            "obs": obs_batch,
            "action": action_batch,
            "reward": reward_batch,
            "cont": cont_batch,
            "first": first_batch,
        }

        return data
