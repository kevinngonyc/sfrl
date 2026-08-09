"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import argparse

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from stable_baselines3.common.callbacks import CheckpointCallback
import stable_retro as retro


class SF3Action(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.valid_actions = [
            [],           # no-op
            [4],          # up
            [5],          # down
            [6],          # left
            [7],          # right
            [4, 7],       # up-right
            [4, 6],       # up-left
            [5, 7],       # down-right
            [5, 6],       # down-left
            [0],          # light kick
            [8],          # medium kick
            [13],         # heavy kick
            [1],          # light punch
            [9],          # medium punch
            [12],         # heavy punch
            [5, 0],       # down + light kick
            [5, 8],       # down + medium kick
            [5, 13],      # down + heavy kick
            [5, 1],       # down + light punch
            [5, 9],       # down + medium punch
            [5, 12],      # down + heavy punch
        ]
        self.action_space = gym.spaces.Discrete(len(self.valid_actions))


    def action(self, act):
        mask = [0] * self.env.action_space.n
        for idx in self.valid_actions[act]:
            mask[idx] = 1
        return mask


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="StreetFighter3rdStrike-Dreamcast-v0")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario, render_mode=None)
        env = WarpFrame(env)
        env = SF3Action(env)
        return env

    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 8), n_stack=4))
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        learning_rate=lambda f: f * 2.5e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000,  # save every 1M steps
        save_path="./models/ppo/checkpoints/",
        name_prefix="sf3_ppo",
    )
    
    model.learn(
        total_timesteps=100_000_000,
        log_interval=1,
        callback=checkpoint_callback,
    )
    model.save(f"models/ppo/{args.game}")


if __name__ == "__main__":
    main()
