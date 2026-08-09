import gymnasium as gym
import stable_retro as retro
import cv2
import numpy as np
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO


class SF3Action(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.valid_actions = [
            [], [4], [5], [6], [7], [4, 7], [4, 6], [5, 7], [5, 6],
            [0], [8], [13], [1], [9], [12],
            [5, 0], [5, 8], [5, 13], [5, 1], [5, 9], [5, 12],
        ]
        self.action_space = gym.spaces.Discrete(len(self.valid_actions))

    def action(self, act):
        mask = [0] * self.env.action_space.n
        for idx in self.valid_actions[act]:
            mask[idx] = 1
        return mask


def make_env():
    def _init():
        env = retro.make(
            game="StreetFighter3rdStrike-Dreamcast-v0",
            render_mode="rgb_array",
            state=retro.State.DEFAULT,
        )
        env = WarpFrame(env)
        env = SF3Action(env)
        return env
    return _init


def main():
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)   # <-- matches the (4, 84, 84) trained shape

    model = PPO.load("models/ppo/checkpoints/sf3_ppo_8000000_steps.zip", env=env)

    observation = env.reset()

    while True:
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)  # VecEnv step API: single `done`
        frame = env.render()
        cv2.imshow('StreetFighter3rdStrike', frame)
        key = cv2.waitKeyEx(1) & 0xFF
        if key == ord('`'):
            break
        if done[0]:
            observation = env.reset()

    env.close()


if __name__ == "__main__":
    main()