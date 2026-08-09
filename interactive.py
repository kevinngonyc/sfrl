from ale_py import env
import gymnasium as gym
from sympy import use
import stable_retro as retro
import cv2
import gzip
import numpy as np


class SF3Action(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.keymap = {
            ord('z'): 0,  # Light Kick
            ord('x'): 8,  # Medium Kick
            ord('c'): 13,  # Heavy Kick
            ord('a'): 1,  # Light Punch
            ord('s'): 9,  # Medium Punch
            ord('d'): 12,  # Heavy Punch
            82: 4,  # Up
            84: 5,  # Down
            81: 6,  # Left
            83: 7,  # Right
            13: 3,  # Start
        }
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


def main():
    
    env = retro.make(
        game="StreetFighter3rdStrike-Dreamcast-v0", 
        render_mode="rgb_array",
        state=retro.State.DEFAULT,
    )
    env = SF3Action(env)
    env.reset()
    

    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        cv2.imshow('StreetFighter3rdStrike', observation)
        key = cv2.waitKeyEx(1) & 0xFF
        action = np.zeros(env.action_space.n, dtype=np.int8)
        if key == ord('`'):
            break
        # elif key == ord('1'):
        #     state_bytes = env.unwrapped.em.get_state()
        #     with gzip.open(
        #         retro.data.get_file_path(env.unwrapped.gamename, "rom.state", retro.data.Integrations.DEFAULT), 
        #         "wb"
        #     ) as f:
        #         f.write(state_bytes)
        # elif key in env.keymap:
        #     action[env.keymap[key]] = 1
        if terminated or truncated:
            env.reset()
    env.close()


if __name__ == "__main__":
    main()