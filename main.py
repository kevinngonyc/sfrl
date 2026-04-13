import gymnasium as gym
import numpy as np
import stable_retro as retro
import cv2

class SF3Action(gym.ActionWrapper):
    

    def action(self, act):
        act[3] = 0
        return act


def main():
    
    env = retro.make(game="StreetFighter3rdStrike-Dreamcast-v0", 
                     render_mode="rgb_array"
                    )
    env = SF3Action(env)
    env.reset()
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        cv2.imshow('StreetFighter3rdStrike', observation)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if terminated or truncated:
            env.reset()
    env.close()


if __name__ == "__main__":
    main()