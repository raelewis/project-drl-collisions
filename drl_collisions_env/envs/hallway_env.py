from enum import Enum

import numpy as np

import gymnasium as gym
from gymnasium import spaces, utils
import xml.etree.ElementTree as ET

from numpy.typing import NDArray
from drl_collisions_env.envs.hallway import Hallway
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces
import os, mujoco, random, sys, time

from .envs import SIMPLE, SIMPLE_CORNER, OPEN, CORNER_OBS, LARGE_OBS, MULTI_OBS

ENVS = {
    "SIMPLE": SIMPLE,
    "SIMPLE_CORNER": SIMPLE_CORNER, 
    "OPEN": OPEN,
    "CORNER_OBS": CORNER_OBS,
    "LARGE_OBS": LARGE_OBS,
    "MULTI_OBS": MULTI_OBS
}

class HallwayEnv(MujocoEnv, utils.EzPickle):
    # Environment metadata
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, 
                 render_mode,
                 episode_len = 500,
                 path = 'SIMPLE',
                 rewards = "sparse",
                 **kwargs):

        utils.EzPickle.__init__(self, **kwargs)
        self.scale = 1.0
        self.height = .4
        self.path = ENVS[path]
        self.render_mode = render_mode
        self.reward_type = rewards
        
        path_dir = os.getcwd() + "/drl_collisions_env/envs/assets/default_env.xml"
        self.room, self.xml_path = Hallway._create_path(self.path, self.scale, self.height, path_dir) 


        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                ),

                desired_goal=spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                ),

                achieved_goal=spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                )
            )
        )

        MujocoEnv.__init__(self, 
                           self.xml_path, 
                           frame_skip=5, 
                           observation_space=self.observation_space,
                           render_mode=self.render_mode)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self.step_number = 0
        self.episode_len = episode_len
        self.obstacles = self._check_for_obstacles()
        self.obstacle_name = []
        for x in range(self.obstacles):
            self.obstacle_name.append(f"door_{x}")
        
        self.goal = self._find_pos('G')
        self.start = self._find_pos('S')

        # Array indexes of start + goal
        self._start_pos = np.array([self.start[0], self.start[1]], dtype=int)
        self._goal_pos = np.array([self.goal[0], self.goal[1]], dtype=int)

        # MuJoCo coords of start + goal
        self._goal = np.array([self.room._get_x_pos(self.goal[1]), self.room._get_y_pos(self.goal[0])])
        self._agent_start = np.array([self.room._get_x_pos(self.start[1]), self.room._get_y_pos(self.start[0])])
        self._agent = self._agent_start.copy()

        self.active = False
        self.prob = .2
        self.data.qpos[0:2] = self._agent_start

    def _find_pos(self, pos):
        """ Helper function to return the index of certain positions (Start, Goal). """
        for sub in self.path:
            if pos in sub:
                return (self.path.index(sub), sub.index(pos)) 

    def _get_observation(self, reset=None):
        """ Returns a dictionary containing the current observation of the agent within the environment.
            Observation:    The agent qvel and qpos
            achieved_goal:  The current position of the agent
            desired_goal:   The current position of the goal (x y coords in mujoco env"""
        return {
            "observation": np.concatenate([self.data.qpos[0:2], self.data.qvel[0:2]]).ravel(),
            "achieved_goal": self.data.qpos[0:2],
            "desired_goal": self._goal,
        }
    
    def _get_info(self):
        return {
            "distance": self._calc_distance(),
            "success": False
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.init_qpos[:2] = self._agent_start
        self.step_number = 0
        self.set_state(self.init_qpos, self.init_qvel)
        observation = self._get_observation()
        info = self._get_info()

        if random.random() < self.prob:
            self.active = True
        else:
            self.active = False
            self.prob += .01

        return observation, info
    
    def reset_model(self):
        self.step_number = 0
        self.set_state(self.init_qpos, self.init_qvel)
        obs = {"observation": np.concatenate([self.data.qpos[0:2], self.data.qvel[0:2]]).ravel()}
        return obs
    
    def _check_for_obstacles(self):
        count = 0
        for p in self.path:
            count += p.count('O')
        return count
    
    def _sim_obstacles(self):
        """Function that randomly determines whether or not an obstacle will activate during a step."""
        obs = np.random.uniform(low=-10.0, high=10.0, size=(self.obstacles,))
        for x in range(len(obs)):
            y = random.randint(1, 10)
            if y < 3 and self.active:
                obs[x] = 0.0
        return obs

    def _check_success(self, achieved, desired):
        return np.linalg.norm(achieved - desired) <= 0.45
    
    # Each time the agent takes a step / action, we must update the environment
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._limit_qvel()
        obs = self._get_observation()

        # If there are obstacles, simulate each obstacle
        if self.obstacles:
            obstacles = self._sim_obstacles()
            action = np.concatenate([action, obstacles]).ravel()
        
        self.do_simulation(action, self.frame_skip)
        observation = self._get_observation()

        reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'])
        self._agent = observation['achieved_goal']

        # Determine if the simulation is finished
        info = self._get_info()
        terminated, success = False, False

        if self._check_success(observation['achieved_goal'], observation['desired_goal']):
            success = True
            terminated = True
            info["success"] = success
        elif self._check_collision():
            terminated = True
            success = False
            reward += -5       # Penalty for collision
        else:
            reward =-.5        # Minor negative reward per timestep added later
        
        info["reward"] = reward

        if self.render_mode == "human":
            self.render()

        self.step_number += 1
        truncated = self.step_number > self.episode_len
        return observation, reward, terminated, truncated, info

    # Determine actual reward function, but i think it should be of closer to goal, gain +.1 reward or something small
    def compute_reward(self, achieved, desired, info=None):
        match(self.reward_type):
            case "dense":
                new_dist = np.linalg.norm(achieved - desired, axis=-1)
                return np.exp(-new_dist)
            case "sparse":
                if self._check_success(achieved, desired):
                    return 1
                else:
                    return 0
            case _:
                raise Exception("No reward type specified.")
    
    def _calc_distance(self):
        return np.linalg.norm(self._agent - self._goal, axis=-1)

    def _limit_qvel(self):
        # Limit the velocity of our agent so it does not move too far on each step
        # https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/maze/point.py#L24
        qvel = np.clip(self.data.qvel, -5.0, 5.0)
        self.set_state(self.data.qpos, qvel)

    def _check_collision(self):
        for x in self.obstacle_name:
            if self.model.geom(x).id in self.data.contact.geom and self.model.geom("agent").id in self.data.contact.geom:
                return True
        return False

    # When the simulation is complete, close the window
    def close(self):
        super().close()
