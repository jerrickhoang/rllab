import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
from rllab.envs.base import Step
from rllab.misc import logger
from rllab.misc.overrides import overrides

from rllab.core.serializable import Serializable
from mbbl.env.gym_env import walker


class BenchmarkHalfCheetahEnv(Serializable):

    def __init__(self, *args, **kwargs):
        Serializable.__init__(self, *args, **kwargs)
        self._env = walker.env('gym_cheetah', 1234, {})
        self.action_space = self._env._env.action_space
        self.observation_space = self._env._env.observation_space

    def reset(self, init_state=None):
        self._env.reset()
        if init_state is not None:
            self.reset_mujoco(init_state)
            self._env._env.env.model.forward()
        return self.get_current_obs()

    def reset_mujoco(self, init_state):
        start = 0
        for datum_name in ["qpos", "qvel", "qacc", "ctrl"]:
            datum = getattr(self._env._env.env.model.data, datum_name)
            datum_dim = datum.shape[0]
            datum = init_state[start: start + datum_dim]
            setattr(self._env._env.env.model.data, datum_name, datum)
            start += datum_dim

    @property
    def model(self):
        return self._env._env.env.model

    def get_current_obs(self):
        return self._env._get_observation()

    def get_body_xmat(self, body_name):
        return self._env._env.env.get_body_xmat(body_name)

    def get_body_comvel(self, body_name):
        return self._env._env.env.get_body_comvel(body_name)

    def get_body_com(self, body_name):
        return self._env._env.env.get_body_com(body_name)

    def step(self, action):
        next_obs, reward, done, _ = self._env.step(action)
        return Step(next_obs, reward, done)

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))


    def cost_np(self, x, u, x_next):
        assert np.amax(np.abs(u)) <= 1.0
        data_dict = {"start_state": x, "action": u, "end_state": x_next}
        return -np.mean(np.clip(self._env.reward(data_dict), -10, 10))

    def cost_tf(self, x, u, x_next):
        data_dict = {"start_state": x, "action": u, "end_state": x_next}
        return -tf.reduce_mean(tf.clip_by_value(self._env.reward_tf(data_dict), -10, 10))

    def cost_np_vec(self, x, u, x_next):
        assert np.amax(np.abs(u)) <= 1.0
        return self.cost_np(x, u, x_next)
