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
        self.ctrl_cost_coeff = 1e-1
        self._env = walker.env('gym_cheetah', 1234, {})
        self.init_qacc = self._env._env.env.model.data.qacc
        self.init_ctrl = self._env._env.env.model.data.ctrl
        self.action_space = self._env._env.action_space
        self.observation_space = self._env._env.observation_space

    def reset(self, init_state=None):
        if init_state is not None:
            self.reset_mujoco(init_state)
            self._env._env.env.model.forward()
        else:
            self._env.reset()
        return self.get_current_obs()

    def reset_mujoco(self, init_state=None):
        if init_state is None:
            self._env._env.env.model.data.qpos = self._env._env.env.init_qpos + \
                                   np.random.normal(size=self._env._env.env.init_qpos.shape) * 0.01
            self._env._env.env.model.data.qvel = self._env._env.env.init_qvel + \
                                   np.random.normal(size=self._env._env.env.init_qvel.shape) * 0.1
            self._env._env.env.model.data.qacc = self.init_qacc
            self._env._env.env.model.data.ctrl = self.init_ctrl
        else:
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
        # return -np.mean(x_next[:, 9] - self.ctrl_cost_coeff * 0.5 * np.sum(np.square(u), axis=1))
        return -np.mean(np.clip(x_next[:, 9] - self.ctrl_cost_coeff * 0.5 * np.sum(np.square(u), axis=1), -10, 10))

    def cost_tf(self, x, u, x_next):
        # return -tf.reduce_mean(x_next[:, 9] - self.ctrl_cost_coeff * 0.5 * tf.reduce_sum(tf.square(u), axis=1))
        return -tf.reduce_mean(tf.clip_by_value(x_next[:, 9] - self.ctrl_cost_coeff * 0.5 * tf.reduce_sum(tf.square(u), axis=1), -10, 10))

    def cost_np_vec(self, x, u, x_next):
        assert np.amax(np.abs(u)) <= 1.0
        # return -(x_next[:, 9] - self.ctrl_cost_coeff * 0.5 * np.sum(np.square(u), axis=1))
        return -np.clip(x_next[:, 9] - self.ctrl_cost_coeff * 0.5 * np.sum(np.square(u), axis=1), -10, 10)
