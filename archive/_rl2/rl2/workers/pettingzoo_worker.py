import _thread as thread
import logging
from copy import deepcopy
from logging import getLogger

import numpy as np
from pettingzoo import ParallelEnv, AECEnv
from tqdm import tqdm

from rl2.workers.base_worker import RolloutWorker
from rl2.workers.gym_worker import GymRolloutWorker


class NDArrayPackage:
    # https://github.com/geek-ai/MAgent/blob/b1fce2799bacd1d29f4a14b3196abc56425a6e6b/python/magent/model.py#L70
    """wrapper for transferring numpy arrays by bytes"""

    def __init__(self, *args):
        if isinstance(args[0], np.ndarray):
            self.data = args
            self.info = [(x.shape, x.dtype) for x in args]
        else:
            self.data = None
            self.info = args[0]

        self.max_len = (1 << 30) / 4

    def send_to(self, conn, use_thread=False):
        assert self.data is not None

        def send_thread():
            for x in self.data:
                if np.prod(x.shape) > self.max_len:
                    seg = int(self.max_len // np.prod(x.shape[1:]))
                    for pt in range(0, len(x), seg):
                        conn.send_bytes(x[pt:pt + seg])
                else:
                    conn.send_bytes(x)

        if use_thread:
            thread.start_new_thread(send_thread, ())
        else:
            send_thread()

    def recv_from(self, conn):
        bufs = []
        for info in self.info:
            buf = np.empty(shape=(int(np.prod(info[0])),), dtype=info[1])

            item_size = int(np.prod(info[0][1:]))
            if np.prod(info[0]) > self.max_len:
                seg = int(self.max_len // item_size)
                for pt in range(0, int(np.prod(info[0])), seg * item_size):
                    conn.recv_bytes_into(buf[pt:pt + seg * item_size])
            else:
                conn.recv_bytes_into(buf)
            bufs.append(buf.reshape(info[0]))
        return bufs


class PettingzooSequential(RolloutWorker):
    def __init__(
            self,
            initial_transition,
            env,  # must be reset outside
            agents,
            max_steps: int,
            log_interval: int
    ):
        assert isinstance(env, AECEnv)
        assert isinstance(agents, dict), f'agents must be dict'
        self.agents = agents
        super().__init__(
            initial_transition=initial_transition,
            env=env,  # must be reset outside
            agent=agents,
            max_steps=max_steps,
            log_interval=log_interval,
        )

    def last_transition(self):
        return self.env.last()

    # def _rollout(self):
    #     self.obs, self.rew, self.done, info = self.env.last()
    #     if self.done:
    #         self.ac = None
    #     self.env.step(self.ac)
    #     self.agent.observe(self.obs, self.ac, self.rew, self.done)
    #     self.ac = self.agent.act(self.obs)
    #
    # def _rollout_all(self):
    #     for agent_id in self.env.agent_iter():
    #         self.agent = self.agents[agent_id]
    #         self._rollout()

    def run(self, steps):
        for step in tqdm(range(steps)):
            self.global_steps += 1

            for agent_id in self.env.agent_iter():
                self.agent = self.agents[agent_id]
                self.ac = self.agent.act(self.obs)
                self.agent.observe(self.obs, self.ac, self.rew, self.done)

                if self.done:
                    self.ac = None

                self.env.step(self.ac)
                self.obs, self.rew, self.done, info = self.env.last()

            if all(self.env.dones.values()):
                self.env.reset()


class PettingzooParallel(RolloutWorker):
    def __init__(
            self,
            initial_transition,
            env,  # must be reset outside
            agents,
            max_steps: int,
            log_interval: int
    ):
        assert isinstance(env, ParallelEnv)
        # assert isinstance(agents, dict), f'agents must be dict'
        self.agents = agents
        super().__init__(
            initial_transition=initial_transition,
            env=env,  # must be reset outside
            agent=agents,
            max_steps=max_steps,
            log_interval=log_interval,
        )

        # self.logger = getLogger(__name__)
        # self.logger.info(str(env.observation_space(0)))
        # self.logger.info(str(env.action_space(0)))

    def last_transition(self):
        return self.obs, self.ac, self.rew, self.done

    # def _rollout(self):
    #     self.ac = self.agent.act(self.obs)
        # self.obs, self.rew, self.done, self.info = self.env.step(self.ac)

    def run(self, steps):
        for step in range(steps):
            self.global_steps += 1

            self.ac = self.agent.act(self.obs)
            self.agent.observe(self.obs, self.ac, self.rew, self.done)
            self.obs, self.rew, self.done, self.info = self.env.step(self.ac)

            if all(self.done.values()):
                self.env.reset()
                self.ac, self.rew, self.done = self.initial_transition[1:]

            # self._rollout()
