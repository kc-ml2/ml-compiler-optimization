import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import wandb

from rl2.interfaces import Agent
from copy import deepcopy


class RolloutWorker:
    def __init__(
            self,
            initial_transition,
            env,  # must be reset outside
            agent: Agent,
            max_steps: int = int(1e6),
            log_interval: int = 0,
    ):
        self.initial_transition = deepcopy(list(initial_transition))
        self.obs, self.ac, self.rew, self.done = initial_transition
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.global_steps = 0

        self.max_return, self.max_length = float('-inf'), float('-inf')  # tmp
        self.episodic_rewards = []
        self.episode_id = 0

    def last_transition(self):
        return self.obs, self.ac, self.rew, self.done

    def run(self, steps):
        self.agent.observe(*self.last_transition())
        ############
        for step in range(steps - 1):
            print(self.global_steps)
            print(self.rew)

            self.episodic_rewards.append(self.rew)

            self.global_steps += 1
            if self.global_steps % 50 == 0:
                self.done = True
            if self.done:
                print(f'============episode {self.episode_id}')
                print(self.env.state)
                print(sum(self.episodic_rewards))
                self.obs, self.rew, self.done = self.env.reset(), 0., False
                self.episodic_rewards = []
                self.episode_id += 1

            else:
                self.obs, self.rew, self.done, info = self.env.step(self.ac)

            self.ac = self.agent.act([self.obs])
            self.agent.observe(self.obs, self.ac, self.rew, self.done)
            if self.global_steps > 0 and self.global_steps % 16 == 0:
                ret = self.agent.learn()
                # log everything
                wandb.log(ret)

    def __enter__(self):
        self.start_dt = time.clock()

        if self.saving_model:
            signal.signal(
                signal.SIGINT,
                lambda sig, frame: self.save_model()
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_dt = time.clock()

        if self.saving_model:
            try:
                self.save_model()
            except Exception as e:
                print('saving failed')
        #         logger.warning(f'saving model failed, {e}')
        #
        # logging.info(f'Time elapsed {self.end_dt - self.start_dt}.')
        # logging.info(f'Ran from {self.start_dt} to {self.end_dt}.')

    @property
    def running_time(self):
        return self.end_dt - self.start_dt

    def default_save_dir(self):
        return f"""{type(self).__name__}_{datetime.now().strftime('%Y%m%d%H%M%S')}"""

    def ensure_save_dir(self, name):
        save_dir = os.path.join(
            self.base_log_dir,
            self.default_save_dir(),
            name
        )

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        return save_dir
    # def _rollout(self):
    #     raise NotImplementedError
