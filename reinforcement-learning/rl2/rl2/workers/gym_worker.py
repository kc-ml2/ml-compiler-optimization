from rl2.interfaces import Agent

from rl2.workers.base_worker import RolloutWorker


class GymRolloutWorker(RolloutWorker):
    def __init__(
            self,
            initial_transition,
            env,  # must be reset outside
            agent: Agent,
            max_steps: int,
            log_interval: int
    ):
        super().__init__(
            initial_transition=initial_transition,
            env=env,  # must be reset outside
            agent=agent,
            max_steps=max_steps,
            log_interval=log_interval,
        )

    def last_transition(self):
        return self.obs, self.ac, self.rew, self.done

    # def _rollout(self):
    #     self.ac = self.agent.act(self.obs)
    #     self.agent.observe(self.obs, self.ac, self.rew, self.done)
    #     self.obs, self.rew, self.done, self.info = self.env.step(self.ac)


    def run(self, steps):
        self.agent.observe(*self.last_transition())
        ############
        for step in range(steps):
            self.global_steps += 1

            self.ac = self.agent.act(self.obs)
            self.agent.observe(self.obs, self.ac, self.rew, self.done)
            self.obs, self.rew, self.done, self.info = self.env.step(self.ac)

            if self.done:
                self.obs, self.rew, self.done = self.env.reset(), 0., False

            self._rollout()
        #############
        # info = {}
        # # if self.env.episode_id >= 100:
        # if self.global_steps % self.log_interval == 0:
        #     # if self.global_step % log_interval == 0:
        #     num_episodes = 100
        #     lengths = np.array(self.env.get_episode_lengths()[-num_episodes:])
        #     returns = np.array(self.env.get_episode_rewards()[-num_episodes:])
        #     info['average_length'] = lengths.mean()
        #     info['average_return'] = returns.mean()
        #
        #     self.max_length = max(self.max_length, lengths.max())
        #     info['max_length'] = self.max_length
        #
        #     self.max_return = max(self.max_return, returns.max())
        #     info['max_return'] = self.max_return
        #
        # return info

