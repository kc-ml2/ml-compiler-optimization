# import numpy as np
# import sys
#
# from rl2.interfaces import Agent
#
#
# class RolloutWorker:
#     def __init__(
#             self,
#             initial_obs,
#             env, # must be reset outside
#             agent: Agent,
#             max_steps: int=int(1e12),
#             log_interval: int=int(1e3)
#     ):
#         self.obs = initial_obs
#         self.env = env
#         self.agent = agent
#         self.max_steps = max_steps
#         self.log_interval = log_interval
#         # self.ac = agent.act(self.obs)
#         self.global_steps = 0
#
#         self.max_return, self.max_length = -sys.maxsize, -sys.maxsize  # tmp
#         self._sanity_check()
#
#     def _sanity_check(self):
#         pass
#
#     def last_transition(self):
#         return self.obs, self.ac, self.rew, self.done
#
#     def run(self, steps):
#         self.agent.observe(*self.last_transition())
#         ############
#         for step in range(steps - 1):
#             self.global_steps += 1
#             if self.done:
#                 self.obs, self.rew, self.done = self.env.reset(), 0., False
#             else:
#                 self.obs, self.rew, self.done, info = self.env.step(self.ac)
#
#             self.ac = self.agent.act(self.obs)
#             self.agent.observe(self.obs, self.ac, self.rew, self.done)
#         #############
#         # info = {}
#         # # if self.env.episode_id >= 100:
#         # if self.global_steps % self.log_interval == 0:
#         #     # if self.global_step % log_interval == 0:
#         #     num_episodes = 100
#         #     lengths = np.array(self.env.get_episode_lengths()[-num_episodes:])
#         #     returns = np.array(self.env.get_episode_rewards()[-num_episodes:])
#         #     info['average_length'] = lengths.mean()
#         #     info['average_return'] = returns.mean()
#         #
#         #     self.max_length = max(self.max_length, lengths.max())
#         #     info['max_length'] = self.max_length
#         #
#         #     self.max_return = max(self.max_return, returns.max())
#         #     info['max_return'] = self.max_return
#         #
#         # return info
