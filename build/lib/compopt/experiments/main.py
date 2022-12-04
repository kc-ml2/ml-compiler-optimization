import sys
from pprint import pprint

import ray
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from compiler_gym.wrappers import CommandlineWithTerminalAction

import common
from . import configs



class Workspace:
    def __init__(
            self,
            config,
            ckpt_dir,
            model_cls=None,
            ckpt_interval=sys.maxsize,
            verbose=False
    ):
        self.config = config
        self.ckpt_dir = ckpt_dir
        self.obs_type = config.env_config['observation_space']
        self.ckpt_interval = ckpt_interval
        self.verbose = verbose

        common.register_all(
            config=config,
            model_cls=model_cls
        )
        self.algo = config.build()

    def train(self, train_iter=int(1e+5)):
        for i in range(train_iter):
            result = self.algo.train()

            if self.verbose:
                pprint(result)
            # if i % self.ckpt_interval == 0:
            #     self.algo.save(self.ckpt_dir)
        self.algo.save(self.ckpt_dir)

    def eval(self):
        eval_llvm_instcount_policy(self._eval)

    def _eval(self, env):
        done = False
        env = CommandlineWithTerminalAction(env)
        while not done:
            ac = self.algo.compute_single_action(env.observation[self.obs_type])
            obs, _, done, _ = env.step(ac)


# algo.restore('/home/anthony/ray_results/PPO_llvm-ic-v0_2022-11-13_00-51-04euk5qbze/checkpoint_000033')


if __name__ == '__main__':
    ray.init()
    # from compopt.rllib_model import Model as model_cls

    config = configs.get_ppo()
    print(config)
    ws = Workspace(config=config, ckpt_dir=common.DATA_DIR)
    ws.train(50000)
    ws.eval()
    ws.train(50000)
    ws.eval()
    ws.algo.save(ws.ckpt_dir)
    ray.shutdown()
