import compiler_gym
import ray
from compiler_gym.wrappers import TimeLimit, CommandlineWithTerminalAction, \
    RuntimePointEstimateReward
from ray import air
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env

from compopt.rllib_utils import CustomCallbacks
from compopt.wrappers import RunnableWrapper, LogNormalizer

# observation_space='InstCountNorm'
observation_space = 'InstCountNorm'


def make_env():
    env = compiler_gym.make(
        "llvm-v0",
        observation_space=observation_space,
    )
    env = TimeLimit(env, max_episode_steps=256)
    env = CommandlineWithTerminalAction(env)
    env = RuntimePointEstimateReward(env)
    env = RunnableWrapper(env)
    if observation_space == 'Autophase':
        env = LogNormalizer(env)

    return env


register_env(
    "llvm", lambda _: make_env()
)
algo = (
    PPOConfig()
    .environment('llvm')
    .framework('torch')
    .training(
        train_batch_size=128,
        # sgd_minibatch_size=8,
        model={"fcnet_hiddens": [2048, 2048, 2048]}
    )
    .rollouts(
        num_envs_per_worker=8,
        num_rollout_workers=8,
        # batch_mode='complete_episodes',
        #rollout_fragment_length=4,
    )
    .resources(num_gpus=2)
    .callbacks(CustomCallbacks)
    # .debugging()
)

stop = {
    "timesteps_total": 50000000,
}
tuner = tune.Tuner(
    'PPO',
    param_space=algo.to_dict(),
    # tune_config=tune.TuneConfig(), # for hparam search
    run_config=air.RunConfig(
        stop=stop,
        local_dir='/data1/llvm-runtime',
        name=observation_space,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end=True,
            num_to_keep=5,
            checkpoint_score_attribute='episode_reward_mean',
            checkpoint_score_order='max',
        ),
    ),
)

ray.init(local_mode=False)
results = tuner.fit()
ray.shutdown()
