from compopt.rllib_model import Model
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

from compopt.wrappers import make_env

torch, nn = try_import_torch()

if __name__ == "__main__":
    default_config = ppo.DEFAULT_CONFIG.copy()
    config = {
        "framework": "torch",
        # "kl_coeff": 0.0,
        "lambda": 0.95,
        "clip_param": 0.1,
        "gamma": 0.9,
        "entropy_coeff": 0.05,
        # "shuffle_sequences": True,
        "log_level": "DEBUG",
        "train_batch_size": 32,
        "rollout_fragment_length": 8,
        "sgd_minibatch_size": 32,
        "num_sgd_iter": 8,
        "num_gpus": 4,
        # 'horizon': 128,
        # "soft_horizon": True,
        "num_workers": 4,
        "num_cpus_per_worker": 15,
        "num_gpus_per_worker": 0,
        "num_envs_per_worker": 1,
        "lr": 5e-5,
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "model": {
            "custom_model": "model",
        },
        "ignore_worker_failures": False,
        "disable_env_checking": True,
        "_disable_preprocessor_api": False,
        "env_config": {
        },
    }
    stop = {
        # "training_iteration": args.stop_iters,
        "timesteps_total": 1024,  # 8192
        # "episode_reward_mean": args.stop_reward,
    }
    # ray.init(ignore_reinit_error=True, include_dashboard=False)
    ModelCatalog.register_custom_model("model", Model)

    env_id = "wrapped-v0"
    tune.register_env(env_id, lambda c: make_env(c))

    config = ppo.PPOTrainer.merge_trainer_configs(default_config, config)
    analysis = tune.run(
        ppo.PPOTrainer,
        name='compopt-appo',
        # config=ppo.,
        reuse_actors=True,
        checkpoint_freq=int(stop["timesteps_total"] / 4),
        checkpoint_at_end=True,
        stop=stop,
        local_dir='/data1/anthony',
        # restore=ckpt,
        # metric='episode_reward_mean',
        # mode='max',
        max_failures=3,
    )
