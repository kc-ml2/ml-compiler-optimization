from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig


def get_ppo():
    config = PPOConfig()
    config.framework(
        framework='torch'
    ).training(
        vf_loss_coeff=0.5,
        clip_param=0.1,
        entropy_coeff=0.001,
        train_batch_size=512,
        # sgd_minibatch_size=2,
    ).resources(
        num_gpus=1,
        # num_cpus_per_worker=8,
    ).rollouts(
        # num_envs_per_worker=4,
        num_rollout_workers=0
    )
    #     .debugging(
    #     logger_config={
    #         'logdir': common.LOG_DIR
    #     }
    # )

    return config


def get_sac():
    config = SACConfig()
    config.framework(
        framework='torch'
    ).training(
        n_step=3,
    ).resources(
        num_gpus=4,
        num_cpus_per_worker=8,
    ).rollouts(
        num_envs_per_worker=16,
        num_rollout_workers=1
    )

    return config
