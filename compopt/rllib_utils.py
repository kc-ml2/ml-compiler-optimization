from typing import Dict, Optional, Union

from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID


class CustomCallbacks(DefaultCallbacks):
    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2, Exception],
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        envs = base_env.get_sub_environments()
        runtimes = []
        for env in envs:
            try:
                # env.reward_space.previous_runtime
                runtime = env.observation['Runtime']
            except:
                env.reset()
                runtime = env.observation['Runtime']
            runtimes.append(runtime)
        avg_runtime = sum(runtimes) / len(runtimes)
        episode.custom_metrics['avg_runtime'] = avg_runtime