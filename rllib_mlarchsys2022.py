# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Evaluation script for Compiler Optimization Competition at MLArchSys 2022.

Use this script to evaluate the agent that you have developed for optimizing
LLVM phase ordering for instruction count reduction. To do this, read through
the section marked "START OF USER CONFIGURATION" below. It contains a function,
optimize_program(), which you should use to implement your agent. When the
script is run, this function will be called on each of the programs in the test
set. There also are a number of configuration options in this section.

As a reminder, this competition uses CompilerGym v0.2.3. Install it using:

    $ python -m pip install compiler_gym==0.2.3

No other dependencies are required.

Once you have hooked in and configured your agent, run this script:

    $ python mlarchsys2022.py

It will download the test set and run your agent on it. A CSV file of results
will be logged to ~/logs/compiler_gym. The exact path will be printed once the
script has finished.

Once the script has finished, upload the CSV file to our OpenReview site, along
with a write-up of your approach (up to 4 pages), describing the algorithm used,
any parameters tuned, etc. Please follow the follow the ISCA'22 Latex Template:

<https://www.iscaconf.org/isca2022/submit/isca2022-latex-template.zip>

The OpenReview site can be found at:

<https://openreview.net/group?id=iscaconf.org/ISCA/2022/Workshop/MLArchSys>

For further details and submission instructions, see:

<https://sites.google.com/view/mlarchsys/isca-2022/compiler-optimization-competition>

For support with this script, please file an issue on
<https://github.com/facebookresearch/CompilerGym/issues> and put
"[MLArchSys2022]" at the start of the subject line.
"""

import io
import shutil
import sys
import tarfile
from pathlib import Path
from typing import List

import compiler_gym
import networkx as nx
import numpy as np
from compiler_gym.compiler_env_state import CompilerEnvStateWriter
from compiler_gym.datasets import Benchmark
from compiler_gym.envs import LlvmEnv
from compiler_gym.service.connection import SessionNotFound
from compiler_gym.util.download import download
from compiler_gym.util.runfiles_path import create_user_logs_dir, site_data_path
from compiler_gym.util.shell_format import emph, plural
from compiler_gym.util.statistics import arithmetic_mean, geometric_mean, stdev
from compiler_gym.wrappers import CompilerEnvWrapper, \
    CommandlineWithTerminalAction, TimeLimit
from fasteners import InterProcessLock
from gym import Wrapper
from gym.spaces import Box
########################################################################
# === 8<  START OF USER CONFIGURATION  >8 ===
########################################################################
from compopt.rllib_model import Model
from gym.spaces import Tuple
from ray import tune
from ray.rllib.agents.ppo import DDPPOTrainer, ddppo
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.spaces.repeated import Repeated

from compopt.constants import VOCAB, MAX_TEXT, MAX_TYPE, MAX_FLOW, MAX_POS
from compopt.wrappers import _Repeated

torch, nn = try_import_torch()


class EvalWrapper(Wrapper):
    def __init__(
            self,
            env,
            vocab,
            num_nodes,
            num_edges
    ):
        super().__init__(env)

        self.observation_space = self.compute_space(num_nodes, num_edges)
        # try:
        #     obs = self.env.observation.Programl()
        # except Exception as e:
        # obs = self.env.reset()
        # self.initial_obs = self.parse_graph(obs)

    def compute_space(self, num_nodes, num_edges):
        return Tuple([
            _Repeated(
                Box(
                    low=np.array([0, 0]),
                    high=np.array([MAX_TEXT, MAX_TYPE]),
                    shape=(2,),
                    dtype=int
                ),
                max_len=num_nodes * 4,  # TODO: find exact bound
            ),
            _Repeated(
                Box(
                    low=np.array([0, 0]),
                    high=np.array([num_nodes - 1, num_nodes - 1]),
                    shape=(2,),
                    dtype=int
                ),
                max_len=num_edges * 4,  # TODO: find exact bound
            ),
            _Repeated(
                Box(
                    low=np.array([0, 0]),
                    high=np.array([MAX_FLOW, MAX_POS]),
                    shape=(2,),
                    dtype=int
                ),
                max_len=num_edges * 4,  # TODO: find exact bound
            )
        ])

    def parse_nodes(self, ns, return_attr=False):
        # in-place
        x = []
        for nid in ns:
            n = ns[nid]
            n.pop('function', None)
            n.pop('block', None)
            n.pop('features', None)
            n['text'] = self.vocab.get(n['text'], self.max_text)

            if return_attr:
                x.append(np.array([n['text'], n['type']]))

        return x

    def parse_edges(self, es, return_attr=False):
        # in-place
        x = []
        for eid in es:
            e = es[eid]
            e['position'] = min(e['position'], self.max_pos)

            if return_attr:
                x.append(np.array([e['flow'], e['position']]))

        return x

    def parse_graph(self, g):
        # TODO: want to avoid for loop
        x = self.parse_nodes(g.nodes, return_attr=True)
        edge_attr = self.parse_edges(g.edges, return_attr=True)
        g = nx.DiGraph(g)

        edge_index = list(g.edges)

        return x, edge_index, edge_attr

    def step(self, ac):
        obs, rew, done, info = self.env.step(ac)
        obs = self.parse_graph(obs)

        return obs, rew, done, info

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        obs = self.parse_graph(obs)

        return obs


def make_dummy_env(config):
    """Return an environment with correct observation space for rllib"""
    env = compiler_gym.make("llvm-ic-v0", observation_space="Programl")
    env = CommandlineWithTerminalAction(env)
    env = EvalWrapper(env, VOCAB, config['num_nodes'], config['num_edges'])

    return env


def optimize_program(env: LlvmEnv) -> None:
    ckpt = '/data1/anthony/eval/DDPPOTrainer_myenv-v0_66aa6_00000_0_2022-05-25_14-24-35/checkpoint_000768/checkpoint-768'
    config = {
        "disable_env_checking": True,
        "_disable_preprocessor_api": False,
        "num_workers": 1,
        # "evaluation_num_workers": 2,
        # "evaluation_interval": 1,
        # Run 10 episodes each time evaluation runs.
        "evaluation_duration": 1,
        "num_cpus_per_worker": 32,
        "observation_filter": "NoFilter",
        "num_envs_per_worker": 1,
        "sgd_minibatch_size": 1,
        "rollout_fragment_length": 16,
        "num_sgd_iter": 4,
        "model": {
            "custom_model": "model",
            "vf_share_layers": False,
        },
        "log_level": "INFO",
        "ignore_worker_failures": True,
        "vf_loss_coeff": 0.5,
        # "grad_clip": 10,
        "clip_param": 0.1,
        "horizon": 1024,
        "env_config": {
            "num_nodes": 100,
            "num_edges": 200,
        }
    }
    ModelCatalog.register_custom_model("model", Model)

    env.observation_space = "Programl"
    env = CommandlineWithTerminalAction(env)
    obs = env.observation.Programl()
    config["env_config"]["num_nodes"] = nn = obs.number_of_nodes()
    config["env_config"]["num_edges"] = ne = obs.number_of_edges()
    config = ddppo.DDPPOTrainer.merge_trainer_configs(
        ddppo.DEFAULT_CONFIG,
        config
    )

    tune.register_env("myenv-v0", lambda c: make_dummy_env(c))
    agent = DDPPOTrainer(config, "myenv-v0")
    agent.restore(ckpt)

    env = TimeLimit(env, 1024)  # TODO: rough limit
    env = EvalWrapper(env, VOCAB, nn, ne)
    # Set observation space as needed:

    # agent.reset_config(config)
    # policy = agent.get_policy()
    # policy.model
    obs = env.reset()
    # obs = env.initial_obs
    done = False
    while not done:
        ac = agent.compute_single_action(obs)
        print(env.action_space.flags[ac])
        obs, _, done, _ = env.step(ac)


# Specifies the limit on the number of calls to env.step() and env.reset() that
# can be made in a single call to optimize_program(). There are two allowed
# values:
#
#   - "short": allows a maximum of 5 resets and 1,000 steps.
#   - "long": allows a maximum of 1,000 resets and 1,000,000 steps.
#
# If the limit for reset or step calls is reached, any further calls to
# env.step() will return True for 'done', and a 'truncated' key will be set in
# the 'info' dict. Once this state has been reached, the optimize_program()
# function should return.
#
# Note these are upper bounds - you do not need to use your entire budget if you
# do not wish to.
AGENT_BUDGET: str = "long"

# If you know that your agent is determinstic, you can set this to True. The
# only difference is that a nondeterministic agent will be evaluated 10 times
# on each benchmark. A deterministic agent will be evaluated only once.
AGENT_IS_DETERMINISTIC: bool = False

# The number of bitcodes to run the evaluation on. You may set this to a smaller
# number for debugging purposes. This value must be set to 100 for producing the
# final results.
BITCODES_TO_OPTIMIZE: int = 100

########################################################################
# === 8<  END OF USER CONFIGURATION  >8 ===
########################################################################
# Do not edit anything below this line.

__version__ = "0.0.2"


class ResetLimitedWrapper(CompilerEnvWrapper):
    """Enforce the maximum number of steps and resets."""

    def __init__(self, env: LlvmEnv, max_step_count: int, max_reset_count: int):
        super().__init__(env)
        self.max_step_count = max_step_count
        self.max_reset_count = max_reset_count
        self.step_count = 0
        self.reset_count = -1  # -1 to allow the initial call to reset()

    def step(self, action, *args, **kwargs):
        return self.multistep([action], *args, **kwargs)

    def multistep(self, actions, *args, **kwargs):
        self.step_count += 1

        if (
                self.step_count >= self.max_step_count
                or self.reset_count >= self.max_reset_count
        ):
            observation, reward, done, info = self.env.multistep(
                actions=[], *args, **kwargs
            )
            info["truncated"] = True
            info["step_limit_reached"] = self.step_count >= self.max_step_count
            info[
                "reset_limit_reached"] = self.reset_count >= self.max_reset_count
            return observation, reward, True, info

        return self.env.multistep(actions, *args, **kwargs)

    def reset(self, *args, **kwargs):
        self.reset_count += 1
        return self.env.reset(*args, **kwargs)


def download_and_unpack_test_set() -> Path:
    url = "https://dl.fbaipublicfiles.com/compiler_gym/mlarchsys2022/llvm_bitcodes-10.0.0-mlarchsys-competition-2022.tar.bz2"
    sha256 = "1a3bbe7d85118ef9dee5f7fb1cd5a1c5d8868f529f9de06b217de98110c68da3"
    download_dir = site_data_path("llvm-v0/mlarchsys")
    download_dir.mkdir(exist_ok=True, parents=True)
    lock_file = download_dir / ".lock"
    marker_file = download_dir / ".extracted"

    if marker_file.is_file():
        return download_dir / "mlarchsys-competition-2022"

    with InterProcessLock(lock_file):
        if marker_file.is_file():
            return download_dir / "mlarchsys-competition-2022"

        shutil.rmtree(download_dir / "mlarchsys-competition-2022",
                      ignore_errors=True)

        tar_data = io.BytesIO(download(url, sha256))
        with tarfile.open(fileobj=tar_data, mode="r:bz2") as arc:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(arc, str(download_dir))

        assert (download_dir / "mlarchsys-competition-2022").is_dir()
        assert (
                download_dir
                / "mlarchsys-competition-2022"
                / "70f7a924d4d49223f63b69e9bd8452a0407ac8d9.bc"
        ).is_file()
        marker_file.touch()

    return download_dir / "mlarchsys-competition-2022"


def get_test_bitcode_paths() -> List[Path]:
    paths = [
        p.absolute()
        for p in download_and_unpack_test_set().iterdir()
        if p.name.endswith(".bc")
    ]
    assert len(paths) == 100
    return paths


def main():
    # Sanity check user configuration.
    if AGENT_BUDGET != "short" and AGENT_BUDGET != "long":
        print(
            f"error: Invalid value for AGENT_BUDGET: {AGENT_BUDGET}. "
            "Must be one of {{short,long}}.",
            file=sys.stderr,
        )
        sys.exit(1)
    if BITCODES_TO_OPTIMIZE < 1:
        print(
            f"error: Invalid value for BITCODES_TO_OPTIMIZE: {BITCODES_TO_OPTIMIZE}. "
            "Must be > 0.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Only Linux is supported.
    if sys.platform == "darwin":
        print("error: macOS is not supported", file=sys.stderr)
        sys.exit(1)

    bitcodes = get_test_bitcode_paths()[:BITCODES_TO_OPTIMIZE]
    outdir = create_user_logs_dir("mlarchsys-competition-2022")
    (outdir / "bitcodes").mkdir()
    results_path = outdir / "results.csv"
    rewards, walltimes, errors = [], [], []

    print(f"This script version: {__version__}")
    print(f"Logging results to: {results_path}")
    print(f"Using agent budget: {AGENT_BUDGET}")
    print(f"Agent is deterministic? {AGENT_IS_DETERMINISTIC}")
    print(f"Number of bitcodes to optimize: {BITCODES_TO_OPTIMIZE}")
    print()
    with CompilerEnvStateWriter(open(results_path, "w")) as writer:
        for i, bitcode in enumerate(bitcodes, start=1):
            uri = f"benchmark://mlarchsys2022/{bitcode.stem}"
            print(emph(f"Evaluating benchmark {i} of {len(bitcodes)} ..."))
            for j in range(1, 2 if AGENT_IS_DETERMINISTIC else 11):
                with compiler_gym.make("llvm-ic-v0") as env:
                    env = ResetLimitedWrapper(
                        env,
                        max_reset_count=5 if AGENT_BUDGET == "short" else 1000,
                        max_step_count=1000 if AGENT_BUDGET == "short" else 1000000,
                    )
                    with open(bitcode, "rb") as f:
                        benchmark = Benchmark.from_file_contents(uri=uri,
                                                                 data=f.read())
                    env.reset(benchmark=benchmark)
                    optimize_program(env)
                    assert (
                            env.benchmark == benchmark
                    ), "Benchmark changed during optimization!"
                    step_count = (
                        f"{env.step_count:,d} {plural(env.step_count, 'step', 'steps')}"
                    )
                    reset_count = f"{env.reset_count:,d} {plural(env.reset_count, 'reset', 'resets')}"
                    print(emph(
                        f"    Agent completed in {step_count}, {reset_count}"))
                    try:
                        ic = env.observation.IrInstructionCount()
                        oz_ic = env.observation.IrInstructionCountOz()
                        reward = oz_ic / max(
                            env.observation.IrInstructionCount(), 1)
                        rewards.append(reward)
                        print(
                            emph(
                                f"    Agent instruction count: {ic:,d}, -Oz: {oz_ic:,d}"
                            )
                        )
                        prefix = (
                            "Reward"
                            if AGENT_IS_DETERMINISTIC
                            else f"Reward for run {j} of 10"
                        )
                        print(emph(f"    {prefix}: {reward:.4f}"))

                        state = env.state.copy()
                        state.reward = reward
                        walltimes.append(state.walltime)

                        writer.write_state(state, flush=True)
                        env.write_bitcode(outdir / "bitcodes" / f"{i}.bc")
                    except SessionNotFound:
                        error = (
                            f"Failed to compute final reward for benchmark {i}, run {j}"
                        )
                        print(emph(f"    {error}!"))
                        errors.append([errors])
                    print()

    if not rewards:
        print("No results gathered!", file=sys.stderr)
        sys.exit(1)
    print(f"Results written to: {results_path}")
    print("Please include this results file with your submission.")
    print(
        "Submit results to: "
        "https://openreview.net/group?id=iscaconf.org/ISCA/2022/Workshop/MLArchSys"
    )
    print()
    walltime_mean = f"{arithmetic_mean(walltimes):.3f}s"
    walltime_std = f"{stdev(walltimes):.3f}s"
    print(
        f"Mean walltime per benchmark: {emph(walltime_mean)} "
        f"(std: {emph(walltime_std)})"
    )
    reward = f"{geometric_mean(rewards):.3f}"
    reward_std = f"{stdev(rewards):.3f}"
    print(f"Geomean reward: {emph(reward)} " f"(std: {emph(reward_std)})")
    if errors:
        print(f"{len(errors)} were encountered during evaluation:")
        for error in error in errors:
            print(f"    {error}")


if __name__ == "__main__":
    main()
