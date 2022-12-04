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

########################################################################
# === 8<  START OF USER CONFIGURATION  >8 ===
########################################################################
import torch
from PPO import ActorCritic
from train import to_pyg


def optimize_program(env: LlvmEnv) -> None:
    #  __________________
    # < Your agent here! >
    #  ------------------
    #         \   ^__^
    #          \  (oo)\_______
    #             (__)\       )\/\
    #                 ||----w |
    #                 ||     ||
    #
    # Run your agent here, interacting with the input env. Once the agent
    # completed, return. Final reward is calculated based on the env state after
    # this function has returned.
    device = 'cuda:3'
    print(env.benchmark)
    env = CommandlineWithTerminalAction(env)
    env = TimeLimit(env, 256)
    print(env)
    # env = TimeLimit(env, env.action_space.n)
    env.observation_space = 'Programl'
    action_dim = env.action_space.n
    agent = ActorCritic(action_dim=action_dim)
    print('loading')
    agent.load_state_dict(
        torch.load(
            '/home/anthony/PPO-PyTorch/PPO_preTrained/llvm-ic-v0/PPO_llvm-ic-v0_0_20220522163939.pth',
            device,
        )
    )
    print('loaded')
    agent = agent.to(device)
    done = False

    obs = env.reset()
    obs = to_pyg(obs).to(device)
    while not done:
        ac, _ = agent.act(obs)
        print(env.step_count, env.action_space.commandline([ac]), env.state)
        obs, rew, done, info = env.step(ac.item())
        obs = to_pyg(obs).to(device)

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
AGENT_BUDGET: str = "short"

# If you know that your agent is determinstic, you can set this to True. The
# only difference is that a nondeterministic agent will be evaluated 10 times
# on each benchmark. A deterministic agent will be evaluated only once.
AGENT_IS_DETERMINISTIC: bool = True # TODO

# The number of bitcodes to run the evaluation on. You may set this to a smaller
# number for debugging purposes. This value must be set to 100 for producing the
# final results.
BITCODES_TO_OPTIMIZE: int = 50


########################################################################
# === 8<  END OF USER CONFIGURATION  >8 ===
########################################################################
# Do not edit anything below this line.

__version__ = "0.0.1"


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

        shutil.rmtree(download_dir / "mlarchsys-competition-2022", ignore_errors=True)

        tar_data = io.BytesIO(download(url, sha256))
        with tarfile.open(fileobj=tar_data, mode=f"r:bz2") as arc:
            arc.extractall(str(download_dir))

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
                        max_reset_count=5 if AGENT_BUDGET == "short" else 1000, # TODO:
                        max_step_count=1000 if AGENT_BUDGET == "short" else 1000000, # TODO:
                    )
                    with open(bitcode, "rb") as f:
                        benchmark = Benchmark.from_file_contents(uri=uri, data=f.read())
                    env.reset(benchmark=benchmark)
                    optimize_program(env)
                    assert (
                        env.benchmark == benchmark
                    ), f"Benchmark changed during optimization!"
                    step_count = (
                        f"{env.step_count:,d} {plural(env.step_count, 'step', 'steps')}"
                    )
                    reset_count = f"{env.reset_count:,d} {plural(env.reset_count, 'reset', 'resets')}"
                    print(emph(f"    Agent completed in {step_count}, {reset_count}"))
                    try:
                        ic = env.observation.IrInstructionCount()
                        oz_ic = env.observation.IrInstructionCountOz()
                        reward = oz_ic / max(env.observation.IrInstructionCount(), 1)
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