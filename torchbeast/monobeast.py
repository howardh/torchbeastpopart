#!/home/ml/users/hhuang63/rl/ENV/bin/python

# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from collections import OrderedDict, defaultdict
import logging
import os
import re
import pprint
import threading
import time
import timeit
import traceback
import typing

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import shelve
import orion
import orion.client
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from torchbeast import atari_wrappers
from torchbeast.core import environment
from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace

from torchbeast.models.attention_augmented_agent import AttentionAugmentedAgent
from torchbeast.models.resnet_monobeast import ResNet
from torchbeast.models.atari_net_monobeast import AtariNet
from torchbeast.models.custom import MonobeastMP2

from torchbeast.analysis.gradient_tracking import GradientTracker

import wandb
import wandb.util

# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--env", type=str, default="PongNoFrameskip-v4",
                    help="Gym environment.")
parser.add_argument("--atari-mode", type=int, default=None,
                    help="Mode of the Atari environment.")
parser.add_argument("--atari-difficulty", type=int, default=None,
                    help="Difficulty of the Atari environment.")
parser.add_argument("--atari-action-repeat", type=float, default=None,
                    help="Probability of action repeat in the Atari environment.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors per environment (default: 4).")
parser.add_argument("--total_steps", default=100000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--num_actions", default=6, type=int, metavar="A",
                    help="Number of actions.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")
parser.add_argument("--agent_type", type=str, default="resnet",
                    help="The type of network to use for the agent.")
parser.add_argument("--frame_height", type=int, default=84,
                    help="Height to which frames are rescaled.")
parser.add_argument("--frame_width", type=int, default=84,
                    help="Width to which frames are rescaled.")
parser.add_argument("--aaa_input_format", type=str, default="gray_stack", choices=["gray_stack", "rgb_last", "rgb_stack"],
                    help="Color format of the frames as input for the AAA.")
parser.add_argument("--use_popart", action="store_true",
                    help="Use PopArt Layer.")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.0006,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")

# Misc settings.
parser.add_argument("--write_profiler_trace", action="store_true",
                    help="Collect and write a profiler trace "
                    "for chrome://tracing/.")
parser.add_argument("--save_model_every_nsteps", default=0, type=int,
                    help="Save model every n steps")
parser.add_argument("--beta", default=0.0001, type=float,
                    help="PopArt parameter")
#parser.add_argument("--wandb", action="store_true",
#                    help="Track the experiment on W&B")
parser.add_argument("--wandb", nargs='?', const='', default='', type=str,
                    help="Track the experiment on W&B")

# Test settings.
parser.add_argument("--num_episodes", default=100, type=int,
                    help="Number of episodes for Testing.")
parser.add_argument("--actions",
                    help="Use given action sequence.")
parser.add_argument("--stochastic", action="store_true",
                    help="Sample actions according to the predicted distribution rather than taking the greedy action.")
parser.add_argument("--test_results_path", default=None, type=str,
                    help="File/directory to save test results to")

# yapf: enable

logging.basicConfig(format="[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s", level=0)
logging.getLogger("matplotlib.font_manager").disabled = True

Buffers = typing.Dict[str, typing.List[torch.Tensor]]

gradient_tracker = GradientTracker()


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        # target=torch.flatten(actions, 0, 1),
        target=torch.flatten(actions, 0, 2),
        reduction="none",
    )
    # cross_entropy = cross_entropy.view_as(advantages)
    cross_entropy = cross_entropy.view_as(actions)
    return torch.sum(cross_entropy * advantages.detach())


def act(
    flags,
    env_name: str,
    task: int,
    full_action_space: bool,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        # create the environment from command line parameters
        # => could also create a special one which operates on a list of games (which we need)
        env_config = { 'full_action_space': full_action_space }
        if flags.atari_mode is not None:
            env_config['mode'] = flags.atari_mode
        if flags.atari_difficulty is not None:
            env_config['difficulty'] = flags.atari_difficulty
        if flags.atari_action_repeat is not None:
            env_config['repeat_action_probability'] = flags.atari_action_repeat
        gym_env = create_env(env_name, frame_height=flags.frame_height, frame_width=flags.frame_width,
                             gray_scale=(flags.aaa_input_format == "gray_stack"),
                             config={ 'full_action_space': full_action_space, **env_config },
                             task=task)

        # generate a seed for the environment (NO HUMAN STARTS HERE!), could just
        # use this for all games wrapped by the environment for our application
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)

        # wrap the environment, this is actually probably the point where we could
        # use multiple games, because the other environment is still one from Gym
        env = environment.Environment(gym_env)

        # get the initial frame, reward, done, return, step, last_action
        env_output = env.initial()

        # perform the first step
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            # get a buffer index from the queue for free buffers (?)
            index = free_queue.get()
            # termination signal (?) for breaking out of this loop
            if index is None:
                break

            # Write old rollout end.
            # the keys here are (frame, reward, done, episode_return, episode_step, last_action)
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            # here the keys are (policy_logits, baseline, action)
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            # I think the agent_state is just the RNN/LSTM state (which will be the "initial" state for the next step)
            # not sure why it's needed though because it really just seems to be the initial state before starting to
            # act; however, it might be randomly initialised, which is why we might want it...
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout
            for t in range(flags.unroll_length):
                timings.reset()

                # forward pass without keeping track of gradients to get the agent action
                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                timings.time("model")

                # agent acting in the environment
                env_output = env.step(agent_output["action"])

                timings.time("step")

                # writing the respective outputs of the current step (see above for the list of keys)
                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")

            # after finishing a trajectory put the index in the "full queue",
            # presumably so that the data can be processed/sent to the learner
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock(),
):
    # need to make sure that we wait until batch_size trajectories/rollouts have been put into the queue
    with lock:
        timings.time("lock")
        # get the indices of actors "offering" trajectories/rollouts to be processed by the learner
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")

    # create the batch as a dictionary for all the data in the buffers (see act() function for list of
    # keys), where each entry is a tensor of these values stacked across actors along the first dimension,
    # which I believe should be the "batch dimension" (see _format_frame())
    batch = {key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers}

    # similar thing for the initial agent states, where I think the tuples are concatenated to become torch tensors
    initial_agent_state = (torch.cat(ts, dim=1) for ts in zip(*[initial_agent_state_buffers[m] for m in indices]))
    timings.time("batch")

    # once the data has been "transferred" into batch and initial_agent_state,
    # signal that the data has been processed to the actors
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")

    # move the data to the right device (e.g. GPU)
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(t.to(device=flags.device, non_blocking=True) for t in initial_agent_state)
    timings.time("device")

    return batch, initial_agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    stats,
    lock=threading.Lock(),
    envs=None
):
    """Performs a learning (optimization) step."""
    with lock:
        # forward pass with gradients
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        # if specified, clip rewards between -1 and 1
        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        # the "~"/tilde operator is apparently kind of a complement or # inverse, so maybe this just reverses
        # the "done" tensor? in that case would discounting only be applied when the game was NOT done?
        discounts = (~batch["done"]).float() * flags.discounting

        # prepare tensors for computation of the loss
        task = F.one_hot(batch["task"].long(), flags.num_tasks).float()
        clipped_rewards = clipped_rewards[:, :, None]
        discounts = discounts[:, :, None]

        # prepare PopArt parameters as well
        mu = model.baseline.mu[None, None, :]
        sigma = model.baseline.sigma[None, None, :]

        # get the V-trace returns; I hope nothing needs to be changed about this, but I think
        # once one has the V-trace returns it can just be plugged into the PopArt equations
        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
            normalized_values=learner_outputs["normalized_baseline"],
            mu=mu,
            sigma=sigma
        )

        # PopArt normalization
        with torch.no_grad():
            normalized_vs = (vtrace_returns.vs - mu) / sigma

        # policy gradient loss
        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages * task,
        )

        # value function/baseline loss (1/2 * squared difference between V-trace and value function)
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            # vtrace_returns.vs - learner_outputs["baseline"]
            (normalized_vs - learner_outputs["normalized_baseline"]) * task
        )

        # entropy loss for getting a "diverse" action distribution (?), "normal entropy" over action distribution
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        # do the backward pass (WITH GRADIENT NORM CLIPPING) and adjust hyperparameters (scheduler, ?)
        optimizer.zero_grad()
        total_loss.backward()
        # plot_grad_flow(model.named_parameters(), flags)
        gradient_tracker.process_backward_pass(model.named_parameters())
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        # update the PopArt parameters, which the optimizer does not take care of
        if flags.use_popart:
            model.baseline.update_parameters(vtrace_returns.vs, task)

        # update the actor model with the new parameters
        actor_model.load_state_dict(model.state_dict())

        # get the returns only for finished episodes (where the game was played to completion)
        if batch['done'].any():
            returns_by_game = defaultdict(lambda: [])
            stats["returns_by_game"] = stats.get("returns_by_game", {})
            for i,r in zip(batch['task'][batch['done']],batch["episode_return"][batch["done"]]):
                returns_by_game[envs[i]].append(r.item())
                stats["returns_by_game"][envs[i]] = stats["returns_by_game"].get(envs[i], [])[:9] + [r.item()]
            #logging.info("returns by game:")
            #logging.info(pprint.pformat(dict(returns_by_game)))
            if flags.wandb is not None:
                try:
                    wandb.log({
                        **{f'returns/{k}': torch.tensor(v).mean().item() for k,v in returns_by_game.items()},
                        **{f'mu/{k}': mu[0, 0, i].item() for i,k in enumerate(envs)},
                        'loss/pg': pg_loss.item(),
                        'loss/baseline': baseline_loss.item(),
                        'loss/entropy': entropy_loss.item(),
                        'loss/total': total_loss.item(),
                        'transition_steps': stats['step'],
                        **{f'env_step/{k}': v for k,v in stats['env_step'].items()},
                    })
                except:
                    pass
        episode_returns = batch["episode_return"][batch["done"]]
        stats["step"] = stats.get("step", 0) + flags.unroll_length * flags.batch_size
        stats["episode_returns"] = tuple(episode_returns.cpu().numpy())
        stats["mean_episode_return"] = torch.mean(episode_returns).item()
        stats["total_loss"] = total_loss.item()
        stats["pg_loss"] = pg_loss.item()
        stats["baseline_loss"] = baseline_loss.item()
        stats["entropy_loss"] = entropy_loss.item()
        stats["mu"] = mu[0, 0, :]
        stats["sigma"] = sigma[0, 0, :]
        if "env_step" not in stats:
            stats["env_step"] = {}
        for task in batch["task"][0].cpu().numpy():
            stats["env_step"][envs[task]] = stats["env_step"].get(envs[task], 0) + flags.unroll_length

        return stats


def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(  # seems like these "inner" dicts could also be something else...
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1, flags.num_tasks), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1, 1), dtype=torch.int64),
        normalized_baseline=dict(size=(T + 1, flags.num_tasks), dtype=torch.float32),
        task=dict(size=(T + 1,), dtype=torch.int64)
    )
    buffers: Buffers = {key: [] for key in specs}

    # basically create a bunch of empty torch tensors according to the sizes in the specs dicts above
    # and do this for the specified number of buffers, so that there will be a list of length flags.num_buffers
    # for each key
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    # prepare for logging and saving models
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir)
    checkpointpath = os.path.expandvars(os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar")))
    if flags.save_model_every_nsteps > 0:
        os.makedirs(checkpointpath.replace("model.tar", "intermediate"), exist_ok=True)

    # get a list and determine the number of environments
    environments = flags.env.split(",")
    flags.num_tasks = len(environments)

    # set the number of buffers
    if flags.num_buffers is None:
        flags.num_buffers = max(2 * flags.num_actors * flags.num_tasks, flags.batch_size)
    if flags.num_actors * flags.num_tasks >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    # set the device to do the training on
    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    # set the environments
    if flags.env == "six":
        flags.env = "AirRaidNoFrameskip-v4,CarnivalNoFrameskip-v4,DemonAttackNoFrameskip-v4," \
                    "NameThisGameNoFrameskip-v4,PongNoFrameskip-v4,SpaceInvadersNoFrameskip-v4"
    elif flags.env == "three":
        flags.env = "AirRaidNoFrameskip-v4,CarnivalNoFrameskip-v4,DemonAttackNoFrameskip-v4"

    # set the right agent class
    if flags.agent_type.lower() in ["aaa", "attention_augmented", "attention_augmented_agent"]:
        Net = AttentionAugmentedAgent
        logging.info("Using the Attention-Augmented Agent architecture.")
    elif flags.agent_type.lower() in ["rn", "res", "resnet", "res_net"]:
        Net = ResNet
        logging.info("Using the ResNet architecture (monobeast version).")
    elif flags.agent_type.lower() in ["custom"]:
        Net = MonobeastMP2
        logging.info("Using custom model.")
    else:
        Net = AtariNet
        logging.warning("No valid agent type specified. Using the default agent.")

    # create a dummy environment, mostly to get the observation and action spaces from
    gym_env = create_env(environments[0], frame_height=flags.frame_height, frame_width=flags.frame_width,
                     gray_scale=(flags.aaa_input_format == "gray_stack"))
    observation_space_shape = gym_env.observation_space.shape
    action_space_n = gym_env.action_space.n
    full_action_space = False
    for environment in environments[1:]:
        gym_env = create_env(environment)
        if gym_env.action_space.n != action_space_n:
            logging.warning("Action spaces don't match, using full action space.")
            full_action_space = True
            action_space_n = 18
            break

    # create the model and the buffers to pass around data between actors and learner
    model = Net(observation_space_shape,
                action_space_n,
                use_lstm=flags.use_lstm,
                num_tasks=flags.num_tasks,
                use_popart=flags.use_popart,
                reward_clipping=flags.reward_clipping,
                rgb_last=(flags.aaa_input_format == "rgb_last"))
    buffers = create_buffers(flags, observation_space_shape, model.num_actions)

    # I'm guessing that this is required (similarly to the buffers) so that the
    # different threads/processes can all have access to the parameters etc. (?)
    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    # create stuff to keep track of the actor processes
    actor_processes = []
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    # create and start actor threads (the same number for each environment)
    for i, environment in enumerate(environments):
        for j in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(
                    flags,
                    environment,
                    i,
                    full_action_space,
                    i*flags.num_actors + j,
                    free_queue,
                    full_queue,
                    model,
                    buffers,
                    initial_agent_state_buffers,
                ),
            )
            actor.start()
            actor_processes.append(actor)

    learner_model = Net(observation_space_shape,
                        action_space_n,
                        use_lstm=flags.use_lstm,
                        num_tasks=flags.num_tasks,
                        use_popart=flags.use_popart,
                        reward_clipping=flags.reward_clipping,
                        rgb_last=(flags.aaa_input_format == "rgb_last")).to(device=flags.device)

    # the hyperparameters in the paper are found/adjusted using population-based training,
    # which might be a bit too difficult for us to do; while the IMPALA paper also does
    # some experiments with this, it doesn't seem to be implemented here
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Logger
    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        "mu",
        "sigma",
    ] + [
        "{}_step".format(e) for e in environments
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    # Load state from a checkpoint, if possible.
    if os.path.exists(checkpointpath):
        checkpoint_states = torch.load(checkpointpath, map_location=flags.device)
        learner_model.load_state_dict(checkpoint_states["model_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_states["scheduler_state_dict"])
        stats = checkpoint_states.get("stats", {})
        logging.info(f"Resuming preempted job, current stats:\n{stats}")

    # Initialize actor model like learner model.
    model.load_state_dict(learner_model.state_dict())

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        # step in particular needs to be from the outside scope, since all learner threads can update
        # it and all learners should stop once the total number of steps/frames has been processed
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
            )
            learn(flags, model, learner_model, batch, agent_state, optimizer, scheduler, stats, envs=environments)
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys if "_step" not in k})
                for e in stats["env_step"]:
                    to_log["{}_step".format(e)] = stats["env_step"][e]
                plogger.log(to_log)
                step += T * B  # so this counts the number of frames, not e.g. trajectories/rollouts

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    # populate the free queue with the indices of all the buffers at the start
    for m in range(flags.num_buffers):
        free_queue.put(m)

    # start as many learner threads as specified => could in principle do PBT
    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,))
        thread.start()
        threads.append(thread)

    # W&B
    if flags.wandb is not None:
        wandb_run_id = None
        if flags.wandb != '':
            wandb_run_id = flags.wandb
        else:
            slurm_job_id = os.environ.get('SLURM_JOB_ID')
            if slurm_job_id is not None:
                wandb_run_id = f'mila-slurm-{slurm_job_id}'
            else:
                wandb_run_id = wandb.util.generate_id()
        wandb.init(project="monobeast", resume='allow', id=wandb_run_id, config=flags)

    def save_latest_model():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": learner_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "stats": stats,
                "flags": vars(flags),
            },
            checkpointpath,
        )

    def save_intermediate_model():
        save_model_path = checkpointpath.replace(
            "model.tar", "intermediate/model." + str(stats.get("step", 0)).zfill(9) + ".tar")
        logging.info("Saving model to %s", save_model_path)
        torch.save(
            {
                "model_state_dict": learner_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "stats": stats,
                "flags": vars(flags),
            },
            save_model_path,
        )

    returns_by_game = {}
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        last_savemodel_nsteps = 0
        while step < flags.total_steps:
            start_step = stats.get("step", 0)
            start_time = timer()
            time.sleep(5)
            end_step = stats.get("step", 0)

            if timer() - last_checkpoint_time > 10 * 60:
                # save every 10 min.
                save_latest_model()
                last_checkpoint_time = timer()

            if flags.save_model_every_nsteps > 0 and end_step >= last_savemodel_nsteps + flags.save_model_every_nsteps:
                # save model every save_model_every_nsteps steps
                save_intermediate_model()
                last_savemodel_nsteps = end_step

            sps = (end_step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = ("Return per episode: %.1f. " % stats["mean_episode_return"])
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                end_step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat({k:v for k,v in stats.items() if k not in ["returns_by_game","episode_returns"]}),
            )
            # Check stats for any finished episodes
        orion.client.report_results(data=[
                {'name': k, 'type': 'statistic', 'value': sum(v)/len(v)}
                for k,v in stats['returns_by_game'].items()
            ]+[{'name': 'dummy_objective', 'type': 'objective', 'value': 0}]
        )
    except KeyboardInterrupt:
        gradient_tracker.print_total()
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)  # send quit signal to actors
        for actor in actor_processes:
            actor.join(timeout=10)
        gradient_tracker.print_total()
    save_latest_model()
    plogger.close()


def test(flags):
    if flags.xpid is None:
        checkpointpath = os.path.expandvars(os.path.expanduser("%s/%s/%s" % (flags.savedir, "latest", "model.tar")))
    elif ".tar" in flags.xpid:
        checkpointpath = os.path.expandvars(os.path.expanduser(flags.xpid))
    else:
        checkpointpath = os.path.expandvars(os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar")))

    print('')
    print('+'+'-'*80)
    print('|  Testing')
    print(f'|  {checkpointpath}')
    print('+'+'-'*80)

    if len(flags.env.split(",")) != 1:
        raise Exception("Only one environment allowed for testing")

    if flags.test_results_path is not None:
        results = shelve.open(flags.test_results_shelve)
    else:
        results = {}

    if checkpointpath not in results:
        results[checkpointpath] = []

    print(f'{len(results[checkpointpath])} results already stored')
    if len(results[checkpointpath]) >= flags.num_episodes:
        return

    # load the original arguments for the loaded network
    flags_orig = file_writer.read_metadata(
        re.sub(r"model.*tar", "meta.json", checkpointpath).replace("/intermediate", ""))
    args_orig = flags_orig["args"]
    agent_type = args_orig.get("agent_type", "resnet")
    num_actions = args_orig.get("num_actions", 6)
    num_tasks = args_orig.get("num_tasks", 1)
    use_lstm = args_orig.get("use_lstm", False)
    use_popart = args_orig.get("use_popart", False)
    reward_clipping = args_orig.get("reward_clipping", "abs_one")
    frame_width = args_orig.get("frame_width", 84)
    frame_height = args_orig.get("frame_height", 84)
    aaa_input_format = args_orig.get("aaa_input_format", "gray_stack")

    # set the right agent class
    if agent_type.lower() in ["aaa", "attention_augmented", "attention_augmented_agent"]:
        Net = AttentionAugmentedAgent
        logging.info("Using the Attention-Augmented Agent architecture.")
        agent_type = "aaa"
    elif agent_type.lower() in ["rn", "res", "resnet", "res_net"]:
        Net = ResNet
        logging.info("Using the ResNet architecture (monobeast version).")
        agent_type = "resnet"
    else:
        Net = AtariNet
        logging.warning("No valid agent type specified. Using the default agent.")
        agent_type = "default"

    # check if the full action space should be used
    full_action_space = False
    if flags.num_actions == 18:
        full_action_space = True

    # create the environment
    env_config = { 'full_action_space': full_action_space }
    if flags.atari_mode is not None:
        env_config['mode'] = flags.atari_mode
    if flags.atari_difficulty is not None:
        env_config['difficulty'] = flags.atari_difficulty
    if flags.atari_action_repeat is not None:
        env_config['repeat_action_probability'] = flags.atari_action_repeat
    gym_env = create_env(flags.env,
                         frame_height=frame_height,
                         frame_width=frame_width,
                         gray_scale=(agent_type != "aaa" or aaa_input_format == "gray_stack"),
                         config=env_config)
    env = environment.Environment(gym_env)

    # create the model and load its parameters
    model = Net(observation_shape=gym_env.observation_space.shape,
                num_actions=num_actions,
                num_tasks=num_tasks,
                use_lstm=use_lstm,
                use_popart=use_popart,
                reward_clipping=reward_clipping,
                rgb_last=(agent_type == "aaa" and aaa_input_format == "rgb_last"))
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    if 'baseline.mu' not in checkpoint["model_state_dict"]:
        checkpoint["model_state_dict"]["baseline.mu"] = torch.zeros(1)
        checkpoint["model_state_dict"]["baseline.sigma"] = torch.ones(1)
    to_ignore = ['baseline.weight', 'baseline.bias', 'baseline.mu','baseline.sigma'] # The size of these tensors depend on the number of environments
    model.load_state_dict(OrderedDict((k,v) for k,v in checkpoint["model_state_dict"].items() if k not in to_ignore), strict=False)

    observation = env.initial()
    core_state = model.initial_state(1)
    returns = results[checkpointpath]
    while len(returns) < flags.num_episodes:
        if flags.mode == "test_render":
            time.sleep(0.05)
            env.gym_env.render()
        agent_outputs = model(observation, core_state, stochastic=flags.stochastic)
        policy_outputs, core_state = agent_outputs
        observation = env.step(policy_outputs["action"])
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
    env.close()
    logging.info("Average returns over %i steps: %.1f", flags.num_episodes, sum(returns) / len(returns))

    # Save results
    if isinstance(results, shelve.Shelf):
        results[checkpointpath] = returns
        results.close()


def create_env(env, frame_height=84, frame_width=84, gray_scale=True, config={'full_action_space':False}, task=0):
    pattern = re.compile(r"^(.*-v(\d))(-m(\d)d(\d))?$")
    m = pattern.match(env)
    if m is None:
        raise Exception("Could not parse environment name: %s" % env)
    env_name = m.group(1)
    if m.group(3) is not None:
        env_config = {
            'mode': int(m.group(4)),
            'difficulty': int(m.group(5)),
        }
    else:
        env_config = {}

    return atari_wrappers.wrap_pytorch_task(
        atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(env_name, config={
                'frameskip': 1,
                **config,
                **env_config,
            }),
            clip_rewards=False,
            frame_stack=True,
            frame_height=frame_height,
            frame_width=frame_width,
            gray_scale=gray_scale,
            scale=False,
        ),
        task=task
    )


def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
