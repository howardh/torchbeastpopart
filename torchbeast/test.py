import argparse
from collections import OrderedDict
import os
import logging
import re
import time

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import torch
from torch import multiprocessing as mp

from torchbeast.core import environment
from torchbeast.core import file_writer

from torchbeast.models.attention_augmented_agent import AttentionAugmentedAgent
from torchbeast.models.resnet_monobeast import ResNet
from torchbeast.models.atari_net_monobeast import AtariNet

from torchbeast.monobeast import create_env

# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--env", type=str, default="PongNoFrameskip-v4",
                    help="Gym environment.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--atari-mode", type=int, default=None,
                    help="Mode of the Atari environment.")
parser.add_argument("--atari-difficulty", type=int, default=None,
                    help="Difficulty of the Atari environment.")
parser.add_argument("--atari-action-repeat", type=float, default=None,
                    help="Probability of action repeat in the Atari environment.")

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

# Misc settings.
parser.add_argument("--write_profiler_trace", action="store_true",
                    help="Collect and write a profiler trace "
                    "for chrome://tracing/.")
parser.add_argument("--save_model_every_nsteps", default=0, type=int,
                    help="Save model every n steps")
parser.add_argument("--beta", default=0.0001, type=float,
                    help="PopArt parameter")
parser.add_argument("--wandb", action="store_true",
                    help="Track the experiment on W&B")

# Test settings.
parser.add_argument("--num_episodes", default=100, type=int,
                    help="Number of episodes for Testing.")
parser.add_argument("--test_results_path", default=None, type=str,
                    help="File/directory to save test results to")
parser.add_argument("--models", type=str, nargs="+",
                    help="Paths to models to test.")
parser.add_argument("--num_workers", type=int, default=0,
                    help="Number of parallel processes to use for testing.")

def test(flags, model_filename):
    print('')
    print('+'+'-'*80)
    print('|  Testing')
    print(f'|  {model_filename}')
    print('+'+'-'*80)

    if len(flags.env.split(",")) != 1:
        raise Exception("Only one environment allowed for testing")

    os.makedirs(flags.test_results_path, exist_ok=True)
    results_filename = os.path.join(
            flags.test_results_path,
            os.path.basename(model_filename).split('.')[1]+'.pt'
    )
    returns = None
    if flags.test_results_path is not None:
        try:
            with open(results_filename, 'rb') as f:
                returns = torch.load(f)
        except:
            print(f'Failed to load file {results_filename}')
    if returns is None:
        returns = []

    print(f'{len(returns)} results already stored')
    if len(returns) >= flags.num_episodes:
        return

    # load the original arguments for the loaded network
    flags_orig = file_writer.read_metadata(
        re.sub(r"model.*tar", "meta.json", model_filename).replace("/intermediate", ""))
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
    checkpoint = torch.load(model_filename, map_location="cpu")
    if 'baseline.mu' not in checkpoint["model_state_dict"]:
        checkpoint["model_state_dict"]["baseline.mu"] = torch.zeros(1)
        checkpoint["model_state_dict"]["baseline.sigma"] = torch.ones(1)
    to_ignore = ['baseline.weight', 'baseline.bias', 'baseline.mu','baseline.sigma'] # The size of these tensors depend on the number of environments
    model.load_state_dict(OrderedDict((k,v) for k,v in checkpoint["model_state_dict"].items() if k not in to_ignore), strict=False)

    observation = env.initial()
    core_state = model.initial_state(1)
    while len(returns) < flags.num_episodes:
        if flags.mode == "test_render":
            time.sleep(0.05)
            env.gym_env.render()
        agent_outputs = model(observation, core_state, stochastic=True)
        policy_outputs, core_state = agent_outputs
        observation = env.step(policy_outputs["action"])
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
        # Save results
        with open(results_filename, 'wb') as f:
            torch.save(returns, f)
    env.close()
    logging.info("Average returns over %i steps: %.1f", flags.num_episodes, sum(returns) / len(returns))

if __name__ == "__main__":
    flags = parser.parse_args()
    if flags.num_workers == 0:
        for model_filename in flags.models:
            test(flags, model_filename)
    else:
        from multiprocessing import Pool
        with Pool(flags.num_workers) as pool:
            pool.starmap(test, [(flags,fn) for fn in flags.models])
