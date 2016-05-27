from parameters import Parameters
from agent import Agent
from network import Linear, Baseline, TrainTarget
from environment import ArrayEnvironment, AtariEnvironment
from monitor import Monitor
from tqdm import tqdm
import numpy as np
import random
import signal

parameters = Parameters()
args = parameters.parse()

# Determinism
random.seed(args.random_seed)
np.random.seed(args.random_seed)

# Initialize
environment = AtariEnvironment(args)
network = TrainTarget(Baseline, args, environment)
agent = Agent(args, environment, network)
monitor = Monitor(args, environment, network, agent)

# Get initial state
state = environment.get_state()

# Harness Variables
force_eval = False
eval_pending = False


# Handle Ctrl + c
def commander(signal, frame):
    command = raw_input("\n\n {} Command [args | eval | verbose | quiet]: ".format(args.name))
    if command == "args":
        for key in sorted(vars(args)):
            print "{0:>40} : {1}".format(key, getattr(args, key))
        raw_input("Press enter to continue.")
    elif command == "verbose":
        args.verbose = True
    elif command == "quiet":
        args.verbose = False
    elif command == "eval":
        global eval_pending
        eval_pending = True

signal.signal(signal.SIGINT, commander)

# Main Loop
for tick in tqdm(range(args.total_ticks), ncols=50, mininterval=.001):

    # Determine if we should evaluate this episode or not
    is_evaluate = (tick + 1) % args.evaluate_frequency == 0 or force_eval

    action, q_values = agent.get_action(state, is_evaluate)
    state, reward, terminal = environment.act(action)
    agent.after_action(state, reward, action, terminal, is_evaluate)

    # Log stats and visualize
    if args.verbose or is_evaluate:
        monitor.monitor(state, reward, terminal, q_values, is_evaluate)

    # Reset if needed
    if terminal or environment.frames >= args.max_frames_per_episode:
        eval_pending = False if is_evaluate else eval_pending
        force_eval = eval_pending
        environment.reset()

# TODO
# Network Weight Visualizer
# Policy explorer for atari
# learning rate annealer is broken

# death ends episode
# check the phi passed into the q value

# if we prioritize -- do we prioritize based on delta, or clipped_delta?