from __future__ import print_function
from parameters import Parameters
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
environment = args.environment_class(args)
network = args.dqn_class(args.network_class, args, environment)
agent = args.agent_class(args, environment, network)
monitor = Monitor(args, environment, network, agent)

# Get initial state
state = environment.get_state()

# Harness Variables
eval_pending = False
is_evaluate = False


# Handle Ctrl + c
def commander(signal, frame):
    command = raw_input("\n\n {} Command [args | eval | verbose | quiet | reset-network]: ".format(args.name))
    if command == "args":
        for key in sorted(vars(args)):
            print("{0:>40} : {1}".format(key, getattr(args, key)))
        raw_input("Press enter to continue.")
    elif command == "verbose":
        args.verbose = True
    elif command == "quiet":
        args.verbose = False
    elif command == "reset-network":
        network.initialize()
    elif command == "eval":
        global eval_pending
        eval_pending = True

signal.signal(signal.SIGINT, commander)

# Main Loop
for tick in tqdm(range(args.total_ticks), ncols=40, mininterval=0.0001, smoothing=.001,
                 bar_format='{percentage:3.0f}% | {bar} | {n_fmt} [{elapsed}, {rate_fmt}]'):

    # Determine if we should evaluate this episode or not
    is_evaluate = is_evaluate or ((tick + 1) % args.evaluate_frequency) == 0
    is_evaluate = is_evaluate and tick > args.iterations_before_training

    action, q_values = agent.get_action(state, is_evaluate)
    state, reward, terminal = environment.act(action)
    agent.after_action(state, reward, action, terminal, is_evaluate)

    # Log stats and visualize
    if tick >= args.iterations_before_training:
        monitor.monitor(state, reward, terminal, q_values, action, is_evaluate)
    elif tick == 0:
        print("   ({} iterations before training)".format(args.iterations_before_training), end="")

    # Reset if needed
    if terminal:
        is_evaluate = eval_pending
        eval_pending = False
        environment.reset()

# TODO
# HIGH
# baselineduel not working great


# MEDIUM
# death ends episode
# negative reward on death
# add constrained priorities
# add with op.defaults({'activation_fn': 'relu', 'floatx': 'float16'}): to ops.py
# rename states_placeholder to state_placeholder and remove all plural inputs
# dynamically build list of networks for parameters.py
# add --network_name-parameter dynamic parameters for each network and agent
# add auto-complete to params


# LOW
# dynamic layer labeling
# add checkpoints
# add gray to custom gym