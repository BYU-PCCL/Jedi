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
    command = raw_input("\n\n {} Command [args | eval | verbose | quiet]: ".format(args.name))
    if command == "args":
        for key in sorted(vars(args)):
            print("{0:>40} : {1}".format(key, getattr(args, key)))
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
# watch q values over course of game in graph - confirm that the variance is constant
# dynamic multi-gpu allocation
# negative reward on death may make q-value sampling better as it would increase the variance during frames that matter



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
# speed in convergence network and prepare for test
# ssert not nan in train assert not np.isnan(loss_value)
# dynamic layer labeling
# dynamic train  ops in network
# add checkpoints
# add gray to custom gym

# discussion topics with dr. wingate
# - high level: what are we testing? where are we hoping this takes us? how can we better prepare for a paper?

# findings:
# clipping the error, but NOT clipping the reward results in very strange behavior, after playing with the clipping
# values it seems that any activation of the clipping function causes problems (clipping at 49.0 vs 50) -- evidence
# that error clipping is a terrible way to handle issues

# training on density network -- because the relationship between state and q-value is 1:1, the optimal sigma is zero.
# it's possible that we could redefine bellman error probablistically and get some idea of variance
# (r + discount * next_qs[random_index]) which would result in a 1:many relationship more suited for a density model
# or perhaps the density model is more suited for environments that are more probablistic where choosing an action could
# be a "risk adjusted" step

# variance is relatively constant throughout the entire training experience, ranging from a stdev of 0.01 ro .1
# i suspect lots of this homogony is due to the fact that MOST actions do not matter at all.

# Sampling according to q-values is actually pretty successful if you sample ~ q^alpha | alpha = 2 -- this has the
# bonus of not needing an explicit epsilon -- however, it doesn't result in a "dramatic" improvement in score for
# breakout we were hoping for. I think this might be because regular training episodes struggle to get as high of a
# score as evaluation I believe it might be possible to find an alpha that closely approximates evaluation-level play,
# but still provides the exploration needed. Update: alpha = 2.5 seems to be quite good, but I think the learner
# has a hard time differentiating between "barely" hitting the ball and "safely" hitting the ball, and it's easy to
# sample a missing shot if you are barely hitting the ball 10 or 15 times. I think adding a "negative reward on death"
# flag may improve our results from a q-sampler because the variance in q-values will be increased, and sampling a
# missing shot will be less likely. The same is likely true for a "negative reward on terminal"


# ideas:
# Is it possible to
# train a model that attempts to minimize the q-value and maximize the variance? would doing so allow our
# variance-based explorer to function better?