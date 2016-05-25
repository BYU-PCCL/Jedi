from parameters import Parameters
from agent import Agent
from network import Linear
from environment import ArrayEnvironment
from monitor import Monitor
from tqdm import tqdm

parameters = Parameters()
args = parameters.parse()

# Initialize
environment = ArrayEnvironment(args)
network = Linear(args, environment)
agent = Agent(args, environment, network)
monitor = Monitor(args, environment, network, agent)

#TODO: remove
agent.monitor = monitor

# Get initial state
state = environment.get_state()

# Main Loop
for _ in tqdm(range(args.max_ticks)):

    # Determine if we should evaluate this episode or not
    is_evaluate = environment.get_episodes() % args.evaluate_frequency == 0

    action, q_values = agent.get_action(state, is_evaluate)
    state, reward, terminal = environment.act(action)
    agent.after_action(state, reward, action, terminal, is_evaluate)

    # Log stats and visualize
    monitor.monitor(state, reward, terminal, q_values, is_evaluate)

    # Reset if needed
    if terminal:
        environment.reset()

# TODO
# Network Weight Visualizer
# Policy Explorer and fixup
# ALE Environment
# Hook up SQL
