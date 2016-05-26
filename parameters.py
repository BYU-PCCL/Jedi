import argparse
import random

class Parameters():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Q-Learner')
        sql_args = self.parser.add_argument_group('SQL Monitor')
        sql_args.add_argument('--bypass_sql', action='store_const', const=True, default=False, help='do not send sql')
        sql_args.add_argument('--sql_host', default="192.168.23.44", help='hostname of postgres db')
        sql_args.add_argument('--sql_db', default="vinci")
        sql_args.add_argument('--sql_port', default=5432, type=int)
        sql_args.add_argument('--sql_user', default="postgres")
        sql_args.add_argument('--sql_password', default="beware the pccl", type=str)

        harness_args = self.parser.add_argument_group('Harness')
        harness_args.add_argument('--vis', action='store_const', const=True, default=False, help='show visualization')
        harness_args.add_argument('--name', default="learner", help='Name associated with experiment')
        harness_args.add_argument('--verbose', action='store_const', const=True, default=False)
        harness_args.add_argument('--deterministic', action='store_const', const=True, default=False, help='set random seeds when appropriate')
        harness_args.add_argument('--random_seed', default=42, type=int, help='apply the weights training to target weights every n callstrains')
        harness_args.add_argument('--total_ticks', default=100000, type=int, help='max iterations in main loop')
        harness_args.add_argument('--evaluate_frequency', default=1000, type=int, help='evaluate every n episodes')
        harness_args.add_argument('--train_frequency', default=4, type=int, help='train every n frames')
        harness_args.add_argument('--console_frequency', default=1000, type=int, help='print every n iterations if verbose')
        harness_args.add_argument('--max_frames_per_episode', default=100, type=int)
        harness_args.add_argument('--threads', default=1, type=int)


        environment_args = self.parser.add_argument_group('ALE Environment')
        environment_args.add_argument('--actions_per_tick', default=4, type=int, help='repeat actions for n frames')
        environment_args.add_argument('--rom', default="breakout", help='filename in /worlds/ALE/roms to run')
        environment_args.add_argument('--max_initial_noop', default=30, type=int, help='randomize initial conditions with some noops')
        environment_args.add_argument('--resize_width', default=84, type=int, help='the width of the input to the network')
        environment_args.add_argument('--resize_height', default=84, type=int, help='the height of the input to the network')
        environment_args.add_argument('--buffer_size', default=2, type=int, help='number of frames to max')

        agent_args = self.parser.add_argument_group('Agent')
        agent_args.add_argument('--phi_frames', default=4, type=int, help='the number of frames in phi')
        agent_args.add_argument('--replay_memory_size', default=1000000, type=int, help='maximum number of frames')
        agent_args.add_argument('--batch_size', default=32, type=int, help='the batch size passed to the network')
        agent_args.add_argument('--iterations_before_training', default=100, type=int, help='the number of frames to collect before begining training')
        agent_args.add_argument('--exploration_epsilon_end', default=.1, type=float, help='the minimum exploration epsilon')
        agent_args.add_argument('--exploration_epsilon_decay', default=1000000, type=int, help='over how many calls to train should epsilon decay')
        agent_args.add_argument('--exploration_epsilon_evaluation', default=.05, type=int, help='over how many calls to train should epsilon decay')

        network_args = self.parser.add_argument_group('Network')
        network_args.add_argument('--discount', default=.50, type=float, help='gamma, the discount rate')
        network_args.add_argument('--learning_rate_start', default=0.00025, type=float, help='the learning rate')
        network_args.add_argument('--learning_rate_end', default=0.00025, type=float, help='the learning rate')
        network_args.add_argument('--learning_rate_decay', default=100000, type=float, help='the learning rate')
        network_args.add_argument('--initializer', default='xavier', type=str, help='xavier | normal | truncated-normal | uniform')
        network_args.add_argument('--rms_eps', default=0.01, type=float, help='the epsilon for rmsprop')
        network_args.add_argument('--rms_decay', default=.90, type=float, help='the decay for rmsprop')
        network_args.add_argument('--rms_momentum', default=0.99, type=float, help='the momentum for rmsprop')
        network_args.add_argument('--gpu_fraction', default=.90, type=float, help='how much gpu to use')
        network_args.add_argument('--copy_frequency', default=1000, type=int, help='copy to target every n trains')

    def parse(self):
        args = self.parser.parse_args()

        changed_args = [key + "=" + str(getattr(args, key)) for key in vars(args) if getattr(args, key) != self.parser.get_default(key)]
        changed_args = "-".join(changed_args) if len(changed_args) > 0 else "defaults"
        args.name = args.name + '-' + changed_args + '-' + str(random.randint(10000000, 99999999))

        if args.deterministic:
            args.random_seed = 42
        else:
            args.random_seed = random.randint(1, 10000)

        return args

# # environment_args.add_argument('--death_ends_episode', action='store_const', const=True, default=False, help='load network and agent')
# environment_args.add_argument('--negative_reward_on_death', action='store_const', const=True, default=False, help='load network and agent')

#harness_args.add_argument('--evaluate_episode_frequency', default=100, type=int, help='evaluate every n episodes')
#harness_args.add_argument('--save_tick_frequency', default=-1, type=int, help='save every n frames')
#harness_args.add_argument('--save_episode_frequency', default=-1, type=int, help='save every n episodes')
#harness_args.add_argument('--console_tick_frequency', default=100, type=int, help='print to screen every n frames')
#harness_args.add_argument('--max_episodes', default=-1, type=int, help='train for n episodes (-1 is infinite)')
#harness_args.add_argument('--max_ticks', default=100000, type=int, help='maximum ticks allowed per episode (-1 is infinite)')
#harness_args.add_argument('--load_checkpoint', action='store_const', const=True, default=False, help='load network and agent')
#harness_args.add_argument('--evaluation_repetition', default=5, type=int, help='number of evaluations to average')


#agent_args.add_argument('--training_iterations', default=1, type=float, help='number of times to train per frame')
#agent_args.add_argument('--training_frequency', default=1, type=float, help='train every n frames')
#agent_args.add_argument('--use_after_state', action='store_const', const=True, default=False, help='use the after state in memory experiences')
#agent_args.add_argument('--exploration_epsilon_testing', default=.05, type=float, help='the exploration epsilon used during evaluations')
#agent_args.add_argument('--priority_epsilon', default=.05, type=float, help='the epsilon associated with h2 priority')
#agent_args.add_argument('--prioritization_type', default="uniform", help='uniform, h0, h1, or h2')
#agent_args.add_argument('--clear_priority_frequency', default=0, type=int, help='how often to reset priorities')


#network_args.add_argument('--network_type', default="baseline", help='baseline, baseline_two, constrained, deep or shallow')
#network_args.add_argument('--clip_delta', action='store_const', const=True, default=False, help='use the fancy clipping method')
#network_args.add_argument('--clip_delta_value', action='store_const', const=True, default=False, help='use the not fancy clipping method')
#network_args.add_argument('--copy_frequency', default=10000, type=int, help='apply the weights training to target weights every n callstrains')
#network_args.add_argument('--lookahead', default=5, type=int, help='the number of frames to look ahead when constraining')
#network_args.add_argument('--constrained_lambda', default=1.0, type=float, help='the lambda used for constrained networks (ignored if network_type is not constrained)')
#network_args.add_argument('--constrained_use_priorities', action='store_const', const=True, default=False, help='use priorities if network_type is constrained')
#network_args.add_argument('--weight_initialization_stdev', default=0.01, type=float, help='the random normal stdev for weight initialization')
#network_args.add_argument('--use_nature_values', action='store_const', const=True, default=False, help='set all parameters to nature paper values')
#network_args.add_argument('--m3', action='store_const', const=True, default=False, help='devsisters m3 model')
#network_args.add_argument('--m4', action='store_const', const=True, default=False, help='devsisters m4 model')
#network_args.add_argument('--learning_rate_end', default=.0002, type=float, help='the learning rate')
#network_args.add_argument('--learning_rate_decay', default=100000, type=float, help='the learning rate')
#network_args.add_argument('--accumulator', default="mean", type=str, help='mean or sum')

