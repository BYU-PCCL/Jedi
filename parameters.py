import argparse
import random
import subprocess
import network
import agent
import environment

from inspect import isclass

class Parameters():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Q-Learner')
        sql_args = self.parser.add_argument_group('PostgreSQL Monitor')
        sql_args.add_argument('--use_sql', action='store_const', const=True, default=False)
        sql_args.add_argument('--sql_db_file', default="jedi.sqlite", type=str)
        sql_args.add_argument('--save_episode_stats', action='store_const', const=True, default=False)
        sql_args.add_argument('--commit_buffer', default=1000, type=int)

        harness_args = self.parser.add_argument_group('Harness')
        harness_args.add_argument('--vis', action='store_const', const=True, default=False)
        harness_args.add_argument('--test', action='store_const', const=True, default=False)
        harness_args.add_argument('--name', default="learner")
        harness_args.add_argument('--verbose', action='store_const', const=True, default=False)
        harness_args.add_argument('--deterministic', action='store_const', const=True, default=False)
        harness_args.add_argument('--random_seed', default=42, type=int)
        harness_args.add_argument('--total_ticks', default=10000000, type=int)
        harness_args.add_argument('--evaluate_frequency', default=10000, type=int, help='in ticks')
        harness_args.add_argument('--console_frequency', default=1000, type=int, help='in ticks')
        harness_args.add_argument('--max_frames_per_episode', default=10000, type=int)

        environment_args = self.parser.add_argument_group('Environment')
        environment_args.add_argument('--actions_per_tick', default=1, type=int)
        environment_args.add_argument('--rom', default="Breakout")
        environment_args.add_argument('--environment_type', default="Atari", choices=self.module_to_dict(environment).keys())
        environment_args.add_argument('--openaigym_environment', default="Pendulum-v0")
        environment_args.add_argument('--max_initial_noop', default=8, type=int)
        environment_args.add_argument('--resize_width', default=84, type=int)
        environment_args.add_argument('--resize_height', default=84, type=int)
        environment_args.add_argument('--buffer_size', default=2, type=int, help='number of frames to max')
        environment_args.add_argument('--negative_reward_on_death', action='store_const', const=True, default=False)

        agent_args = self.parser.add_argument_group('Agent')
        agent_args.add_argument('--agent_type', default='Agent', type=str, choices=self.module_to_dict(agent).keys())
        agent_args.add_argument('--phi_frames', default=4, type=int)
        agent_args.add_argument('--replay_memory_size', default=1000000, type=int)
        agent_args.add_argument('--batch_size', default=32, type=int)
        agent_args.add_argument('--iterations_before_training', default=50000, type=int, help='in frames')
        agent_args.add_argument('--exploration_epsilon_end', default=.1, type=float)
        agent_args.add_argument('--exploration_epsilon_decay', default=1000000, type=int, help='in calls to train')
        agent_args.add_argument('--exploration_epsilon_evaluation', default=.05, type=float, help='epsilon for evaluation')
        agent_args.add_argument('--train_frequency', default=3, type=int, help='in ticks')
        agent_args.add_argument('--threads', default=12, type=int)
        agent_args.add_argument('--lookahead', default=10, type=int, help='in frames')
        agent_args.add_argument('--use_prioritization', action='store_const', const=True, default=False)
        agent_args.add_argument('--priority_temperature', default=2.0, type=float, help='n where tderror^n')

        network_args = self.parser.add_argument_group('Network')
        network_args.add_argument('--network_type', default='Baseline', type=str, choices=self.module_to_dict(network, [network.Network]).keys())
        network_args.add_argument('--discount', default=.99, type=float)
        network_args.add_argument('--learning_rate_start', default=0.00025, type=float)
        network_args.add_argument('--learning_rate_end', default=0.00025, type=float)
        network_args.add_argument('--learning_rate_decay', default=.96, type=float)
        network_args.add_argument('--learning_rate_decay_step', default=100000, type=int)
        network_args.add_argument('--initializer', default='truncated-normal', type=str, choices=['xavier', 'normal', 'truncated-normal', 'uniform'])
        network_args.add_argument('--rms_eps', default=0.01, type=float)
        network_args.add_argument('--rms_decay', default=.90, type=float)
        network_args.add_argument('--rms_momentum', default=0.99, type=float)
        network_args.add_argument('--gpu_fraction', default=.90, type=float)
        network_args.add_argument('--target_network_alpha', default=0.0, type=float)
        network_args.add_argument('--copy_frequency', default=2500, type=int, help='in calls to train')
        network_args.add_argument('--clip_reward', default=1, type=int)
        network_args.add_argument('--clip_tderror', default=1, type=int)
        network_args.add_argument('--tf_summary_path', default="/tmp/network", type=str)
        network_args.add_argument('--tf_checkpoint_path', default="/tmp/checkpoints", type=str)

    def parse(self):
        args = self.parser.parse_args()
        ignored_args = ['verbose', 'sql_host', 'sql_db', 'sql_port', 'sql_user', 'sql_password',
                        'vis', 'name', 'total_ticks', 'evaluate_frequency', 'use_sql']
        changed_args = ['rom'] + [key + "=" + str(getattr(args, key)) for key in vars(args)
                                  if key not in ignored_args and getattr(args, key) != self.parser.get_default(key)]
        changed_args = "-".join(changed_args) if len(changed_args) > 0 else "defaults"

        args.job_id = str(random.randint(10000000, 99999999))
        args.name = args.name + '-' + changed_args + '-' + args.job_id

        if args.test:
            args.environment_type = 'Array'
            args.network_type = 'Linear'
            args.agent_type = 'Test'
            args.copy_frequency = 10
            args.iterations_before_training = 1000

        args.environment_class = self.parse_environment(args.environment_type)
        args.network_class = self.parse_network_type(args.network_type)
        args.agent_class = self.parse_agent_type(args.agent_type)

        try:
            args.commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
            if subprocess.check_output(['git', 'status']).find("Changes not staged for commit") >= 0:
                args.commit_hash += "-with-uncommited-changes"
        except:
            args.commit_hash = 'no-git'
            pass

        if args.vis and not self.can_vis():
            args.vis = False

        if args.deterministic:
            args.random_seed = 42
        else:
            args.random_seed = random.randint(1, 10000)

        return args

    def can_vis(self):
        try:
            import cv2
            return True
        except ImportError:
            return False

    def module_to_dict(self, module, exclude=[]):
        return dict([(x, getattr(module, x)) for x in dir(module)
                     if isclass(getattr(module, x))
                     and x not in exclude
                     and getattr(module, x) not in exclude])

    def parse_environment(self, env_string):
        return self.module_to_dict(environment)[env_string]

    def parse_agent_type(self, agent_string):
        return self.module_to_dict(agent)[agent_string]

    def parse_network_type(self, network_string):
        return self.module_to_dict(network, [network.Network])[network_string]

#environment_args.add_argument('--death_ends_episode', action='store_const', const=True, default=False, help='load network and agent')

#harness_args.add_argument('--load_checkpoint', action='store_const', const=True, default=False, help='load network and agent')

#agent_args.add_argument('--priority_epsilon', default=.05, type=float, help='the epsilon associated with h2 priority')
#agent_args.add_argument('--prioritization_type', default="uniform", help='uniform, h0, h1, or h2')
#agent_args.add_argument('--clear_priority_frequency', default=0, type=int, help='how often to reset priorities')

#network_args.add_argument('--constrained_lambda', default=1.0, type=float, help='the lambda used for constrained networks (ignored if network_type is not constrained)')
