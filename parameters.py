import argparse
import random
import subprocess
import network
import agent

class Parameters():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Q-Learner')
        sql_args = self.parser.add_argument_group('PostgreSQL Monitor')
        sql_args.add_argument('--bypass_sql', action='store_const', const=True, default=False)
        sql_args.add_argument('--sql_host', default="192.168.23.44")
        sql_args.add_argument('--sql_db', default="vinci")
        sql_args.add_argument('--sql_port', default=5432, type=int)
        sql_args.add_argument('--sql_user', default="postgres")
        sql_args.add_argument('--sql_password', default="beware the pccl", type=str)

        harness_args = self.parser.add_argument_group('Harness')
        harness_args.add_argument('--vis', action='store_const', const=True, default=False)
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
        environment_args.add_argument('--rom', default="breakout")
        environment_args.add_argument('--max_initial_noop', default=8, type=int)
        environment_args.add_argument('--resize_width', default=84, type=int)
        environment_args.add_argument('--resize_height', default=84, type=int)
        environment_args.add_argument('--buffer_size', default=2, type=int, help='number of frames to max')

        agent_args = self.parser.add_argument_group('Agent')
        agent_args.add_argument('--agent_type', default='agent', type=str, choices=['agent', 'qexplorer', 'mdnexplorer'])
        agent_args.add_argument('--phi_frames', default=4, type=int)
        agent_args.add_argument('--replay_memory_size', default=1000000, type=int)
        agent_args.add_argument('--batch_size', default=32, type=int)
        agent_args.add_argument('--iterations_before_training', default=50000, type=int, help='in frames')
        agent_args.add_argument('--exploration_epsilon_end', default=.1, type=float)
        agent_args.add_argument('--exploration_epsilon_decay', default=1000000, type=int, help='in calls to train')
        agent_args.add_argument('--exploration_epsilon_evaluation', default=.05, type=int, help='epsilon for evaluation')
        agent_args.add_argument('--train_frequency', default=3, type=int, help='in ticks')
        agent_args.add_argument('--threads', default=4, type=int)
        agent_args.add_argument('--lookahead', default=10, type=int, help='in frames')

        network_args = self.parser.add_argument_group('Network')
        network_args.add_argument('--network_type', default='baseline', type=str, choices=['baseline', 'linear', 'density', 'causal', 'constrained'])
        network_args.add_argument('--discount', default=.99, type=float)
        network_args.add_argument('--learning_rate_start', default=0.00025, type=float)
        network_args.add_argument('--learning_rate_end', default=0.00025, type=float)
        network_args.add_argument('--learning_rate_decay', default=.96, type=float)
        network_args.add_argument('--learning_rate_decay_step', default=50000, type=float)
        network_args.add_argument('--initializer', default='truncated-normal', type=str, choices=['xavier', 'normal', 'truncated-normal', 'uniform'])
        network_args.add_argument('--rms_eps', default=0.01, type=float)
        network_args.add_argument('--rms_decay', default=.90, type=float)
        network_args.add_argument('--rms_momentum', default=0.99, type=float)
        network_args.add_argument('--gpu_fraction', default=.90, type=float)
        network_args.add_argument('--target_network_alpha', default=1.0, type=float)
        network_args.add_argument('--copy_frequency', default=2500, type=int, help='in calls to train')
        network_args.add_argument('--clip_reward', default=True, type=bool)
        network_args.add_argument('--clip_tderror', default=True, type=bool)
        network_args.add_argument('--tf_summary_path', default="/tmp/network", type=str)
        network_args.add_argument('--tf_checkpoint_path', default="/tmp/checkpoints", type=str)

    def parse(self):
        args = self.parser.parse_args()
        ignored_args = ['verbose', 'sql_host', 'sql_db', 'sql_port', 'sql_user', 'sql_password',
                        'vis', 'name', 'total_ticks', 'evaluate_frequency', 'bypass_sql']
        changed_args = [key + "=" + str(getattr(args, key)) for key in vars(args) if key not in ignored_args and getattr(args, key) != self.parser.get_default(key)]
        changed_args = "-".join(changed_args) if len(changed_args) > 0 else "defaults"
        args.name = args.name + '-' + changed_args + '-' + str(random.randint(10000000, 99999999))

        args.network_class = self.parse_network_type(args.network_type)
        args.agent_class = self.parse_agent_type(args.agent_type)

        try:
            args.commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
            if subprocess.check_output(['git', 'status']).find("Changes not staged for commit") >= 0:
                args.commit_hash += "-with-uncommited-changes"
        except:
            args.commit_hash = 'no-git'
            pass

        if args.deterministic:
            args.random_seed = 42
        else:
            args.random_seed = random.randint(1, 10000)

        return args

    def parse_agent_type(self, agent_string):
        return {'agent': agent.Agent,
                'qexplorer': agent.QExplorer,
                'densityexplorer': agent.DensityExplorer}[agent_string]

    def parse_network_type(self, network_string):
        return {'baseline': network.Baseline,
                'linear': network.Linear,
                'density': network.Density,
                'causal': network.Causal,
                'constrained': network.Constrained}[network_string]


#environment_args.add_argument('--death_ends_episode', action='store_const', const=True, default=False, help='load network and agent')
#environment_args.add_argument('--negative_reward_on_death', action='store_const', const=True, default=False, help='load network and agent')

#harness_args.add_argument('--load_checkpoint', action='store_const', const=True, default=False, help='load network and agent')

#agent_args.add_argument('--priority_epsilon', default=.05, type=float, help='the epsilon associated with h2 priority')
#agent_args.add_argument('--prioritization_type', default="uniform", help='uniform, h0, h1, or h2')
#agent_args.add_argument('--clear_priority_frequency', default=0, type=int, help='how often to reset priorities')

#network_args.add_argument('--lookahead', default=5, type=int, help='the number of frames to look ahead when constraining')
#network_args.add_argument('--constrained_lambda', default=1.0, type=float, help='the lambda used for constrained networks (ignored if network_type is not constrained)')
