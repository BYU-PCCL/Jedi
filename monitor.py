from __future__ import print_function
import numpy as np
from colorama import Fore, Style
import sqlite3
import time

class Stats:
    def __init__(self):
        self.list_stats = {}

    def update(self, stat, value):
        if stat not in self.list_stats:
            self.list_stats[stat] = []

        self.list_stats[stat].append(value)

    def __getitem__(self, item):
        if item in self.list_stats:
            return self.list_stats[item]

        partition = item.partition("_")
        key = partition[2]
        operator = partition[0]

        if key in self.list_stats:
            return {"max": np.max(self.list_stats[key]),
                    "min": np.min(self.list_stats[key]),
                    "mean": np.mean(self.list_stats[key])}[operator]

        return 0.0


class Monitor:
    def __init__(self, args, environment, network, agent):
        self.args = args
        self.environment = environment
        self.network = network
        self.agent = agent
        self.iterations = 0
        self.score = 0
        self.eval_stats = None
        self.console_stats = Stats()
        self.episode_stats = Stats()
        self.commit_ready = False

        self.start = None
        self.inserts = []

        self.cv2 = None

        if not self.args.bypass_sql:
            self.conn = sqlite3.connect(args.sql_db_file, timeout=1.0)
            self.cursor = self.conn.cursor()

        self.save_config(args)

        print("\n\nInitialized")
        print("{0:>20} : {1:,} ".format("Network Parameters", self.network.total_parameters()))
        print("{0:>20} : {1} ".format("Name", args.name))
        print("\n")

    def save_stat(self, stat_name, value, is_evaluation):
        if not self.args.bypass_sql:
            self.inserts.append((self.args.name, self.environment.get_episodes(), stat_name, float(value), is_evaluation))

    def commit(self):
        if not self.args.bypass_sql:
            if len(self.inserts) > 0:
                for _ in range(100):
                    try:
                        self.cursor.executemany("INSERT INTO stats (agent_name, episode, stat_name, value, is_evaluation) VALUES (?, ?, ?, ?, ?)", self.inserts)
                        self.conn.commit()
                        self.inserts = []
                        return
                    except sqlite3.OperationalError:
                        print("Error storing stats")
                        time.sleep(np.random.random() * 5.0)

                print("Fatal error - couldn't store stat data")
                quit()

    def save_config(self, settings):
        if not self.args.bypass_sql:
            for _ in range(100):
                try:
                    self.cursor.execute("INSERT INTO configs (agent_name, configs) VALUES (?, ?)", (self.args.name, str(settings)))
                    self.conn.commit()
                    return
                except sqlite3.OperationalError:
                    print("Trying to store config")
                    time.sleep(np.random.random() * 5.0)

            print("Fatal error - couldn't store config data")
            quit()

                      
    def save_stats(self, stats, evaluation=False):
        stats_to_save = ['max_q', 'max_score', 'min_lr', 'min_epsilon', 'max_q', 'min_q']
        for key in stats_to_save:
            self.save_stat(key, stats[key], evaluation) if stats[key] != None else None

    def print_stats(self, stats, evaluation=False):

        actions = np.zeros(1, dtype=np.int)
        if evaluation:
            actions = np.zeros(self.environment.get_num_actions())
            stat_actions = stats['action'] if type(stats['action']) is list else []

            for action in stat_actions:
                actions[action] += 1.0

            actions = np.array((actions / np.sum(actions)) * 100, dtype=np.uint32)

        log = " |  episodes: {}  " \
              "max q: {:<8.4f} " \
              "q std: {:<8.4f} " \
              "score: [{:>2g},{:<2g}]  " \
              "lr: {:<11.7f} " \
              "eps: {:<9.5} " \
              "loss: {:<10.6f}  " \
              "frames: {}  " \
              "actions: {}".format(self.environment.get_episodes(),
                                   float(stats['max_q']),
                                   float(stats['max_std']),
                                   stats['min_score'],
                                   stats['max_score'],
                                   float(self.network.learning_rate),
                                   float(self.agent.epsilon),
                                   float(self.network.batch_loss),
                                   self.environment.frames,
                                   np.array_str(actions, precision=2))

        self.console_stats = Stats()

        if not self.args.verbose:
            iterations_per_second = round(self.iterations / (time.time() - self.start), ndigits=2)
            self.start = None
            self.iterations = 0
            log = "{:<4.2f}it/s ".format(iterations_per_second) + log

        if evaluation:
            print(Fore.GREEN, log, Style.RESET_ALL)

        elif self.args.verbose:
            # Space accounts for the Fore.GREEN space that gets printed by evaluation
            print(" " + log, end="\r")

    def start_visualizer(self):
        if self.args.vis:
            import cv2
            self.cv2 = cv2
            cv2.startWindowThread()
            cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    def end_visualizer(self):
        self.cv2.destroyAllWindows()

    def monitor(self, state, reward, terminal, q_values, action, is_evaluate):
        self.iterations += 1

        if self.start is None:
            self.start = time.time()

        if self.cv2 is not None and self.args.vis:
            pixels = self.environment.render()
            if pixels is not None:
                self.cv2.imshow("preview", pixels)

        for stats in [self.console_stats, self.episode_stats, self.eval_stats]:
            if stats is not None:
                if q_values is not None:
                    stats.update('q', np.max(q_values))
                    stats.update('std', np.sqrt(np.var(q_values)))
                stats.update('action', action)

                if terminal:
                    stats.update('score', self.environment.get_score())

        if is_evaluate:
            if self.eval_stats is None:
                self.eval_stats = Stats()

            if terminal:
                self.print_stats(self.eval_stats, True)
                self.eval_stats = None

        elif (self.iterations + 1) % self.args.console_frequency == 0:
            self.print_stats(self.console_stats)
            self.commit()

        if terminal:
            self.save_stats(self.episode_stats, is_evaluate)
            self.episode_stats = Stats()



