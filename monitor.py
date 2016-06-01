from __future__ import print_function
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import cv2
import psycopg2
from colorama import Fore, Style

class Stats:
    def __init__(self):
        self.stats = {}
        self.list_stats = {}

    def update(self, stat, value):
        if stat not in self.list_stats:
            self.list_stats[stat] = []

        self.list_stats[stat].append(value)

    def __getitem__(self, item):
        if item in self.stats:
            return self.stats[item]

        partition = item.partition("_")
        key = partition[2]
        operator = partition[0]

        if key in self.list_stats:
            return {"max": np.max(self.list_stats[key]),
                    "min": np.min(self.list_stats[key])}[operator]

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

        # Always try to connect - this avoids issues where you forget to
        # pip install psycopg2, or werid DNS issues, etc.
        self.conn = psycopg2.connect(host=args.sql_host,
                                     port=args.sql_port,
                                     database=args.sql_db,
                                     user=args.sql_user,
                                     password=args.sql_password)
        self.cur = self.conn.cursor()
        self.save_config(args)

        self.policy_test = environment.generate_test()
        if self.policy_test:
            ideal_states, ideal_actions, ideal_rewards, ideal_next_states, ideal_terminals = self.policy_test
            self.ideal_states = [[state] for i, state in enumerate(ideal_states) if ideal_actions[i] == 0]

        if args.vis:
            self.initialize_visualization()

        print("\n\nInitialized")
        print("{0:>20} : {1:,} ".format("Network Parameters", len(self.network)))
        print("{0:>20} : {1} ".format("Name", args.name))
        print("\n")

    def initialize_visualization(self):
        self.args.vis = True

        # State Visualizer
        cv2.startWindowThread()
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

        # QT and CV2 seem to dislike working together
        # Q-Value Visualizer
        # if self.policy_test:
        #     self.history = np.zeros((len(self.ideal_states), 500, self.environment.get_num_actions()))
        #
        #     self.app = QtGui.QApplication([])
        #     self.q_win = pg.GraphicsWindow(title="Q Monitor")
        #     self.q_win.resize(1000, 1000)
        #     pg.setConfigOptions(antialias=True)
        #
        #     if self.policy_test is not None:
        #         self.q_plots = []
        #         for i, _ in enumerate(self.ideal_states):
        #             plot = self.q_win.addPlot()
        #             plot.hideAxis('bottom')
        #             plot.hideAxis('left')
        #             #plot.labelAxis('')
        #             self.q_win.nextRow() if (i + 1) % 3 == 0 else None
        #             self.q_plots.append([plot.plot(pen=(a + 1) * 5) for a in range(self.environment.get_num_actions())])

    def save_stat(self, stat_name, value):
        if not self.args.bypass_sql:
            self.cur.execute("INSERT INTO stats (agent_name, episode, stat_name, value) VALUES (%s, %s, %s, %s)",
                             (self.args.name, self.environment.get_episodes(), stat_name, float(value)))
            self.commit_ready = True

    def commit(self):
        if self.commit_ready:
            self.conn.commit()

    def save_config(self, settings):
        if not self.args.bypass_sql:
            self.cur.execute("INSERT INTO configs (agent_name, configs) VALUES (%s, %s)",
                             (self.args.name, str(settings)))
            self.commit_ready = True
            self.commit()

    def destroy_visualization(self):
        cv2.destroyAllWindows()
        self.args.vis = False

    def visualize_qs(self):
        policy, batch_qs = self.network.q(self.ideal_states)

        self.history[:, 0:-1, :] = self.history[:, 1:, :]
        self.history[:, -1, :] = batch_qs

        for i, qs in enumerate(batch_qs):
            for j, q in enumerate(qs):
                self.q_plots[i][j].setData(self.history[i, :, j])

        pg.QtGui.QApplication.processEvents()

    def save_stats(self, stats, evaluation=False):
        stats_to_save = ['max_q', 'max_score', 'min_lr', 'min_epsilon', 'max_q', 'min_q']
        for key in stats_to_save:
            self.save_stat(key, stats[key]) if stats[key] != None else None

    def print_stats(self, stats, evaluation=False):

        policy = "-"
        qs = [0.0]
        if self.policy_test:
            policy, qs = self.network.q(self.ideal_states)
            policy = "".join(str(p) if i != self.environment.goal else '-' for i, p in enumerate(policy))

        log = " |  episodes: {:<8} " \
              "max q: {:<8.4f} " \
              "score: [{:>4},{:<4}]  " \
              "lr: {:<11.7f} " \
              "eps: {:<9.5} " \
              "loss: {:<10.6f} " \
              "policy q: {:<6.4}" \
              "policy: {} ".format(self.environment.get_episodes(),
                                  float(stats['max_q']) if stats['max_q'] is not None else 0.0,
                                  stats['min_score'],
                                  stats['max_score'],
                                  float(self.network.lr),
                                  float(self.agent.epsilon),
                                  float(self.network.batch_loss),
                                  np.max(qs),
                                  policy)

        self.console_stats = Stats()

        if evaluation:
            print(Fore.GREEN, log, Style.RESET_ALL)
        else:
            print(" " + log, end="")


    def monitor(self, state, reward, terminal, q_values, is_evaluate):
        self.iterations += 1

        if self.args.vis:
            cv2.imshow("preview", state)

        # if self.iterations % 50 == 0:
        #     self.visualize_qs()

        for stats in [self.console_stats, self.episode_stats, self.eval_stats]:
            if stats is not None:
                stats.update('q', np.max(q_values))

                if terminal:
                    stats.update('score', self.environment.get_score())

        if is_evaluate:
            if self.eval_stats is None:
                self.eval_stats = Stats()

            if terminal:
                self.print_stats(self.eval_stats, True)
                self.eval_stats = None

        elif self.iterations % self.args.console_frequency == 1:
            self.print_stats(self.console_stats)
            self.commit()

        if terminal:
            self.save_stats(self.episode_stats, is_evaluate)
            self.episode_stats = Stats()



