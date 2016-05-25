import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import random
import time
import cv2

class Monitor():
    def __init__(self, args, environment, network, agent):
        self.args = args
        self.environment = environment
        self.network = network
        self.iterations = 0
        self.max_q = -99999
        self.max_score = -99999
        self.score = 0

        if args.vis:
            self.initialize_visualization()

    def initialize_visualization(self):

        self.args.vis = True

        # State Visualizer
        cv2.startWindowThread()
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

        # QT and CV2 seem to dislike working together
        # Q-Value Visualizer
        self.test_data = self.environment.generate_test()
        if self.test_data:
            self.history = np.zeros((len(self.test_data[0]), 500, self.environment.get_num_actions()))

            self.app = QtGui.QApplication([])
            self.q_win = pg.GraphicsWindow(title="Q Monitor")
            self.q_win.resize(1000, 1000)
            pg.setConfigOptions(antialias=True)

            if self.test_data is not None:
                self.q_plots = []
                for i, _ in enumerate(self.test_data[0]):
                    plot = self.q_win.addPlot()
                    plot.hideAxis('bottom')
                    plot.hideAxis('left')
                    #plot.labelAxis('')
                    self.q_win.nextRow() if (i + 1) % 3 == 0 else None
                    self.q_plots.append([plot.plot(pen=(a + 1) * 5) for a in range(self.environment.get_num_actions())])

    def destroy_visualization(self):
        cv2.destroyAllWindows()
        self.args.vis = False

    def visualize_qs(self):
        _, batch_qs = self.network.q(self.test_data[0])

        self.history[:, 0:-1, :] = self.history[:, 1:, :]
        self.history[:, -1, :] = batch_qs

        for i, qs in enumerate(batch_qs):
            for j, q in enumerate(qs):
                self.q_plots[i][j].setData(self.history[i, :, j])

        pg.QtGui.QApplication.processEvents()

    def monitor(self, state, reward, terminal, q_values, is_evaluate):
        self.iterations += 1

        if self.args.vis:
            cv2.imshow("preview", state)

        if is_evaluate:
            self.max_q = max(self.max_q, np.max(q_values))
            self.max_score = max(self.max_score, self.environment.get_score())

            if terminal:
                print "\nEvaluation: Iterations {}, Episodes {}, Max-Q {}, Max Score {}".format(self.iterations, self.environment.get_episodes(), self.max_q, self.max_score)
                self.max_q = -99999
                self.max_score = -99999

