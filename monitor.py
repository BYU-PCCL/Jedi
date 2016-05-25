import numpy as np
import pyqtgraph as pg

class Monitor():
    def __init__(self, args, environment, network, agent):
        self.args = args
        self.environment = environment
        self.network = network
        self.iterations = 0
        self.max_q = -99999
        self.max_score = -99999
        self.score = 0

        self.test_data = self.environment.generate_test()

        if args.vis:
            self.win = pg.GraphicsWindow(title="Monitor")
            pg.setConfigOptions(antialias=True)
            graph = self.win.addPlot(title="Q-Values for Test Data")

            self.q_graphs = []
            for _ in self.test_data[0]:
                self.q_graphs.append(graph.plot())

    def visualize(self):
        if self.args.vis:
            if self.test_data is not None:
                _, qs = self.network.q(self.test_data[0])

                for i, q in enumerate(qs):
                    self.q_graphs[i].setData(qs[self.test_data[1][i]])

    def monitor(self, state, reward, terminal, q_values, is_evaluate):
        self.iterations += 1

        if is_evaluate:
            self.max_q = max(self.max_q, np.max(q_values))
            self.max_score = max(self.max_score, self.environment.get_score())

            if terminal:
                print "\nEvaluation: Iterations {}, Episodes {}, Max-Q {}, Max Score {}".format(self.iterations, self.environment.get_episodes(), self.max_q, self.max_score)
                self.max_q = -99999
                self.max_score = -99999
        else:
            print q_values

        # save stats

        #if self.args.vis:
            # visualize
