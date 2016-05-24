import numpy as np

class Monitor():
    def __init__(self, args, environment, network, agent):
        self.args = args
        self.environment = environment
        self.iterations = 0
        self.max_q = -99999
        self.max_score = -99999
        self.score = 0

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
