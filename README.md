Installation
============

```bash
apt-get install -y python-numpy python-pip python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

git clone https://github.com/el3ment/gym
cd gym
pip install -e '.[all]'

pip install tqdm pyqtgraph

# The easiest way to install cv2 is with conda. Do -conda install opencv-

pip install psycopg2 # if you get an error, -sudo apt-get install libpq-dev- may solve it

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL
```

You should also consider adding the ./ directory to $PATH so that "jedi" will function like a command

```
export PATH=$PATH:.
```


SQL Tables
==========
```sql
DROP TABLE stats;
DROP TABLE agents;
CREATE TABLE stats(
  id INTEGER PRIMARY KEY AUTOINCREMENT, 
  agent_id INTEGER,
  episode INTEGER,
  stat_name VARCHAR(25),
  value NUMERIC,
  is_evaluation BOOL,
  FOREIGN KEY(agent_id) REFERENCES agents(id)
);

CREATE TABLE agents(
  id INTEGER PRIMARY KEY AUTOINCREMENT, 
  agent_name TEXT,
  configs TEXT
);

CREATE INDEX agent_id_idx ON stats (agent_id);
```

Findings
========
Clipping the error, but NOT clipping the reward results in a network that does not learn at all.
It seems ANY activation of the clipping function causes this to occur (clipping at 49.0 vs 50) and is evidence that
error clipping is not the proper way to handle exploding q-values.


Density Network
---------------

Because the relationship between state and q-value is 1:1, the optimal sigma is zero.
It might be possible to redefine bellman error probabilistically and introduce some idea of variance for example:

```
Q = r + discount * next_qs[random_index]
```

This would result in the 1:many relationship more suited for a density model. Likewise, the density model is more suited
for environments that are stochastic in nature. In these environments, a density model may allow us to choose
/risk adjusted/ actions that considers variance and mean q-value together.

Variance in q-values is relatively constant throughout the entire training experience, ranging from a maximum
standard deviation of 0.01 to .1.
 - **Idea:** Perhaps it's possible to train with a loss function that minimizes q-error and maximizes q-variance.


Q-Explorer (Sampling according to Q-values)
-------------------------------------------
Sampling according to q-values is actually pretty successful when sampling

```
p(action) ~ q^alpha | alpha = 2 or 2.5
```

This has the bonus of not needing an explicit epsilon but it doesn't result in a "dramatic" improvement in score.
However, if q-sampling is equivalent to epsilon greedy exploration, q-sampling should be preferred.
- **Task:** Run 50-game experiment to confirm that q-sampling is equivalent to epsilon decay across all games.

It was thought that the network had a hard time differentiating between *barely* and *safely* hitting the ball
resulting in an higher-than-expected chance of missing the ball during action sampling. A negative reward on death
was added to increase q-value variance at critical decision points. This resulted in increased score variance, but not
substantial improvement in max-score or frequency of max-score.
- **Task:** Prove the impact negative reward on death.
- **Idea:** Try a higher alpha to encourage the explorer and evaluator to be equivalent


Float-16
--------
```
In : np.array([100000], dtype=np.float16)
Out: array([ inf], dtype=float16)

In : np.array([10000], dtype=np.float16)
Out: array([ 10000.], dtype=float16)

In : np.array([100000], dtype=np.float16)
Out: array([ inf], dtype=float16)

In : np.array([65504], dtype=np.float16)
Out: array([ 65504.], dtype=float16)

In : np.array([65505], dtype=np.float16)
Out: array([ 65504.], dtype=float16)

In : np.array([65515], dtype=np.float16)
Out: array([ 65504.], dtype=float16)

In : np.array([66504], dtype=np.float16)
Out: array([ inf], dtype=float16)

In : np.array([65504], dtype=np.float16)
Out: array([ 65504.], dtype=float16)
```



Ideas Without Context
-----------------------
- Sample 15 priorities and 15 random
- parameterize with fft (convolution) - try to find paper and consider implementing
- maximum margin
- http://arxiv.org/pdf/1512.02011v2.pdf - increase discount rate during training,
- http://arxiv.org/pdf/1605.05365v2.pdf - consider adding a "duration" output for how long to apply the command
- http://arxiv.org/pdf/1602.07714v1.pdf - gradient/reward clipping sucks
- (Based on http://jmlr.org/proceedings/papers/v37/schaul15.html) 
  Transform V(s, theta) to V(s, g, theta) and train using V(s, s_prime, theta) using s_prime sampled from experience database
  Exploration can be guided by choosing g differently
- Importance Weighting of Experience
  drop Argmin_x(KL(experience db with x, experience db without x)) from database
- Macro Actions - train network (LSTM) to output sequences of actions
- Episodic Control - Pass the best-sequence-so-far in as input into the network
- Episodic Control - Pass the best-sequence-so-far in as input into the network, learn a probabilistic gate between predicted action and bssf action
- Uncertianty - Give neurons a pulse frequency that is governed by activation (high action potential = fire, low action potential = fire sometimes with probability = f(frequency), very low action potential = don't fire)



Experiment Data to be deleted
=============================

