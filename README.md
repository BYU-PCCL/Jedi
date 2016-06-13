Installation
============

```bash
apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
git clone https://github.com/el3ment/gym
cd gym
pip install -e '.[all]'
pip install tqdm pyqtgraph
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

```Q = r + discount * next_qs[random_index]```

This would result in the 1:many relationship more suited for a density model. Likewise, the density model is more suited
for environments that are stochastic in nature. In these environments, a density model may allow us to choose
/risk adjusted/ actions that considers variance and mean q-value together.

Variance in q-values is relatively constant throughout the entire training experience, ranging from a maximum
standard deviation of 0.01 to .1.
 - **Idea:** Perhaps it's possible to train with a loss function that minimizes q-error and maximizes q-variance.


Q-Explorer (Sampling according to Q-values)
-------------------------------------------
Sampling according to q-values is actually pretty successful when sampling

```p(action) ~ q^alpha | alpha = 2 or 2.5```

This has the bonus of not needing an explicit epsilon but it doesn't result in a "dramatic" improvement in score.
However, if q-sampling is equivalent to epsilon greedy exploration, q-sampling should be preferred.
- **Task:** Run 50-game experiment to confirm that q-sampling is equivalent to epsilon decay across all games.

It was thought that the network had a hard time differentiating between /barely/ and /safely/ hitting the ball
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
