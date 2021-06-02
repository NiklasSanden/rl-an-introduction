# rl-an-introduction
Small tests of the theory presented in: http://incompleteideas.net/book/RLbook2020.pdf

For reference, the origin of the "hack" used by some code files to import uninstalled packages in parent directories was taken from here: https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder#comment23054549_11158224

## Future improvements
* Chapter 2
  - [x] The environments in chapter 2 should take more parameters in their constructor such as the starting mean.
  - [x] The utilities framework in chapter 2 was not designed with *Gradient Bandits* in mind. The implementation for it is therefore currently a stand-alone class in the *agent.py* file.
  - [x] The results of *2.10 The 10-armed Testbed Parameter Study Moving* used a very low NUM_RUNS because of the significant time for a single run. Therefore, the results are not reliable until it is rerun with a much higher NUM_RUNS.

## Ideas
* Chapter 3
  * (When reading 3.2) Giving a reward of -1 at every time step to encourage the agent to escape a maze seems suboptimal from a sense of how we want the agent to "think", although it works great in the tabular case. Even though it does receive -1, it has no idea that it can do better or that there is a goal in the end with a different reward (or just a terminal state at all). When dealing with sparse rewards (I'm mostly thinking about just giving a different reward when transitioning to a terminal state), perhaps it makes sense to let the agent know the reward function and see the states (or something similar) where the reward differs. From there it could perhaps construct some model or internal reward system to encourage exploring and making "progress". The point of this whole idea is that if we only have one sparse reward at the end (and all the others are the same such as -1 or 0 for example), it will almost always make sense to let the agent know when that occurs or at least that it can occur (even if I'm not sure how it can be exploited yet). That's what we would do if a human were to try to complete such a task in most relevant situations.
