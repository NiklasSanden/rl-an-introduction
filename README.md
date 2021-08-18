# rl-an-introduction
Small tests of the theory presented in http://incompleteideas.net/book/RLbook2020.pdf, serving as practice for me.

This mostly just recreates the figures shown in the book, with a few differences every now and then, along with solutions to the programming exercises. Code that recreates more figures and even more precisely can be found here: https://github.com/ShangtongZhang/reinforcement-learning-an-introduction

## Future improvements
* Chapter 2-9
  - [ ] Use string interpolation and other style-changes present in Part II.

* Chapter 5
  - [ ] Rerun the experiments to make sure code changes didn't break them.
  - [ ] I realised something very interesting with off-policy MC Control in *5.7 Racetrack MC Control*. Having initial Q-values as 0 is highly optimistic, which means an unexplored state-action pair will always be the greedy action when available (the environment gives you -1 as a reward on each timestep). This can be problematic for this kind of off-policy learning since the learning signal for state *s* becomes 0 if the behaviour policy picked a different action than what the target policy says after visiting *s*. This means that even after training for a long time, there might still be a few state-action pairs that have not gotten a learning signal through (even if the behaviour policy is improved as well), which makes it very likely that the deterministic target policy won't ever reach the finish line. Currently the fix is to ensure that we have pessimistic values so that the greedy action is always of a state-action pair that has been explored (which is equivalent to ignoring unexplored ones when performing greedy action selection). The fact that this solves a problem proves that we haven't trained enough to get a good estimate for every state-action pair. Therefore we won't get optimal solutions but they will be good given our limited Q function. A future experiment could instead give a reward of +1 when reaching the finish line and 0 otherwise and have a discount factor < 1 to incentivise speed while allowing 0 to be pessimistic. Then you could also use *Discounting-aware Importance Sampling*.
  - [ ] Perhaps there should be an experiment using *Discounting-aware Importance Sampling* if it isn't used in the previous point.
  - [ ] Perhaps there should be an experiment using *Per-decision Importance Sampling*. Not a high priority since it is covered in Chapter 7 in the n-step setting.

* Chapter 7
  - [ ] Rewrite *n_step_sarsa_off_policy* and potentially *n_step_sarsa_off_policy_Q_sigma* as well. The pseudo code in the book includes one variant of off-policy where the importance sampling is calculated with the lastest Q-estimates and one where the PI(a|s)/b(a|s) is saved when the action was picked. Decide which implementation should be used and make sure that all the indices are correct. The experiment in 7.4 could be extended to include the *Tree Backup* from 7.5 and *Q(sigma)* from 7.6. 7.4 and 7.5 is implemented using Q(sigma=1) and Q(sigma=0) respectively.
  - [ ] Potentially rewrite all n-step methods to use index access and store mod n+1. Perhaps even some which currently use deques should consider using this for the sake of consistency. This is the implementation used in the later Chapters.
  - [ ] Do exercise 7.10. Note that this is in the prediction case!
  - [ ] The 7.1 experiment should reset the predetermined actions after each run (otherwise there is no point in doing multiple runs). See *9.4 State Aggregation n-step* for how it should be handled.

* Chapter 8
  - [ ] The results in Prioritized Sweeping are not very fair. Normal Dyna-Q should not make updates that wouldn't change anything, or we should count Prioritized Sweep as doing updates even if the queue is empty. We also only check for it being optimal after an episode, so long episodes can give a high error, which is worse for Dyna-Q. The experiment was cumbersome because with a low epsilon the algorithms, especially Prioritized Sweeping could easily get stuck on an unoptimal solution for very long. One should also experiment a bit with smaller alpha values, because right now the queue is empty most of the time. 

* Chapter 10
  - [ ] The book most likely used epsilon=0 for *10.1 Semi-gradient Sarsa Alpha*, so that experiment should be rerun.
  - [ ] *10.2 n-step Parameter Sweep* should be rerun with NUM_RUNS=100, but there is a risk that a run won't converge and therefore loops indefinitely. Make an environment wrapper that makes the episode end after a set number of steps so that the experiment can finish despite this. Such a wrapper can be found in later Chapters.
