# rl-an-introduction
Small tests of the theory presented in: http://incompleteideas.net/book/RLbook2020.pdf

For reference, the origin of the "hack" used by some code files to import uninstalled packages in parent directories was taken from here: https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder#comment23054549_11158224

## Future improvements
* Chapter 2-9
  - [ ] Use string interpolation.

* Chapter 2
  - [x] The environments in chapter 2 should take more parameters in their constructor such as the starting mean.
  - [x] The utilities framework in chapter 2 was not designed with *Gradient Bandits* in mind. The implementation for it is therefore currently a stand-alone class in the *agent.py* file.
  - [x] The results of *2.10 Parameter Study Moving* used a very low NUM_RUNS because of the significant time for a single run. Therefore, the results are not reliable until it is rerun with a much higher NUM_RUNS.

* Chapter 5
  - [ ] Rerun the experiments to make sure code changes didn't break them.
  - [ ] I realised something very interesting with off-policy MC Control in *5.7 Racetrack MC Control*. Having initial Q-values as 0 is highly optimistic, which means an unexplored state-action pair will always be the greedy action when available (the environment gives you -1 as a reward on each timestep). This can be problematic for this kind of off-policy learning since the learning signal for state s becomes 0 if the behaviour policy picked a different action than what the target policy says after visiting s. This means that even after training for a long time, there might still be a few state-action pairs that have not gotten a learning signal through (even if the behaviour policy is improved as well), which makes it very likely that the deterministic target policy won't ever reach the finish line. Currently the fix is to ensure that we have pessimistic values so that the greedy action is always of a state-action pair that has been explored (which is equivalent to ignoring unexplored ones when performing greedy action selection). The fact that this solves a problem proves that we haven't trained enough to get a good estimate for every state-action pair. Therefore we won't get optimal solutions but they will be good given our limited Q function. A future experiment could instead give a reward of +1 when reaching the finish line and 0 otherwise and have a discount factor < 1 to incentivise speed while allowing 0 to be pessimistic. Then you could also use *Discounting-aware Importance Sampling*.
  - [ ] Perhaps there should be a test using *Discounting-aware Importance Sampling* if it isn't used in the previous point (my understanding of it theoretically should at least improve.)
  - [ ] Perhaps there should be a test using *Per-decision Importance Sampling*. Not a high priority since it is covered in Chapter 7 in the n-step setting.

* Chapter 7
  - [x] The 7.3 experiment is temporary. It is supposed to compare *Per-decision Methods* with ordinary importance sampling (i.e. 7.4 & 7.3 respectively) as described in exercise 7.10. This experiment is post-poned until I understand the theory in 7.4 with respect to action values (equation 7.14). 
  - [ ] Rewrite *n_step_sarsa_off_policy* and potentially *n_step_sarsa_off_policy_Q_sigma* as well. The pseudo code in the book includes one variant of off-policy where the importance sampling is calculated with the lastest Q-estimates and one where the PI(a|s)/b(a|s) is saved when the action was picked. Decide which implementation should be used and make sure that all the indices are correct. The experiment in 7.4 could be extended to include the *Tree Backup* from 7.5 and *Q(sigma)* from 7.6. 7.4 and 7.5 is implemented using Q(sigma=1) and Q(sigma=0) respectively.
  - [ ] Potentially rewrite all n-step methods to use index access and store mod n+1. Perhaps even some which currently use deques should consider using this for the sake of consistency.
  - [ ] Do exercise 7.10. Note that this is in the prediction case!
  - [ ] The 7.1 experiment should reset the predetermined actions after each run (otherwise there is no point in doing multiple runs). See *9.4 State Aggregation n-step* for how it should be handled.

* Chapter 8
  - [ ] The results in Prioritized Sweeping are not very fair. Normal Dyna-Q should not make updates that wouldn't change anything, or we should count Prioritized Sweep as doing updates even if the queue is empty. We also only check for it being optimal after an episode, so long episodes can give a high error, which is worse for Dyna-Q. The experiment was cumbersome because with a low epsilon the algorithms, especially Prioritized Sweeping could easily get stuck on an unoptimal solution for very long. One should also experiment a bit with smaller alpha values, because right now the queue is empty most of the time. 

* Chapter 10
  - [ ] The book most likely used epsilon=0 for *10.1 Semi-gradient Sarsa Alpha*, so that experiment should be rerun.
  - [ ] *10.2 n-step Parameter Sweep* should be rerun with NUM_RUNS=100, but there is a risk that a run won't converge and therefore loops indefinitely. Make an environment wrapper that makes the episode end after a set number of steps so that the experiment can finish despite this.

* Chapter 13
  - [x] Run the same experiment as in *Example 13.1*

## Ideas
* Chapter 3
  * (When reading 3.2) Giving a reward of -1 at every time step to encourage the agent to escape a maze seems suboptimal from a sense of how we want the agent to "think", although it works great in the tabular case. Even though it does receive -1, it has no idea that it can do better or that there is a goal in the end with a different reward (or just a terminal state at all). When dealing with sparse rewards (I'm mostly thinking about just giving a different reward when transitioning to a terminal state), perhaps it makes sense to let the agent know the reward function and see the states (or something similar) where the reward differs. From there it could perhaps construct some model or internal reward system to encourage exploring and making "progress". The point of this whole idea is that if we only have one sparse reward at the end (and all the others are the same such as -1 or 0 for example), it will almost always make sense to let the agent know when that occurs or at least that it can occur (even if I'm not sure how it can be exploited yet). That's what we would do if a human were to try to complete such a task in most relevant situations.

## Notes
* Chapter 3
  * Discounting can be necessary for iterative policy evaluation (or value iteration) to converge to the unique solution v_pi or v_* in an episodic task. Imagine a gridworld that allows you to go back to a previous state (which we usually think of as episodic since it has a terminal state). If your policy will have a cycle, meaning the environment and the policy's actions together create a deterministic infinite trajectory from some state, then there is not necessarily a uniqe solution of v for those states unless discounting is used. If you receive a reward of 0 on each transition in this trajectory, then as long as the value of each of these states is the same, the bellman equations would hold for any such value. There is also an obvious problem with a non-zero reward, since the returns would be infinite. In this case a discount factor is necessary and will make the solutions unique. Therefore, it follows that in the prediction problem, the discount factor cannot be one if there exists a starting state where your policy will *never* terminate (if there is a chance it will because of the dynamics function, then that is similar to a discount factor as discussed much later in the book).  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the control problem, you might get away with it, as demonstrated by using a -1 reward in a gridworld without discounting. Value iteration is fine in that example because the algorithm would never converge to such a policy where the episode would never terminate, and sample based bootstrapping methods would always end up exploring the entire state space if it had to in order to find the terminal state. Running policy evaluation until it converges could be problematic if the starting policy would go in cycles as described earlier because then it would diverge. The same goes for monte carlo methods. Unless the episode is "cut short", you can never finish an episode and therefore never learn to get out of the cycle. Monte carlo methods require episodes to terminate eventually regardless, and will therefore always explore to some degree, so once again this is usually avoided in practice.

* Chapter 6
  * Batch TD(0) & batch MC are introduced in 6.3. They claim that any alpha "sufficiently small" will make them converge to their respective *correct* values. As a few comments in the source code of that chapter state, there is an optimal alpha. If the batch size for state s is b, then 1/b is the best you can do for updating V(s). Any value above that risks diverging and any value below that will take longer to converge because you'll leave some of your previous values in your new estimates (which induces an unwanted bias). This means that you would have to use a different step-size for each state s since their batch sizes will most likely differ, which the source code here does. If alpha is 1/b, then MC becomes V(S_t) = avg(G_t) and TD becomes V(S_t) = avg(R_t)+gamma\*avg(V(S_{t+1})), where we average all samples we have for that value.
