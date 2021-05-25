# rl-an-introduction
Small tests of the theory presented in: http://incompleteideas.net/book/RLbook2020.pdf

For reference, the origin of the "hack" used by some code files to import uninstalled packages in parent directories was taken from here: https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder#comment23054549_11158224

## Future improvements
* Chapter 2
  - [ ] The environments in chapter 2 should take more parameters in their constructor such as the starting mean.
  - [ ] The utilities framework in chapter 2 was not designed with *Gradient Bandits* in mind. The implementation for it is therefore currently a stand-alone class in the *agent.py* file.
  - [ ] The results of *2.10 The 10-armed Testbed Parameter Study Moving* used a very low NUM_RUNS because of the significant time for a single run. Therefore, the results are not reliable until it is rerun with a much higher NUM_RUNS.
