# hobotrl

Common RL framework and utilities.

## Design considerations

* maximize module reuse under RL framework
* accumulate experiment baselines for algorithm, with hyperparam setup, easily reproducible
* accumulate experiment baselines for problems, with algorithm/hyperparam/engineering efforts
* easily implement more algorithms
* easily combine different works

## Initial algorithm set

* [v] DQN
* [v] DDPG
* [v] Replay Buffer
* [v] Prioritized Exp Replay
* [v] Double DQN
* [v] Duel DQN
* [v] Actor Critic
* [v] Optimality Tightening
* [v] A3C
* Bootstrap DQN


## Getting started

All single process experiments resides in `test` folder.

```
python test/exp_tabular.py run --name TabularGrid
```
for starter.

```
python test/exp_tabular.py list
python test/exp_deeprl.py list
```
to get a list of experiments in each file.

```
. scripts/a3c_pong.sh
```
to start processes to run a3c algorithm.

### Driving Simulator Environment
The steps for starting the driving simulator environment:
1. Open up a new shell, exececute `roscore` to launch ROS master.
2. Open up yet another shell, first `source [catkin_ws_dir]/devel/setup.bash` to register simulator ROS packages, then run `python rviz_restart.py` to fire up the simulator launcher.
3. The last shell if for running the actual main script, where a `DrivingSimulatorEnv` is instanced to commnunicate with the previously opened nodes as well as the agent.

Note these steps are tentitive and subject to change.

## Developers Guide
### Sharing network parameters across modules
~~See this [wiki entry](https://github.com/zaxliu/hobotrl/wiki#sharing-network-weights-across-modules) for a recommended way via global variable scope reuse.~~ [Setting global scope reference will break the creation of target network.]
