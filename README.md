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
* DDPG
* [v] Replay Buffer
* Prioritized Exp Replay
* Double DQN
* Duel DQN
* Actor Critic


## Getting started

All single process experiments resides in `test` folder.

```
python test/exp_tabular.py run --name TabularGrid
```
for starter.

```
python test/exp_tabular.py list
```
to get a list of experiments.

## Developers Guide
### Sharing network parameters across modules
See this [wiki entry](https://github.com/zaxliu/hobotrl/wiki#sharing-network-weights-across-modules) for a recommended way via global variable scope reuse.
