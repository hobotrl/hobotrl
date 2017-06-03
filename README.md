# hobotrl

Common RL framework and utilities.

## Design considerations

* maximize module reuse under RL framework
* accumulate experiment baselines for algorithm, with hyperparam setup, easily reproducible
* accumulate experiment baselines for problems, with algorithm/hyperparam/engineering efforts
* easily implement more algorithms
* easily combine different works

## Initial algorithm set

* DQN
* DDPG
* Replay Buffer
* Prioritized Exp Replay
* Double DQN
* Duel DQN

## Considerations Regarding Mixins
Jun 3: The current implementation relies heavily on mixins. So far this is fine due to the linear nature of single-agent settings, but it is questionable whether or not mixin-style programming will still be proper for multi-agent settings (maybe a *subscriber*-style interface is better?). For this reason, we refactor all mixin classes as glue code that build the mixing classes and mixin-independent heavy-lifting procudures. All of the heavy-lifting procedures (ValueFunc & Policies) are put into the `utils` module, and the glue code will import and instantiate them as members. If in the future mixins should become obsolete, we can easilly reuse the heavy-lifting procedures without breaking anything.

