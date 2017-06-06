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
### Lewis, Jun 3
The current implementation relies heavily on mixins. So far this is fine due to the linear nature of single-agent settings, but it is questionable whether or not mixin-style programming will still be proper for multi-agent settings (maybe a *subscriber*-style interface is better?).

For this reason, we refactor all mixin classes as glue code that build the mixing classes and mixin-independent heavy-lifting procudures. All of the heavy-lifting procedures (ValueFunc & Policies) are put into the `utils` module, and the glue code will import and instantiate them as members. If in the future mixins should become obsolete, we can easilly reuse the heavy-lifting procedures without breaking anything.

## Considerations regarding decoupled design in TF-based implementation
### Lewis, Jun 4
None of the TF-based RL libs provides well decoupled implementations for common modules seen in an RL algorithm (value func. and etc.). Actually, I myself never tried to build such modules too. Most common are function-style templates for a sub-graph, but never have I seen a module that encapsues *both the subgraph and related ops*.

I guess the reason lies in the fact that TF Graphs are built in an *append-only* fashion. Imagine you have two RL modules which needs the output of each other as part of its input for a related training op. Concretely let's just use DDPG as a example: the DQN needs the action output of the policy network for both training and inference, while the gradient of the Q fcn. w.r.t to the its action input is needed by the policy network to calculate DPG. So you simply cannot initialize DQN module ahead of Policy network module and vise versa. Due to this reason, the first instinct of anyone using TF should be using function-style template and separately implement a sub-graph and its trainning ops.

I don't know if this is the right way to do, but we have the following work-arounds:
1. Set those inputs that couple the outputs of other modules as `PlaceHolders` and glue them together with feed and fetch. (Slow since have to move data back & forth betweeen the frontend and backend)
1. Set those inputs that couple the outputs of other modules as `Variables` and use `tf.assign()` as a glue op to move data from outpus to inputs in TF backend. Might also consider using `control_dependency()` to ensure `tf.assign()` is called before the op depending on the inputs are fetched.
1. Clone those disconnected graphs into a new graph and connect ouputs to dependent inputs at the same time. This method can be implemented easily using the `tf.contrib.graph_editor` module.

