#
# -*- coding: utf-8 -*-

import hobotrl as hrl
import hobotrl.tf_dependent as tf_dep


class ActorCritic(
    tf_dep.policy.DiscreteNNPolicy,
    hrl.mixin.ReplayMixin,
    tf_dep.mixin.DeepQFuncMixin,
    hrl.core.BaseAgent
):
    def __init__(self, state_shape, num_actions, f_create_policy, f_create_value,
                 entropy=0.01, gamma=0.9, train_interval=8,
                 training_params=None, schedule=None,
                 greedy_policy=True, ddqn=False,
                 buffer_class=hrl.playback.MapPlayback,
                 buffer_param_dict={"capacity": 1000, "sample_shapes": {}},
                 batch_size=1,
                 **kwargs):
        kwargs.update({
            "state_shape": state_shape,
            "num_actions": num_actions,
            "f_create_net": f_create_policy,
            "f_net": f_create_value,
            "entropy": entropy,
            "gamma": gamma,
            "train_interval": train_interval,
            "training_params": training_params,
            "schedule": schedule,
            "greedy_policy": greedy_policy,
            "ddqn": ddqn,
            "buffer_class": buffer_class,
            "buffer_param_dict": buffer_param_dict,
            "batch_size": batch_size
        })
        super(ActorCritic, self).__init__(**kwargs)
