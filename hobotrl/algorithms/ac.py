#
# -*- coding: utf-8 -*-

import hobotrl as hrl
import hobotrl.tf_dependent as tf_dep


class ActorCritic(
    hrl.tf_dependent.mixin.NNStochasticPolicyMixin,
    hrl.mixin.ReplayMixin,
    tf_dep.mixin.DeepQFuncMixin,
    hrl.tf_dependent.base.BaseDeepAgent
):
    def __init__(self, state_shape, is_continuous_action, num_actions, f_create_policy, f_create_value,
                 entropy=0.01, gamma=0.9, train_interval=8,
                 training_params=None, schedule=None,
                 greedy_policy=True, ddqn=False,
                 buffer_class=hrl.playback.MapPlayback,
                 buffer_param_dict={"capacity": 1000, "sample_shapes": {}},
                 batch_size=1,
                 **kwargs):
        """
        list all supported ctor parameters here for user reference.
        :param state_shape:
        :param is_continuous_action:
        :param num_actions:
        :param f_create_policy:
        :param f_create_value:
        :param entropy:
        :param gamma:
        :param train_interval:
        :param training_params:
        :param schedule:
        :param greedy_policy:
        :param ddqn:
        :param buffer_class:
        :param buffer_param_dict:
        :param batch_size:
        :param kwargs:
        """
        kwargs.update({
            "state_shape": state_shape,
            "is_continuous_action": is_continuous_action,
            "is_action_in": is_continuous_action,  # for Q function
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
        if is_continuous_action:
            kwargs.update({"action_shape": [num_actions]})  # for Q function

        super(ActorCritic, self).__init__(**kwargs)
