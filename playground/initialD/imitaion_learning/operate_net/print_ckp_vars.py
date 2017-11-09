import tensorflow as tf


def get_vars(checkpoint_dir):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        var_dict = {}
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            var_dict[var_name] = var
    return var_dict


ckpt_list = [
             # "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_pi",
             # "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_pq_rename",
             # "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_learn_q_obj80_adam001_stop_gradient_v0",
             # "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_pq_learn_q_v1",
             # "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_learn_q_adam0001_not_stop_gradient",
             # "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_pq_ac",
             # "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_ac_learn_q_stop_gradient_wait_10s_new_func_reward",
             # "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_ac_wait_40s_new_func_reward_only_q_loss",
             # "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_pq_rename"
             #    "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_ac_wait_40s_new_func_reward"
            # "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_check_learn_q_wait_40s_new_func_reward",
            # "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_check_learn_q_wait_40s_new_func_reward_no_q",
            # "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_check_no_q_wait_40s_new_func_reward_learning_off",
            #     "/mnt/a/rl.work/agents/hobotrl/log/A3CCarDiscrete",
            #     "/home/pirate03/Downloads/A3CCarDiscrete2",
                "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_ac_with_q_learned_from_wait40s",
             ]

fnames = [
          # "resnet_pi.txt",
          # "resnet_pq_rename.txt",
          #   "resnet_pq_learn_q_obj80_adam001_stop_gradient_v0.txt",
          # "resnet_pq_learn_q_v1.txt",
          # "resnet_not_stop_gradient_learn_q.txt",
          #   "resnet_ac_learn_q_stop_gradient_wait_10s_new_func_reward.txt",
          #   "resnet_ac_wait_40s_new_func_reward_only_q_loss.txt"
          # "resnet_ac_wait_40s_new_func_reward.txt"
    # "resnet_check_learn_q_wait_40s_new_func_reward_2.txt",
    # "resnet_check_learn_q_wait_40s_new_func_reward_no_q_2.txt"
    # "resnet_check_no_q_wait_40s_new_func_reward_learning_off.txt"
    #         "mnt_a_rl.work_agents_hobotrl_log_A3CCarDiscrete.txt",
    #         "home_pirate03_Downloads_A3CCarDiscrete2.txt",
            "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_ac_with_q_learned_from_wait40s/resnet_ac_with_q_learned_from_wait40s.txt"

         ]

files = [open(fname, "w") for fname in fnames]

for ckpt, file in zip(ckpt_list, files):
    print ckpt
    var_dict = get_vars(ckpt)
    for name in sorted(var_dict):
        print name
        # print var_dict[name]
        # file.write("%s\n"%(name))
        file.write("%s\n%s\n"%(name,var_dict[name]))
    file.close()
