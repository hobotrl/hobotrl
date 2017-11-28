import numpy as np

def combine_rewards(vec_rewards, actions):
    momentum_ped = 0.0
    momentum_opp = 0.0
    comb_rewards = []
    for vec_r, action in zip(vec_rewards, actions):
        r0 = 0.0 * vec_r[0]
        r1 = -10 * (vec_r[1] > 2.0)
        r2 = 10 * vec_r[2]
        momentum_opp = (vec_r[3] < 0.5) * (momentum_opp + 1 - vec_r[3])
        momentum_opp = min(momentum_opp, 20)
        r3 = -20 * (0.9 + 0.1 * momentum_opp) * (momentum_opp > 1.0)
        momentum_ped = (vec_r[4] > 0.5) * (momentum_ped + vec_r[4])
        momentum_ped = min(momentum_ped, 12)
        r4 = -40 * (0.9 + 0.1 * momentum_ped) * (momentum_ped > 1.0)
        r5 = -100.0 * vec_r[5]
        r6 = -10.0 * float(action==1 or action==2)
        comb_r = (r0 + r1 + r2 + r3 + r4 + r5 + r6) / 100.0
        comb_rewards.append(comb_r)
    return np.array(comb_rewards)


def eps_rewards_actions_v0(eps_dir):
    vec_rewards = []
    actions = []
    scalar_rewards = []
    with open(eps_dir + "/0000.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # if i > len(lines):
                #     break
            if i % 2 == 0:
                actions.append(float(line.split("\n")[0].split(",")[1]))
                scalar_rewards.append(float(line.split("\n")[0].split(",")[-1]))
            else:
                vec_reward = map(float, line.split(',')[:-1])
                vec_rewards.append(vec_reward)
    return np.array(vec_rewards), np.array(actions), np.array(scalar_rewards)


def eps_rewards_actions_v1(eps_dir):
    vec_rewards = []
    actions = []
    scalar_rewards = []
    with open(eps_dir + "/0000.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # if i > len(lines):
            #     break
            if i % 4 == 0:
                actions.append(float(line.split("\n")[0].split(",")[1]))
                scalar_rewards.append(float(line.split("\n")[0].split(",")[-1]))
            elif i % 4 == 2:
                vec_reward = map(float, line.split(',')[:-1])
                vec_rewards.append(vec_reward)
            else:
                pass
    return np.array(vec_rewards), np.array(actions), np.array(scalar_rewards)



if __name__ == "__main__":
    # eps_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_ac_records/0001"
    # eps_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_ac_stop_conv4_include_red_line_reward/0001"
    # eps_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_ac_check_restore_records/0002"
    # eps_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_learn_q_obj80_adam001_stop_gradient_v0_records/0001"
    # eps_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_learn_q_obj80_adam001_stop_gradient_v1_records/0001"
    eps_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_learn_q_adam0001_not_stop_gradient_records/0001"
    # vec_rewards, actions, scalar_rewards = eps_rewards_actions_v1(eps_dir)
    vec_rewards, actions, scalar_rewards = eps_rewards_actions_v0(eps_dir)
    comb_rewards = combine_rewards(vec_rewards, actions)
    i = 1
    for r1, r2 in zip(comb_rewards, scalar_rewards):
        print i, ", ", r1, ", ", r2
        i += 1


