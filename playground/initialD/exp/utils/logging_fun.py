import tensorflow as tf
import numpy as np

def print_qvals(n_steps, agent, state, action, AGENT_ACTIONS):
    q_vals = agent.learn_q(np.asarray(state)[np.newaxis, :])[0]
    # ('s', q), ('d', q), ('a', q)
    p_dict = sorted(zip(
        map(lambda x: x[0], AGENT_ACTIONS), q_vals
    ))
    max_idx = np.argmax([v for _, v in p_dict])
    p_str = "(({:3d})[Q_vals]: ".format(n_steps)
    for i, (a, v) in enumerate(p_dict):
        if a == AGENT_ACTIONS[action][0]:
            sym = '|x|' if i == max_idx else ' x '
        else:
            sym = '| |' if i == max_idx else '   '
        p_str += '{}{:3d}: {:8.4f} '.format(sym, a, v)
    print p_str

def log_info(agent_info, env_info,
             done,
             cum_reward,
             n_ep, n_ep_steps, n_env_steps, n_agent_steps):
    summary_proto = tf.Summary()
    for tag in agent_info:
        summary_proto.value.add(
            tag=tag, simple_value=np.mean(agent_info[tag])
        )
    if done:
        summary_proto.value.add(tag='exp/n_ep_steps', simple_value=n_ep_steps)
        summary_proto.value.add(tag='exp/n_env_steps', simple_value=n_env_steps)
        summary_proto.value.add(tag='exp/n_agent_steps', simple_value=n_agent_steps)
        summary_proto.value.add(
            tag='exp/num_episodes', simple_value=n_ep)
        summary_proto.value.add(
            tag='exp/total_reward', simple_value=cum_reward)
        summary_proto.value.add(
            tag='exp/per_step_reward', simple_value=cum_reward/n_ep_steps)
        if 'flag_success' in env_info:
            summary_proto.value.add(
                tag='exp/flag_success', simple_value=env_info['flag_success'])

    return summary_proto