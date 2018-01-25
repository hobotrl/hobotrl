import numpy as np
import os
import math

def no_nan(rewards):
    for r in rewards:
        if np.isnan(r):
            return False
    return True


class ObjStat(object):

    def __init__(self, eps_n=100, disc=0.9):
        self._eps_n = eps_n
        self._disc = disc
        self._obj_dir = None

    @staticmethod
    def read_eps_rewards(eps_dir):
        f = open(eps_dir + "/" + "0000.txt", "r")
        lines = f.readlines()
        eps_rewards = []
        scarlar_rewards = []
        for i, line in enumerate(lines):
            # if i > len(lines):
            #     break
            t = i % 3
            if t == 0:
                scarlar_rewards.append(float(line.split("\n")[0].split(",")[-1]))
            elif t == 1:
                vec_rewards = map(float, line.split(',')[:-1])
                eps_rewards.append(vec_rewards)
            else:
                pass

        for i in range(len(scarlar_rewards)):
            eps_rewards[i].append(scarlar_rewards[i])

        # return filter(no_nan, stat_rewards)
        return np.array(eps_rewards)

    def disc_rewards(self, rewards):
        ret = 0.0
        for r in rewards[::-1]:
            ret = r + self._disc * ret
        return ret

    def stat_eps_rewards(self, eps_rewards):
        # ret = self.disc_rewards(eps_rewards[:, -1])
        ave_reward = np.mean(eps_rewards, axis=0)
        ave_reward = list(ave_reward)
        # ave_reward.append(ret)
        return np.array(ave_reward)

    def read_obj_rewards(self):
        eps_names = sorted(os.listdir(self._obj_dir))[:-1]
        eps_names.remove('0000.txt')
        obj_rewards = []
        i = 0
        for eps_name in eps_names:
            eps_dir = self._obj_dir + "/" + eps_name
            eps_rewards = self.read_eps_rewards(eps_dir)
            obj_rewards.append(eps_rewards)
            i += 1
            if i >= self._eps_n:
                break
        return obj_rewards

    def stat_success(self):
        f = open(self._obj_dir + "/0000.txt", "r")
        lines = f.readlines()
        success_list = []
        for line in lines:
            flag_success = line.split(',')[-2]
            # flag_success = ' True' or ' False'
            int_flag_success = 1 if 'True' in flag_success else 0
            success_list.append(int_flag_success)
        return np.mean(success_list)

    def stat_obj_rewards(self):
        obj_rewards = self.read_obj_rewards()
        obj_info = []
        for eps_rewards in obj_rewards:
            eps_info = self.stat_eps_rewards(eps_rewards)
            obj_info.append(eps_info)

        # Here need to consider more
        obj_info = np.mean(obj_info, axis=0)
        success_rate = self.stat_success()
        obj_info = list(obj_info)
        obj_info.append(success_rate)
        return np.array(obj_info)

    def __call__(self, obj_dir, *args, **kwargs):
        self._obj_dir = obj_dir
        return self.stat_obj_rewards()


class ExpStat(object):
    def __init__(self):
        self._disc = 0.9
        self._eps_n = 100

    def __call__(self, exp_dir, st=1, ed=10):
        obj_names = sorted(os.listdir(exp_dir), key=lambda x: int(x))[st-1:ed]
        objStat = ObjStat(self._eps_n, self._disc)
        exp_info = []
        for obj_name in obj_names:
            obj_dir = exp_dir+"/"+obj_name + "/logging_0"
            print obj_name
            obj_info = objStat(obj_dir)
            exp_info.append(obj_info)
        # return list(np.mean(exp_info, axis=0)), list(np.var(exp_info, axis=0)), list(np.percentile(exp_info, 50, axis=0))
        return list(np.percentile(exp_info, 10, axis=0)), list(np.percentile(exp_info, 90, axis=0))

if __name__ == "__main__":
    # exp_dir = "/home/pirate03/work/agents/Compare/AgentStepAsCkpt/exp26"
    expStat = ExpStat()
    # ret = expStat(exp_dir, st=1, ed=10)
    # for ret_i in ret:
    #     print repr(ret_i).replace(',', ' ')

    exp_names = ['exp00', 'exp01', 'exp05', 'exp09', 'exp10', 'exp12', 'exp13', 'exp14',
                 'exp19', 'exp21', 'exp23', 'exp24', 'exp25', 'exp26']
    # exp_names = ['exp12']
    exp_dirs = []
    prefix = '/home/pirate03/work/agents/Compare/AgentStepAsCkpt/'
    for exp_name in exp_names:
        exp_dirs.append(prefix+exp_name)

    import csv
    fs = [open("stat_perc_10.csv", 'w'), open("stat_perc_90.csv", "w")]
    f_csvs = [csv.writer(f) for f in fs]
    headers = ['exp_id', 'vel_front', 'dist_to_lp', 'obs_factor', 'cur_road_valid', 'entering_intersection',
               'last_on_opposite_path', 'on_biking_lane', 'on_innerest_lane', 'on_outterest_lane', 'car_velocity_oth',
               'ave_reward', 'success_rate']
    for f_csv in f_csvs:
        f_csv.writerow(headers)
    for exp_dir in exp_dirs:
        ret = expStat(exp_dir, st=1, ed=10)
        for i in range(2):
            ret[i].insert(0, exp_dir[-5:])
            f_csvs[i].writerow(ret[i])
    for f in fs:
        f.close()



    # for i in range(1, 11):
    #     ret = expStat(exp_dir, st=i, ed=i)
    #     print "==========="
    #     print i
    #     print ret[0]
