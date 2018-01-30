from multiprocess import Pool
import os
# import test1
# import test2

from exp_DrSimKub_AsyncDQN_LanePaper_ExpID01 import exp as exp01
from exp_DrSimKub_AsyncDQN_LanePaper_ExpID09 import exp as exp09
from exp_DrSimKub_AsyncDQN_LanePaper_ExpID10 import exp as exp10
from exp_DrSimKub_AsyncDQN_LanePaper_ExpID19 import exp as exp19
from exp_DrSimKub_AsyncDQN_LanePaper_ExpID21 import exp as exp21


def concat_names(prefix, exp_info):
    ret = {}
    for exp_id in sorted(exp_info.keys()):
        exp_id_range = exp_info[exp_id]
        tmp = []
        for i in exp_id_range:
            tmp.append(os.sep.join([prefix, exp_id, str(i)]))
        ret[exp_id] = tmp
    return ret

def lunzhun(infos):
    ret = []
    l_list = [len(info) for info in infos]
    i = 0
    for _ in range(max(l_list)):
        for j in range(len(infos)):
            if i < l_list[j]:
                ret.append(infos[j][i])
        i += 1
    return ret

if __name__ == "__main__":
    # exp_list = [exp01, exp01]
    test_prefix_dir = "../../../../../../work/agents/Compare/AgentStepAsCkpt"
    exp_infos = {'exp01': range(1, 11), 'exp09': range(0, 10), 'exp10': range(0, 10), 'exp19': range(1, 11)}
    exp_infos = concat_names(test_prefix_dir, exp_infos)
    exp_list = []
    # para_list = [['../../../../../../work/agents/Comapre/AgentStepAsCkpt/exp01/1'],['../../../../../../work/agents/Comapre/AgentStepAsCkpt/exp01/2']]

    to_do_exp_infos = {}
    for i in exp_infos:
        to_do_exp_infos[i] = []

    for i in exp_infos:
        exps = exp_infos[i]
        for exp in exps:
            if os.path.exists(exp) and not os.path.exists(exp+"/logging_0"):
                to_do_exp_infos[i].append(exp)
    to_do_exp_list = to_do_exp_infos.values()
    lunzhuan_exp_list = lunzhun(to_do_exp_list)
    from pprint import pprint
    pprint(lunzhuan_exp_list)

    for para in lunzhuan_exp_list:
        if 'exp01' in para:
            exp_list.append(exp01)
        elif 'exp09' in para:
            exp_list.append(exp09)
        elif 'exp10' in para:
            exp_list.append(exp10)
        elif 'exp19' in para:
            exp_list.append(exp19)
        elif 'exp21' in para:
            exp_list.append(exp21)
        # elif 'exp22' in para:
        #     exp_list.append(exp22)
        # elif 'exp23' in para:
        #     exp_list.append(exp23)
        # elif 'exp24' in para:
        #     exp_list.append(exp24)
        else:
            print "Wrong exp id"
    print "exp_list: ", exp_list
    pool = Pool(4)
    for exp, para in zip(exp_list, lunzhuan_exp_list):
        pool.apply_async(exp, (para, ))
    # pool.map(exp_list, lunzhuan_exp_list)
    pool.close()
    pool.join()
    #

    # print concat_names(test_prefix_dir, exp_infos)