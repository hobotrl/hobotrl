from multiprocess import Pool
import os
import subprocess


def concat_names(prefix, exp_ids, num=11):
    ret = {}
    for exp_id in exp_ids:
        tmp = []
        for i in range(num):
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


def to_script_name(exp_info):
    script_name = None
    if 'exp01' in exp_info:
        script_name = 'ExpID01'
    elif 'exp09' in exp_info:
        script_name = 'ExpID09'
    elif 'exp10' in exp_info:
        script_name = 'ExpID10'
    elif 'exp19' in exp_info:
        script_name = 'ExpID19'
    elif 'exp21' in exp_info:
        script_name = 'ExpID21'
    else:
        print "Wrong Exp Name"
    script_name = 'exp_DrSimKub_AsyncDQN_LanePaper_'+script_name+'.py'
    return script_name


def run_exp(exp_info):
    script_name = to_script_name(exp_info)
    f = open(os.sep.join([exp_info, "test.txt"]), "w")
    fe = open(os.sep.join([exp_info, "test_err.txt"]), "w")
    proc = subprocess.Popen(['python', script_name, '--dir_prefix', exp_info],
                     stdout=f, stderr=fe)
    ret = proc.wait()
    print "Open process "+exp_info+", ret:"+ret

if __name__ == "__main__":
    # exp_list = [exp01, exp01]
    test_prefix_dir = "../../../../../../work/agents/Compare/AgentStepAsCkpt"
    # exp_ids = ['exp01', 'exp09', 'exp10', 'exp19']
    exp_ids = ['exp21']
    exp_infos = concat_names(test_prefix_dir, exp_ids, num=11)

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

    pool = Pool(4)
    # pool.map(run_exp, lunzhuan_exp_list)
    for para in lunzhuan_exp_list:
        pool.apply_async(run_exp, (para, ))
    print "What happened"
    pool.close()
    pool.join()
