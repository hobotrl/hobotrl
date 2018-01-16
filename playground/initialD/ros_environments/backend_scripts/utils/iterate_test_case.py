"""Iterate test files.

:author: Jingchu LIU
:data: 2017-11-16
"""
import os
import shutil
import argparse
import pickle
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test File Iterator.')
    parser.add_argument('--test_folder', type=str, default="/Projects/hobotrl/playground/initialD/test/experiment/test_cases")
    parser.add_argument('--ckpt_folder', type=str, default="/Projects/catkin_ws/src/Planning/planning/launch")
    parser.add_argument('--fail_duration', type=float, default=60.0)
    args = parser.parse_args()

    try:
        with open(os.sep.join([args.ckpt_folder, 'last.time']), 'rb') as f:
            t1, t2 = pickle.load(f)
        print "[iterate_test_case]: duration of last episode {}".format(t2 - t1)
        if t2 - t1 <= args.fail_duration:
            last_launch_fail = True
        else:
            last_launch_fail = False
    except IOError:
        print "[iterate_test_case]: file 'last.time' not found."
        last_launch_fail = False

    if not last_launch_fail:
        with open(os.sep.join([args.test_folder, 'test.list']), 'rb') as f:
            tests = f.readlines()

        try:
            with open(os.sep.join([args.ckpt_folder, 'finished.list']), 'rb') as f:
                finished = f.readlines()
        except IOError:
            print "finished list not found."
            finished = []

        for test in tests:
            if test not in finished:
                break

        shutil.copy(
            os.sep.join([args.test_folder, test[:-1]]),
            os.sep.join([args.ckpt_folder, 'next.launch'])
        )

        with open(os.sep.join([args.ckpt_folder, 'finished.list']), 'wb') as f:
            finished.append(test)
            f.writelines(finished)
    else:
        print "Last launch fail, reusing launch file."

    print "[iterate_test_case]: start loop."
    try:
        t1 = time.time()
        while True:
            time.sleep(1.0)
    except:
        pass
    finally:
        t2 = time.time()
        with open(os.sep.join([args.ckpt_folder, 'last.time']), 'wb') as f:
            pickle.dump((t1, t2), f)
        print "[iterate_test_case]: out, episode duration {}".format(t2 - t1)

