"""Iterate test files.

:author: Jingchu LIU
:data: 2017-11-16
"""
import os
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test File Iterator.')
    parser.add_argument('test_folder', type=str)
    parser.add_argument('ckpt_folder', type=str, default="/root/")
    parser.add_argument('planning_path', type=str)
    args = parser.parse_args()

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
        os.sep.join([args.planning_path, 'next.launch'])
    )

    with open(os.sep.join([args.test_folder, 'finished.list']), 'wb') as f:
        finished.append(test)
        f.writelines(finished)

