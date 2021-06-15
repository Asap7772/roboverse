import argparse
import time
import subprocess
import datetime
import os

from roboverse.utils import get_timestamp


def get_data_save_directory(args):
    data_save_directory = args.data_save_directory

    data_save_directory += '_{}'.format(args.env)

    if args.num_trajectories > 1000:
        data_save_directory += '_{}K'.format(int(args.num_trajectories/1000))
    else:
        data_save_directory += '_{}'.format(args.num_trajectories)

    if args.save_all:
        data_save_directory += '_save_all'

    data_save_directory += '_noise_{}'.format(args.noise)
    data_save_directory += '_{}'.format(get_timestamp())

    return data_save_directory


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, required=True)
    parser.add_argument("-pl", "--policy-name", type=str, required=True)
    parser.add_argument("-a", "--accept-trajectory-key", type=str, required=True)
    parser.add_argument("-n", "--num-trajectories", type=int, required=True)
    parser.add_argument("-t", "--num-timesteps", type=int, required=True)
    parser.add_argument("-d", "--data-save-directory", type=str, required=True)
    parser.add_argument("--save-all", action='store_true', default=False)
    parser.add_argument("--target-object", type=str, default="shed")
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument('--mult_reset', action='store_true')
    parser.add_argument('--num_reset', type=int, default=5)
    parser.add_argument('--num_sample', type=int, default=10)

    args = parser.parse_args()

    num_trajectories_per_thread = int(
        args.num_trajectories / args.num_parallel_threads)
    if args.num_trajectories % args.num_parallel_threads != 0:
        num_trajectories_per_thread += 1

    timestamp = get_timestamp()
    save_directory = get_data_save_directory(args)

    script_name = "scripted_collect.py"
    command = ['python',
               'scripts/{}'.format(script_name),
               '--policy-name={}'.format(args.policy_name),
               '-a{}'.format(args.accept_trajectory_key),
               '-e{}'.format(args.env),
               '-n {}'.format(num_trajectories_per_thread),
               '-t {}'.format(args.num_timesteps),
               '-o{}'.format(args.target_object),
               '-d{}'.format(save_directory),
               '--num_reset={}'.format(args.num_reset),
               '--num_sample={}'.format(args.num_sample),
               ]

    if args.save_all:
        command.append('--save-all')
    if args.mult_reset:
        command.append('--mult_reset')

    subprocesses = []
    for i in range(args.num_parallel_threads):
        subprocesses.append(subprocess.Popen(command))
        time.sleep(1)

    exit_codes = [p.wait() for p in subprocesses]

    merge_command = ['python',
                     'scripts/combine_trajectories.py',
                     '-d{}'.format(save_directory)]

    subprocess.call(merge_command)

