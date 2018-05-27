import argparse
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def get_times(fname):
    start = False
    times = collections.defaultdict(list)
    with open(fname, 'r') as f:
        for line in f:
            if '====' in line:
                start = True
                continue
            if not start:
                continue
            if 'DONE' in line:
                break
            lst = line.split(' ')  # split by space
            mebibytes = int(lst[1])
            sec = float(lst[4])
            times[mebibytes].append(sec)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log', nargs='+')
    args = parser.parse_args()

    ax = plt.subplot()
    for log in args.log: 
        times = get_times(log)
        x = sorted(times.keys())
        y = []
        for k, v in times.items():
            seconds = np.array(v)
            y.append(seconds.mean())
        if '8170' in log:
            label = 'ncclReduce'
        if '8171' in log:
            label = 'ncclAllReduce'
        ax.plot(x, y, label=label)

    ax.set_xlabel('MiB')
    ax.set_ylabel('second')
    ax.legend()
    plt.savefig('plot.pdf')


if __name__ == '__main__':
    main()
