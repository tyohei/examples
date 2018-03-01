#!/usr/bin/env python3
import concurrent.futures
import time


def func(n, s):
    for i in range(1024 * 1024):
        n = n * 1.00001
    return n + s


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for i in range(100):
            futures.append(executor.submit(func, i, 3))
        for i in range(100):
            print(futures[i].result())


if __name__ == '__main__':
    main()
