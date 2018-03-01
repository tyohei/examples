#!/usr/bin/env python3
import concurrent.futures
import time


def func():
    print('func')
    time.sleep(1)


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        s_time = time.time()
        for i in range(10):
            executor.submit(func)
        print('Took {} [sec]'.format(time.time() - s_time))

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        s_time = time.time()
        for i in range(10):
            executor.submit(func)
        print('Took {} [sec]'.format(time.time() - s_time))


if __name__ == '__main__':
    main()
