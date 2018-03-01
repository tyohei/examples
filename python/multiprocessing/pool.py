import multiprocessing
import time


def task(i):
    time.sleep(10)
    print('DONE {}'.format(i))
    

def main():
    args = list(range(10000))
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(task, args)
        

if __name__ == '__main__':
    main()
