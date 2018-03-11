import numpy as np

import test


def main():
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    arr3 = test.add_array(arr1, arr2)
    print(arr3)


if __name__ == '__main__':
    main()
