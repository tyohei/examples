import chainer
import numpy as np

import models


def main():
    model = models.load_resnet50()

    x = np.zeros((1, 3, model.insize, model.insize), dtype=np.float32)
    t = np.zeros((1,), dtype=np.int32)
    y = model(x, t)
    y.backward()

    chainer.serializers.save_npz('resnet50.npz', model)


if __name__ == '__main__':
    main()
