import chainer.functions as F
import chainer.links as L
import requests


class MLP(object):
    def __init__(self):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(2, 3)
            self.l2 = L.Linear(3, 2)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        return F.relu(self.l2(h1))


def _load(url):
    print(f'Downloading file from {url}...')
    page = requests.get(url)
    exec(page.text, globals())


def load_resnet50():
    url = 'https://raw.githubusercontent.com/chainer/chainer/master/examples/imagenet/resnet50.py'  # NOQA
    _load(url)
    return ResNet50()  # NOQA


def load_googlenet():
    url = 'https://raw.githubusercontent.com/chainer/chainer/master/examples/imagenet/googlenet.py'  # NOQA
    _load(url)
    return GoogLeNet()  # NOQA


def load_vgg():
    url = 'https://raw.githubusercontent.com/chainer/chainer/master/examples/cifar/models/VGG.py'  # NOQA
    _load(url)
    return VGG()  # NOQA


def load_mlp():
    return MLP()
