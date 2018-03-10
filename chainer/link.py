import chainer
import chainer.functions as F
import chainer.links as L
from collections import OrderedDict
import numpy as np

from googlenetbn import GoogLeNetBN


class SubMLP(chainer.Chain):

    def __init__(self):
        super(SubMLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(10, 10)
            self.conv1 = L.Convolution2D(1, 1, 1)

    def __call__(self, x):
        h = self.l1(x)
        h = h[np.newaxis, np.newaxis, :, :]
        h = self.conv1(h)
        h = h[0, 0, :, :]
        return h


class MLP(chainer.Chain):

    def __init__(self):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l0 = SubMLP()
            self.l1 = L.Linear(10, 100)
            self.l2 = L.Linear(100, 100)
            self.l3 = L.Linear(100, 10)

    def __call__(self, x):
        h0 = F.relu(self.l0(x))
        h1 = F.relu(self.l1(h0))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

    def _secret(self):
        print('Secret called')


def _kfac_backward(link, backward_main, retain_grad=True,
                   enable_double_backprop=False, loss_scale=None):
    with chainer.using_config('enable_backprop', enable_double_backprop):
        # To obtain grads, we need to edit the origianl file (`variable.py`)
        grads = backward_main(retain_grad, loss_scale)

    namedparams = list(link.namedparams())

    def get_linkname(param):
        for _name, _param in namedparams:
            if param is _param:
                return _name[:_name.rfind('/')]
        return None

    data = {}
    for node, grad in grads.items():
        creator_node = node.creator_node  # parent function node
        if creator_node is not None:  # ignore leaf node
            if getattr(creator_node, '_input_indexes_to_retain') is not None:
                a, param = creator_node.get_retained_inputs()
                linkname = get_linkname(param)
                if linkname is not None:
                    # params that its linkname is None, are output layer (e.g.
                    # softmax layer). These layers do not have laernable
                    # param inside.
                    data[linkname] = (creator_node.rank, a.data, grad.data)
    data = OrderedDict(sorted(data.items(), key=lambda x: x[1][0]))
    return data


def stack_grad(link, data):

    def get_param(path):
        for _name, _param in link.namedparams():
            if _name == path:
                return _param
        return None

    for linkname in data.keys():
        param_W = get_param(linkname + '/W')
        param_b = get_param(linkname + '/b')
        if param_W is None or param_b is None:
            if param_W is not None:
                pass
                #print('b is None...' + linkname)
            elif param_b is not None:
                print('W is None...' + linkname)
            else:
                pass
                #print('W and b is None...' + linkname)
        else:
            print('No None!...' + linkname)
    return


def main():
    model = MLP()
    x = np.ones((10, 10), dtype=np.float32) 
    y = model(chainer.Variable(x)) 
    print(y.creator_node)
    print(y.creator)         # object of LinearFunction(FunctionNode)
    print(y.creator.inputs)  # (VariableNode, VariableNode, VariableNode)
    print(y.creator.inputs[0])  # ``inputs`` is tuple
    print(y.creator.inputs[1])
    print(y.creator.inputs[2])
    print(y.creator.inputs[0].label)
    print(y.creator.inputs[1].label)
    print(y.creator.inputs[2].label)
    # print(y.creator.inputs[0].get_variable())  # x -> h2
    # print(y.creator.inputs[1].get_variable())  # W
    # print(y.creator.inputs[2].get_variable())  # b
    print('================')
    print(model.l3)
    print(len(list(model.l3.params())))      # 2 (W and b)
    print(type(list(model.l3.params())[0]))  # Parameter
    print(type(list(model.l3.params())[1]))  # Parameter
    # print(list(model.l3.params())[0])  # Parameter
    # print(list(model.l3.params())[1])  # Parameter
    print('================')
    W = y.creator.inputs[1].get_variable()
    b = y.creator.inputs[2].get_variable()
    W_ = list(model.l3.params())[0]
    b_ = list(model.l3.params())[1]
    print(W is W_)  # True (sometimes False)
    print(b is b_)  # True (sometimes False)

    h2 = y.creator.inputs[0].get_variable()
    print(h2.creator)  # chainer.functions.activation.relu.ReLU(FunctionNode)
    print(len(h2.creator.inputs))  # 1
    print(h2.creator.inputs[0])
    # print(h2.creator.inputs[0].get_variable())  # self.l2(h1) -> Variable(None)
    print(h2.creator.inputs[0].get_variable().creator)
    print(h2.creator.inputs[0].get_variable().creator.inputs)
    print(len(h2.creator.inputs[0].get_variable().creator.inputs))  # 3
    # print(h2.creator.inputs[0].get_variable().creator.inputs[0].get_variable())  # x
    # print(h2.creator.inputs[0].get_variable().creator.inputs[1].get_variable())  # W
    # print(h2.creator.inputs[0].get_variable().creator.inputs[2].get_variable())  # b
    W = h2.creator.inputs[0].get_variable().creator.inputs[1].get_variable()
    b = h2.creator.inputs[0].get_variable().creator.inputs[2].get_variable()
    W_ = list(model.l2.params())[0]
    b_ = list(model.l2.params())[1]
    print(W is W_)  # True (sometimes False)
    print(b is b_)  # True (sometimes False)
    print('================')
    links = list(model.namedlinks())
    print(links)
    print('================')

    l1 = L.Linear(2, 1, initialW=np.array([[0.5, 0.5]]))
    l2 = L.Linear(1, 1, initialW=np.array([0.5]))
    x = chainer.Variable(np.array([[3, 4],], dtype=np.float32))
    y = l1(x)
    z = l2(y)
    z.backward(retain_grad=True)
    print(z)
    print(z.creator_node)  # 
    print(z.creator_node.get_retained_inputs())  # y and l2.W
    print(z.creator_node.get_retained_inputs()[0])
    print(z.creator_node.get_retained_inputs()[1])
    print(z.creator_node.get_retained_inputs()[0] is y)    # True
    print(z.creator_node.get_retained_inputs()[1] is l2.W) # True
    print(z.creator_node.inputs)
    print(z.creator_node.inputs[0])
    print(z.creator_node.inputs[1])
    print(z.creator_node.inputs[2])
    print(z.creator_node.inputs[0].get_variable() is y)  # True

    print('================')
    batchsize = 1
    c1 = L.Convolution2D(3, 2, 2)
    b1 = L.BatchNormalization(2)
    x = chainer.Variable(np.ones((batchsize, 3, 4, 4), dtype=np.float32))
    y1 = c1(x)  # y.backward() is not valid since its not 1 dim
    y2 = b1(y1)
    z = F.sum(y2)
    z.backward(retain_grad=True)
    print(z.rank)
    print(y.rank)
    print(x.rank)
    print(z)
    print(z.creator_node.inputs[0].get_variable() is y2)  # True
    print(z.creator_node.inputs[0].get_variable().creator_node.inputs[0].get_variable() is y1)  # True
    print(z.creator_node.inputs[0])
    print(z.creator_node.inputs[0].__class__ is chainer.variable.VariableNode)

    print('================')
    model = MLP()
    x = chainer.Variable(np.ones((batchsize, 10), dtype=np.float32)) 
    y = model(x) 
    z = F.sum(y)
    backward_main = getattr(z, '_backward_main')
    print('================')
    data = _kfac_backward(model, backward_main)
    for param_name, (rank, a, g) in data.items():
        print(param_name, rank)

    print(z.rank)
    print(y.rank)
    print(x.rank)
    print(z.grad)

    print(y.creator_node.inputs[1].get_variable() is model.l3.W)  # True
    for param in model.params():
        print(param.rank)
    print(y.creator_node)
    print(y.creator)
    print(y.creator_node.inputs[0].get_variable().\
          creator_node.inputs[0].get_variable().\
          creator_node.inputs[1].get_variable() is model.l2.W)  # True
    rank9 = y.creator_node.inputs[0].get_variable()
    rank8 = rank9.creator_node.inputs[0].get_variable()
    rank7 = rank8.creator_node.inputs[0].get_variable()
    rank6 = rank7.creator_node.inputs[0].get_variable()
    rank5 = rank6.creator_node.inputs[0].get_variable()
    print(rank9.grad)
    print(rank8.grad)
    print(rank7.grad)
    print(rank6.grad)
    print(rank5.grad)
    print(model.l3.W.__class__)

    def check(v, link):
        for param_name, param in link.namedparams():
            if v is param:
                return param_name, param
        return None, None
    print(check(y, model))
    print(check(y.creator_node.inputs[0].get_variable(), model)[0])
    print(check(y.creator_node.inputs[1].get_variable(), model)[0])
    print(check(y.creator_node.inputs[2].get_variable(), model)[0])

    print('================')
    googlenetbn = GoogLeNetBN()
    x = np.zeros((1, 3, googlenetbn.insize, googlenetbn.insize), dtype=np.float32)
    t = np.zeros((1,), dtype=np.int32)
    y = googlenetbn(x, t)

    backward_main = getattr(y, '_backward_main')
    data = _kfac_backward(googlenetbn, backward_main)
    stack_grad(googlenetbn, data)
    # print(y.rank)  # 113


if __name__ == "__main__":
    main()
