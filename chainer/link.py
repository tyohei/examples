import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


class SubMLP(chainer.Chain):

    def __init__(self):
        super(SubMLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(10, 10)

    def __call__(self, x):
        return self.l1(x)


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
        print(self.l1)
        print(self.l2)
        print(self.l3)
        return self.l3(h2)


def main():
    model = MLP()
    x = np.zeros((10, 10), dtype=np.float32) 
    y = model(chainer.Variable(x)) 
    print(y.creator)         # object of LinearFunction(FunctionNode)
    print(y.creator.inputs)  # (VariableNode, VariableNode, VariableNode)
    print(y.creator.inputs[0])  # ``inputs`` is tuple
    print(y.creator.inputs[1])
    print(y.creator.inputs[2])
    print(y.creator.inputs[0].label)
    print(y.creator.inputs[1].label)
    print(y.creator.inputs[2].label)
    print(y.creator.inputs[0].get_variable())  # x -> h2
    print(y.creator.inputs[1].get_variable())  # W
    print(y.creator.inputs[2].get_variable())  # b
    print('================')
    print(model.l3)
    print(len(list(model.l3.params())))      # 2 (W and b)
    print(type(list(model.l3.params())[0]))  # Parameter
    print(type(list(model.l3.params())[1]))  # Parameter
    print(list(model.l3.params())[0])  # Parameter
    print(list(model.l3.params())[1])  # Parameter
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
    print(h2.creator.inputs[0].get_variable())  # self.l2(h1) -> Variable(None)
    print(h2.creator.inputs[0].get_variable().creator)
    print(h2.creator.inputs[0].get_variable().creator.inputs)
    print(len(h2.creator.inputs[0].get_variable().creator.inputs))  # 3
    print(h2.creator.inputs[0].get_variable().creator.inputs[0].get_variable())  # x
    print(h2.creator.inputs[0].get_variable().creator.inputs[1].get_variable())  # W
    print(h2.creator.inputs[0].get_variable().creator.inputs[2].get_variable())  # b
    W = h2.creator.inputs[0].get_variable().creator.inputs[1].get_variable()
    b = h2.creator.inputs[0].get_variable().creator.inputs[2].get_variable()
    W_ = list(model.l2.params())[0]
    b_ = list(model.l2.params())[1]
    print(W is W_)  # True (sometimes False)
    print(b is b_)  # True (sometimes False)


    links = list(model.namedlinks())
    print(links)

if __name__ == "__main__":
    main()
