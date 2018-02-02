import inspect


def func_f():
    return func_g()


def func_g():
    return func_h()


def func_h():
    return func_i()


def func_i():
    return get_frames(), get_outerframes()


def get_frames():
    frames = []
    frames.append(inspect.currentframe())
    frames.append(inspect.currentframe().f_back)
    frames.append(inspect.currentframe().f_back.f_back)
    frames.append(inspect.currentframe().f_back.f_back.f_back)
    frames.append(inspect.currentframe().f_back.f_back.f_back.f_back)
    return [inspect.getframeinfo(f) for f in frames]


def get_outerframes():
    return inspect.getouterframes(inspect.currentframe())


def main():
    stack_frames = func_f()
    for f0, f1 in zip(stack_frames[0], stack_frames[1]):
        print(f0)  # -> List of named tuples Traceback(...)
        print(f1)  # -> List of named tuples FrameInfo(...)
    """
    The deepest stack frame is on the head of the list of stack frames.
    """


if __name__ == '__main__':
    main()
