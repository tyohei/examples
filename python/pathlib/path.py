import pathlib


def ls(root: str):
    path = pathlib.Path(root)
    return list(path.iterdir())


def ls_abs(root: str):
    path = pathlib.Path(root)
    head = list(path.iterdir())[0]
    return pathlib.Path(head).absolute()


def main():
    print(ls('.'))  # -> PosixPath
    print(ls_abs('.'))  # -> str


if __name__ == '__main__':
    main()
