from contextlib import contextmanager


@contextmanager
def cd(newdir):
    import os

    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
