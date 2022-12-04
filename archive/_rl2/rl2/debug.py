from rl2.ctx import ctx_debug_flag

DEBUG = ctx_debug_flag.get()


def _p(x):
    if DEBUG:
        print(x)
