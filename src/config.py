from functools import partial
import sys

print_flushed = partial(print, flush=True)

LOAD_OUTPUT = 0
RESULT_OUTPUT = 0
METHOD = 'NAIVEHDD'

if METHOD == 'KDTREE':
    sys.setrecursionlimit(10000)