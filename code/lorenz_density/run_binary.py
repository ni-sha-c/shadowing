# density/run_binary.py
import os
import sys
import json
import subprocess
from numpy import *

config = json.load(open(sys.argv[1]))
config = config[sys.argv[2]]
binary = config['binary']
params = list(config['parameters'])

procs = []
nIter = 256
gpus = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
numGpus = len(gpus.strip().splitlines())
for i in range(numGpus):
    p = subprocess.Popen(binary, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.stdin.write(array([i, random.randint(1 << 30)], uint32).tobytes())
    p.stdin.write(array(params, float32).tobytes())
    p.stdin.write(array(nIter, uint32).tobytes())
    p.stdin.flush()
    procs.append(p)

n = 2048
out = [frombuffer(p.stdout.read(n * n * 8), double) for p in procs]
histogram = sum(out, 0).reshape([n, n])
myname = subprocess.check_output('hostname').decode().strip()
params = ('_'.join(['{}'] * len(params))).format(*tuple(params))
fname = '{}_{}_{}_{}_{}.npy'.format(os.path.basename(binary),
        params, myname, nIter, random.rand())
save(fname, histogram)
