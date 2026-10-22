#!/bin/bash
cd /root/ana
pkill -f p29_walk_par || true
pkill -f dog_evolve || true
sleep 2
cat > /root/ana/_seed_once.py <<'PY'
import json, os, sys
sys.path.insert(0, '.')
import numpy as np
from experiments.p27_walk_sweep import one

rng = np.random.default_rng(7)
best = None
for i in range(40):
    kw = dict(seed=int(rng.integers(1, 10**6)),
              k_near=int(rng.integers(4, 25)),
              region_links=int(rng.integers(0, 129)),
              w_lo=float(rng.uniform(0.02, 0.5)),
              w_hi=float(rng.uniform(0.3, 2.0)))
    r = one(kw, 100)
    if r is None:
        continue
    dx = abs(r['dx'])
    if best is None or dx > best[0]:
        best = (dx, kw)
        print('#%d 新高 %.4f' % (i + 1, dx), flush=True)
print('best %.4f' % best[0])
json.dump({'dx': best[0], 'fallen': False, 'kw': best[1]},
          open('walk_best.json', 'w'), indent=2)
print('written walk_best.json')
PY
export ANA_GO2_XML=/root/go2models/scene.xml
timeout 600 python3 _seed_once.py | tail -5
