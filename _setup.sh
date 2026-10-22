#!/bin/bash
set -e
cd /root
rm -rf ana
mkdir -p ana
cd ana
python3 - <<'PY'
import zipfile
zipfile.ZipFile('/root/ana.zip').extractall('.')
print('extracted')
PY
ls
python3 -c "import sys; sys.path.insert(0,'.'); import agi; print('import ok')"
python3 tests/test_ana.py | tail -1
python3 -m agi check --preset xl | tail -1
python3 experiments/foundation_check.py | tail -1
