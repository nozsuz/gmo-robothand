"""Train the Diffusion Policy, then evaluate it in closed loop (one job, e.g. one GPU kernel).

    uv run python scripts/train_eval.py --demos holder --steps 30000 --eval --policy holder --episodes 50
    uv run python kaggle/remote.py --slug rh-train --gpu --source rh-demos scripts/train_eval.py -- \
        --demos holder --eval --policy holder
Arguments before --eval go to train_policy.py, the rest to eval_policy.py.
"""

import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
scripts = here if (here / "train_policy.py").exists() else here / "scripts"  # repo or Kaggle job copy
args = sys.argv[1:]
cut = args.index("--eval") if "--eval" in args else len(args)
subprocess.run([sys.executable, str(scripts / "train_policy.py"), *args[:cut]], check=True)
subprocess.run([sys.executable, str(scripts / "eval_policy.py"), *args[cut + 1:]], check=True)
