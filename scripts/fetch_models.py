"""Fetch the hand models into third_party/ (pinned commits, sparse checkouts).

* Shadow Hand: MuJoCo Menagerie (Apache-2.0)
* BrainCo Revo 2: brainco-description (license not yet published upstream; fetched for local
  simulation only and not redistributed in this repository)

    uv run python scripts/fetch_models.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from robothand_sim.paths import BRAINCO_DIR, MENAGERIE_DIR, REVO2_RIGHT_XML, SHADOW_RIGHT_HAND_XML

SOURCES = [
    # (target dir, repository, commit, sparse path, file that marks a complete checkout)
    (MENAGERIE_DIR, "https://github.com/google-deepmind/mujoco_menagerie.git",
     "0059d4335f8156206f63a35662313385f7ad6d74", "shadow_hand/", SHADOW_RIGHT_HAND_XML),
    (BRAINCO_DIR, "https://github.com/BrainCoTech/brainco-description.git",
     "f197d9d720b1b106a3298e2438ff07e055bf15ec", "revo2_system/", REVO2_RIGHT_XML),
]


def fetch(target: Path, repo: str, commit: str, sparse: str, marker: Path) -> None:
    if marker.exists():
        print(f"already present: {marker}")
        return

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(target), *args], check=True)

    target.mkdir(parents=True, exist_ok=True)
    git("init", "-q")
    git("remote", "add", "origin", repo)
    git("config", "core.sparseCheckout", "true")
    (target / ".git" / "info" / "sparse-checkout").write_text(sparse + "\n")
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
    git("checkout", "-q", "FETCH_HEAD")
    print(f"fetched {marker}")


if __name__ == "__main__":
    for source in SOURCES:
        fetch(*source)
