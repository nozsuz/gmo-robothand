"""Fetch the Shadow Hand model from MuJoCo Menagerie (Apache-2.0) into third_party/.

    uv run python scripts/fetch_menagerie.py
"""

from __future__ import annotations

import subprocess

from robothand_sim.paths import MENAGERIE_DIR, SHADOW_RIGHT_HAND_XML

REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "0059d4335f8156206f63a35662313385f7ad6d74"  # pinned for reproducibility


def git(*args: str) -> None:
    subprocess.run(["git", "-C", str(MENAGERIE_DIR), *args], check=True)


def main() -> None:
    if SHADOW_RIGHT_HAND_XML.exists():
        print(f"already present: {SHADOW_RIGHT_HAND_XML}")
        return
    MENAGERIE_DIR.mkdir(parents=True, exist_ok=True)
    git("init", "-q")
    git("remote", "add", "origin", REPO)
    git("config", "core.sparseCheckout", "true")
    (MENAGERIE_DIR / ".git" / "info" / "sparse-checkout").write_text("shadow_hand/\n")
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    git("checkout", "-q", "FETCH_HEAD")
    print(f"fetched {SHADOW_RIGHT_HAND_XML}")


if __name__ == "__main__":
    main()
