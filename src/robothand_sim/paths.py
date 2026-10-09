from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MENAGERIE_DIR = REPO_ROOT / "third_party" / "mujoco_menagerie"
SHADOW_RIGHT_HAND_XML = MENAGERIE_DIR / "shadow_hand" / "right_hand.xml"
BRAINCO_DIR = REPO_ROOT / "third_party" / "brainco-description"
REVO2_RIGHT_XML = BRAINCO_DIR / "revo2_system" / "mjcf" / "revo2_right.xml"
OUTPUT_DIR = REPO_ROOT / "outputs"
