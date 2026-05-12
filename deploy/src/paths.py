from pathlib import Path

# Root of the real_g1 package (one level up from this file's directory)
REAL_G1_ROOT = Path(__file__).resolve().parents[1]

# Centralized directory for data assets (ckpts, data, g1, plot, visuals, npy logs)
ASSETS_DIR = REAL_G1_ROOT / "assets"

# Ensure the assets directory exists when imported in scripts that write outputs
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def to_assets_path(rel: str | Path) -> Path:
    """Return an absolute path under ASSETS_DIR for a given relative path."""
    p = Path(rel)
    return p if p.is_absolute() else (ASSETS_DIR / p)

