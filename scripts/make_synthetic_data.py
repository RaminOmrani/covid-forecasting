"""Write the synthetic stand-in series to data/synthetic_covid.csv."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from covidfc.data import make_synthetic  # noqa: E402

path = ROOT / "data" / "synthetic_covid.csv"
make_synthetic().to_csv(path, index=False)
print(f"wrote {path}")
