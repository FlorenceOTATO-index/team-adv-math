"""Merge the per-method result files into results/results.csv.

Each person can safely write ``results/results_<method>.csv`` without editing
the same file during parallel work.  Person 5 runs this script after new
method results arrive and uses the generated aggregate file for plotting.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from evaluate import load_all_results


def main() -> int:
    config_path = Path(config.RESULTS_DIR)
    config_path.mkdir(parents=True, exist_ok=True)
    output_path = config_path / "results.csv"
    table = load_all_results()
    table.to_csv(output_path, index=False)
    print(f"Merged {len(table)} rows into {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
