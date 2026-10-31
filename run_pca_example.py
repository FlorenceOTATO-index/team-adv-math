"""Backward-compatible entry point for Person 2's experiment.

Use ``python scripts/run_person2.py`` for the full documented workflow.  This
file remains so the original repository command still runs the shared raw and
PCA evaluations, and it now saves raw latents for Person 5's t-SNE work.
"""

from scripts.run_person2 import main


if __name__ == "__main__":
    raise SystemExit(main())
