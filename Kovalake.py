# Kovalake.py
# Creates CSV and JSON exports of a single snapshot, for debugging data errors.

from Kovap import Kovap
from datetime import datetime

snapshot_date = "2024-05-10"  # in-game date
asof = datetime.strptime(snapshot_date, "%Y-%m-%d").date()

processor = Kovap(
    r"Betis\BetisMay24.html",
    dedupe_policy="max_mins",
    asof_date=asof,
)

processor.process()
print(processor.df.shape)
print(processor.meta)

paths = processor.export(out_dir="out", stem="V4")
print(paths)