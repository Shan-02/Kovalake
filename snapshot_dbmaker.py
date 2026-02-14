# snapshot_dbmaker.py
# Imports a processed FM snapshot into SQLite (single responsibility)

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict

import pandas as pd

from Kovap import Kovap


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def sql_value(x):
    """Convert pandas/NaN values safely for SQLite."""
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def ensure_schema(conn: sqlite3.Connection, schema_path: Path) -> None:
    conn.executescript(schema_path.read_text(encoding="utf-8"))


def get_player_id_map(
    conn: sqlite3.Connection, df: pd.DataFrame
) -> Dict[str, int]:
    """Map FM UID -> player_id."""
    uids = [str(x) for x in df["UID"].tolist()]
    if not uids:
        return {}

    qmarks = ",".join(["?"] * len(uids))
    rows = conn.execute(
        f"""
        SELECT player_id, fm_uid
        FROM players
        WHERE fm_uid IN ({qmarks})
        """,
        uids,
    ).fetchall()

    return {fm_uid: player_id for player_id, fm_uid in rows}


# --------------------------------------------------
# Players
# --------------------------------------------------

def upsert_players(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    rows = [
        (
            str(r["UID"]),
            r.get("Name"),
            r.get("DoB"),
            r.get("dob_date"),
            r.get("Nat"),
            r.get("Preferred Foot"),
        )
        for _, r in df.iterrows()
    ]

    conn.executemany(
        """
        INSERT INTO players (fm_uid, name, dob, dob_date, nat, preferred_foot)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(fm_uid) DO UPDATE SET
          name=excluded.name,
          dob=excluded.dob,
          dob_date=excluded.dob_date,
          nat=excluded.nat,
          preferred_foot=excluded.preferred_foot
        """,
        rows,
    )


# --------------------------------------------------
# Snapshots
# --------------------------------------------------

def insert_snapshot(
    conn: sqlite3.Connection,
    snapshot_date: str,
    source_file: str,
    source_hash: str,
    snapshot_type: str = "player_search",
    notes: str | None = None,
) -> int:
    conn.execute(
        """
        INSERT INTO snapshots
        (snapshot_date, source_file, source_hash, snapshot_type, notes)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source_hash) DO NOTHING
        """,
        (snapshot_date, source_file, source_hash, snapshot_type, notes),
    )

    row = conn.execute(
        "SELECT snapshot_id FROM snapshots WHERE source_hash = ?",
        (source_hash,),
    ).fetchone()

    assert row is not None
    return int(row[0])


# --------------------------------------------------
# Snapshot stats
# --------------------------------------------------

def insert_snapshot_stats(
    conn: sqlite3.Connection,
    snapshot_id: int,
    df: pd.DataFrame,
    uid_to_pid: Dict[str, int],
) -> None:
    payload = []

    for _, r in df.iterrows():
        player_id = uid_to_pid.get(str(r["UID"]))
        if player_id is None:
            continue

        stats_json = json.dumps(
            {k: sql_value(v) for k, v in r.to_dict().items()},
            ensure_ascii=False,
        )

        payload.append(
            (
                snapshot_id,
                player_id,
                sql_value(r.get("Club")),
                sql_value(r.get("Division")),
                sql_value(r.get("Grade_Pos")),
                sql_value(r.get("comp_grade")),
                sql_value(r.get("comp_bucket")),
                sql_value(r.get("On Loan From")),
                sql_value(r.get("Position")),
                sql_value(r.get("Best Pos")),
                sql_value(r.get("Home-Grown Status")),
                sql_value(r.get("wage_weekly")),
                sql_value(r.get("wage_raw")),
                sql_value(r.get("transfer_value_min")),
                sql_value(r.get("transfer_value_max")),
                sql_value(r.get("transfer_value_avg")),
                sql_value(r.get("transfer_value_status")),
                sql_value(r.get("transfer_value_raw")),
                sql_value(r.get("Mins")),
                sql_value(r.get("age_years")),
                sql_value(r.get("Apps")),
                sql_value(r.get("Starts")),
                sql_value(r.get("Av Rat")),
                sql_value(r.get("xG")),
                sql_value(r.get("xG/90")),
                sql_value(r.get("xA")),
                sql_value(r.get("xA/90")),
                sql_value(r.get("Pas %")),
                sql_value(r.get("Shot %")),
                sql_value(r.get("Sv %")),
                sql_value(r.get("xSv %")),
                sql_value(r.get("npxg_plus_xa_per90")),
                sql_value(r.get("dist_per90_num")),
                sql_value(r.get("net_poss_per90")),
                sql_value(r.get("net_save_pct")),
                sql_value(r.get("def_actions_per_min")),
                sql_value(r.get("tackle_efficacy")),
                sql_value(r.get("aerial_efficacy")),
                sql_value(r.get("position_score")),
                sql_value(r.get("position_pct")),
                stats_json,
            )
        )

    conn.executemany(
        """
        INSERT OR REPLACE INTO player_snapshot_stats (
          snapshot_id, player_id,
          club, division, grade_pos, comp_grade, comp_bucket,
          on_loan_from, position, best_pos, home_grown_status,
          wage_weekly, wage_raw,
          transfer_value_min, transfer_value_max, transfer_value_avg,
          transfer_value_status, transfer_value_raw,
          mins, age_years, apps, starts,
          av_rat,
          xg, xg_per90, xa, xa_per90,
          pas_pct, shot_pct, sv_pct, xsv_pct,
          npxg_plus_xa_per90, dist_per90_num, net_poss_per90, net_save_pct,
          def_actions_per_min, tackle_efficacy, aerial_efficacy,
          position_score, position_pct,
          stats_json
        )
        VALUES (
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        payload,
    )


# --------------------------------------------------
# Grade scores (normalized table)
# --------------------------------------------------

def insert_grade_scores(
    conn: sqlite3.Connection,
    snapshot_id: int,
    df: pd.DataFrame,
    uid_to_pid: Dict[str, int],
) -> None:
    rows = []

    for _, r in df.iterrows():
        player_id = uid_to_pid.get(str(r["UID"]))
        if player_id is None:
            continue

        for col in df.columns:
            if col.endswith("_score") and col != "position_score":
                val = r[col]
                if pd.isna(val):
                    continue

                rows.append(
                    (
                        snapshot_id,
                        player_id,
                        col.replace("_score", ""),
                        float(val),
                    )
                )

    conn.executemany(
        """
        INSERT OR REPLACE INTO player_grade_scores
        (snapshot_id, player_id, grade_name, grade_score)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )


# --------------------------------------------------
# Main entry point
# --------------------------------------------------

def main() -> None:
    html = r"Betis\BetisMay24.html"
    snapshot_date = "2024-05-10"
    db_path = Path("data/kovalake.sqlite")
    schema_path = Path("schema.sql")

    asof = datetime.strptime(snapshot_date, "%Y-%m-%d").date()

    print("Processing snapshot...")
    proc = Kovap(html, asof_date=asof).process()
    df = proc.df
    assert df is not None

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        ensure_schema(conn, schema_path)

        print("Upserting players...")
        upsert_players(conn, df)

        snapshot_id = insert_snapshot(
            conn,
            snapshot_date=snapshot_date,
            source_file=proc.meta["source_file"],
            source_hash=proc.meta["source_hash"],
        )

        uid_to_pid = get_player_id_map(conn, df)

        print("Inserting snapshot stats...")
        insert_snapshot_stats(conn, snapshot_id, df, uid_to_pid)

        print("Inserting grade scores...")
        insert_grade_scores(conn, snapshot_id, df, uid_to_pid)

        conn.commit()
        print(f"✅ Snapshot {snapshot_id} imported ({len(df)} players)")

    finally:
        conn.close()


if __name__ == "__main__":
    main()