from __future__ import annotations

from pathlib import Path

import pandas as pd

from Kovap import Kovap


def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else (n / d) * 100.0


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(df)

    rows = []
    for col in df.columns:
        s = df[col]
        nulls = int(s.isna().sum())
        non_null = n_rows - nulls

        # For object columns, also treat blank strings as missing-ish
        blanks = 0
        if s.dtype == "object":
            blanks = int(s.astype(str).str.strip().eq("").sum())

        distinct = int(s.nunique(dropna=True))

        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "rows": n_rows,
                "nulls": nulls,
                "null_rate_%": round(pct(nulls, n_rows), 2),
                "blanks": blanks,
                "blank_rate_%": round(pct(blanks, n_rows), 2),
                "non_null": non_null,
                "distinct_non_null": distinct,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["null_rate_%", "blank_rate_%"], ascending=False
    )


def add_parse_error_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a few domain-specific checks:
    - wage_raw present but wage_weekly missing
    - transfer_value_raw present but parsed min/max missing
    - percent strings present but parsed value missing (if you kept raw)
    """
    n = len(df)
    checks = []

    def add_check(name: str, mask: pd.Series) -> None:
        bad = int(mask.sum())
        checks.append(
            {
                "check": name,
                "bad_rows": bad,
                "bad_rate_%": round(pct(bad, n), 2),
            }
        )

    # Wage parse failures
    if "wage_raw" in df.columns and "wage_weekly" in df.columns:
        raw_present = df["wage_raw"].notna() & df["wage_raw"].astype(str).str.strip().ne(
            ""
        )
        parsed_missing = df["wage_weekly"].isna()
        add_check("wage_raw_present_but_wage_weekly_missing", raw_present & parsed_missing)

    # Transfer value parse failures (ignoring Unknown/Not for Sale statuses)
    if (
        "transfer_value_raw" in df.columns
        and "transfer_value_min" in df.columns
        and "transfer_value_status" in df.columns
    ):
        raw_present = (
            df["transfer_value_raw"].notna()
            & df["transfer_value_raw"].astype(str).str.strip().ne("")
        )
        status_ok = df["transfer_value_status"].isna()  # only count failures where status isn't unknown/not_for_sale
        parsed_missing = df["transfer_value_min"].isna()
        add_check(
            "transfer_value_raw_present_but_transfer_value_min_missing_(excluding_unknown/not_for_sale)",
            raw_present & status_ok & parsed_missing,
        )

    return pd.DataFrame(checks).sort_values("bad_rate_%", ascending=False)


def main() -> None:
    html_path = r"Betis\BetisMay24.html"
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    proc = Kovap(html_path).process()
    df = proc.df
    assert df is not None

    # Basic sanity
    print("Processed shape:", df.shape)
    print("UID unique:", df["UID"].nunique(dropna=True) if "UID" in df.columns else "N/A")

    # Column-level null report
    col_report = quality_report(df)
    print("\nTop 25 columns by null rate:")
    print(col_report.head(25).to_string(index=False))

    # Domain parse checks
    checks = add_parse_error_checks(df)
    if len(checks) > 0:
        print("\nParse checks:")
        print(checks.to_string(index=False))

    # Save reports
    col_report.to_csv(out_dir / "quality_columns.csv", index=False, encoding="utf-8-sig")
    if len(checks) > 0:
        checks.to_csv(out_dir / "quality_checks.csv", index=False, encoding="utf-8-sig")

    # Flag “problem columns” (tune thresholds)
    problem_cols = col_report[
        (col_report["null_rate_%"] >= 30.0)
        | (col_report["blank_rate_%"] >= 10.0)
    ]
    problem_cols.to_csv(
        out_dir / "quality_problem_columns.csv", index=False, encoding="utf-8-sig"
    )

    print("\nWrote:")
    print(" - out/quality_columns.csv")
    print(" - out/quality_checks.csv (if any)")
    print(" - out/quality_problem_columns.csv")


if __name__ == "__main__":
    main()