# Kovap.py
# Parses a Football Manager HTML export for use in Kovalake and snapshot_dbmaker.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from io import StringIO
from datetime import date, datetime

import hashlib
import re
import numpy as np
import pandas as pd

DIVISION_GRADE_MAP: dict[str, str] = {
    # AAA
    "LALIGA EA SPORTS": "AAA",
    "Serie A Enilive": "AAA",
    "Bundesliga": "AAA",
    "Premier League": "AAA",
    # A
    "Eredivisie": "A",
    "Brasileirão Assaí Série A": "A",
    "Sky Bet Championship": "A",
    "Liga Portugal Betclic": "A",
    "Trendyol Süper Lig": "A",
    "Ligue 1 McDonald's": "A",
    # BBB
    "Liga Profesional de Fútbol": "BBB",
    "Admiral Bundesliga": "BBB",
    "Raiffeisen Super League": "BBB",
    "Jupiler Pro League": "BBB",
    "Liga MX": "BBB",
    "Tinkoff Russian Premier League": "BBB",
    # CCC
    "3F Superliga": "CCC",
    "Chance Liga": "CCC",
    "Eliteserien": "CCC",
    "Allsvenskan": "CCC",
    "William Hill Premiership": "CCC",
    "Mozzart Bet SuperLiga": "CCC",
    "PKO Ekstraklasa": "CCC",
    "FavBet Liha": "CCC",
    "Meiji Yasuda J1 League": "CCC",
    "Major League Soccer": "CCC",
    "SuperSport Hrvatska nogometna liga": "CCC",
    "Stoiximan Super League": "CCC",
    # C
    "SuperLiga": "C",
    "LALIGA HYPERMOTION": "C",
    "Ligue 2 BKT": "C",
    "Sky Bet League One": "C",
    "Serie BKT": "C",
    "Hana 1Q K League 1": "C",
    "Campeonato AFP PlanVital": "C",
    "Liga BetPlay Dimayor": "C",
    "DSTV Premiership": "C",
    "OTP Bank Liga": "C",
    "Torneo Uruguayo Copa Coca-Cola": "C",
    "2. Bundesliga": "C",
    # D
    "Keuken Kampioen Divisie": "D",
    "Sky Bet League Two": "D",
    "Liga Portugal 2 Meu Super": "D",
    "Brasileirão Serie B Chevrolet": "D",
    "Isuzu UTE A-League": "D",
    "PrvaLiga Telemach": "D",
    "Challenger Pro League": "D",
    "SSE Airtricity League Premier Division": "D",
    "Enterprise National League": "D",
    "Ryman League Premier Division": "D",
    "Fortuna Liga": "D",
}

GRADE_WEIGHTS = {
    "Scoring": {
        "components": ["xG/90", "Gls/90", "xG-OP"],
        "weights": [0.6, 0.3, 0.1],
    },
    "Shooting": {
        "components": ["Shot/90", "Shot %", "xG/shot"],
        "weights": [0.3, 0.2, 0.5],
    },
    "Chance Creation": {
        "components": ["xA/90", "OP-KP/90", "Ch C/90"],
        "weights": [0.2, 0.4, 0.4],
    },
    "Passing": {
        "components": ["Ps C/90", "Pas %"],
        "weights": [0.4, 0.6],
    },
    "Progression": {
        "components": ["Pr passes/90", "prog_pass_eff"],
        "weights": [0.25, 0.75],
    },
    "Crossing": {
        "components": ["OP-Crs C/90", "OP-Cr %"],
        "weights": [0.4, 0.6],
    },
    "Dribbling": {
        "components": ["Drb/90"],
        "weights": [1.0],
    },
    "Work Rate": {
        "components": ["Sprints/90", "Dist/90"],
        "weights": [0.35, 0.65],
    },
    "Defensive Activity": {
        "components": ["def_actions_per_min", "Poss Won/90"],
        "weights": [0.5, 0.5],
    },
    "Tackling Quality": {
        "components": ["tackle_efficacy"],
        "weights": [1.0],
    },
    "Aerial Ability": {
        "components": ["aerial_efficacy"],
        "weights": [1.0],
    },
    "Pressing Effectiveness": {
        "components": ["pressing_eff"],
        "weights": [1.0],
    },
    "Box Defense": {
        "components": ["Shts Blckd/90", "Blk/90", "Clr/90"],
        "weights": [0.4, 0.4, 0.2],
    },
    "Net Possession": {
        "components": ["net_poss_per90"],
        "weights": [1.0],
    },
    "GK Shot Stopping": {
        "components": ["net_save_pct", "xGP/90"],
        "weights": [0.5, 0.5],
    },
    "GK Handling": {
        "components": ["Svh/90", "Svp/90", "Svt/90"],
        "weights": [0.5, 0.25, 0.25],
    },
}

POSITION_GRADE_WEIGHTS = {
    "Striker": {
        "Scoring": 0.3,
        "Shooting": 0.2,
        "Chance Creation": 0.15,
        "Passing": 0.05,
        "Progression": 0.05,
        "Dribbling": 0.1,
        "Work Rate": 0.05,
        "Aerial Ability": 0.1,
    },
    "Winger": {
        "Scoring": 0.15,
        "Shooting": 0.1,
        "Chance Creation": 0.35,
        "Progression": 0.1,
        "Crossing": 0.1,
        "Dribbling": 0.2,
    },
    "Attacking Mid": {
        "Scoring": 0.05,
        "Shooting": 0.1,
        "Chance Creation": 0.4,
        "Passing": 0.15,
        "Progression": 0.15,
        "Dribbling": 0.15,
    },
    "Midfielder": {
        "Scoring": 1 / 22,
        "Chance Creation": 2 / 22,
        "Passing": 3 / 22,
        "Progression": 3 / 22,
        "Dribbling": 2 / 22,
        "Work Rate": 3 / 22,
        "Defensive Activity": 2 / 22,
        "Tackling Quality": 2 / 22,
        "Aerial Ability": 1 / 22,
        "Pressing Effectiveness": 1 / 22,
        "Net Possession": 2 / 22,
    },
    "Full Back": {
        "Chance Creation": 2 / 24,
        "Passing": 2 / 24,
        "Progression": 3 / 24,
        "Crossing": 1 / 24,
        "Dribbling": 2 / 24,
        "Work Rate": 3 / 24,
        "Defensive Activity": 2 / 24,
        "Tackling Quality": 2 / 24,
        "Aerial Ability": 1 / 24,
        "Pressing Effectiveness": 2 / 24,
        "Box Defense": 1 / 24,
        "Net Possession": 3 / 24,
    },
    "Center Back": {
        "Chance Creation": 2 / 20,
        "Passing": 2 / 20,
        "Progression": 3 / 20,
        "Dribbling": 1 / 20,
        "Defensive Activity": 1 / 20,
        "Tackling Quality": 4 / 20,
        "Aerial Ability": 3 / 20,
        "Pressing Effectiveness": 1 / 20,
        "Box Defense": 1 / 20,
        "Net Possession": 2 / 20,
    },
    "Goalkeeper": {
        "Net Possession": 0.05,
        "Progression": 0.15,
        "GK Shot Stopping": 0.6,
        "GK Handling": 0.2,
    },
}

BEST_POS_TO_GRADE_POS = {
    "ST (C)": "Striker",
    "AM (L)": "Winger",
    "AM (R)": "Winger",
    "AM (C)": "Attacking Mid",
    "M (C)": "Midfielder",
    "DM": "Midfielder",
    "D (L)": "Full Back",
    "D (R)": "Full Back",
    "D (C)": "Center Back",
    "GK": "Goalkeeper",
}

COMP_BUCKET_MAP = {
    "AAA": "TOP",
    "A": "TOP",
    "BBB": "MID",
    "CCC": "MID",
    "C": "LOW",
    "D": "LOW",
}

@dataclass(frozen=True)
class TransferValueParsed:
    vmin: int | None
    vmax: int | None
    vavg: int | None
    status: str | None  # 'unknown', 'not_for_sale', or None


class Kovap:
    """
    Parse a Football Manager HTML export (single large table), clean/normalize fields,
    compute derived metrics, and export in app-friendly formats.

    Deduplication is enforced by UID (FM player unique id).
    """

    MISSING = {"", "—", "-", "N/A", "NaN", "None", None}

    _money_re = re.compile(
        r"""
        ^\s*
        (?P<cur>[£€$])?
        \s*
        (?P<num>(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)
        \s*
        (?P<suf>[KkMm])?
        \s*$
        """,
        re.VERBOSE,
    )

    def __init__(
        self,
        html_path: str | Path,
        *,
        table_pick: Literal["largest"] = "largest",
        dedupe_policy: Literal["max_mins", "first"] = "max_mins",
        asof_date: date | None = None,
    ) -> None:
        self.path = Path(html_path)
        self.table_pick = table_pick
        self.dedupe_policy = dedupe_policy
        self.asof_date = asof_date
        self.df_raw: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None
        self.meta: dict[str, Any] = {}

    # ----------  API ----------

    def load(self) -> "Kovap":
        html = self.path.read_text(encoding="utf-8", errors="replace")
        tables = pd.read_html(StringIO(html), flavor="lxml")
        if not tables:
            raise ValueError("No HTML tables found.")



        if self.table_pick == "largest":
            df = max(tables, key=len).copy()
        else:
            df = tables[0].copy()

        df.columns = [self._clean_header(c) for c in df.columns]
        df = df.dropna(how="all")

        self.df_raw = df
        return self
    
    def parse_fm_dob_to_date(self, dob_str: Any) -> date | None:
        """
        "29/7/1990 (33 years old)" -> date(1990, 7, 29)
        """
        if dob_str is None:
            return None
        s = str(dob_str).strip()
        if s in self.MISSING:
            return None

        # take the bit before the space + "("
        # "29/7/1990 (33 years old)" -> "29/7/1990"
        base = s.split("(", 1)[0].strip()

        try:
            # FM format here is d/m/YYYY (not US)
            dt = datetime.strptime(base, "%d/%m/%Y").date()
            return dt
        except Exception:
            return None
    
    def add_percentiles(
        self,
        df: pd.DataFrame,
        *,
        pos_col: str = "Best Pos",
        grade_col: str = "comp_grade",
        min_group_size: int = 20,
    ) -> pd.DataFrame:
        lower_is_better = {"Poss Lost/90"}

        percentile_cols = [
            "xG",
            "Pens S",
            "Pens Faced",
            "Pens",
            "Mins",
            "Gls/90",
            "Conc",
            "Gls",
            "Fls",
            "FA",
            "xG/90",
            "xG-OP",
            "xA/90",
            "xA",
            "Con/90",
            "Clean Sheets",
            "Av Rat",
            "Mins/Gl",
            "Ast",
            "Hdrs A",
            "Apps",
            "Tck/90",
            "Tck W",
            "Tck A",
            "Tck R",
            "Shot/90",
            "Shot %",
            "ShT/90",
            "ShT",
            "Shots Outside Box/90",
            "Shts Blckd/90",
            "Shts Blckd",
            "Shots",
            "Svt",
            "Svp",
            "Svh",
            "Sv %",
            "Pr passes/90",
            "Pr Passes",
            "Pres C/90",
            "Pres C",
            "Pres A/90",
            "Pres A",
            "Poss Won/90",
            "Poss Lost/90",
            "Ps C/90",
            "Ps C",
            "Ps A/90",
            "Pas A",
            "Pas %",
            "OP-KP/90",
            "OP-KP",
            "OP-Crs C/90",
            "OP-Crs C",
            "OP-Crs A/90",
            "OP-Crs A",
            "OP-Cr %",
            "Off",
            "Gl Mst",
            "K Tck/90",
            "K Tck",
            "K Ps/90",
            "K Pas",
            "K Hdrs/90",
            "Int/90",
            "Itc",
            "Sprints/90",
            "Hdr %",
            "Hdrs W/90",
            "Hdrs",
            "Hdrs L/90",
            "Goals Outside Box",
            "xSv %",
            "xGP/90",
            "xGP",
            "Drb/90",
            "Drb",
            "Dist/90",
            "Distance",
            "Cr C/90",
            "Cr C",
            "Crs A/90",
            "Cr A",
            "Cr C/A",
            "Conv %",
            "Clr/90",
            "Clear",
            "CCC",
            "Ch C/90",
            "Blk/90",
            "Blk",
            "Asts/90",
            "Aer A/90",
            "xG/shot",
            "dist_per90_num",
            "net_poss_per90",
            "net_save_pct",
            "def_actions_per_min",
            "tackle_efficacy",
            "aerial_efficacy",
            "prog_pass_eff",
            "pressing_eff",
        ]

        percentile_cols = [c for c in percentile_cols if c in df.columns]

        if pos_col not in df.columns:
            raise ValueError(f"Missing {pos_col} column for percentile grouping.")
        if grade_col not in df.columns:
            raise ValueError(f"Missing {grade_col} column for percentile grouping.")

        out = df.copy()

        for col in percentile_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        grouped = out.groupby([pos_col, grade_col], dropna=False)

        for col in percentile_cols:
            pct = grouped[col].rank(pct=True, method="average")
            group_sizes = grouped[col].transform("count")
            pct = pct.where(group_sizes >= min_group_size, np.nan)

            if col in lower_is_better:
                pct = 1.0 - pct

            out[f"{col}_pct"] = pct

        return out


    @staticmethod
    def age_years_decimal(dob: date, asof: date) -> float:
        """
        Returns age in years as a decimal using days/365.2425.
        """
        days = (asof - dob).days
        return days / 365.2425

    def process(self) -> "Kovap":
        if self.df_raw is None:
            self.load()

        assert self.df_raw is not None
        df = self.df_raw.copy()

        self.meta = {
            "source_file": str(self.path),
            "source_hash": self.sha256_file(self.path),
        }

        if self.asof_date is not None and "DoB" in df.columns:
            dob_dates = df["DoB"].map(self.parse_fm_dob_to_date)

            # Store real ISO dates like "1990-07-29"
            df["dob_date"] = dob_dates.map(lambda d: d.isoformat() if d else None)

            # Decimal age at snapshot date
            df["age_years"] = dob_dates.map(
                lambda d: self.age_years_decimal(d, self.asof_date) if d else np.nan
            )

        print(df["Dist/90"].head(20).tolist())
        print(df["Dist/90"].value_counts(dropna=False).head(10))

        # UID Check
        bad = df["UID"].isna().sum()
        print("UID NaNs after coercion:", bad, "out of", len(df))

        print(df.loc[df["UID"].isna(), "UID"].head(20))

        # Ensure UID exists and is numeric
        if "UID" not in df.columns:
            raise ValueError("Expected column 'UID' not found.")

        df["UID"] = pd.to_numeric(df["UID"], errors="coerce").astype("Int64")

        # Drop rows without UID
        df = df[df["UID"].notna()]

        # Remove blank Best Pos rows (matches your previous filtering intent)
        if "Best Pos" in df.columns:
            df = df[df["Best Pos"].notna()]
            df = df[df["Best Pos"].astype(str).str.strip() != ""]

        # Normalize Wage
        if "Wage" in df.columns:
            df["wage_raw"] = df["Wage"].astype(str)
            df["wage_weekly"] = df["Wage"].map(self.parse_wage_weekly).astype(
                "Int64"
            )

        # Normalize Transfer Value (min/max/avg/status)
        if "Transfer Value" in df.columns:
            df["transfer_value_raw"] = df["Transfer Value"].astype(str)
            parsed = df["Transfer Value"].map(self.parse_transfer_value)
            df["transfer_value_min"] = parsed.map(lambda p: p.vmin).astype(
                "Int64"
            )
            df["transfer_value_max"] = parsed.map(lambda p: p.vmax).astype(
                "Int64"
            )
            df["transfer_value_avg"] = parsed.map(lambda p: p.vavg).astype(
                "Int64"
            )
            df["transfer_value_status"] = parsed.map(lambda p: p.status)

        # Percent columns -> float
        for c in self._percent_columns(df):
            df[c] = df[c].map(self.parse_percent)

        # Normalize Best Pos
        if "Best Pos" in df.columns and "Position" in df.columns:
            df["Best Pos"] = [
                self.normalize_best_pos(bp, pos)
                for bp, pos in zip(df["Best Pos"], df["Position"])
            ]

        # Coerce numeric-ish columns safely
        text_cols = {
            "Name",
            "Position",
            "Best Pos",
            "Club",
            "Division",
            "Nat",
            "Home-Grown Status",
            "Preferred Foot",
            "On Loan From",
            "Inf",
        }

        if "Division" in df.columns:
            df["comp_grade"] = df["Division"].map(
                lambda d: DIVISION_GRADE_MAP.get(str(d).strip(), "D") if pd.notna(d) else "D"
            )
        else:
            df["comp_grade"] = "D"

        for col in df.columns:
            if col in text_cols:
                continue
            df[col] = self.coerce_numeric_series(df[col])

        # Derived columns (from your Power Query)
        df = self.add_derived_columns(df)
        df = self.dedupe_by_uid(df)
        df = self.add_percentiles(df, min_group_size=20)

        df["Grade_Pos"] = df["Best Pos"].map(BEST_POS_TO_GRADE_POS).fillna("Unknown")
        df["comp_bucket"] = df["comp_grade"].map(COMP_BUCKET_MAP).fillna("Unknown")

        df = Kovap.compute_grade_scores(df)
        df = Kovap.compute_position_score(df)
        df = Kovap.compute_position_percentile(df)

        df = df.copy()


        self.df = df
        return self

    def export(
        self,
        out_dir: str | Path = "out",
        *,
        stem: str | None = None,
        to_csv: bool = True,
        to_json: bool = True,
    ) -> dict[str, str]:
        """
        Exports processed dataframe. Returns dict of output paths.
        """
        if self.df is None:
            self.process()
        assert self.df is not None

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        name = stem or self.path.stem
        outputs: dict[str, str] = {}

        if to_csv:
            p = out / f"{name}.csv"
            self.df.to_csv(p, index=False, encoding="utf-8-sig")
            outputs["csv"] = str(p)

        if to_json:
            p = out / f"{name}.json"
            json_text = self.df.to_json(orient="records", force_ascii=False)
            p.write_text(json_text, encoding="utf-8")
            outputs["json"] = str(p)

        return outputs

    # ---------- Deduplication ----------

    def dedupe_by_uid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce one row per UID in this snapshot.
        Policy:
          - max_mins: keep the row with the highest Mins (if available)
          - first: keep the first occurrence
        """
        if df["UID"].isna().any():
            df = df[df["UID"].notna()]

        if self.dedupe_policy == "first" or "Mins" not in df.columns:
            return df.drop_duplicates(subset=["UID"], keep="first").reset_index(
                drop=True
            )

        # max_mins
        mins = pd.to_numeric(df["Mins"], errors="coerce").fillna(-1)
        df2 = df.copy()
        df2["_mins_sort"] = mins
        df2 = df2.sort_values(["UID", "_mins_sort"], ascending=[True, False])
        df2 = df2.drop_duplicates(subset=["UID"], keep="first")
        df2 = df2.drop(columns=["_mins_sort"])
        return df2.reset_index(drop=True)

    # ---------- Derived columns ----------

    def add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        def safe_div(a, b):
            return np.where((b == 0) | pd.isna(b), np.nan, a / b)
        
        # 90s = minutes / 90
        if "Mins" in df.columns:
            mins = pd.to_numeric(df["Mins"], errors="coerce")
            df["nineties"] = np.where(pd.isna(mins), np.nan, mins / 90.0)

        # npxG + xA per 90
        if "NP-xG/90" in df.columns and "xA/90" in df.columns:
            df["npxg_plus_xa_per90"] = pd.to_numeric(
                df["NP-xG/90"], errors="coerce"
            ) + pd.to_numeric(df["xA/90"], errors="coerce")

        # DistNum from Dist/90 like "10.2 km"
        if "Distance" in df.columns and "nineties" in df.columns:
            # parse numeric km from strings like "460.9km"
            dist_txt = df["Distance"].astype(str).str.replace(",", "", regex=False)
            distance_km = pd.to_numeric(
                dist_txt.str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0],
                errors="coerce",
            )

            nineties = pd.to_numeric(df["nineties"], errors="coerce")

            # km per 90 = total_km / (mins/90)
            dist_per90 = np.where(
                (nineties <= 0) | pd.isna(nineties) | pd.isna(distance_km),
                np.nan,
                distance_km / nineties,
            )

            # overwrite flawed raw "Dist/90" with computed numeric value
            df["Dist/90"] = dist_per90

            # keep your numeric helper too (optional)
            df["dist_per90_num"] = dist_per90

            df = df.drop(columns=["Distance"])

        # Progressive Pass Efficiency
        if "Pr passes/90" in df.columns and "Ps A/90" in df.columns:
            pr = pd.to_numeric(df["Pr passes/90"], errors="coerce")
            pa = pd.to_numeric(df["Ps A/90"], errors="coerce")

            df["prog_pass_eff"] = (pr + pa) / 2
        
        # Pressing Efficacy
        if "Pres A/90" in df.columns and "Pres C/90" in df.columns:
            pres_a = pd.to_numeric(df["Pres A/90"], errors="coerce")
            pres_c = pd.to_numeric(df["Pres C/90"], errors="coerce")

            df["pressing_eff"] = np.where(
                (pres_a <= 0) | pd.isna(pres_a) | pd.isna(pres_c),
                np.nan,
                (pres_c / pres_a) * pres_c,
            )
        # Net Poss per 90
        if "Poss Won/90" in df.columns and "Poss Lost/90" in df.columns:
            df["net_poss_per90"] = pd.to_numeric(
                df["Poss Won/90"], errors="coerce"
            ) - pd.to_numeric(df["Poss Lost/90"], errors="coerce")

        # Net Save %
        if "Sv %" in df.columns and "xSv %" in df.columns:
            df["net_save_pct"] = pd.to_numeric(
                df["Sv %"], errors="coerce"
            ) - pd.to_numeric(df["xSv %"], errors="coerce")

        # Defensive Actions per minute
        needed = {"Tck A", "Itc", "Blk", "Pres A", "Mins"}
        if needed.issubset(df.columns):
            num = (
                pd.to_numeric(df["Tck A"], errors="coerce")
                + pd.to_numeric(df["Itc"], errors="coerce")
                + pd.to_numeric(df["Blk"], errors="coerce")
                + pd.to_numeric(df["Pres A"], errors="coerce")
            )
            den = pd.to_numeric(df["Mins"], errors="coerce")
            df["def_actions_per_min"] = safe_div(num, den)

        # Tackle Efficacy
        needed = {"Tck W", "Mins", "Tck R"}
        if needed.issubset(df.columns):
            tck_w = pd.to_numeric(df["Tck W"], errors="coerce")
            mins = pd.to_numeric(df["Mins"], errors="coerce")

            # Tck R is often like "78%" -> parse as 78.0
            tck_r = df["Tck R"].map(self.parse_percent)

            df["tackle_efficacy"] = np.where(
                (mins == 0) | pd.isna(mins) | pd.isna(tck_w) | pd.isna(tck_r),
                np.nan,
                (tck_w / mins) * tck_r,
            )

        # Aerial Efficacy
        if "Hdrs W/90" in df.columns and "Hdr %" in df.columns:
            hdrs_w90 = pd.to_numeric(df["Hdrs W/90"], errors="coerce")
            hdr_pct = df["Hdr %"].map(self.parse_percent)

            df["aerial_efficacy"] = np.where(
                pd.isna(hdrs_w90) | pd.isna(hdr_pct),
                np.nan,
                hdrs_w90 * hdr_pct,
            )




        return df
    # ---------- Parsing helpers ----------

    @staticmethod
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _clean_header(c: Any) -> str:
        return str(c).strip().replace("\n", " ").replace("  ", " ")

    def parse_money_amount(self, s: str) -> int | None:
        if s is None:
            return None
        s = str(s).strip()
        if s in self.MISSING:
            return None

        m = self._money_re.match(s.replace(" ", ""))
        if not m:
            return None

        num = float(m.group("num").replace(",", ""))
        suf = (m.group("suf") or "").upper()

        mult = 1
        if suf == "K":
            mult = 1_000
        elif suf == "M":
            mult = 1_000_000

        return int(round(num * mult))

    def parse_wage_weekly(self, wage_raw: Any) -> int | None:
        if wage_raw is None:
            return None
        s = str(wage_raw).strip()
        if s in self.MISSING:
            return None

        s = (
            s.replace("p/w", "")
            .replace("p.w.", "")
            .replace("/w", "")
            .replace("per week", "")
            .strip()
        )
        return self.parse_money_amount(s)

    def parse_transfer_value(self, tv_raw: Any) -> TransferValueParsed:
        if tv_raw is None:
            return TransferValueParsed(None, None, None, None)

        s = str(tv_raw).strip()
        if s in self.MISSING:
            return TransferValueParsed(None, None, None, None)

        sl = s.lower()
        if "not for sale" in sl:
            return TransferValueParsed(None, None, None, "not_for_sale")
        if "unknown" in sl:
            return TransferValueParsed(None, None, None, "unknown")

        parts = [p.strip() for p in re.split(r"\s*[-–]\s*", s)]
        if len(parts) == 1:
            v = self.parse_money_amount(parts[0])
            return TransferValueParsed(v, v, v, None)

        if len(parts) == 2:
            v1 = self.parse_money_amount(parts[0])
            v2 = self.parse_money_amount(parts[1])
            if v1 is None or v2 is None:
                return TransferValueParsed(None, None, None, None)
            vmin, vmax = (v1, v2) if v1 <= v2 else (v2, v1)
            vavg = int(round((vmin + vmax) / 2))
            return TransferValueParsed(vmin, vmax, vavg, None)

        return TransferValueParsed(None, None, None, None)

    def parse_percent(self, x: Any) -> float | None:
        """
        Parses percent-like values into a decimal fraction.

        Examples:
        "78%"  -> 0.78
        "78"   -> 0.78   (treated as percent points)
        78     -> 0.78
        0.78   -> 0.78   (already decimal)
        """
        if x is None:
            return None

        s = str(x).strip()
        if s in self.MISSING:
            return None

        has_pct = "%" in s
        s = s.replace("%", "").strip()

        try:
            v = float(s)
        except Exception:
            return None

        # If it explicitly had a percent sign, interpret as percent points.
        if has_pct:
            return v / 100.0

        # If it's already in [0,1], assume it's a decimal.
        if 0.0 <= v <= 1.0:
            return v


        # Otherwise treat it as percent points.
        return v / 100.0

    def coerce_numeric_series(self, s: pd.Series) -> pd.Series:
        s2 = s.astype(str).str.strip()
        s2 = s2.replace(list(self.MISSING), np.nan)
        candidate = s2.str.replace(",", "", regex=False)

        as_num = pd.to_numeric(candidate, errors="coerce")
        non_null = candidate.notna().sum()
        numeric = as_num.notna().sum()

        if non_null > 0 and numeric / non_null > 0.9:
            return as_num
        return s

    def normalize_best_pos(self, best_pos: Any, position: Any) -> str | None:
        if best_pos is None:
            return None
        bp = str(best_pos)
        pos = "" if position is None else str(position)

        if ("M (L)" in bp) or ("WB (L)" in bp):
            return "AM (L)" if "AM" in pos else "D (L)"
        if ("M (R)" in bp) or ("WB (R)" in bp):
            return "AM (R)" if "AM" in pos else "D (R)"
        return bp if bp.strip() != "" else None

    @staticmethod
    def _percent_columns(df: pd.DataFrame) -> list[str]:
        cols = []
        for c in df.columns:
            if str(c).endswith("%"):
                cols.append(c)
        # ensure key ones included even if naming varies
        for c in ["Sv %", "Pas %", "Shot %", "Hdr %", "xSv %", "Tck R", "Cr C/A"]:
            if c in df.columns and c not in cols:
                cols.append(c)
        return cols
    
    @staticmethod
    def compute_grade_scores(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        for grade, spec in GRADE_WEIGHTS.items():
            components = spec["components"]
            weights = spec["weights"]
            score_col = f"{grade}_score"

            # initialise column
            out[score_col] = np.nan

            # which rows this grade applies to
            if grade.startswith("GK"):
                mask = out["Grade_Pos"] == "Goalkeeper"
            else:
                mask = out["Grade_Pos"] != "Goalkeeper"

            # build full-length score series
            values = []
            for stat, w in zip(components, weights):
                pct_col = f"{stat}_pct"
                if pct_col in out.columns:
                    values.append(out[pct_col] * w)

            if values:
                grade_score = np.nansum(values, axis=0)  # full-length
                out.loc[mask, score_col] = grade_score[mask]

        return out

    @staticmethod
    def compute_position_score(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # initialise column
        out["position_score"] = np.nan

        for role, weights in POSITION_GRADE_WEIGHTS.items():
            role_mask = out["Grade_Pos"] == role

            parts = []
            for grade, w in weights.items():
                col = f"{grade}_score"
                if col in out.columns:
                    parts.append(out[col] * w)

            if parts:
                role_score = np.nansum(parts, axis=0)  # full-length array
                out.loc[role_mask, "position_score"] = role_score[role_mask]

        return out
    
    @staticmethod
    def compute_position_percentile(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["position_pct"] = (
            out
            .groupby(["Grade_Pos", "comp_bucket"])["position_score"]
            .rank(pct=True, method="average")
        )

        return out