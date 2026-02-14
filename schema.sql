-- schema.sql
PRAGMA foreign_keys = ON;

--------------------------------------------------
-- PLAYERS (stable identity across snapshots)
--------------------------------------------------
CREATE TABLE players (
  player_id INTEGER PRIMARY KEY,
  fm_uid TEXT NOT NULL UNIQUE,              -- FM unique player ID
  name TEXT,
  dob TEXT,                                 -- raw FM DoB string
  dob_date TEXT,                            -- ISO date YYYY-MM-DD
  nat TEXT,
  preferred_foot TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);

--------------------------------------------------
-- SNAPSHOTS (one per import)
--------------------------------------------------
CREATE TABLE snapshots (
  snapshot_id INTEGER PRIMARY KEY,
  snapshot_date TEXT NOT NULL,              -- in-game date (ISO)
  source_file TEXT NOT NULL,
  source_hash TEXT NOT NULL UNIQUE,          -- prevents duplicate imports
  snapshot_type TEXT DEFAULT 'player_search',
  notes TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);

--------------------------------------------------
-- PLAYER SNAPSHOT STATS
-- (raw + derived metrics that vary per snapshot)
--------------------------------------------------
CREATE TABLE player_snapshot_stats (
  snapshot_id INTEGER NOT NULL,
  player_id INTEGER NOT NULL,

  -- Context / grouping
  club TEXT,
  division TEXT,
  grade_pos TEXT,                           -- Striker, Winger, etc
  comp_grade TEXT,                          -- AAA, A, BBB, CCC, C, D
  comp_bucket TEXT,                         -- TOP / MID / LOW
  on_loan_from TEXT,
  position TEXT,
  best_pos TEXT,
  home_grown_status TEXT,

  -- Contract / value
  wage_weekly INTEGER,
  wage_raw TEXT,
  transfer_value_min INTEGER,
  transfer_value_max INTEGER,
  transfer_value_avg INTEGER,
  transfer_value_status TEXT,
  transfer_value_raw TEXT,

  -- Playing time / demographics
  mins INTEGER,
  age_years REAL,
  apps INTEGER,
  starts INTEGER,

  -- Core stats (selected numeric fields)
  av_rat REAL,
  xg REAL,
  xg_per90 REAL,
  xa REAL,
  xa_per90 REAL,
  pas_pct REAL,
  shot_pct REAL,
  sv_pct REAL,
  xsv_pct REAL,

  -- Derived metrics
  npxg_plus_xa_per90 REAL,
  dist_per90_num REAL,
  net_poss_per90 REAL,
  net_save_pct REAL,
  def_actions_per_min REAL,
  tackle_efficacy REAL,
  aerial_efficacy REAL,

  -- Final model outputs
  position_score REAL,
  position_pct REAL,

  -- Raw snapshot row (full fidelity backup)
  stats_json TEXT,

  PRIMARY KEY (snapshot_id, player_id),
  FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
  FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE
);

--------------------------------------------------
-- GRADE SCORES (normalized, extensible)
--------------------------------------------------
CREATE TABLE player_grade_scores (
  snapshot_id INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  grade_name TEXT NOT NULL,                 -- e.g. 'Scoring'
  grade_score REAL NOT NULL,                -- 0..1

  PRIMARY KEY (snapshot_id, player_id, grade_name),
  FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
  FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE
);

--------------------------------------------------
-- INDEXES (performance critical)
--------------------------------------------------
CREATE INDEX idx_players_uid
  ON players(fm_uid);

CREATE INDEX idx_snapshots_date
  ON snapshots(snapshot_date);

CREATE INDEX idx_stats_snapshot
  ON player_snapshot_stats(snapshot_id);

CREATE INDEX idx_stats_snapshot_gradepos
  ON player_snapshot_stats(snapshot_id, grade_pos);

CREATE INDEX idx_stats_snapshot_division
  ON player_snapshot_stats(snapshot_id, division);

CREATE INDEX idx_stats_snapshot_bucket
  ON player_snapshot_stats(snapshot_id, comp_bucket);

CREATE INDEX idx_stats_snapshot_position_score
  ON player_snapshot_stats(snapshot_id, position_score);

CREATE INDEX idx_grade_scores_snapshot
  ON player_grade_scores(snapshot_id);

CREATE INDEX idx_grade_scores_grade
  ON player_grade_scores(grade_name);

--------------------------------------------------
-- TAGGING / SHORTLISTS (future-proof)
--------------------------------------------------
CREATE TABLE tags (
  tag_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  color TEXT
);

CREATE TABLE player_tags (
  player_id INTEGER NOT NULL,
  tag_id INTEGER NOT NULL,
  created_at TEXT DEFAULT (datetime('now')),

  PRIMARY KEY (player_id, tag_id),
  FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
  FOREIGN KEY (tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
);