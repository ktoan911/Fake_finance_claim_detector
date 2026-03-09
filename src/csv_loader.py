"""
CSV Labeled Data Loader

Expected CSV columns (minimum):
  - text (string) OR claim (string)
  - evidence (string)
  - label (int or string: Đúng/Sai, True/False, Supported/Refuted)

Optional:
  - timestamp (ISO string or unix seconds)
"""

from datetime import datetime, timezone

import pandas as pd
from loguru import logger


class CSVLabeledLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        if "text" not in df.columns and "claim" in df.columns:
            df = df.rename(columns={"claim": "text"})

        required_cols = {"text", "evidence", "label"}
        if not required_cols.issubset(df.columns):
            missing = ", ".join(sorted(required_cols - set(df.columns)))
            raise ValueError(
                f"CSV must contain columns: text (or claim), evidence, label. Missing: {missing}"
            )

        df = df.copy()
        df["evidence"] = df["evidence"].fillna("").astype(str)

        # Handle evidence that looks like a list string: "['item1', 'item2']"
        def parse_evidence(ev):
            ev_str = str(ev).strip()
            if ev_str.startswith("[") and ev_str.endswith("]"):
                try:
                    import ast

                    parsed = ast.literal_eval(ev_str)
                    if isinstance(parsed, list):
                        # Use ||| as separator for clear article boundaries
                        return "|||".join(str(item) for item in parsed)
                except (ValueError, SyntaxError):
                    pass
            return ev_str

        df["evidence"] = df["evidence"].apply(parse_evidence)
        # Allow string/boolean labels - map to stable binary IDs (0=Đúng, 1=Sai)
        if df["label"].dtype == object or df["label"].dtype == bool:
            label_map = {
                # Positive/support variants (ID: 0)
                "ĐÚNG": 0,
                "DUNG": 0,
                "TRUE": 0,
                "SUPPORTED": 0,
                "LEGIT": 0,
                "LEGITIMATE": 0,
                "0": 0,
                # Negative/refuted variants (ID: 1)
                "SAI": 1,
                "FALSE": 1,
                "REFUTED": 1,
                "SCAM": 1,
                "1": 1,
            }
            df["label"] = df["label"].astype(str).str.upper().map(label_map)

        unmapped = df["label"].isna().sum()
        if unmapped > 0:
            logger.warning(
                f"{unmapped} labels could not be mapped. Defaulting to negative class (1)."
            )
            df["label"] = df["label"].fillna(1)

        df["label"] = df["label"].astype(int)

        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].apply(self._parse_timestamp)

        return df

    def _parse_timestamp(self, value):
        if pd.isna(value):
            return datetime.now(timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return datetime.now(timezone.utc)
