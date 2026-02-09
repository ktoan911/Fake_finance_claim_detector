"""
CSV Labeled Data Loader

Label Convention (matches trainer normalization):
  - TRUE / SUPPORTED / LEGIT / "1" (string) → Label ID 0 (positive class)
  - FALSE / REFUTED / SCAM / "0" (string) → Label ID 1 (negative class)
  - NEUTRAL / NEI / UNKNOWN → Label ID 2 (not used in binary classification)

IMPORTANT: Numeric string "1" means TRUE (positive), "0" means FALSE (negative).
This matches standard ML convention where 1 = positive class.

Expected CSV columns (minimum):
  - text (string) OR claim (string)
  - evidence (string)
  - label (int, bool, or string: 1/0, true/false, True/False, SUPPORTED/REFUTED, etc.)

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

        # Normalize labels to integer IDs
        # CRITICAL: Convention matches trainer expectation:
        #   - Numeric/String "1" = True/Supported → ID 0 (positive class)
        #   - Numeric/String "0" = False/Refuted → ID 1 (negative class)

        # Handle both string labels (e.g., "TRUE", "FALSE") and numeric labels (e.g., 1, 0)
        # Pandas may auto-convert quoted strings like "1" to int64(1)

        if pd.api.types.is_numeric_dtype(df["label"]):
            # Labels are already numeric (e.g., int64(1), int64(0))
            # Apply the flip: 1 → 0 (True), 0 → 1 (False)
            def normalize_numeric_label(val):
                if pd.isna(val):
                    return 2  # Default to NEI
                val_int = int(val)
                if val_int == 1:
                    return 0  # 1 → True (ID 0)
                elif val_int == 0:
                    return 1  # 0 → False (ID 1)
                else:
                    return val_int  # Keep other values (e.g., 2 for NEI)

            df["label"] = df["label"].apply(normalize_numeric_label)
        else:
            # Labels are strings (e.g., "TRUE", "FALSE", "SUPPORTED")
            label_map = {
                # True/Supported/Positive class → ID 0
                "TRUE": 0,
                "SUPPORTED": 0,
                "LEGIT": 0,
                "LEGITIMATE": 0,
                "1": 0,  # String "1" means True (positive class)
                # False/Refuted/Negative class → ID 1
                "FALSE": 1,
                "REFUTED": 1,
                "SCAM": 1,
                "0": 1,  # String "0" means False (negative class)
                # Neutral/Unknown → ID 2 (not used in binary classification)
                "NEUTRAL": 2,
                "NEI": 2,
                "NOT": 2,
                "UNKNOWN": 2,
                "2": 2,
            }
            df["label"] = df["label"].astype(str).str.upper().map(label_map)

            unmapped = df["label"].isna().sum()
            if unmapped:
                logger.warning(
                    f"{unmapped} labels could not be mapped. Defaulting to False (1)."
                )
                df["label"] = df["label"].fillna(1)  # Default to False instead of NEI

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
