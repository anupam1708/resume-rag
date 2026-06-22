"""
Downloads the Kaggle Resume Dataset and trims to 100 resumes.
Output: data/resumes.csv with columns [candidate_id, category, resume_text]

Usage:
    python scripts/download_data.py

Requires Kaggle credentials in ~/.kaggle/kaggle.json
(create one at https://www.kaggle.com/settings under "API").

If you can't get Kaggle creds working, manually download from:
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
and place Resume.csv in data/
"""
import os
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
TARGET = DATA_DIR / "resumes.csv"


def main():
    try:
        import kagglehub
        path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
        src_csv = Path(path) / "Resume" / "Resume.csv"
        if not src_csv.exists():
            # Some versions of the dataset use a flat layout
            src_csv = next(Path(path).rglob("Resume.csv"))
    except Exception as e:
        print(f"kagglehub download failed: {e}")
        print(f"Falling back to existing file at {DATA_DIR / 'Resume.csv'}")
        src_csv = DATA_DIR / "Resume.csv"
        if not src_csv.exists():
            raise SystemExit(
                "Download Resume.csv manually from "
                "https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset "
                f"and place at {src_csv}"
            )

    df = pd.read_csv(src_csv)
    # Dataset uses 'Resume_str' for plain text and 'Category' for label
    text_col = "Resume_str" if "Resume_str" in df.columns else "Resume"
    cat_col = "Category" if "Category" in df.columns else "category"

    # Sample stratified across categories so retrieval evals are interesting.
    # Done with an explicit loop so the grouping column is preserved across
    # pandas versions (groupby.apply drops it in pandas >= 2.2).
    parts = [
        g.sample(min(len(g), 5), random_state=42)
        for _, g in df.groupby(cat_col)
    ]
    df = pd.concat(parts).reset_index(drop=True)

    df = df.head(100).reset_index(drop=True)
    df["candidate_id"] = [f"c_{i:04d}" for i in range(len(df))]
    df = df[["candidate_id", cat_col, text_col]]
    df.columns = ["candidate_id", "category", "resume_text"]
    df.to_csv(TARGET, index=False)
    print(f"Wrote {len(df)} resumes to {TARGET}")
    print("Category distribution:")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()
