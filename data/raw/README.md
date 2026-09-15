# Raw Kaggle Data Directory

This folder contains the raw Kaggle dataset for the take-home project:
- **Dataset**: [Customer Support on Twitter (Kaggle)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- **File**: `twcs.csv` (~516 MB, ~2.8M customer support tweets)

## Instructions
1. Download `twcs.csv` from Kaggle.
2. Place `twcs.csv` directly into this folder (`data/raw/twcs.csv`).
3. Run `python3 src/data_processor.py` (or `./run_pipeline.sh`) to extract authentic multi-turn threads and build the golden evaluation set for `@AmazonHelp`.
