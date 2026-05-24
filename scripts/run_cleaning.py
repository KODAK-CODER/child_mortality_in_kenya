# ─────────────────────────────────────────────
# scripts/run_cleaning.py
# Run the full data cleaning pipeline
# ─────────────────────────────────────────────
"""
Usage:
    python scripts/run_cleaning.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mortality_cleaner import MortalityCleaner

def main():
    cleaner = MortalityCleaner(data_dir="data/raw")

    # Clean all three datasets in order
    cleaner.clean_county_indicators()
    cleaner.clean_interventions()
    cleaner.clean_deployments()

    # Print summary
    cleaner.summary()

    # Save cleaned outputs
    cleaner.save_cleaned(output_dir="data/processed")
    print("Pipeline complete. Cleaned files saved to data/processed/")

if __name__ == "__main__":
    main()
