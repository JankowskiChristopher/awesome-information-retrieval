import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert CSV to JSON")
    parser.add_argument("--input_dir", default="./backup_data", type=str, help="Input dir")
    parser.add_argument("--output", default="merged_results.csv", type=str, help="Output CSV file path")
    args = parser.parse_args()

    input_path = args.input_dir
    output_path = args.output

    # Check that file ends with csv or add
    if not output_path.endswith(".csv"):
        output_path += ".csv"

    # Check if input path exists
    if not os.path.exists(input_path):
        print("Input file does not exist.")
        return

    # Get all the sorted files from input dir
    files = sorted(os.listdir(input_path))
    dfs = []
    for file in files:
        with open(Path(input_path) / file, "r") as f:
            data = dict(sorted(json.load(f).items()))
            data["experiment_name"] = Path(file).stem
            partial_df = pd.DataFrame([data.values()], columns=list(data.keys()), index=["experiment_name"])
            dfs.append(partial_df)

    df = pd.concat(dfs)
    # set index to experiment_name
    df.set_index("experiment_name", inplace=True)

    # save to csv
    df.to_csv(output_path)


if __name__ == "__main__":
    main()
