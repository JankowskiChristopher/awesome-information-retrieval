import argparse
import csv
import json
import os


def csv_to_json(csv_file: str) -> str:
    """
    Convert CSV file to JSON. Integer and float values are converted to appropriate types.
    :param csv_file: Path to CSV file
    :return: JSON string
    """
    # Read CSV file
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        data = next(reader)
        # convert to float if possible
        for key in data:
            try:
                data[key] = int(data[key])
            except ValueError:
                try:
                    data[key] = float(data[key])
                except ValueError:
                    pass

    # Convert to JSON
    json_data = json.dumps(data)

    return json_data


def save_json(json_data: str, output_file: str) -> None:
    """
    Save JSON string to file
    :param json_data: Data of the csv in json format to save
    :param output_file: Path to output file
    """
    # Save JSON to file
    with open(output_file, "w") as f:
        f.write(json_data)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert CSV to JSON")
    parser.add_argument("input", type=str, help="Input CSV file path")
    args = parser.parse_args()

    input_path = args.input

    # Check if input file exists
    if not os.path.exists(input_path):
        print("Input file does not exist.")
        return

    # Get output file name
    output_path = os.path.splitext(input_path)[0] + ".json"

    # Convert CSV to JSON
    json_data = csv_to_json(input_path)

    # Save JSON to file
    save_json(json_data, output_path)

    print(f"Conversion successful. JSON file saved as: {output_path}")


if __name__ == "__main__":
    main()
