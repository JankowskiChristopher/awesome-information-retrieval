import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator
from termcolor import colored

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def _relu(x: float) -> float:
    """
    Rectified Linear Unit (ReLU) function.
    :param x: Input value
    :return: Output value
    """
    return max(0.0, x)


def compare_experiments_with_baseline(
    experiments_dfs: List[pd.DataFrame],
    baseline_df: Optional[pd.DataFrame],
    baseline_function: str,
    metric: str,
    relu: bool = False,
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Compares the experiments with the baseline DataFrame and returns the processed DataFrames.
    :param experiments_dfs: List of DataFrames for the experiments
    :param baseline_df: Baseline DataFrame in the same format
    :param baseline_function: Function to compare with the baseline. 'none', 'delta', or 'percent'.
    :param metric: Metric to compare
    :param relu: Take only positive value improvement when comparing delta or percent.
    :return: List of processed DataFrames for the experiments and the aggregated DataFrame values
    """
    if baseline_function == "none":
        concatenated_df = pd.concat(experiments_dfs)
        aggregated_values = concatenated_df.groupby("experiment_name")[metric].mean().reset_index()
        aggregated_values["dataset"] = "MTEB retrieval"
        return [df.sort_values(by="dataset") for df in experiments_dfs], aggregated_values

    assert baseline_df is not None, "Baseline DataFrame is None. Please provide a baseline DataFrame."

    sorted_baseline_df = baseline_df.sort_values(by="dataset", ignore_index=True)
    processed_dfs = []
    partial_aggregated_values = []
    for df in experiments_dfs:
        df = df.copy()
        # sort df to have the same order
        df.sort_values(by="dataset", inplace=True, ignore_index=True)
        # shapes must match
        assert df.shape == sorted_baseline_df.shape, (
            f"Cannot compare - shapes of the DataFrames do not match.\n"
            f"Got {df.shape} and {sorted_baseline_df.shape}."
        )
        if relu:
            df[metric] = df[metric].combine(sorted_baseline_df[metric], max)

        # compare the mean values
        mean_of_df = df.groupby("experiment_name")[metric].mean().reset_index()
        mean_of_baseline = sorted_baseline_df.groupby("experiment_name")[metric].mean().reset_index()
        difference_df = mean_of_df.copy()

        if baseline_function == "delta":
            df[metric] = df[metric] - sorted_baseline_df[metric]
            if relu:
                df[metric] = df[metric].apply(_relu)
            difference_df[metric] = mean_of_df[metric] - mean_of_baseline[metric]
            partial_aggregated_values.append(difference_df)

        elif baseline_function == "percent":
            df[metric] = ((df[metric] - sorted_baseline_df[metric]) / sorted_baseline_df[metric]) * 100
            if relu:
                df[metric] = df[metric].apply(_relu)
            difference_df[metric] = ((mean_of_df[metric] - mean_of_baseline[metric]) / mean_of_baseline[metric]) * 100
            partial_aggregated_values.append(difference_df)

        processed_dfs.append(df)

    aggregated_values = pd.concat(partial_aggregated_values)
    aggregated_values["dataset"] = "MTEB retrieval"

    return processed_dfs, aggregated_values


def get_and_preprocess_experiment_files(
    experiment_name: str,
    experiment_path: Path,
    metric: str,
    baseline_function: str,
    add_baseline_to_experiments: bool,
    include_germanquad: bool = False,
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Function iterates over all the files in the experiment path and creates DataFrames for each file.
    If the directory contains a baseline file, it is returned separately.
    If multiple baseline files are found, it raises an exception.
    :param experiment_name: Name of the experiment
    :param experiment_path: Path to the experiment directory
    :param metric: Metric to filter only the appropriate values
    :param baseline_function: Function to compare with the baseline. 'none', 'delta', or 'percent'.
    :param add_baseline_to_experiments: Whether the baseline should be plotted with the experiments
    (only if the function to compare is "none")
    :param include_germanquad: Include GermanQuad in the plot. By default, it is excluded.
    :raise ValueError: If multiple baseline files are found
    :return: List of DataFrames for the experiments and the baseline DataFrame
    """
    experiments_dfs = []
    baseline_df = None
    for file in sorted(experiment_path.iterdir()):
        if file.suffix not in [".json", ".csv"]:
            logger.warning(f"Skipping the file {file.name} as it is not a json or csv file.")
            continue

        with open(file) as f:
            if file.suffix == ".json":
                json_values: Dict[str, float] = json.load(f)
                values_tuples = [(k.split("/")[0], v, file.stem) for k, v in json_values.items() if (metric in k)]

            elif file.suffix == ".csv":
                one_row_df = pd.read_csv(f)
                values_tuples = [
                    (k.split("/")[0], v, file.stem) for k, v in one_row_df.iloc[0].items() if (metric in k)
                ]

            if not include_germanquad:
                values_tuples = [t for t in values_tuples if "germanquad" not in t[0]]

            # Change to range [0. 100]
            values_tuples = [(dataset, v * 100, experiment_name) for dataset, v, experiment_name in values_tuples]

            df = pd.DataFrame(values_tuples, columns=["dataset", metric, "experiment_name"])
            logger.info(f"Created DataFrame for {file.name} with shape {df.shape}.")

            # check if the file is a baseline file
            if "baseline" in file.name:
                if baseline_df is None:
                    baseline_df = df
                    if (baseline_function == "none") and add_baseline_to_experiments:
                        experiments_dfs.append(baseline_df)
                else:
                    raise ValueError(f"Multiple baseline files found in the experiment {experiment_name}.")
            else:
                experiments_dfs.append(df)

    return experiments_dfs, baseline_df


def create_experiment_df(
    experiment_name: str,
    input_dir: str,
    metric: str,
    baseline_function: str,
    add_baseline_to_experiments: bool = True,
    include_germanquad: bool = False,
    relu: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates the dataframe for the given experiment name. Reads, preprocesses and merges the files.
    :param experiment_name: Name of the experiment
    :param input_dir: Input directory where the experiments are stored
    :param metric: Metric to plot
    :param baseline_function: Function to compare with the baseline. 'none', 'delta', or 'percent'.
    :param add_baseline_to_experiments: Add the baseline DataFrame to the experiments when 'none' is selected.
    :param include_germanquad: Include GermanQuad in the plot. By default, it is excluded.
    :param relu: Take only max value when comparing delta or percent.
    :raise ValueError: If the experiment does not exist or multiple baseline files are found
    :return: Concatenated DataFrame and aggregated DataFrame values
    """
    experiment_path = Path(input_dir) / experiment_name

    # check if the experiment exists
    if not experiment_path.exists():
        logger.error(f"Experiment {experiment_name} does not exist in {str(experiment_path)}")
        return pd.DataFrame(), pd.DataFrame()

    logger.info(f"Experiment Name: {experiment_name} in {experiment_path}. Iterating over all the files...")

    # can raise an exception if multiple baseline files are found
    experiments_dfs, baseline_df = get_and_preprocess_experiment_files(
        experiment_name, experiment_path, metric, baseline_function, add_baseline_to_experiments, include_germanquad
    )

    processed_experiments_dfs, aggregated_df_values = compare_experiments_with_baseline(
        experiments_dfs, baseline_df, baseline_function, metric, relu
    )
    all_experiment_df = pd.concat(processed_experiments_dfs)
    logger.info(f"Concatenated all the DataFrames. Final shape: {all_experiment_df.shape}.")

    return all_experiment_df, aggregated_df_values


def baseline_function_to_string(baseline_function: str) -> str:
    """
    Converts the baseline function to a string for the plot title or file.
    :param baseline_function: Baseline function
    :return: String representation of the baseline function
    """
    if baseline_function == "none":
        return ""
    return baseline_function + "_"


def plot_results(
    df: pd.DataFrame,
    aggregated_df_values: pd.DataFrame,
    metric: str,
    experiment_name: str,
    output_dir: str,
    file_format: str,
    baseline_function: str,
    figsize: Tuple[int, int] = (25, 8),
    relu: bool = False,
) -> None:
    """
    Plots the results for the given DataFrame.
    :param df: Dataframe with the results
    :param aggregated_df_values: Aggregated DataFrame with the values to plot on the right side
    :param metric: Metric to plot
    :param experiment_name: Name of the experiment
    :param output_dir: Output directory to store the plots
    :param file_format: File format to store the plots
    :param baseline_function: Function to compare with the baseline. 'none', 'delta', or 'percent'.
    :param figsize: Figure size for the plot
    """
    # plot the results
    logger.info(f"Plotting the results for {experiment_name} experiment with metric {metric} in {output_dir}.\n")
    x_ticks_fontsize = 16
    y_ticks_fontsize = 14

    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [4, 1]})

    # Plot the bar plot on ax1
    sns.set(style="whitegrid")
    sns.set_context("talk")
    plt.xticks(fontsize=x_ticks_fontsize)
    plt.yticks(fontsize=y_ticks_fontsize)

    ax1 = sns.barplot(x="dataset", y=metric, data=df, hue="experiment_name", ax=ax1)

    if baseline_function == "none":
        ax1.set_title(f"{metric} for {experiment_name} experiment")
    else:
        ax1.set_title(f"{baseline_function} difference in {metric} for {experiment_name} experiment")

    ax1.set_xlabel("")
    ax1.xaxis.set_major_locator(FixedLocator(ax1.get_xticks()))  # necessary to avoid warning
    ax1.yaxis.set_major_locator(FixedLocator(ax1.get_yticks()))  # necessary to avoid warning
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=60, fontsize=x_ticks_fontsize)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=y_ticks_fontsize)

    # Plot the mean values as a bar plot on ax2
    sns.barplot(x="dataset", y=metric, data=aggregated_df_values, hue="experiment_name", ax=ax2)
    ax2.set_title("Mean Values")
    ax2.set_xlabel("")
    ax2.set_ylabel("")  # No need for y-axis label in the subfigure
    ax2.legend().remove()  # Remove legend from subfigure

    if baseline_function == "none":
        ax1.set_ylabel(metric)
    else:
        ax1.set_ylabel(f"{baseline_function} {metric}")
        if baseline_function == "percent":
            import matplotlib.ticker as mtick

            ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()

    # Move legend above the plot
    handles, labels = ax2.get_legend_handles_labels()

    def _rename_legend_label(_label: str) -> str:
        if not _label:
            return _label
        if _label[0].isnumeric():
            return _label[1:]
        return _label

    new_labels = [_rename_legend_label(label) for label in labels]
    ax1.legend(
        handles, new_labels, bbox_to_anchor=(0.5, -0.5), loc="upper center", ncol=len(df["experiment_name"].unique())
    )
    plt.subplots_adjust(bottom=0.4)

    # save fig
    output_function = baseline_function_to_string(baseline_function)
    relu_str = "relu_" if relu else ""
    output_path = Path(output_dir) / experiment_name / f"{relu_str}{output_function}{metric}.{file_format}"
    os.makedirs(output_path.parent, exist_ok=True)
    if file_format == "pdf":
        fig.savefig(str(output_path), bbox_inches="tight")
    else:
        fig.savefig(str(output_path), dpi=250)
    plt.close(fig)  # Close the figure to release memory


def main():
    parser = argparse.ArgumentParser(description="Arguments for the plotter.")
    parser.add_argument(
        "--experiment_names",
        type=str,
        default=None,
        nargs="+",
        help="Names of the experiment in the input directory. None means all the experiments.",
    )
    parser.add_argument("--input_dir", type=str, default="data", help="Directory that stores the experiments data.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory that stores the experiments data.")
    parser.add_argument("--metric", type=str, default="NDCG@10", help="Metric to plot. Default is NDCG@10.")
    parser.add_argument(
        "--file_format",
        type=str,
        default="png",
        choices=["pdf", "png"],
        help="File format of the output. Default is pdf.",
    )
    parser.add_argument(
        "--baseline_function",
        type=str,
        default="none",
        choices=["delta", "percent", "none"],
        help="Function to compare with the baseline",
    )
    parser.add_argument("--no_baseline_added", action="store_true", help="Do not add the baseline to the experiments.")
    parser.add_argument("--include_germanquad", action="store_true", help="Include GermanQuad in the plot.")
    parser.add_argument("--relu", action="store_true", help="Take only max value when comparing delta or percent.")

    args = parser.parse_args()
    if args.relu:
        assert args.baseline_function in ["delta", "percent"], "ReLU can be used only with delta or percent."

    # check that input directory exists and is a directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory {args.input_dir} does not exist or is not a directory.")
        return

    # create output_dir if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Script running with arguments {args}.\n")

    # get the names of the experiments
    if args.experiment_names is None:
        experiments_names = [d.name for d in input_dir.iterdir()]
        logger.info(f"Using all the experiments in the directory. Found: {experiments_names}")
    else:
        experiments_names = args.experiment_names

    # prepare the dataframe and plot the results
    skipped_experiments = 0
    for experiment_name in sorted(experiments_names):
        try:
            all_experiment_df, aggregated_df_values = create_experiment_df(
                experiment_name,
                args.input_dir,
                args.metric,
                args.baseline_function,
                not args.no_baseline_added,
                args.include_germanquad,
                relu=args.relu,
            )
            logger.debug(all_experiment_df.head())
            logger.debug(aggregated_df_values.head())
            plot_results(
                all_experiment_df,
                aggregated_df_values,
                args.metric,
                experiment_name,
                args.output_dir,
                args.file_format,
                args.baseline_function,
                relu=args.relu,
            )
        except ValueError as e:
            logger.error(f"Error in experiment {experiment_name}: {e}")
            skipped_experiments += 1
            continue

    if skipped_experiments > 0:
        logger.error(
            colored(f"Skipped {skipped_experiments} experiments due to errors. Consult logs for more details.", "red")
        )
    else:
        logger.info(colored("Plotted all the experiments.", "green"))


if __name__ == "__main__":
    main()
