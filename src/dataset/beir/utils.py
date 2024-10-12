import logging
import os
from pathlib import Path
from shutil import rmtree
from typing import Tuple

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from src.constants import BEIR_DATASETS, DATASETS_ROOT_DIR
from src.dataset.beir.classes import RetrievalDataset
from src.utils import get_absolute_path

logger = logging.getLogger(__name__)


def download_beir_dataset(
    dataset_name: str,
    out_dir: str,
    source_url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip",
):
    url = source_url.format(dataset_name)
    datasets_root_dir = os.path.dirname(out_dir)
    try:
        dataset_dir = util.download_and_unzip(url=url, out_dir=datasets_root_dir)
        os.remove(f"{dataset_dir}.zip")
    except Exception as e:
        logger.info(f"Dataset: {dataset_name} not found at: {url}, Removing cached dir")
        rmtree(out_dir)
        raise ValueError(f"invalid BEIR dataset: {dataset_name}") from e

    logger.info(f"Dataset: {dataset_name}, downloaded at: {url}, stored at {dataset_dir}")
    return dataset_dir


def download_dataset(dataset_name: str, out_dir: str):
    if dataset_name in BEIR_DATASETS:
        return download_beir_dataset(dataset_name, out_dir)
    else:
        raise Exception("Wrong dataset name!")


def get_dataset(dataset_name: str, split: str, local: bool = False) -> RetrievalDataset:
    """
    Load a dataset by name, with a specified data split..

    :param dataset_name: The identifier for the dataset.
    :param split: The specific split of the dataset to load (e.g., 'train', 'test').
    :param local: If True, loads from local directory, else loads from /nas

    :return: An object representing the loaded dataset, which includes corpora, queries, and relevance judgments.
    """
    logger.info(f"Loading dataset {dataset_name}")

    datasets_absolute_dir = Path(get_absolute_path(DATASETS_ROOT_DIR, local=local))
    dataset_load_path, dataset_save_path = resolve_dataset_path(dataset_name, datasets_absolute_dir)

    if not dataset_load_path.exists():
        logger.info(f"Dataset does not exist, downloading to {dataset_save_path}")
        download_dataset(dataset_name, str(dataset_save_path))

    return load_dataset_from_path(dataset_load_path, split)


def resolve_dataset_path(dataset_name: str, root_dir: Path) -> Tuple[Path, Path]:
    """
    Determine the paths for loading and saving the dataset based on its name and the root directory.

    :param dataset_name: The name of the dataset.
    :param root_dir: The root directory where datasets are stored.

    :return: A tuple containing the paths for loading and saving the dataset, respectively.
    """
    if dataset_name.startswith("cqadupstack"):
        return resolve_cqadupstack_path(dataset_name, root_dir)
    dataset_save_path = dataset_load_path = root_dir / dataset_name
    return dataset_load_path, dataset_save_path


def resolve_cqadupstack_path(dataset_name: str, root_dir: Path) -> Tuple[Path, Path]:
    """
    Handle path resolution for 'cqadupstack' datasets which are structured based on subforums.

    :param dataset_name: The complete name of the cqadupstack dataset, including the subforum.
    :param root_dir: The base directory where the dataset should be stored.

    :return: A tuple where the first element is the path to load the dataset and the second is the path to save it.
    """
    parts = dataset_name.split("_")
    if len(parts) != 2:
        raise ValueError(
            "Dataset name for CQADupStack must include a subforum, separated by an underscore, e.g., 'cqadupstack_android'"
        )
    subforum = parts[1]
    dataset_save_path = root_dir / "cqadupstack"
    dataset_load_path = dataset_save_path / subforum
    return dataset_load_path, dataset_save_path


def load_dataset_from_path(dataset_path: Path, split: str) -> RetrievalDataset:
    """
    Load the dataset from a specified path and data split.

    :param dataset_path: The path where the dataset is stored.
    :param split: The split of the dataset to load.

    :return: The dataset loaded from the specified path and split.
    """
    logger.info(f"Dataset exists, loading from {dataset_path}")
    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split=split)
    return RetrievalDataset(corpus, queries, qrels)
