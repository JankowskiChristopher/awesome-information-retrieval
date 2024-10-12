import logging
from pathlib import Path
from typing import Dict, Tuple, Union

from datasets import Dataset, IterableDataset, load_dataset

from src.constants import QA_DATASETS, QA_DATASETS_ROOT_DIR
from src.dataset.qa.dataset_configs import qa_data
from src.utils import get_absolute_path

logger = logging.getLogger(__name__)


def download_dataset(dataset_name: str, out_dir: str):
    if dataset_name in QA_DATASETS.keys():
        dataset_conf = QA_DATASETS[dataset_name]
        if dataset_conf.revision:
            data = load_dataset(
                dataset_conf.name,
                dataset_conf.revision,
                split=dataset_conf.split,
            )
        else:
            data = load_dataset(dataset_conf.name, split=dataset_conf.split)

        data.save_to_disk(out_dir)
    else:
        raise Exception("Wrong dataset name!")


def get_qa_dataset(dataset_name: str, local: bool = False) -> Tuple[Union[Dataset, IterableDataset], Dict[str, str]]:
    logger.info(f"Loading dataset {dataset_name}")
    datasets_absolute_dir = get_absolute_path(QA_DATASETS_ROOT_DIR, local=local)
    dataset_path = Path(datasets_absolute_dir) / dataset_name

    if dataset_path.exists():
        logger.info(f"Dataset exists, loading from {dataset_path}")
    else:
        # Download the dataset
        logger.info(f"Dataset does not exist, downloading from web")
        download_dataset(dataset_name, str(dataset_path))

    dataset, columns = qa_data(QA_DATASETS[dataset_name], str(dataset_path))

    return dataset, columns
