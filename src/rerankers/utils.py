import logging
import os
from typing import Optional

from omegaconf import DictConfig, omegaconf

logger = logging.getLogger(__name__)


def _reranker_name_to_config_and_args(reranker_name: str, config: Optional[DictConfig] = None) -> DictConfig:
    """
    Function converts the part of the EnsembleReranker config to the config of an individual reranker
    that will be later instantiated.
    :param reranker_name: Name of the reranker, useful but not necessary
    :param config: Config for the reranker
    :return: DictConfig of a single reranker
    """
    logger.info(f"Convert reranker name {reranker_name} to config path and additional args.")
    config_path: Optional[str] = None
    additional_args = {}
    for key, value in config.items():
        if key == "config_path":
            config_path = value
        else:
            additional_args[key] = value

    assert config_path is not None, f"Config path not found in the reranker config: {config}"

    # Create a DictConfig and override values
    config = omegaconf.OmegaConf.load(config_path)
    for key, value in additional_args.items():
        omegaconf.OmegaConf.update(config, key, value)

    return config


def scan_huggingface_cache(cache_dir: str = "/repo/hf/models") -> None:
    """
    Function used for debugging purposes. Scans the Huggingface cache and prints the paths.
    If the cache is not present, it prints a message.

    @param cache_dir: String with path to the Huggingface cache.
    """
    logger.info("Scanning Huggingface cache.")
    # check if path exists
    if not os.path.exists(cache_dir):
        logger.info(f"{cache_dir} does not exist.")
        return

    def _recursive_scan(path: str) -> None:
        """
        Helper function to scan the cache recursively.

        :param path: Subdir path to scan recursively
        """
        for subpath in os.scandir(path):
            logger.info(f"\t{subpath.path}")
            if subpath.is_dir():
                _recursive_scan(subpath.path)

    for path in os.scandir(cache_dir):
        logger.info(f"Main path {path.path}:")
        if path.is_dir():
            _recursive_scan(path.path)
