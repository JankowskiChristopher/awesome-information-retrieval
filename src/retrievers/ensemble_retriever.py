import logging
from typing import List

logger = logging.getLogger(__name__)


class EnsembleRetriever:
    def __init__(self, configs_paths: List[str], nums_of_retrieved_passages: List[int]):
        """
        Initializes an ensemble retriever with a dictionary of retrievers.
        Thanks to recursive initialization by Hydra, the retrievers are already instantiated.
        We store them in a dictionary, so we can access them by name.
        :param configs_paths: List of paths to the retriever configs.
        """
        logger.info(
            f"EnsembleRetriever initialized with {len(configs_paths)} retrievers paths."
            f" and nums_of_retrieved_passages: {nums_of_retrieved_passages}"
        )
        assert len(configs_paths) == len(nums_of_retrieved_passages), (
            f"Number of retrievers {len(configs_paths)} does not match the number of nums_of_retrieved_passages "
            f"{len(nums_of_retrieved_passages)}"
        )

        self.nums_of_retrieved_passages = nums_of_retrieved_passages
        self.configs_paths = configs_paths
