import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Union

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """
    Base class for different generators that can be used in the pipeline.
    """

    @abstractmethod
    def query(self, prompts: Union[str, List[str], Dict[str, str]]) -> Union[str, List[str], Dict[str, str]]:
        """
        Function queries the generator with a list of prompts. Overriden by subclasses.

        :param prompts: string, list of strings or a dict with id : prompt
        :return: New results in the same format as the input.
        """
        pass
