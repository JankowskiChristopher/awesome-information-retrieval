import collections
from typing import Dict, List

import datasets
import evaluate

from src.constants import QA_METRICS

# Copied and adjusted from huggingface's SquadV2 eval
# TODO: In future accept multiple reference answers


class QAEval(evaluate.Metric):
    """Generic Question Answering metric computing F1 and EM scores given golden answer"""

    def __init__(self, metrics: List[str] = ["f1", "em"]):
        super().__init__()

        common_metrics = set(metrics) & set(QA_METRICS.keys())

        if common_metrics:
            self.metrics = list(common_metrics)
        else:
            raise ValueError(
                f"None of the provided metrics are supported. \n Supported metrics: {list(QA_METRICS.keys())}"
            )

    def _info(self):
        return evaluate.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": {"id": datasets.Value("string"), "prediction_text": datasets.Value("string")},
                    "references": {"id": datasets.Value("string"), "answers": datasets.Value("string")},
                },
            ),
        )

    def _compute(self, *, predictions, references):
        predictions = {pred["id"]: pred["prediction_text"] for pred in predictions}
        references = {ref["id"]: ref["answers"] for ref in references}
        return self._compute_score(predictions, references)

    def _compute_score(
        self,
        preds: Dict[str, str],
        refs: Dict[str, str],
    ) -> Dict[str, float]:
        results = collections.defaultdict(list)
        for id in preds.keys():
            for metric in self.metrics:
                results[metric].append(QA_METRICS[metric](refs[id], preds[id]))

        return {name: sum(scores) / len(scores) for name, scores in results.items()}
