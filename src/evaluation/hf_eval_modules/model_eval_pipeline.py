from typing import Dict

from datasets import Dataset
from evaluate import QuestionAnsweringEvaluator
from evaluate.evaluator.utils import DatasetColumn
from tqdm import tqdm

from src.evaluation.hf_eval_modules.qa_metric import QAEval
from src.generators.llm_generator import LLMGenerator


class GenericModelQAEvalPipeline:
    def __init__(
        self,
        generator: LLMGenerator,
        question_string: str = "question:",
        context_string: str = "context:",
        question_first: bool = False,
    ):
        self.generator = generator
        self.question_string = question_string if question_string is not None else "question:"
        self.context_string = context_string if context_string is not None else "context:"
        self.question_first = question_first
        self.task = "question-answering"

    def __call__(self, **kwargs):
        answers = []

        inputs = self._parse_inputs(**kwargs)

        responses, tok_ids = self.generator.query(inputs, return_tok_ids=True)

        prev_response_iter = 0

        for tok_id in tok_ids:
            answers = answers + [
                {"score": 0.0, "start": id[0], "end": id[-1], "answer": ans}
                for id, ans in zip(tok_id, responses[prev_response_iter : prev_response_iter + len(tok_id)])
            ]
            prev_response_iter += len(tok_id)

        return answers

    def _parse_inputs(self, **kwargs):
        if "context" in kwargs.keys():
            if self.question_first:
                inputs = [
                    (
                        f"{self.question_string} {Q} {self.context_string} {C}"
                        if C is not None
                        else f"{self.question_string} {Q}"
                    )
                    for Q, C in zip(kwargs["question"], kwargs["context"])
                ]
            else:
                inputs = [
                    (
                        f"{self.context_string} {Q} {self.question_string} {C}"
                        if C is not None
                        else f"{self.question_string} {Q}"
                    )
                    for Q, C in zip(kwargs["question"], kwargs["context"])
                ]

        else:
            inputs = ["question:" + Q for Q in kwargs["question"]]

        return inputs


# class GenericModelQAEvalPipeline:
#    def __init__(
#        self,
#        model,
#        tokenizer,
#        model_name: str,
#        tokenizer_name: str,
#        batch_size: int,
#        tokenizer_args: Dict,
#        generate_args: Dict,
#        question_string: str = "question:",
#        context_string: str = "context:",
#        question_first: bool = False,
#    ):
#        self.model = model
#        self.tokenizer = tokenizer
#        self.model_name = model_name
#        self.tokenizer_name = tokenizer_name
#        self.batch_size = batch_size
#        self.tokenizer_args = tokenizer_args
#        self.generate_args = generate_args
#        self.question_string = question_string if question_string is not None else "question:"
#        self.context_string = context_string if context_string is not None else "context:"
#        self.question_first = question_first
#        if self.tokenizer.pad_token is None:
#            self.tokenizer.pad_token = self.tokenizer.eos_token
#        self.task = "question-answering"
#
#    def __call__(self, **kwargs):
#        answers = []
#
#        inputs = self._parse_inputs(**kwargs)
#
#        for i in tqdm(range(len(inputs))[:: self.batch_size]):
#            prompt = inputs[i : i + self.batch_size]
#
#            if self.tokenizer.chat_template is not None:
#                prompt = [
#                    self.tokenizer.apply_chat_template(
#                        [{"role": "user", "content": input}],
#                        tokenize=False,
#                    )
#                    for input in prompt
#                ]
#            ids = self.tokenizer(prompt, **self.tokenizer_args).to(self.model.device)
#
#            generated_ids = self.model.generate(**ids, **self.generate_args)
#
#            answer = self.tokenizer.batch_decode(
#                generated_ids[:, ids["input_ids"].shape[1] :]
#            )  # needed to eliminate original prompt from the generated answer
#            answers = answers + [
#                {"score": 0.0, "start": generated_id[0].cpu(), "end": generated_id[-1].cpu(), "answer": ans}
#                for generated_id, ans in zip(generated_ids, answer)
#            ]
#
#        return answers
#
#    def _parse_inputs(self, **kwargs):
#        if "context" in kwargs.keys():
#            if self.question_first:
#                inputs = [
#                    (
#                        f"{self.question_string} {Q} {self.context_string} {C}"
#                        if C is not None
#                        else f"{self.question_string} {Q}"
#                    )
#                    for Q, C in zip(kwargs["question"], kwargs["context"])
#                ]
#            else:
#                inputs = [
#                    (
#                        f"{self.context_string} {Q} {self.question_string} {C}"
#                        if C is not None
#                        else f"{self.question_string} {Q}"
#                    )
#                    for Q, C in zip(kwargs["question"], kwargs["context"])
#                ]
#
#        else:
#            inputs = ["question:" + Q for Q in kwargs["question"]]
#
#        return inputs
#


class QAEvaluator(QuestionAnsweringEvaluator):
    """QA evaluator that given a question and potentially context computes
    metrics based on golden answer and prediction

    Usage:
    QAEvaluator.compute(
                    model_or_pipeline,
                    tokenizer,
                    data,
                    metric : QAEval,
                    id_column : str = "id",
                    context_column : str = "context",
                    label_column : str = "label",
                    question_column : str = "question",
                     ):
    """

    def __init__(self):
        super().__init__()

    def compute(
        self,
        model_or_pipeline: GenericModelQAEvalPipeline,
        data: Dataset,
        metric: QAEval,
        id_column: str = "id",
        context_column: str = "context",
        label_column: str = "label",
        question_column: str = "question",
    ):
        return super().compute(
            model_or_pipeline=model_or_pipeline,
            data=data,
            question_column=question_column,
            id_column=id_column,
            context_column=context_column,
            label_column=label_column,
            metric=metric,
            squad_v2_format=False,
        )

    def prepare_data(
        self,
        data,
        question_column: str = "question",
        context_column: str = "context",
        id_column: str = "id",
        label_column: str = "answer",
    ):
        if data is None:
            raise ValueError("Please specify a valid `data` object - either a `str` with a name or a `Dataset` object.")

        pipe_ret = {"question": DatasetColumn(data, question_column)}

        if context_column:
            self.check_required_columns(
                data,
                {
                    "question_column": question_column,
                    "context_column": context_column,
                    "id_column": id_column,
                    "label_column": label_column,
                },
            )
            pipe_ret["context"] = DatasetColumn(data, context_column)
        else:
            self.check_required_columns(
                data,
                {
                    "question_column": question_column,
                    "id_column": id_column,
                    "label_column": label_column,
                },
            )

        metric_inputs = dict()
        metric_inputs["references"] = [{"id": element[id_column], "answers": element[label_column]} for element in data]

        return metric_inputs, pipe_ret
