from collections import namedtuple

from datasets import load_from_disk

QADataConfig = namedtuple(
    "QADataConfig",
    [
        "name",
        "question_column",
        "context_column",
        "id_column",
        "label_column",
        "split",
        "revision",
        "preprocess_func",
    ],
)


def qa_data(config: QADataConfig, dataset_dir: str):
    id_column = config.id_column

    data = load_from_disk(dataset_dir)

    if config.preprocess_func is not None:
        data = config.preprocess_func(data)

    if id_column is None:
        data = data.add_column("id", [str(i) for i in range(len(data))])
        id_column = "id"

    return data, {
        "question_column": config.question_column,
        "context_column": config.context_column,
        "id_column": id_column,
        "label_column": config.label_column,
    }


# Preprocessing functions, mainly to convert columns containing multiple possible answers
# to column containing a single answer


def openbookqa_preprocess(data):
    def process_func(example):
        labels = example["choices"]["label"]
        labels = dict(zip(labels, range(len(labels))))
        example["choices"] = example["choices"]["text"][labels[example["answerKey"]]]
        return example

    return data.map(process_func)


def wikiqa_preprocess(data):
    return data.filter(lambda example: example["label"] == 1)


def medmcqa_preprocess(data):
    answer_columns = {0: "opa", 1: "opb", 2: "opc", 3: "opd"}

    def process_func(example):
        example["opa"] = example[answer_columns[example["cop"]]]
        return example

    return data.map(process_func)


def mathqa_preprocess(data):
    def process_func(example):
        example["Problem"] = "\n".join([example["Problem"], example["options"]])
        return example

    return data.map(process_func)


TRUTHFULQA = QADataConfig(
    **{
        "name": "truthful_qa",
        "question_column": "question",
        "context_column": None,
        "label_column": "best_answer",
        "id_column": None,
        "split": "validation",
        "revision": "generation",
        "preprocess_func": None,
    }
)

OPENBOOKQA = QADataConfig(
    **{
        "name": "openbookqa",
        "question_column": "question_stem",
        "context_column": "fact1",
        "id_column": "id",
        "label_column": "choices",
        "split": "test",
        "revision": "additional",
        "preprocess_func": openbookqa_preprocess,
    }
)

WIKIQA = QADataConfig(
    **{
        "name": "wiki_qa",
        "question_column": "question",
        "context_column": "document_title",
        "id_column": "question_id",
        "label_column": "answer",
        "split": "test",
        "revision": None,
        "preprocess_func": wikiqa_preprocess,
    }
)

MEDMCQA = QADataConfig(
    **{
        "name": "medmcqa",
        "question_column": "question",
        "context_column": "exp",
        "id_column": "id",
        "label_column": "opa",
        "split": "validation",
        "revision": None,
        "preprocess_func": medmcqa_preprocess,
    }
)

MATHQA = QADataConfig(
    **{
        "name": "math_qa",
        "question_column": "Problem",
        "context_column": None,
        "id_column": None,
        "label_column": "correct",
        "split": "test",
        "revision": None,
        "preprocess_func": mathqa_preprocess,
    }
)
