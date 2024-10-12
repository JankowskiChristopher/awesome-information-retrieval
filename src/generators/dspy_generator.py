import json
import logging

import dspy
from dsp import LM
from hydra.utils import instantiate
from omegaconf import omegaconf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DSPYGeneratorWrapper(LM):
    def __init__(self, generator_path: str):
        self.provider = "default"
        self.history = []
        self.generator = instantiate(omegaconf.OmegaConf.load(generator_path))
        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 12,  # 9 should be enough, but 12 is safer
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }

    def basic_request(self, prompt, **kwargs):
        response = self.generator.query(prompt, **kwargs)
        self.history.append({"prompt": prompt, "response": response[0], "kwargs": kwargs})
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        return self.request(prompt, **kwargs)

    def __exit__(self, *args, **kwargs):
        """
        We do not call garbage collector nor clean CUDA cache as it is done later.
        :param args:
        :param kwargs:
        :return:
        """
        logger.info("__exit__ Cleaning up DSPYGeneratorWrapper")
        self.generator.__exit__(*args, **kwargs)
        del self.generator
        self.generator = None


class RerankTwoPassages(dspy.Signature):
    """Choose which passage is more relevant to the question and output JSON. If passage1 is more relevant return {"passage":1}, otherwise return {"passage":2}. Do not provide any explanation output, only valid JSON. JSON should have only 1 key."""

    question = dspy.InputField()
    passage1 = dspy.InputField()
    passage2 = dspy.InputField()
    answer = dspy.OutputField(
        desc='JSON output with only 1 key "passage". {"passage":1} if passage1 is better {"passage":2} if passage2 is better. Output only valid JSON. If both passages are equally good output {"passage":1} or {"passage":2} randomly.'
    )


def output_validation(output) -> bool:
    try:
        output_json = json.loads(output)
        if output_json == {"passage": 1} or output_json == {"passage": 2}:
            return True
        return False
    except json.JSONDecodeError:
        logger.debug(f"DSPY output failed check. Output: {output}")
        return False


class Rerank(dspy.Module):
    def __init__(self, signature: dspy.Signature):
        super().__init__()
        self.generate_answer = dspy.Predict(signature)
        self.number_of_dspy_corrections = 0

    def forward(self, question, passage1, passage2):
        answer = self.generate_answer(question=question, passage1=passage1, passage2=passage2).answer
        output_validation_result = output_validation(answer)
        dspy.Suggest(
            output_validation_result, 'Shorter. Output should be a valid JSON. Allowed {"passage":1} or {"passage":2}.'
        )
        if not output_validation_result:
            self.number_of_dspy_corrections += 1

        return answer
