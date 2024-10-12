import gc
import logging
from typing import Dict, List, Optional, Union

import torch
from tqdm import tqdm

from src.generators.base_generator import BaseGenerator
from src.utils import get_model, get_tokenizer

logger = logging.getLogger(__name__)


class LLMGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        batch_size: int,
        tokenizer_args: Dict,
        generate_args: Dict,
        local: bool = False,
        device: str = "cuda",
    ):
        self.tokenizer = get_tokenizer(tokenizer_name, local)
        self.model = get_model(model_name, device, local, question_answering=True)
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.tokenizer_args = tokenizer_args
        self.generate_args = generate_args
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def query(
        self, prompts: Union[str, List[str], Dict[str, str]], return_tok_ids: bool = False
    ) -> Union[List[str], Dict[str, str]]:
        input_prompts = prompts

        if type(prompts) == str:
            input_prompts = [input_prompts]

        if type(prompts) == Dict:
            input_prompts = list(prompts.values())

        responses = self._generate_response(input_prompts, return_tok_ids)

        if type(prompts) == dict:
            return {id: response for id, response in zip(prompts.keys(), responses)}

        return responses

    def _generate_response(self, input_prompts: List[str], return_tok_ids: bool = False) -> List[str]:
        messages = [[{"role": "user", "content": prompt}] for prompt in input_prompts]
        messages = [
            self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]

        responses = []
        response_tok_ids = []

        for i in tqdm(range(len(messages))[:: self.batch_size]):
            message_batch = messages[i : i + self.batch_size]
            input_ids = self.tokenizer(message_batch, **self.tokenizer_args).to(self.model.device)
            generated_ids = self.model.generate(
                **input_ids,
                **self.generate_args,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            response = self.tokenizer.batch_decode(
                generated_ids[:, input_ids["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )  # needed to eliminate original prompt from the generated answer
            if return_tok_ids:
                response_tok_ids.append(generated_ids.cpu())
            responses.extend(response)

        if return_tok_ids:
            return responses, response_tok_ids

        return responses

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        logger.info("__exit__ Cleaning up LLMGenerator")
        del self.tokenizer
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
