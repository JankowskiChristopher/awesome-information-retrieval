import logging

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.utils import get_model, get_tokenizer

logger = logging.getLogger(__name__)


class LLMLoraTrainer(Trainer):
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        lora_cfg: DictConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        training_args: TrainingArguments,
        local: bool = False,
    ):
        tokenizer = get_tokenizer(tokenizer_name, local)
        tokenizer.pad_token = tokenizer.eos_token
        model = get_model(model_name, local=local, question_answering=True)
        lora_config = LoraConfig(**OmegaConf.to_container(lora_cfg, resolve=True))
        model = get_peft_model(model, peft_config=lora_config)

        trainable_params, all_param = model.get_nb_trainable_parameters()

        logger.info(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
