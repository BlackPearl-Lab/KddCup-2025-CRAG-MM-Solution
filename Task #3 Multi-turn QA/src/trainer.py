import os
import torch
from transformers import Trainer, Seq2SeqTrainer
from transformers.modeling_utils import unwrap_model, PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


class LoRATrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # model_to_save = self.model
        # state_dict = {k: v.to("cpu") for k, v in model_to_save.named_parameters() if v.requires_grad}
        # # Using Hugging Face's save_pretrained instead of PyTorch's torch.save
        # model_to_save.save_pretrained(output_dir, state_dict=state_dict, save_function=torch.save,safe_serialization=False)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        
        # Save tokenizer and training arguments as usual
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # # print(self.args)
        # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME, ))
