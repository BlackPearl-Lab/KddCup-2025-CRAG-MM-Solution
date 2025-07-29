import torch
import torch.nn as nn

class Loss(object):
    '''
    损失函数的父类
    '''
    def __call__(self, model, inputs, training_args, return_outputs=False):
        raise NotImplemented
        
class generalLMLoss(Loss):
    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        
    def __call__(self, model, inputs, training_args, return_outputs=False, do_eval=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        pixel_values = inputs['pixel_values']
        aspect_ratio_ids = inputs['aspect_ratio_ids']
        aspect_ratio_mask = inputs['aspect_ratio_mask']
        cross_attention_mask = inputs['cross_attention_mask']
        target_mask = inputs['target_mask']
        
        if do_eval:
            model.eval()
            with torch.no_grad():
                # feed forward
                # outputs = model(input_ids=input_ids, 
                #                 attention_mask=attention_mask,
                #                 return_dict=True)
                outputs = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    pixel_values = pixel_values,
                    aspect_ratio_ids = aspect_ratio_ids,
                    aspect_ratio_mask = aspect_ratio_mask,
                    cross_attention_mask = cross_attention_mask,
                    return_dict=True
                )
            model.train()
        else:
            # feed forward
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                pixel_values = pixel_values,
                aspect_ratio_ids = aspect_ratio_ids,
                aspect_ratio_mask = aspect_ratio_mask,
                cross_attention_mask = cross_attention_mask,
                return_dict=True
            )
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]

        # mask and compute loss
        labels = torch.where(target_mask==1, input_ids, self.ignore_index)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss
        