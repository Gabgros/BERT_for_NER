"""
# Author: Yinghao Li
# Modified: September 30th, 2023
# ---------------------------------------
# Description: collate function for batch processing
"""

import torch
from transformers import DataCollatorForTokenClassification

from .batch import unpack_instances, Batch


class DataCollator(DataCollatorForTokenClassification):
    def __call__(self, instance_list: list[dict]):
        tk_ids, attn_masks, lbs = unpack_instances(instance_list, ["bert_tk_ids", "bert_attn_masks", "bert_lbs"])

        # Update `tk_ids`, `attn_masks`, and `lbs` to match the maximum length of the batch.
        # The updated type of the three variables should be `torch.int64``.
        # Hint: some functions and variables you may want to use: `self.tokenizer.pad()`, `self.label_pad_token_id`.
        # --- TODO: start of your code ---
        max_batch_length = max(len(token) for token in tk_ids)

        tk_ids_padded = []
        attn_masks_padded = []
        lbs_padded = []

        for tk, attn_mask, lb in zip(tk_ids, attn_masks, lbs):
            padding_length = max_batch_length - len(tk)

            tk_padded = tk + [self.tokenizer.pad_token_id] * padding_length
            attn_mask_padded = attn_mask + [0] * padding_length
            lb_padded = lb + [self.label_pad_token_id] * padding_length

            tk_ids_padded.append(tk_padded)
            attn_masks_padded.append(attn_mask_padded)
            lbs_padded.append(lb_padded)

        tk_ids_padded = torch.tensor(tk_ids_padded, dtype=torch.int64)
        attn_masks_padded = torch.tensor(attn_masks_padded, dtype=torch.int64)
        lbs_padded = torch.tensor(lbs_padded, dtype=torch.int64)
        # --- TODO: end of your code ---

        return Batch(input_ids=tk_ids_padded, attention_mask=attn_masks_padded, labels=lbs_padded)
