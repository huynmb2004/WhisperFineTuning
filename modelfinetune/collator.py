import torch
from config import processor
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __init__(self, processor: Any, decoder_start_token_id: int):
        # store processor and decoder start id for padding/token handling
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# class DataCollator:
#    def __call__(self, features):
#        input_features = [torch.tensor(feature["input_features"]).squeeze(0) for feature in features]
#        labels = [torch.tensor(feature["labels"]).squeeze(0) for feature in features]
#        # Pad sequences
#        input_features_padded = pad_sequence(input_features, batch_first=True)
#        labels_padded = pad_sequence(labels, batch_first=True)
#        return {"input_features": input_features_padded, "labels": labels_padded}

# def collate(batch):
#     input_features = [item['input_features'] for item in batch]
#     labels = [item['labels'] for item in batch]

#     input_features = pad_sequence(input_features, batch_first=True).to(device)
#     labels = pad_sequence(labels, batch_first=True)

#     return {"input_features": input_features, "labels": labels}