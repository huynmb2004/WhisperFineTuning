from datasets import Dataset
from prepare import prepare_dataset
import gc
from datasets import load_from_disk, concatenate_datasets
import os
from config import processor
import evaluate

def chunking_and_mapping(dataset, dir, num_shards=1, last_index=0):
    for i in range(last_index, num_shards):
        print(f"Processing shard {i+1}/{num_shards} for {dir}")
        small_ds = dataset.shard(num_shards=num_shards, index=i)
        
        # Process without multiprocessing to avoid Windows audio serialization issues
        small_ds = small_ds.map(
            prepare_dataset,
            batched=True,
            batch_size=4,  # Smaller batch size for memory
            num_proc=None,  # Disable multiprocessing
            keep_in_memory=False
        )
        
        # Save each shard
        save_path = f"data/Dataset/{dir}/shard_{i}"
        print(f"Saving to {save_path}")
        small_ds.save_to_disk(save_path)
        
        # Clean up
        del small_ds
        gc.collect()

def loading_and_concatenating(dir, num_shards=1):
    paths = [f"data/Dataset/{dir}/shard_{i}" for i in range(num_shards) if os.path.exists(f"data/Dataset/{dir}/shard_{i}")]
    datasets_list = [load_from_disk(p) for p in paths]
    dataset = concatenate_datasets(datasets_list)

    del datasets_list
    gc.collect()

    return dataset


metric = evaluate.load("evaluate/metrics/wer/wer.py")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

