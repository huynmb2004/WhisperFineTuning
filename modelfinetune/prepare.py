from config import processor
import librosa
import pandas as pd
import numpy as np

def add_path_column(audio_dir, df):
    paths = []
    ids_df = df['id']
    ids = ids_df.values.tolist()

    for id in ids:
        path = audio_dir + '/' + id[:id.find('_')] + '/' + id + ".wav"
        paths.append(path)

    df['path'] = paths

    return df

def prepare_dataset(batch):
    audio_arrays = [x["array"] for x in batch["path"]]
    sampling_rate = batch["path"][0]["sampling_rate"]

    batch["input_features"] = [
        processor.feature_extractor(a, sampling_rate=sampling_rate).input_features[0]
        for a in audio_arrays
    ]

    batch["labels"] = [
        processor.tokenizer(t, return_tensors="np", padding="longest").input_ids[0]
        for t in batch["text"]
    ]

    return batch

# def extract_features(batch):
#     waveform, _ = librosa.load(batch["path"], sr=16000)
#     batch["speech"] = waveform
#     return batch

# def tokenize_label(batch):
#     audio_inputs = processor.feature_extractor(batch["speech"], sampling_rate=16000, return_tensor="pt")
#     batch["input_features"] = audio_inputs.input_features[0]

#     labels = processor.tokenizer(batch["text"], return_tensor="pt", padding="longest")
#     batch["labels"] = labels.input_ids[0]

#     return batch

def extract_features(batch):
    speeches = []
    for path in batch["path"]:
        waveform, _ = librosa.load(path, sr=16000)
        speeches.append(waveform)
    batch["speech"] = speeches
    return batch

def tokenize_label(batch):
    # Convert speech → input_features
    audio_inputs = processor.feature_extractor(
        batch["speech"], sampling_rate=16000, return_tensors="pt"
    )
    batch["input_features"] = [x for x in audio_inputs.input_features]

    # Convert text → token IDs
    labels = processor.tokenizer(
        batch["text"], return_tensors="pt", padding="longest"
    )
    batch["labels"] = [x for x in labels.input_ids]
    
    return batch
