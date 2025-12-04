import pandas as pd
import os

def rename_dataset(prompt_file, audio_dir):
  types = ["speech+speech", "speech+noise", "speech+music"]
  with open(prompt_file, "r", encoding="utf8") as f:
      for line in f:
          audio_id, text = line.strip().split(" ", 1)
          prefix = audio_dir + "/" + audio_id[:audio_id.find("_")] + "/" + audio_id
          for type in types:
            old_path = prefix + type + ".wav"
            new_path = prefix + "_" + type + ".wav"
            os.rename(old_path, new_path)

rename_dataset("C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/train/prompts.txt", "C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/train/waves")