import os
import pandas as pd
import librosa
from augment import add_noise, save_audio
import random

class DatasetBuilder:
    def __init__(self, base_csv):
        self.base_csv = pd.read_csv(base_csv)

    def build_with_noise(self, output_csv, test=False):
        new_rows = []
        types = ["speech+speech", "speech+noise", "speech+music"]
        if test:
            
            for _, row in self.base_csv.iterrows():
                new_rows.append({"path": row["path"], "transcript": row["transcript"]})

                type = random.choice(types)
                audio, _ = librosa.load(row["path"], sr=None)

                noise_speech_audio  = add_noise(audio, type)
                noise_speech_path = save_audio(row["path"], noise_speech_audio, type, test=True)
                new_rows.append({"path": noise_speech_path, "transcript": row["transcript"]})

            df_new = pd.DataFrame(new_rows)
            df_new.to_csv(output_csv, index=False)
            return df_new


        for _, row in self.base_csv.iterrows():
            new_rows.append({"path": row["path"], "transcript": row["transcript"]})

            audio, _ = librosa.load(row["path"], sr=None)
            for type in types:
                noise_speech_audio  = add_noise(audio, type)
                noise_speech_path = save_audio(row["path"], noise_speech_audio, type)
                new_rows.append({"path": noise_speech_path, "transcript": row["transcript"]})

        df_new = pd.DataFrame(new_rows)
        df_new.to_csv(output_csv, index=False)
        return df_new
