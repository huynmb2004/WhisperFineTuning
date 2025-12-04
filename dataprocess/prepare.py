import pandas as pd
import os

def prepare_dataset(prompt_file, audio_dir, output_csv):
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    data = []
    with open(prompt_file, "r", encoding="utf8") as f:
        for line in f:
            audio_id, text = line.strip().split(" ", 1)
            wav_path = audio_dir + "/" + audio_id[:audio_id.find("_")] + "/" + audio_id + ".wav"
            data.append({"path": wav_path, "transcript": text})
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    return df
