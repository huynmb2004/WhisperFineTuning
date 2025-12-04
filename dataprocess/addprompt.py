import os
import pandas as pd

def create_prompt(audio_dir, prompt_file, output_csv):
  ids = []
  # Read all prompts at once
  with open(prompt_file, "r", encoding="utf8") as f:
    prompts = f.readlines()
  
  for x in os.listdir(audio_dir):
    dirpath = audio_dir + "/" + x
    j = 0
    for i, id in enumerate(os.listdir(dirpath)):
      if i > 0 and i % 4 == 0:
        j += 1
      _, text = prompts[j].strip().split(" ", 1)
      ids.append({"id": id[:id.find('.wav')], "text": text})
  
  df = pd.DataFrame(ids)
  df.to_csv(output_csv, index=False)

def create_prompt_test(prompt_file, output_csv):
  ids = []
  
  # Read all prompts at once
  with open(prompt_file, "r", encoding="utf8") as f:
    for line in f:
      id, text = line.strip().split(" ", 1)
      ids.append({"id": id, "text": text})
  
  df = pd.DataFrame(ids)
  df.to_csv(output_csv, index=False)

def create_prompt_test_noise(audio_dir, prompt_file, output_csv):
  ids = []

  with open(prompt_file, "r", encoding="utf8") as f:
    prompts = f.readlines()
  
  for x in os.listdir(audio_dir):
    dirpath = audio_dir + "/" + x
    for id in os.listdir(dirpath):
      key1 = id[:id.find('_speech')]
      for prompt in prompts:
        if key1 in prompt:
          _, text = prompt.strip().split(" ", 1)
          ids.append({"id": id[:id.find('.wav')], "text": text})

  df = pd.DataFrame(ids)
  df.to_csv(output_csv, index=False)

# create_prompt("C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/train/waves", "C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/train/prompts.txt", "C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/train/prompts.csv")
# create_prompt_test("C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/test/prompts.txt", "C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/test/prompts.csv")
# create_prompt_test_noise("C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/test/waves_noise", "C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/test/prompts.txt", "C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/test/prompts_noise.csv")

# dir = os.listdir("C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/test/waves_noise/VIVOSDEV02")

# print(dir)