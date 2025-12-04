from prepare import prepare_dataset
from builder import DatasetBuilder
import kagglehub
import os
from ffmpy import FFmpeg

# musan_path = kagglehub.dataset_download("nhattruongdev/musan-noise")

# print("Path to VIVOS dataset files:", musan_path)

# vivos_path = kagglehub.dataset_download("kynthesis/vivos-vietnamese-speech-corpus-for-asr")

# print("Path to VIVOS dataset files:", vivos_path)

# def audio_transfor(audio_path, output_dir):
#   ext = os.path.basename(audio_path).strip().split('.')[-1]
#   if ext != 'mp3':
#       raise Exception('format is not mp3')

#   result = os.path.join(output_dir, '{}.{}'.format(os.path.basename(audio_path).strip().split('.')[0], 'wav'))
#   filter_cmd = '-f wav -ac 1 -ar 16000'
#   ff = FFmpeg(
#       inputs={
#           audio_path: None}, outputs={
#           result: filter_cmd})
#   print(ff.cmd)
#   ff.run()
#   return result

# def handle(audio_dir):
#   output_dir = audio_dir.replace("clips", "wav")
#   os.makedirs(output_dir, exist_ok=True)
#   print("Path to dataset files:",audio_dir)
#   for i, x in enumerate(os.listdir(audio_dir)):
#     if i == 500:
#        break
#     audio_transfor(os.path.join(audio_dir, x), output_dir)

# path = kagglehub.dataset_download("minhquangphamle/common-voice-vi-21")

# handle("C:/Users/huyng/.cache/kagglehub/datasets/minhquangphamle/common-voice-vi-21/versions/1/vi/clips")

# base_csv = "output/base_train.csv"
# prepare_dataset("C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/train/prompts.txt", "C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/train/waves", base_csv)

# builder = DatasetBuilder(base_csv)
# builder.build_with_noise("output/dataset_train.csv")

base_csv = "output/base_test.csv"
prepare_dataset("C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/test/prompts.txt", "C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/test/waves", base_csv)

builder = DatasetBuilder(base_csv)
builder.build_with_noise("output/dataset_test.csv", test=True)