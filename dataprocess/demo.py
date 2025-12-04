import librosa
from augment import add_noise, save_audio
path = "C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1/vivos/train/waves/VIVOSSPK02/VIVOSSPK02_R003.wav"
audio, sr = librosa.load(path, sr=16000)
audio_noise = add_noise(audio, type="speech+speech")
save_audio("noise_audio", audio_noise, type="speech+speech")
