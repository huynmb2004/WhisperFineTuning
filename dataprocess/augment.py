import librosa, numpy as np, random
import soundfile as sf
import os

def process_sound(clean, noise, snr_db=10, index=0):
    if index > len(clean):
        index=0
    if len(noise) < len(clean) + index:
        repeats = int(np.ceil((len(clean) + index) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[index:len(clean) + index]

    rms_clean = np.sqrt(np.mean(clean**2))
    rms_noise = np.sqrt(np.mean(noise**2))
    desired_rms_noise = rms_clean / (10**(snr_db/25))
    noise = noise * (desired_rms_noise / rms_noise)

    return noise

def add_noise(clean_audio, type, clean_sr=16000):
    
    noise_dir="C:/Users/huyng/.cache/kagglehub/datasets/nhattruongdev/musan-noise/versions/1/musan/noise/free-sound"
    speech_dir_1="C:/Users/huyng/.cache/kagglehub/datasets/nhattruongdev/musan-noise/versions/1/musan/speech"
    speech_dir_2="C:/Users/huyng/.cache/kagglehub/datasets/minhquangphamle/common-voice-vi-21/versions/1/vi/wav"
    music_dir = "C:/Users/huyng/.cache/kagglehub/datasets/nhattruongdev/musan-noise/versions/1/musan/music/hd-classical"

    if type == "speech+speech":
        print("Speech+Speech...")
        noise_files_1 = librosa.util.find_files(speech_dir_1, ext=['wav'])
        noise_files_2 = librosa.util.find_files(speech_dir_2, ext=['wav'])
    elif type == "speech+noise":
        print("Speech+Noise")
        noise_files_1 = librosa.util.find_files(speech_dir_1, ext=['wav'])
        noise_files_2 = librosa.util.find_files(noise_dir, ext=['wav'])
    elif type == "speech+music":
        print("Speech+Music")
        noise_files_1 = librosa.util.find_files(speech_dir_1, ext=['wav'])
        noise_files_2 = librosa.util.find_files(music_dir, ext=['wav'])

    # 1. Chọn file random
    if not noise_files_1:
        raise FileNotFoundError(f"No .wav files found in type {type}")
    noise_file_1 = random.choice(noise_files_1)

    if not noise_files_2:
        raise FileNotFoundError(f"No .wav files found in type {type}")
    noise_file_2 = random.choice(noise_files_2)

    # noise_file_1 = "C:/Users/huyng/.cache/kagglehub/datasets/nhattruongdev/musan-noise/versions/1/musan/noise/free-sound/noise-free-sound-0064.wav"
    # noise_file_2 = "c:/Users/huyng/.cache/kagglehub/datasets/minhquangphamle/common-voice-vi-21/versions/1/vi/wav/common_voice_vi_21833244.wav"

    # 2. Load noise
    noise_1, _ = librosa.load(noise_file_1, sr=clean_sr)
    noise_2, _ = librosa.load(noise_file_2, sr=clean_sr)

    # 3. Speed up
    if type == "speech+noise":
        noise_1 = librosa.effects.time_stretch(noise_1, rate=1.5)

    # 3. Xử lý âm thanh
    noise_2 = process_sound(clean_audio, noise_2)
    noise_1 = process_sound(clean_audio, noise_1, index=len(noise_1)//4)
    
    # 4. Mix
    min_len = min(len(clean_audio), len(noise_1), len(noise_2))
    noisy_audio = clean_audio[:min_len] + noise_1[:min_len] + noise_2[:min_len]
    return noisy_audio

# def add_noise(clean_audio, noise_dir="C:/Users/huyng/.cache/kagglehub/datasets/nhattruongdev/musan-noise/versions/1/musan/noise", speech_dir="C:/Users/huyng/.cache/kagglehub/datasets/nhattruongdev/musan-noise/versions/1/musan/speech", clean_sr=16000):
    

#     # 1. Chọn file random
#     noise_files = librosa.util.find_files(noise_dir, ext=['wav'])
#     noise_file = random.choice(noise_files)

#     speech_files = librosa.util.find_files(speech_dir, ext=['wav'])
#     speech_file = random.choice(speech_files)

#     # 2. Load noise
#     noise, _ = librosa.load(noise_file, sr=clean_sr)
#     speech, _ = librosa.load(speech_file, sr=clean_sr)

#     # 3. Speed up
#     noise = librosa.effects.time_stretch(noise, rate=1.5)

#     # 3. Xử lý âm thanh
#     noise = process_sound(clean_audio, noise)
#     speech = process_sound(clean_audio, speech, index=len(speech)//4)
    
#     # 4. Mix
#     noisy_audio = clean_audio + noise + speech
#     return noisy_audio

def save_audio(clean_path, noise_audio, type, sr=16000, test=False):
    key = ".wav"
    pos = clean_path.find(key)
    path = clean_path[:pos]
    noise_path = path + "_" + type + key

    if test:
        temp_path = clean_path.replace("waves", "waves_noise")
        pos = temp_path.find(key)
        path = temp_path[:pos]
        noise_path = path + "_" + type + key


    # noise_path = clean_path + "/" + "noise_audio15.wav"

    print(f"Path: {noise_path}")
    os.makedirs(os.path.dirname(noise_path), exist_ok=True)
    sf.write(noise_path, noise_audio, sr)
    return noise_path

