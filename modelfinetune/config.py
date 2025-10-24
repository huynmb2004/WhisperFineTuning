# config.py
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_PATH = "vinai/PhoWhisper-base"

processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)