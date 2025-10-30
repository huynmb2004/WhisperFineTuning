from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

processor = AutoProcessor.from_pretrained("model/PhoWhisper-base")

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

model = WhisperForConditionalGeneration.from_pretrained("model/PhoWhisper-base", quantization_config=quantization_config, device_map="auto")

model.config.use_cache = False
model.gradient_checkpointing_disable()

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

# def make_inputs_require_grad(module, input, output):
#     output.requires_grad_(True)

# model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()