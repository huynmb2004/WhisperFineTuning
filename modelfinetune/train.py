from config import model, processor
from prepare import add_path_column, extract_features, tokenize_label, prepare_dataset
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset, Audio
import torch
from collator import DataCollatorSpeechSeq2SeqWithPadding
import optuna
import gc
from utils import chunking_and_mapping, loading_and_concatenating
# from hypertune import objective
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id
)

def objective(trial):
    # Các tham số cần tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-4)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [2, 5, 10])
    num_train_epochs = trial.suggest_int('num_train_epochs', 1, 3)

    # Khai báo TrainingArguments — sử dụng giá trị từ Optuna
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"/kaggle/working/whisper-base-optuna-{trial.number}",  
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,  
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_steps=100,
        max_steps=400,
        eval_steps=50,
        logging_steps=100,
        # disable gradient checkpointing and mixed precision for debugging double-backward issues
        gradient_checkpointing=False,
        fp16=True,
        save_strategy="steps",
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["none"],  # tránh gửi tensorboard cho mỗi trial
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,  # nên tắt khi tuning
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    # Enable anomaly detection to get clearer tracebacks for autograd errors
    try:
        with torch.autograd.detect_anomaly():
            trainer.train()
    except Exception:
        # Re-raise after printing a marker so Optuna captures the trial failure
        import traceback
        traceback.print_exc()
        raise
    eval_results = trainer.evaluate(eval_dataset=dataset["validation"])
    return eval_results["eval_wer"]

if __name__ == "__main__":
    # path = "C:/Users/huyng/.cache/kagglehub/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/versions/1"

    # train_csv_path = path + "/vivos/train/prompts.csv"
    # clean_test_csv_path = path + "/vivos/test/prompts.csv"
    # noise_test_csv_path = path + "/vivos/test/prompts_noise.csv"

    # train_df = pd.read_csv(train_csv_path, dtype={"text": "category"})
    # clean_test_df = pd.read_csv(clean_test_csv_path)
    # noise_test_df = pd.read_csv(noise_test_csv_path)

    # train_df = add_path_column(path + "/vivos/train/waves", train_df)
    # clean_test_df = add_path_column(path + "/vivos/test/waves", clean_test_df)
    # noise_test_df = add_path_column(path + "/vivos/test/waves_noise", noise_test_df)

    # train_df = train_df[['text', 'path']]
    # clean_test_df = clean_test_df[['text', 'path']]
    # noise_test_df = noise_test_df[['text', 'path']]

    # train_dataset = Dataset.from_pandas(train_df)
    # clean_test_dataset = Dataset.from_pandas(clean_test_df)
    # noise_test_dataset = Dataset.from_pandas(noise_test_df)

    # split = train_dataset.train_test_split(test_size=0.05, seed=0)
    # train_dataset = split["train"]
    # val_dataset = split["test"]


    # dataset = DatasetDict({
    #     'train': train_dataset,
    #     'validation': val_dataset,
    #     'test_clean': clean_test_dataset,
    #     'test_noise': noise_test_dataset,
    # })

    # # Cast to Audio type with forced sampling rate
    # print("Casting audio columns...")
    # dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    # # Process each split
    # print("\nProcessing train split...")
    # chunking_and_mapping(dataset["train"], "train", num_shards=1)

    # print("\nProcessing validation split...")
    # chunking_and_mapping(dataset["validation"], "validation", num_shards=1)
    
    # print("\nProcessing test_clean split...")
    # chunking_and_mapping(dataset["test_clean"], "test_clean", num_shards=1)
    
    # print("\nProcessing test_noise split...")
    # chunking_and_mapping(dataset["test_noise"], "test_noise", num_shards=1)
    
    # print("\nAll splits processed successfully!")

    train_dataset = loading_and_concatenating("train", num_shards=20)
    val_dataset = loading_and_concatenating("validation", num_shards=1)
    clean_test_dataset = loading_and_concatenating("test_clean", num_shards=1)
    noise_test_dataset = loading_and_concatenating("test_noise", num_shards=1)

    global dataset 
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test_clean': clean_test_dataset,
        'test_noise': noise_test_dataset,
    })

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)

    best_params = study.best_params