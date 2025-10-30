import optuna
from config import model, processor, tokenizer
from dataset import load_metric
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from collator import DataCollator

def objective(trial, dataset):
    # Các tham số cần tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-4)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [2, 5, 10])
    num_train_epochs = trial.suggest_int('num_train_epochs', 1, 3)

    # Khai báo TrainingArguments — sử dụng giá trị từ Optuna
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./hypertuning/whisper-base-optuna-{trial.number}",  
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,  
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_steps=100,
        max_steps=400,
        eval_steps=50,
        logging_steps=100,
        gradient_checkpointing=True,
        fp16=True,
        save_strategy="steps",
        evaluation_strategy="steps",
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
        data_collator=DataCollator,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    eval_results = trainer.evaluate(eval_dataset=dataset["validation"])
    return eval_results["eval_wer"]



 