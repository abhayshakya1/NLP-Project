import torch
from transformers import BartTokenizerFast, BartForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_from_disk


train_data = load_from_disk("conv_data/therapy_train")
val_data = load_from_disk("conv_data/therapy_val")

tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

def tokenize_function(items):
    items["input_text"] = [x.replace(" [SEP]", "") for x in items["input_text"]]
    items["target_text"] = [x.replace(" [SEP]", "") for x in items["target_text"]]
    inputs = tokenizer(items["input_text"], text_target=items["target_text"],
                       truncation=True, padding="max_length", max_length=512)
    return inputs

train_data = train_data.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])
val_data = val_data.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])

train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training Arguments
batch_size = 2

training_args = TrainingArguments(
    output_dir="_artifacts/bart_therapy",
    overwrite_output_dir=True,
    eval_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    save_total_limit=1,
    logging_dir="_artifacts/logs",
    warmup_steps=100,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True if torch.cuda.is_available() else False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()

model.save_pretrained("_artifacts/bart_therapy")
tokenizer.save_pretrained("_artifacts/bart_therapy")
