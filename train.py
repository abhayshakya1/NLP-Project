from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from evaluate import load
import numpy as np
import torch

# Step 1: Load the dataset
train_dataset = load_from_disk('_data/therapy_train')  # Update path if needed
test_dataset = load_from_disk('_data/therapy_test')    # Update path if needed
val_dataset = load_from_disk('_data/therapy_val')      # Update path if needed

# Step 2: Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Step 3: Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(
        examples["input_text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )
    targets = tokenizer(
        examples["target_text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(["input_text", "target_text"])
test_dataset = test_dataset.remove_columns(["input_text", "target_text"])
val_dataset = val_dataset.remove_columns(["input_text", "target_text"])

# Step 4: Load the pre-trained model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Step 5: Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=False,  # Disabled for MPS
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=500,
)

# Step 6: Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
)

# Step 7: Fine-tune the model
trainer.train()

# Step 8: Load evaluation metrics
bleu = load("bleu")
bertscore = load("bertscore")

# Step 9: Evaluate the model on the test dataset
chunk_size = 100
all_predictions = []
all_references = []

for i in range(0, len(test_dataset), chunk_size):
    chunk = test_dataset.select(range(i, min(i + chunk_size, len(test_dataset))))
    
    # Prepare inputs for generation
    input_ids = torch.tensor(chunk["input_ids"]).to(model.device)
    attention_mask = torch.tensor(chunk["attention_mask"]).to(model.device)
    
    # Generate predictions using model.generate
    pred_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=64,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    # Get ground truth labels
    label_ids = chunk["labels"]
    
    # Decode predictions and labels to text
    pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids if len(ids) > 0]
    label_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in label_ids if len(ids) > 0]
    
    # Append to lists
    all_predictions.extend(pred_texts)
    all_references.extend(label_texts)

# Compute BLEU and BERTScore
bleu_score = bleu.compute(predictions=all_predictions, references=all_references)
bleu_score_float = bleu_score["bleu"]
bertscore_score = bertscore.compute(predictions=all_predictions, references=all_references, lang="en")

# Print results
print(f"BLEU Score: {bleu_score}")
print(f"BERTScore: {bertscore_score}")

# Step 10: Function to generate the next utterance
def generate_next_utterance(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = inputs.input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_length=128,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Step 11: Test the model
input_text = "T: Hi, how are you doing today? [SEP] P: I'm feeling a bit anxious."
generated_response = generate_next_utterance(input_text)
print(f"Input: {input_text}")
print(f"Generated Response: {generated_response}")

# Step 12: Save the model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")