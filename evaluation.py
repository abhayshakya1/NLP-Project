import os
import torch
from transformers import BartTokenizerFast, BartForConditionalGeneration
from datasets import load_from_disk
from bart_train import create_preprocessor
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score

# Compute the BLEU score between a reference sentence and a candidate sentence.
def compute_bleu_score(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    return sentence_bleu(reference_tokens, candidate_tokens)


# Compute the BERT score between a list of reference sentences and candidate sentences.
def compute_bert_score(references, candidates):
    P, R, F1 = bert_score(candidates, references, lang="en")
    avg_precision = P.mean().item()
    avg_recall = R.mean().item()
    avg_f1 = F1.mean().item()

    return avg_precision, avg_recall, avg_f1

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load the trained model and tokenizer
    tokenizer = BartTokenizerFast.from_pretrained("/content/drive/MyDrive/Colab Notebooks/NLP_PROJECT/_artifacts/bart")
    model = BartForConditionalGeneration.from_pretrained("/content/drive/MyDrive/Colab Notebooks/NLP_PROJECT/_artifacts/bart")

    # Load the validation dataset
    val_data = load_from_disk("/content/drive/MyDrive/Colab Notebooks/NLP_PROJECT/data/therapy_test")

    # Create the preprocessor function
    pp_func = create_preprocessor(tokenizer)

    # Lists to store references and candidates
    references = []
    candidates = []

    # Iterate over the validation dataset
    for idx in range(len(val_data)):
        text = val_data[idx]["input_text"]
        target = val_data[idx]["target_text"]

        # Preprocess the input text
        inp = pp_func(text)

        # Ensure the input is within the tokenizer's max length
        if "input_ids" in inp:
            input_ids = inp["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.squeeze(0)  # Remove batch dimension if present
            if len(input_ids) > tokenizer.model_max_length:
                print(f"Input text at index {idx} exceeds max length. Truncating...")
                inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)

        # Generate the output text
        try:
            out = model.generate(
                **inp,
                max_length=1024,
                # do_sample=True,
                # num_beams=5,
                # no_repeat_ngram_size=2,
                # early_stopping=True,
            )
            out_text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"Error generating output for index {idx}: {e}")
            continue

        references.append(target)
        candidates.append(out_text)

    bleu_scores = [compute_bleu_score(ref, cand) for ref, cand in zip(references, candidates)]
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

    avg_precision, avg_recall, avg_f1 = compute_bert_score(references, candidates)

    print(f"Average BLEU Score: {avg_bleu_score}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F1 Score: {avg_f1}")

if __name__ == "__main__":
    main()