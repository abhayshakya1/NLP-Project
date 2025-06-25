import os

from transformers import BartTokenizerFast, BartForConditionalGeneration
from datasets import load_from_disk

from bart_train import create_preprocessor


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = BartTokenizerFast.from_pretrained("_artifacts/bart")
    model = BartForConditionalGeneration.from_pretrained("_artifacts/bart")

    val_data = load_from_disk("_data/therapy_val")
    idx = 13
    text = val_data[idx]["input_text"]
    target = val_data[idx]["target_text"]

    pp_func = create_preprocessor(tokenizer)
    inp = pp_func(text)

    out = model.generate(
        **inp,
        max_length=200,
        # do_sample=True,
        # num_beams=5,
        # no_repeat_ngram_size=2,
        # early_stopping=True,
    )
    out_text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    print("Input:\n", text, "\n")
    print("Output:\n", out_text, "\n")
    print("Target:\n", target)


if __name__ == "__main__":
    main()
