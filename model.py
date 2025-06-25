import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import BartForConditionalGeneration, BartTokenizerFast
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


class NextUttPreprocessor:
    def __init__(self, num_tokens: int = 1024):
        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
        self.tokenizer.truncation_side = "left"
        self.special_ids = self.tokenizer.all_special_ids
        self.num_tokens = num_tokens

        # Therapist will have id 0, Patient id 1 in the utterer_ids
        self.start_utterer_labels = self.tokenizer(
            ["T:", "P:"],
            add_special_tokens=False,
        )["input_ids"]
        self.utterer_labels = self.tokenizer(
            [" T:", " P:"],
            add_special_tokens=False,
        )["input_ids"]

    def _find_utterer_label(self, ids: list[int], start_i: int):
        if start_i == 0:
            for ui, utt_l in enumerate(self.start_utterer_labels):
                if ids[:2] == utt_l: return 0, ui

        for i in range(start_i, len(ids)-1):
            id_pair = ids[i:i+2]
            for ui, utt_l in enumerate(self.utterer_labels):
                if id_pair == utt_l: return i, ui

        return len(ids), None

    def generate_utter_ids(self, input_ids: Tensor) -> Tensor:
        assert input_ids.ndim == 2, input_ids.shape

        mask = torch.zeros_like(input_ids).to(torch.int32)
        ids_batch = input_ids.tolist()
        for i, ids in enumerate(ids_batch):
            idx, utt_id = self._find_utterer_label(ids, 0)
            if utt_id is not None:
                mask[i, :idx] = 0 if utt_id == 1 else 1
            else: mask[i, :idx] = 1

            while idx < len(ids):
                next_idx, next_utt_id = self._find_utterer_label(ids, idx+2)
                mask[i, idx : next_idx] = utt_id
                idx, utt_id = next_idx, next_utt_id

        for sp_id in self.special_ids:
            mask[input_ids==sp_id] = 2 # Padding for utterer_ids
        return mask

    def __call__(
        self,
        input_text: list[str],
        target_text: list[str] = None
    ) -> dict[str, Tensor]:

        if isinstance(input_text, str):
            raise ValueError("Inputs must always be list of strings")

        input_text = [t.replace("[SEP]", self.tokenizer.bos_token) for t in input_text]
        inputs = self.tokenizer(
            input_text,
            text_target=target_text,
            max_length=self.num_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs["utterer_ids"] = self.generate_utter_ids(inputs["input_ids"])
        return inputs


@torch.no_grad()
def modify_max_pos_emb(model: BartForConditionalGeneration, new_max_pos_emb: int):
    sd = model.state_dict()

    new_config = model.config
    new_config.max_position_embeddings = new_max_pos_emb
    new_model = model.__class__(new_config)

    prev_pos_embeds = sd['model.encoder.embed_positions.weight']
    new_pos_embeds = new_model.model.encoder.embed_positions.weight
    embed_size = min(prev_pos_embeds.shape[0], new_pos_embeds.shape[0])
    new_pos_embeds[:embed_size] = prev_pos_embeds[:embed_size]
    sd['model.encoder.embed_positions.weight'] = new_pos_embeds

    prev_pos_embeds = sd['model.decoder.embed_positions.weight']
    new_pos_embeds = new_model.model.decoder.embed_positions.weight
    embed_size = min(prev_pos_embeds.shape[0], new_pos_embeds.shape[0])
    new_pos_embeds[:embed_size] = prev_pos_embeds[:embed_size]
    sd['model.decoder.embed_positions.weight'] = new_pos_embeds

    new_model.load_state_dict(sd)
    return new_model


class NextUtterenceModel(nn.Module):
    def __init__(self, max_tokens: int = 1024):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.embedder = self.bart.get_encoder().embed_tokens
        self.utterer_encoder = nn.Embedding(3, self.embedder.embedding_dim, 2)

        if max_tokens != self.bart.config.max_position_embeddings:
            self.bart = modify_max_pos_emb(self.bart, max_tokens)
        max_tokens = torch.tensor(max_tokens, dtype=torch.uint32)
        self.register_buffer("max_tokens", max_tokens)

    def forward(
        self,
        input_ids: Tensor,
        utterer_ids: Tensor,
        attention_mask: Tensor = None,
        labels: Tensor | None = None,
        **generate_kwargs,
    ) -> Tensor:
        if input_ids.shape != utterer_ids.shape:
            raise ValueError("Shape of `patient_mask` should be same as `input_ids`")

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        embeds = self.embedder(input_ids)
        utt_enc = self.utterer_encoder(utterer_ids)
        embeds += utt_enc

        if labels is not None:
            out = self.bart(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            return out.loss
        else:
            out = self.bart.generate(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
            return out

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        utterer_ids: Tensor,
        attention_mask: Tensor = None,
        **generate_kwargs,
    ):
        self.eval()
        return self(
            input_ids,
            utterer_ids,
            attention_mask,
            **generate_kwargs,
        )

    def load_weights(self, path: Path | str, map_location: str = None):
        wgts = torch.load(path, map_location)
        max_tokens = wgts["max_tokens"].item()
        if max_tokens != self.bart.config.max_position_embeddings:
            self.bart = modify_max_pos_emb(self.bart, max_tokens)
        self.load_state_dict(wgts)


def train_model(
    model: NextUtterenceModel,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int,
    lr: float,
    device: str,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for e in range(epochs):
        print(f"\nEpoch {e+1}/{epochs}:")
        total_loss, num_samples = 0, 0
        model.train()
        for data in tqdm(train_dl, "Training"):
            inputs = {k: v.to(device) for k, v in data.items()}
            loss = model(**inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs["input_ids"].shape[0]
            num_samples += inputs["input_ids"].shape[0]

        train_losses.append(total_loss / num_samples)
        total_loss, num_samples = 0, 0

        model.eval()
        with torch.inference_mode():
            for data in tqdm(val_dl, "Validating"):
                inputs = {k: v.to(device) for k, v in data.items()}
                loss = model(**inputs)
                total_loss += loss.item() * inputs["input_ids"].shape[0]
                num_samples += inputs["input_ids"].shape[0]

        val_losses.append(total_loss / num_samples)
        print("Training Loss:", train_losses[-1])
        print("Validation Loss:", val_losses[-1])

    return train_losses, val_losses


def save_model_data(
    model,
    train_losses: list[float],
    val_losses: list[float],
    save_dir: Path | str,
) -> None:
    assert len(train_losses) == len(val_losses), \
        f"{len(train_losses)} != {len(val_losses)}"

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "weights.pt")

    epochs = len(train_losses)
    loss_df = pd.DataFrame({
        "epoch": list(range(1, epochs+1)),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    loss_df.to_csv(save_dir / "losses.csv", index=False)

    plt.plot(range(1, epochs+1), train_losses, "r", label="Train Loss")
    plt.plot(range(1, epochs+1), val_losses, "g", label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_plot.png")


def train(save_dir: str):
    max_tokens = 1024
    epochs = 10
    lr = 1e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl_params = {
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": device == "cuda",
    }
    torch.manual_seed(89421)

    pp = NextUttPreprocessor(max_tokens)
    train_ds = load_from_disk("_data/therapy_train")
    val_ds = load_from_disk("_data/therapy_val")
    train_ds = train_ds.map(
        lambda x: pp(**x),
        batched=True,
        remove_columns=["input_text", "target_text"],
    )
    val_ds = val_ds.map(
        lambda x: pp(**x),
        batched=True,
        remove_columns=["input_text", "target_text"],
    )
    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    train_dl = DataLoader(train_ds, shuffle=True, **dl_params)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_params)
    model = NextUtterenceModel(max_tokens)

    if os.path.exists(save_dir):
        print(f"!!Warning!! Contents of {save_dir} will be overwritten!")
    tloss, vloss = train_model(model, train_dl, val_dl, epochs, lr, device)
    save_model_data(model, tloss, vloss, save_dir)
    print("\nModel artifacts saved at", save_dir)


def eval_model(load_dir: str):
    from evaluation import evaluate

    load_dir = Path(load_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NextUtterenceModel()
    model.load_weights(load_dir / "weights.pt", "cpu")
    model.to(device)
    pp = NextUttPreprocessor(model.max_tokens.item())

    bleu_score, (bert_p, bert_r, bert_f1) = evaluate(
        model, pp.tokenizer, "_data/therapy_test", pp, None, dict(
            max_length = model.max_tokens.item(),
            # do_sample=True,
            # num_beams=5,
            # no_repeat_ngram_size=2,
            # early_stopping=True,
        )
    )
    print(f"\nBLEU Score: {bleu_score}")
    print(f"BERT Precision: {bert_p}")
    print(f"BERT Recall: {bert_r}")
    print(f"BERT F1-Score: {bert_f1}")


def inference(load_dir: str):
    load_dir = Path(load_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NextUtterenceModel()
    model.load_weights(load_dir / "weights.pt", "cpu")
    model.to(device)
    pp = NextUttPreprocessor(model.max_tokens.item())

    test_data = load_from_disk("_data/therapy_test")
    idx = int(sys.argv[1]) if sys.argv[1].isnumeric() else 100
    text = test_data[idx]["input_text"]
    target = test_data[idx]["target_text"]

    inp = {k: v.to(device) for k, v in pp([text]).items()}

    out = model.generate(
        **inp,
        max_length=model.max_tokens.item(),
        # do_sample=True,
        # num_beams=5,
        # no_repeat_ngram_size=2,
        # early_stopping=True,
    )
    out_text = pp.tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    print("Sample index:", idx, "\n")
    print("Input:\n", text, "\n")
    print("Output:\n", out_text, "\n")
    print("Target:\n", target)


def main():
    model_dir = "_artifacts/model/base"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if len(sys.argv) == 1:
        train(model_dir)
    elif sys.argv[1] == "eval":
        eval_model(model_dir)
    else:
        inference(model_dir)


if __name__ == "__main__":
    main()
