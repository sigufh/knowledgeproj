from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer

from kg_pipeline.utils.spans import bio_to_entities, entities_to_bio, score_entities


@dataclass
class BERTCRFConfig:
    model_name: str = "bert-base-chinese"
    max_length: int = 128
    lr: float = 5e-5
    batch_size: int = 8
    epochs: int = 3
    weight_decay: float = 0.01


class _NERDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.samples[idx]
        text = row["text"]
        char_tags = entities_to_bio(text, row.get("entities", []))

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        offsets = encoded["offset_mapping"].squeeze(0).tolist()

        labels = torch.zeros(self.max_length, dtype=torch.long)
        valid_mask = torch.zeros(self.max_length, dtype=torch.bool)

        for i, (start, end) in enumerate(offsets):
            if attention_mask[i].item() == 0:
                continue
            if start == end:
                continue
            valid_mask[i] = True
            if start < len(char_tags):
                tag = char_tags[start]
            else:
                tag = "O"
            labels[i] = self.label2id.get(tag, self.label2id["O"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "valid_mask": valid_mask,
        }


class _BertCRF(nn.Module):
    def __init__(self, model_name: str, num_labels: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))
        # CRF requires first timestep in each sequence to be valid.
        # Use attention_mask for CRF, and keep valid_mask only for text-span mapping.
        mask = attention_mask.bool()

        if labels is not None:
            nll = -self.crf(emissions, labels, mask=mask, reduction="mean")
        else:
            nll = None

        decoded = self.crf.decode(emissions, mask=mask)
        return nll, decoded


class BERTCRFNamedEntityRecognizer:
    def __init__(self, config: BERTCRFConfig | None = None) -> None:
        self.config = config or BERTCRFConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.labels: List[str] = ["O"]
        self.label2id: Dict[str, int] = {"O": 0}
        self.id2label: Dict[int, str] = {0: "O"}
        self.model: _BertCRF | None = None

    @staticmethod
    def _collect_labels(samples: List[Dict]) -> List[str]:
        labels = {"O"}
        for s in samples:
            for e in s.get("entities", []):
                labels.add(f"B-{e['label']}")
                labels.add(f"I-{e['label']}")
        return sorted(labels)

    def _build_label_maps(self, samples: List[Dict]) -> None:
        self.labels = self._collect_labels(samples)
        self.label2id = {x: i for i, x in enumerate(self.labels)}
        self.id2label = {i: x for x, i in self.label2id.items()}

    def fit(self, train_samples: List[Dict], dev_samples: List[Dict] | None = None) -> Dict[str, float]:
        self._build_label_maps(train_samples)
        self.model = _BertCRF(self.config.model_name, len(self.labels))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        train_ds = _NERDataset(train_samples, self.tokenizer, self.label2id, self.config.max_length)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        for _ in range(self.config.epochs):
            self.model.train()
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, _ = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    valid_mask=batch["valid_mask"],
                    labels=batch["labels"],
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if dev_samples:
            return self.evaluate(dev_samples)
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def predict_entities(self, text: str) -> List[Dict]:
        if self.model is None:
            raise RuntimeError("Model not initialized. Train or load first.")

        self.model.eval()
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        offsets = encoded["offset_mapping"].squeeze(0).tolist()

        valid_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for i, (start, end) in enumerate(offsets):
            if attention_mask[0, i].item() == 0:
                continue
            if start == end:
                continue
            valid_mask[0, i] = True

        with torch.no_grad():
            _, decoded = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                valid_mask=valid_mask,
                labels=None,
            )

        pred_tag_ids = decoded[0] if decoded else []
        char_tags = ["O"] * len(text)
        ptr = 0
        for i, (start, end) in enumerate(offsets):
            if i >= attention_mask.shape[1]:
                break
            if attention_mask[0, i].item() == 0:
                continue
            if start == end:
                continue
            if ptr >= len(pred_tag_ids):
                break
            tag = self.id2label[pred_tag_ids[ptr]]
            ptr += 1
            for c in range(start, min(end, len(char_tags))):
                if c == start:
                    char_tags[c] = tag
                elif tag.startswith("B-"):
                    char_tags[c] = "I-" + tag[2:]
                else:
                    char_tags[c] = tag

        return bio_to_entities(text, char_tags)

    def evaluate(self, samples: List[Dict]) -> Dict[str, float]:
        gold = [s.get("entities", []) for s in samples]
        pred = [self.predict_entities(s["text"]) for s in samples]
        return score_entities(gold, pred)

    def save(self, output_dir: str) -> None:
        if self.model is None:
            raise RuntimeError("No trained model to save.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(out)
        torch.save(self.model.state_dict(), out / "model.pt")
        meta = {
            "config": self.config.__dict__,
            "labels": self.labels,
        }
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, output_dir: str) -> "BERTCRFNamedEntityRecognizer":
        out = Path(output_dir)
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        config = BERTCRFConfig(**meta["config"])
        inst = cls(config=config)
        inst.labels = meta["labels"]
        inst.label2id = {x: i for i, x in enumerate(inst.labels)}
        inst.id2label = {i: x for x, i in inst.label2id.items()}
        inst.model = _BertCRF(inst.config.model_name, len(inst.labels))
        state = torch.load(out / "model.pt", map_location="cpu")
        inst.model.load_state_dict(state)
        inst.model.eval()
        return inst
