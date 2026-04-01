"""Smoke test for generation length/vocab guard in MedicalDigitalTwinModel."""

from types import SimpleNamespace

import torch

from models.mdt_model import MedicalDigitalTwinModel


class DummyBatch(dict):
    def to(self, device):
        return DummyBatch({k: v.to(device) for k, v in self.items()})


class DummyTokenizer:
    def __init__(self):
        self.model_max_length = 16
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "</s>"

    def __call__(self, text, return_tensors="pt", padding=True, truncation=True, max_length=None):
        token_count = min(max_length or 8, 8)
        ids = torch.arange(1, token_count + 1, dtype=torch.long).unsqueeze(0)
        return DummyBatch({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, ids, skip_special_tokens=False):
        return "decoded"


class DummyEmb:
    def __init__(self, n):
        self.num_embeddings = n


class DummyModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(max_position_embeddings=12)
        self.last_kwargs = None

    def get_input_embeddings(self):
        return DummyEmb(100)

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        return torch.tensor([[1, 2, 3]], dtype=torch.long)


obj = MedicalDigitalTwinModel.__new__(MedicalDigitalTwinModel)
obj.config = SimpleNamespace(max_length=20, temperature=0.7, top_p=0.9, top_k=50)
obj.device = torch.device("cpu")
obj.tokenizer = DummyTokenizer()
obj.model = DummyModel()

text = obj.generate("hello", max_length=20)
assert text == "decoded"
assert "max_new_tokens" in obj.model.last_kwargs
assert obj.model.last_kwargs["max_new_tokens"] >= 1
print(f"SMOKE_OK max_new_tokens={obj.model.last_kwargs['max_new_tokens']}")
