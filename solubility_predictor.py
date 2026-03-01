import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MAX_LENGTH = 195

class SolubilityPredictor:
    def __init__(self, model_dir: str, device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_one(self, smiles: str) -> float:
        inputs = self.tokenizer(
            smiles,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        return float(out.logits.squeeze().item())

    @torch.no_grad()
    def predict_batch(self, smiles_list: list[str]) -> list[float]:
        inputs = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        return [float(x) for x in out.logits.squeeze(-1).cpu().numpy().tolist()]