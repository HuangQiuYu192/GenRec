import time
import torch
from .evaluation import ranking_metrics


def make_batches(examples, batch_size, device):
    for start in range(0, len(examples), batch_size):
        part = examples[start:start + batch_size]; width = max(len(history) for history, _ in part)
        history = torch.zeros(len(part), width, dtype=torch.long)
        for row, (values, _) in enumerate(part): history[row, -len(values):] = torch.tensor(values)
        yield {"history": history.to(device), "target": torch.tensor([target for _, target in part], device=device)}


class Trainer:
    def __init__(self, epochs=5, batch_size=32, lr=1e-3, device="cpu"):
        self.epochs, self.batch_size, self.lr, self.device = epochs, batch_size, lr, torch.device(device)
    def fit(self, model, dataset):
        model.to(self.device); optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr); started = time.perf_counter()
        history = []
        examples = [(sequence[:-1], sequence[-1]) for sequence in dataset.train_sequences if len(sequence) > 1]
        for _ in range(self.epochs):
            model.train(); losses = []
            for batch in make_batches(examples, self.batch_size, self.device):
                result = model.training_step(batch); optimizer.zero_grad(); result["loss"].backward(); optimizer.step(); losses.append(result["loss"].item())
            history.append(sum(losses) / max(len(losses), 1))
        return {"train_seconds": time.perf_counter() - started, "train_loss": history[-1] if history else None}
    @torch.no_grad()
    def evaluate(self, model, examples):
        model.eval(); predictions, targets = [], []
        for batch in make_batches(examples, self.batch_size, self.device): predictions.append(model.recommend(batch, 20).cpu()); targets.append(batch["target"].cpu())
        return ranking_metrics(torch.cat(predictions), torch.cat(targets))

