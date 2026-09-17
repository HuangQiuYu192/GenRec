import time
import torch
from .evaluation import ranking_metrics


def prefix_examples(sequences, max_history=50):
    """Standard sequential protocol: every train prefix predicts its next item."""
    return [(sequence[max(0, position - max_history):position], sequence[position])
            for sequence in sequences for position in range(1, len(sequence))]


def make_batches(examples, batch_size, device):
    for start in range(0, len(examples), batch_size):
        part = examples[start:start + batch_size]; width = max(len(history) for history, _ in part)
        history = torch.zeros(len(part), width, dtype=torch.long)
        for row, (values, _) in enumerate(part): history[row, -len(values):] = torch.tensor(values)
        yield {"history": history.to(device), "target": torch.tensor([target for _, target in part], device=device)}


class Trainer:
    def __init__(self, epochs=5, batch_size=128, lr=3e-4, device="cpu", max_history=50):
        self.epochs, self.batch_size, self.lr, self.device, self.max_history = epochs, batch_size, lr, torch.device(device), max_history
    def fit(self, model, dataset):
        model.to(self.device); optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr); started = time.perf_counter()
        if self.device.type == "cuda": torch.cuda.reset_peak_memory_stats(self.device)
        examples, history = prefix_examples(dataset.train_sequences, self.max_history), []
        for _ in range(self.epochs):
            model.train(); losses = []
            for batch in make_batches(examples, self.batch_size, self.device):
                result = model.training_step(batch); optimizer.zero_grad(); result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(result["loss"].item())
            history.append(sum(losses) / max(len(losses), 1))
        resource = {"train_seconds": time.perf_counter() - started, "train_loss": history[-1] if history else None,
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "train_examples": len(examples)}
        if self.device.type == "cuda":
            resource["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2; resource["gpu_name"] = torch.cuda.get_device_name(self.device)
        return resource
    @torch.no_grad()
    def evaluate(self, model, examples):
        model.eval(); predictions, targets = [], []
        for batch in make_batches(examples, self.batch_size, self.device): predictions.append(model.recommend(batch, 20).cpu()); targets.append(batch["target"].cpu())
        return ranking_metrics(torch.cat(predictions), torch.cat(targets))
