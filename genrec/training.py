import time
import json
from pathlib import Path
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
    def __init__(self, epochs=None, steps=None, batch_size=256, lr=.01, optimizer="adagrad", warmup_steps=10_000,
                 device="cpu", max_history=20, log_path=None):
        if steps is None and epochs is None: epochs = 5
        if optimizer not in {"adagrad", "adamw"}: raise ValueError("optimizer must be adagrad or adamw.")
        self.epochs, self.steps, self.batch_size, self.lr = epochs, steps, batch_size, lr
        self.optimizer_name, self.warmup_steps = optimizer, warmup_steps
        self.device, self.max_history = torch.device(device), max_history
        self.log_path = Path(log_path) if log_path else None

    def _log(self, event, **values):
        if not self.log_path: return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"event": event, "time": time.time(), **values}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def fit(self, model, dataset):
        started = time.perf_counter()
        model.to(self.device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=self.lr) if self.optimizer_name == "adagrad" else torch.optim.AdamW(model.parameters(), lr=self.lr)
        if self.device.type == "cuda": torch.cuda.reset_peak_memory_stats(self.device)
        examples, history = prefix_examples(dataset.train_sequences, self.max_history), []
        target_steps = self.steps; epoch = step = 0
        self._log("train_started", epochs=self.epochs, steps=target_steps, batch_size=self.batch_size, lr=self.lr, optimizer=self.optimizer_name,
                  warmup_steps=self.warmup_steps, device=str(self.device), train_examples=len(examples))
        while target_steps is None or step < target_steps:
            epoch += 1
            if self.epochs is not None and epoch > self.epochs: break
            epoch_started = time.perf_counter()
            model.train(); losses = []
            for batch in make_batches(examples, self.batch_size, self.device):
                if target_steps is not None and step >= target_steps: break
                # Paper schedule: 0.01 for the first 10k steps, then inverse-sqrt decay.
                current_lr = self.lr if step < self.warmup_steps else self.lr * (self.warmup_steps / max(step, 1)) ** .5
                for group in optimizer.param_groups: group["lr"] = current_lr
                result = model.training_step(batch); optimizer.zero_grad(); result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(result["loss"].item())
                step += 1
            history.append(sum(losses) / max(len(losses), 1))
            event = {"epoch": epoch, "epochs": self.epochs, "step": step, "target_steps": target_steps, "train_loss": history[-1],
                     "lr": current_lr if losses else None, "epoch_seconds": time.perf_counter() - epoch_started, "batches": len(losses)}
            if self.device.type == "cuda": event["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
            self._log("epoch_completed", **event)
        resource = {"train_seconds": time.perf_counter() - started, "train_loss": history[-1] if history else None, "steps": step,
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "train_examples": len(examples)}
        if self.device.type == "cuda":
            resource["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2; resource["gpu_name"] = torch.cuda.get_device_name(self.device)
        self._log("train_completed", **resource)
        return resource
    @torch.no_grad()
    def evaluate(self, model, examples):
        model.eval(); predictions, targets = [], []
        for batch in make_batches(examples, self.batch_size, self.device): predictions.append(model.recommend(batch, 20).cpu()); targets.append(batch["target"].cpu())
        return ranking_metrics(torch.cat(predictions), torch.cat(targets))
