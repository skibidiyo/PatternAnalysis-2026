"""
Train, validate, test and save the ADNI AD vs CN classifier.

Trains either ConvNeXt or the CNN baseline (--model) on the same
patient-disjoint split, so the two can be compared fairly. As well as accuracy
it records per-class precision and recall, macro-F1, AUROC and calibration
error (ECE).

Two choices matter here:

  * The best epoch is picked by validation AUROC, not accuracy. Only 32% of
    patients are AD, so accuracy rewards always guessing CN. AUROC does not
    depend on a threshold.
  * Temperature scaling is fitted on validation only. Fitting it on test would
    leak test data into the reported calibration, the same mistake as letting a
    patient appear in two splits.

Metrics are written in plain PyTorch, so every number can be traced back to its
formula.

Usage:
    python train.py --model convnext --data-root /home/groups/comp3710/ADNI/AD_NC
    python train.py --model baseline --epochs 30

Author: s4980060
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from dataset import IDX_TO_CLASS, get_dataloaders
from modules import build_model, count_parameters


def parse_args():
    """Command-line configuration for a training run."""
    parser = argparse.ArgumentParser(description="Train the ADNI AD vs CN classifier.")
    parser.add_argument("--model", choices=["convnext", "baseline"], default="convnext")
    parser.add_argument("--data-root", default="/home/groups/comp3710/ADNI/AD_NC")
    parser.add_argument("--metadata", default=None, help="meta_data_with_label.json path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--drop-path", type=float, default=0.1, help="ConvNeXt only")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision.")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--assets-dir", default="assets")
    return parser.parse_args()


def set_seed(seed):
    """Seed Python and PyTorch, and request deterministic cuDNN kernels.

    Required for the reproducibility claims made in the README. Determinism
    costs some throughput but makes a reported result reproducible from the
    stated seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Metrics (implemented from their definitions, in PyTorch)
# --------------------------------------------------------------------------- #


def _average_ranks(values):
    """Rank values ascending, averaging the ranks within tied groups.

    Ties matter for AUROC: a model that assigns the same score to a positive and
    a negative case should be credited half a correct ordering, not a whole one.
    """
    n = values.numel()
    order = torch.argsort(values)
    sorted_values = values[order]

    ranks = torch.empty(n, dtype=torch.float64, device=values.device)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        # Ranks are 1-based; assign the mean rank across the tied block.
        ranks[i : j + 1] = (i + j + 2) / 2.0
        i = j + 1

    out = torch.empty(n, dtype=torch.float64, device=values.device)
    out[order] = ranks
    return out


def binary_auroc(scores, labels):
    """AUROC via the Mann-Whitney U statistic.

    AUROC is the probability that a randomly chosen positive case is ranked
    above a randomly chosen negative one, so it can be computed from rank sums
    without sweeping thresholds. Returns NaN when only one class is present.
    """
    scores = scores.detach().to(torch.float64).flatten()
    labels = labels.detach().flatten()

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _average_ranks(scores)
    rank_sum_pos = ranks[labels == 1].sum()
    u_statistic = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_statistic / (n_pos * n_neg))


def compute_metrics(probs, labels):
    """Accuracy, per-class precision/recall/F1, macro-F1 and AUROC.

    Reported per class because a triage system that misses AD cases is far more
    dangerous than one that over-refers CN cases, and aggregate accuracy hides
    that asymmetry entirely.
    """
    probs = probs.detach()
    labels = labels.detach()
    predictions = probs.argmax(dim=1)

    metrics = {"accuracy": float((predictions == labels).float().mean())}

    f1_scores = []
    for class_idx, class_name in IDX_TO_CLASS.items():
        true_positive = int(((predictions == class_idx) & (labels == class_idx)).sum())
        predicted_positive = int((predictions == class_idx).sum())
        actual_positive = int((labels == class_idx).sum())

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[f"precision_{class_name}"] = precision
        metrics[f"recall_{class_name}"] = recall
        metrics[f"f1_{class_name}"] = f1
        f1_scores.append(f1)

    metrics["macro_f1"] = sum(f1_scores) / len(f1_scores)
    metrics["auroc"] = binary_auroc(probs[:, 1], labels)
    return metrics


def expected_calibration_error(probs, labels, n_bins=15):
    """ECE: mean |confidence - accuracy| over equal-width confidence bins.

    The headline number for the confidence dilemma. A well-calibrated model that
    reports 0.9 confidence should be correct about 90% of the time; a large ECE
    means the confidence score cannot be used to decide what to automate.
    """
    probs = probs.detach()
    labels = labels.detach()

    confidences, predictions = probs.max(dim=1)
    correct = (predictions == labels).float()

    ece = torch.zeros((), dtype=torch.float64, device=probs.device)
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        # Lower bound is exclusive so each confidence falls in exactly one bin;
        # the first bin also takes confidences equal to its lower edge.
        in_bin = (confidences > low) & (confidences <= high)
        if i == 0:
            in_bin = in_bin | (confidences == low)
        proportion = in_bin.float().mean()
        if proportion > 0:
            bin_accuracy = correct[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            ece += proportion.double() * (bin_accuracy - bin_confidence).abs().double()
    return float(ece)


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #


def train_one_epoch(model, loader, criterion, optimiser, device, scaler=None):
    """One pass over the training split. Returns mean loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    seen = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()

        running_loss += loss.item() * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum())
        seen += labels.size(0)

    return running_loss / seen, correct / seen


@torch.no_grad()
def collect_predictions(model, loader, device, temperature=1.0):
    """Run inference over a loader, returning logits, probabilities, labels, ids.

    Shared by validation, testing and predict.py so that every reported metric
    is computed from the same code path. `temperature` divides the logits before
    the softmax; the default of 1.0 leaves them uncalibrated.
    """
    model.eval()
    all_logits = []
    all_labels = []
    all_patient_ids = []

    for images, labels, patient_ids in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).float().cpu()
        all_logits.append(logits)
        all_labels.append(labels)
        all_patient_ids.extend(patient_ids)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.softmax(logits / temperature, dim=1)
    return logits, probs, labels, all_patient_ids


def fit_temperature(val_logits, val_labels, max_iter=100):
    """Fit a single temperature on validation logits by minimising NLL.

    Temperature scaling (Guo et al., 2017) divides the logits by one learned
    scalar. It leaves the argmax - and therefore accuracy - unchanged while
    rescaling confidence, so it isolates calibration from discrimination.
    """
    log_temperature = torch.zeros(1, requires_grad=True)  # optimise log T > 0
    optimiser = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimiser.zero_grad()
        loss = criterion(val_logits / log_temperature.exp(), val_labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_temperature.exp().item())


def evaluate(model, loader, device, temperature=1.0, criterion=None):
    """Collect predictions and summarise them as metrics plus ECE."""
    logits, probs, labels, _ = collect_predictions(model, loader, device, temperature)
    metrics = compute_metrics(probs, labels)
    metrics["ece"] = expected_calibration_error(probs, labels)
    if criterion is not None:
        metrics["loss"] = float(criterion(logits, labels))
    return metrics


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def plot_history(history, out_path):
    """Plot loss, accuracy and AUROC per epoch for train and validation."""
    import matplotlib

    matplotlib.use("Agg")  # headless: Rangpur batch jobs have no display
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Loss")

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="validation")
    axes[1].set_title("Accuracy")

    axes[2].plot(epochs, history["val_auroc"], label="validation", color="tab:green")
    axes[2].set_title("AUROC (validation)")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_checkpoint(path, model, args, epoch, metrics, temperature, data_info):
    """Persist weights, the fitted temperature and the metrics that produced them.

    The temperature travels with the weights so predict.py reproduces the
    calibrated confidences reported here rather than raw softmax values.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model,
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "temperature": temperature,
            "data_info": data_info,
            "args": vars(args),
        },
        path,
    )


def _format_metrics(metrics):
    """One-line metric summary for the epoch log."""
    return (
        f"acc {metrics['accuracy']:.4f}  "
        f"macroF1 {metrics['macro_f1']:.4f}  "
        f"AUROC {metrics['auroc']:.4f}  "
        f"recall(AD) {metrics['recall_AD']:.4f}  "
        f"ECE {metrics['ece']:.4f}"
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | model: {args.model} | seed: {args.seed}")

    train_loader, val_loader, test_loader, data_info = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
        metadata_path=args.metadata,
    )
    print(
        f"Patients  train {data_info['n_patients']['train']} / "
        f"val {data_info['n_patients']['val']} / test {data_info['n_patients']['test']}"
    )
    print(
        f"Slices    train {data_info['n_slices']['train']} / "
        f"val {data_info['n_slices']['val']} / test {data_info['n_slices']['test']}"
    )

    model_kwargs = {"drop_path_rate": args.drop_path} if args.model == "convnext" else {}
    model = build_model(args.model, **model_kwargs).to(device)
    params = count_parameters(model)
    print(f"Parameters: {params['total'] / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Linear warmup then cosine decay: the warmup keeps the early updates small
    # while the LayerNorm statistics and the classifier head settle.
    warmup_steps = max(args.warmup_epochs, 0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimiser,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimiser, start_factor=0.1, total_iters=max(warmup_steps, 1)
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=max(args.epochs - warmup_steps, 1)
            ),
        ],
        milestones=[max(warmup_steps, 1)],
    )
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device == "cuda") else None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auroc": [],
        "val_macro_f1": [],
        "val_ece": [],
    }

    out_dir = Path(args.out_dir)
    checkpoint_path = out_dir / f"{args.model}_best.pth"
    best_auroc = float("-inf")
    best_epoch = -1
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimiser, device, scaler
        )
        val_metrics = evaluate(model, val_loader, device, criterion=criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_auroc"].append(val_metrics["auroc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_ece"].append(val_metrics["ece"])

        # Model selection on AUROC: see module docstring.
        marker = ""
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path, model, args, epoch, val_metrics, 1.0, data_info
            )
            marker = "  <- best"

        print(
            f"epoch {epoch:>3}/{args.epochs}  "
            f"train loss {train_loss:.4f} acc {train_acc:.4f}  |  val "
            f"{_format_metrics(val_metrics)}  ({time.time() - epoch_start:.0f}s){marker}"
        )

    train_minutes = (time.time() - start_time) / 60
    print(f"\nTraining finished in {train_minutes:.1f} min; best epoch {best_epoch}")

    # Restore the selected epoch before calibrating and testing.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    # Calibrate on validation only - never on test.
    val_logits, _, val_labels, _ = collect_predictions(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    print(f"Fitted temperature on validation: {temperature:.4f}")

    test_raw = evaluate(model, test_loader, device, temperature=1.0)
    test_calibrated = evaluate(model, test_loader, device, temperature=temperature)

    print("\nTest set (uncalibrated): " + _format_metrics(test_raw))
    print("Test set (calibrated):   " + _format_metrics(test_calibrated))
    print(
        f"ECE {test_raw['ece']:.4f} -> {test_calibrated['ece']:.4f} after temperature scaling"
    )

    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    summary = {
        "model": args.model,
        "parameters": params,
        "best_epoch": best_epoch,
        "temperature": temperature,
        "val_metrics": checkpoint["metrics"],
        "test_metrics_uncalibrated": test_raw,
        "test_metrics_calibrated": test_calibrated,
        "train_minutes": train_minutes,
        "peak_vram_gb": peak_vram,
        "data_info": data_info,
        "args": vars(args),
    }

    save_checkpoint(
        checkpoint_path, model, args, best_epoch, checkpoint["metrics"], temperature, data_info
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.model}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / f"{args.model}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, Path(args.assets_dir) / f"{args.model}_training.png")

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Peak VRAM: {peak_vram:.2f} GB")


if __name__ == "__main__":
    main()
