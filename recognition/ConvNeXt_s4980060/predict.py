"""
Example use of the trained model: inference, triage policy and failure autopsy.

Loads saved weights, runs them on the held-out patient-disjoint test split, and
produces the evidence behind the engineering recommendation:

  * metrics per class, both per slice and per patient
  * a reliability diagram showing where confidence drifts from accuracy
  * a threshold sweep - how many scans can be automated at a given accuracy,
    and how many go to a specialist
  * example failure cases with their confidence scores

The referral threshold is picked on validation, then applied to test. Picking it
on test would report an operating point that does not hold on new data.

Usage:
    python predict.py --checkpoint outputs/convnext_best.pth
    python predict.py --checkpoint outputs/convnext_best.pth --n-failures 5

Author: s4980060
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    get_dataloaders,
    index_all_slices,
    patient_level_split,
)
from modules import build_model, count_parameters
from train import collect_predictions, compute_metrics, expected_calibration_error


def parse_args():
    """Command-line configuration for an inference run."""
    parser = argparse.ArgumentParser(description="Evaluate a trained ADNI classifier.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved .pth file.")
    parser.add_argument("--data-root", default="/home/groups/comp3710/ADNI/AD_NC")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-failures", type=int, default=5)
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument("--min-ad-recall", type=float, default=0.95)
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--out-dir", default="outputs")
    return parser.parse_args()


def load_checkpoint(path, device):
    """Restore the model and its fitted calibration temperature.

    The temperature is stored alongside the weights so inference reproduces the
    calibrated confidences reported in the README, not the raw softmax values.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    saved_args = checkpoint.get("args", {})

    kwargs = {}
    if model_name.startswith("convnext") and "drop_path" in saved_args:
        # Drop path is inactive at eval, but rebuild the identical architecture.
        kwargs["drop_path_rate"] = saved_args["drop_path"]

    model = build_model(model_name, **kwargs).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    temperature = float(checkpoint.get("temperature", 1.0))
    return model, temperature, checkpoint


def print_report(metrics, ece, title):
    """Print metrics as a readable block."""
    print(f"\n{title}")
    print("-" * len(title))
    print(f"  accuracy        {metrics['accuracy']:.4f}")
    print(f"  macro-F1        {metrics['macro_f1']:.4f}")
    print(f"  AUROC           {metrics['auroc']:.4f}")
    for class_name in IDX_TO_CLASS.values():
        print(
            f"  {class_name:<3} precision {metrics[f'precision_{class_name}']:.4f}"
            f"   recall {metrics[f'recall_{class_name}']:.4f}"
            f"   F1 {metrics[f'f1_{class_name}']:.4f}"
        )
    print(f"  ECE             {ece:.4f}")


def print_confusion(probs, labels):
    """Print the 2x2 confusion matrix with CN/AD row and column headers."""
    predictions = probs.argmax(dim=1)
    print("\n  confusion (rows = truth, cols = prediction)")
    print(f"           {'CN':>8}{'AD':>8}")
    for true_idx, true_name in IDX_TO_CLASS.items():
        counts = [
            int(((labels == true_idx) & (predictions == pred_idx)).sum())
            for pred_idx in IDX_TO_CLASS
        ]
        print(f"    {true_name:<5}{counts[0]:>8}{counts[1]:>8}")


def aggregate_by_patient(probs, labels, patient_ids):
    """Pool every slice of a patient into one prediction by mean probability.

    Clinically the unit of decision is the patient, not the slice: a report is
    issued per scan. Slice-level accuracy can also disguise patients whose
    slices are split roughly 50/50, which are exactly the cases a triage system
    should refer.
    """
    probs = probs.detach().cpu()
    labels = labels.detach().cpu()

    sums = {}
    counts = {}
    truths = {}
    for i, patient_id in enumerate(patient_ids):
        sums[patient_id] = sums.get(patient_id, torch.zeros(probs.shape[1])) + probs[i]
        counts[patient_id] = counts.get(patient_id, 0) + 1
        truths[patient_id] = int(labels[i])

    ordered = sorted(sums)
    patient_probs = torch.stack([sums[p] / counts[p] for p in ordered])
    patient_labels = torch.tensor([truths[p] for p in ordered], dtype=torch.long)
    return patient_probs, patient_labels, ordered


def plot_reliability_diagram(probs, labels, out_path, n_bins=15, title="Reliability"):
    """Confidence against observed accuracy, with the perfect-calibration diagonal.

    Bars below the diagonal are overconfidence - the failure mode that makes an
    automated triage threshold unsafe, because the model's stated certainty
    overstates how often it is actually right.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confidences, predictions = probs.max(dim=1)
    correct = (predictions == labels).float().numpy()
    confidences = confidences.numpy()

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    centres, accuracies, confidence_means, weights = [], [], [], []
    for i in range(n_bins):
        mask = (confidences > edges[i]) & (confidences <= edges[i + 1])
        if i == 0:
            mask |= confidences == edges[i]
        if mask.sum() == 0:
            continue
        centres.append((edges[i] + edges[i + 1]) / 2)
        accuracies.append(correct[mask].mean())
        confidence_means.append(confidences[mask].mean())
        weights.append(mask.sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot([0.5, 1], [0.5, 1], "k--", label="perfect calibration")
    axes[0].bar(centres, accuracies, width=0.5 / n_bins * 0.9, label="observed accuracy")
    axes[0].plot(confidence_means, accuracies, "o-", color="tab:red", label="binned")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].bar(centres, weights, width=0.5 / n_bins * 0.9, color="tab:grey")
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Number of cases")
    axes[1].set_title("Confidence distribution")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def threshold_sweep(probs, labels, thresholds=None):
    """Accuracy, coverage and AD recall as a function of the referral threshold.

    Cases whose confidence falls below the threshold are routed to a human
    specialist and excluded from the automated metrics. Coverage is the fraction
    retained; the remaining fraction is the specialist's workload.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.50, 1.00, 0.01), 2)

    confidences, predictions = probs.max(dim=1)
    correct = (predictions == labels)
    is_ad = labels == CLASS_TO_IDX["AD"]

    rows = []
    total = labels.numel()
    for threshold in thresholds:
        retained = confidences >= float(threshold)
        n_retained = int(retained.sum())
        if n_retained == 0:
            continue

        retained_ad = retained & is_ad
        ad_recall = (
            float((correct & retained_ad).sum() / retained_ad.sum())
            if int(retained_ad.sum()) > 0
            else float("nan")
        )
        # Recall over *all* AD cases, counting referred ones as not auto-cleared:
        # this is the number that matters clinically, since a referred AD case
        # still reaches a specialist.
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": n_retained / total,
                "n_retained": n_retained,
                "n_referred": total - n_retained,
                "accuracy": float(correct[retained].float().mean()),
                "ad_recall_retained": ad_recall,
            }
        )
    return rows


def recommend_threshold(sweep, min_accuracy=0.95, min_ad_recall=0.95):
    """Select the lowest threshold meeting the safety floor on retained cases.

    The lowest qualifying threshold is chosen because it maximises coverage -
    the fraction safely automated - subject to the safety constraint. Returns
    None when no threshold satisfies both floors, which is itself a result: it
    means the model cannot be operated safely at any operating point.
    """
    qualifying = [
        row
        for row in sweep
        if row["accuracy"] >= min_accuracy
        and (
            row["ad_recall_retained"] >= min_ad_recall
            or row["ad_recall_retained"] != row["ad_recall_retained"]  # NaN
        )
    ]
    if not qualifying:
        return None
    return min(qualifying, key=lambda row: row["threshold"])


def plot_threshold_sweep(sweep, out_path, chosen=None):
    """Plot accuracy and coverage against the referral threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [row["threshold"] for row in sweep]
    accuracy = [row["accuracy"] for row in sweep]
    coverage = [row["coverage"] for row in sweep]

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(thresholds, accuracy, label="accuracy on retained cases")
    axis.plot(thresholds, coverage, label="coverage (fraction automated)")
    if chosen is not None:
        axis.axvline(
            chosen["threshold"],
            color="tab:red",
            linestyle="--",
            label=f"recommended ({chosen['threshold']:.2f})",
        )
    axis.set_xlabel("Confidence threshold for automation")
    axis.set_ylabel("Rate")
    axis.set_title("Reject-option trade-off")
    axis.legend()
    axis.grid(alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def select_failure_cases(probs, labels, records, n=5, one_per_patient=True):
    """Pick the most instructive errors: confident mistakes, worst first.

    False negatives are listed before false positives because a confidently
    wrong AD-as-CN prediction is the most dangerous output this system can
    produce - it auto-clears a patient who needs review.

    By default at most one slice per patient is returned. Without this the list
    fills up with consecutive slices of whichever single patient the model found
    hardest, which diagnoses one patient rather than the failure modes.
    """
    confidences, predictions = probs.max(dim=1)
    wrong = (predictions != labels).nonzero(as_tuple=True)[0]

    def sort_key(i):
        # False negatives (true AD, predicted CN) first, then by confidence.
        is_false_negative = int(labels[i]) == CLASS_TO_IDX["AD"]
        return (0 if is_false_negative else 1, -float(confidences[i]))

    ordered = sorted(wrong.tolist(), key=sort_key)
    chosen = []
    seen_patients = set()
    for i in ordered:
        if one_per_patient and records[i].patient_id in seen_patients:
            continue
        seen_patients.add(records[i].patient_id)
        chosen.append(i)
        if len(chosen) == n:
            break

    return [
        {
            "index": i,
            "record": records[i],
            "true": IDX_TO_CLASS[int(labels[i])],
            "predicted": IDX_TO_CLASS[int(predictions[i])],
            "confidence": float(confidences[i]),
            "p_ad": float(probs[i, CLASS_TO_IDX["AD"]]),
        }
        for i in chosen
    ]


def plot_failure_cases(cases, out_path):
    """Render failure slices with true label, prediction and confidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    if not cases:
        return

    fig, axes = plt.subplots(1, len(cases), figsize=(3.2 * len(cases), 3.8))
    if len(cases) == 1:
        axes = [axes]

    for axis, case in zip(axes, cases):
        record = case["record"]
        with Image.open(record.path) as img:
            axis.imshow(np.array(img.convert("L")), cmap="gray")
        axis.set_title(
            f"true {case['true']} / pred {case['predicted']}\n"
            f"confidence {case['confidence']:.2f}\n"
            f"{record.patient_id}  scan {record.scan_id}  slice {record.slice_index}",
            fontsize=8,
        )
        axis.axis("off")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, temperature, checkpoint = load_checkpoint(args.checkpoint, device)
    saved_args = checkpoint.get("args", {})
    seed = saved_args.get("seed", 42)
    image_size = saved_args.get("image_size", 224)

    print(f"Model: {checkpoint['model_name']} | device: {device}")
    print(f"Parameters: {count_parameters(model)['total'] / 1e6:.2f}M")
    print(f"Calibration temperature (fitted on validation): {temperature:.4f}")

    # Rebuild the identical split the checkpoint was trained on.
    _, val_loader, test_loader, data_info = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        image_size=image_size,
        seed=seed,
        num_workers=args.num_workers,
        metadata_path=args.metadata,
    )

    # Test-split records, in loader order (shuffle=False), for the autopsy.
    records = index_all_slices(args.data_root, metadata_path=args.metadata)
    test_records = patient_level_split(records, seed=seed)["test"]

    _, test_probs, test_labels, test_patient_ids = collect_predictions(
        model, test_loader, device, temperature=temperature
    )

    # ---- slice level ----
    slice_metrics = compute_metrics(test_probs, test_labels)
    slice_ece = expected_calibration_error(test_probs, test_labels)
    print_report(slice_metrics, slice_ece, "Test set - slice level")
    print_confusion(test_probs, test_labels)

    # ---- patient level ----
    patient_probs, patient_labels, _ = aggregate_by_patient(
        test_probs, test_labels, test_patient_ids
    )
    patient_metrics = compute_metrics(patient_probs, patient_labels)
    patient_ece = expected_calibration_error(patient_probs, patient_labels)
    print_report(patient_metrics, patient_ece, "Test set - patient level")
    print_confusion(patient_probs, patient_labels)

    majority = float((patient_labels == CLASS_TO_IDX["NC"]).float().mean())
    print(f"\n  majority-class baseline (predict CN for all): {majority:.4f}")

    # ---- reject option: choose on validation, report on test ----
    _, val_probs, val_labels, _ = collect_predictions(
        model, val_loader, device, temperature=temperature
    )
    val_sweep = threshold_sweep(val_probs, val_labels)
    chosen = recommend_threshold(
        val_sweep, min_accuracy=args.min_accuracy, min_ad_recall=args.min_ad_recall
    )

    print("\nReject-option policy")
    print("--------------------")
    if chosen is None:
        print(
            f"  No threshold reaches {args.min_accuracy:.0%} accuracy with "
            f"{args.min_ad_recall:.0%} AD recall on validation."
        )
        print("  The model cannot be operated safely at any confidence threshold.")
    else:
        print(
            f"  Chosen on validation: threshold {chosen['threshold']:.2f} "
            f"(validation accuracy {chosen['accuracy']:.4f}, "
            f"coverage {chosen['coverage']:.4f})"
        )
        test_sweep = threshold_sweep(test_probs, test_labels)
        applied = min(
            test_sweep, key=lambda row: abs(row["threshold"] - chosen["threshold"])
        )
        print(
            f"  Applied to test:      accuracy {applied['accuracy']:.4f} on "
            f"{applied['coverage']:.1%} of slices "
            f"({applied['n_referred']} of "
            f"{applied['n_retained'] + applied['n_referred']} referred to a specialist)"
        )

    # ---- figures and artefacts ----
    assets = Path(args.assets_dir)
    name = checkpoint["model_name"]
    plot_reliability_diagram(
        test_probs,
        test_labels,
        assets / f"{name}_reliability.png",
        title=f"{name} reliability (test, calibrated)",
    )
    plot_threshold_sweep(
        threshold_sweep(test_probs, test_labels), assets / f"{name}_threshold.png", chosen
    )

    failures = select_failure_cases(
        test_probs, test_labels, test_records, n=args.n_failures
    )
    plot_failure_cases(failures, assets / f"{name}_failures.png")

    print(f"\nFailure cases ({len(failures)})")
    print("-" * 20)
    for case in failures:
        record = case["record"]
        # Identified by scan, not just patient: one patient contributes several
        # scans, so a patient id and slice index together are not unique.
        print(
            f"  {record.patient_id} scan {record.scan_id} slice {record.slice_index:>3}  "
            f"true {case['true']} -> pred {case['predicted']}  "
            f"confidence {case['confidence']:.3f}  P(AD) {case['p_ad']:.3f}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}_evaluation.json", "w") as f:
        json.dump(
            {
                "model": name,
                "temperature": temperature,
                "slice_metrics": slice_metrics | {"ece": slice_ece},
                "patient_metrics": patient_metrics | {"ece": patient_ece},
                "majority_baseline": majority,
                "recommended_threshold": chosen,
                "failures": [
                    {
                        "patient_id": c["record"].patient_id,
                        "scan_id": c["record"].scan_id,
                        "slice": c["record"].slice_index,
                        "true": c["true"],
                        "predicted": c["predicted"],
                        "confidence": c["confidence"],
                    }
                    for c in failures
                ],
                "data_info": data_info,
            },
            f,
            indent=2,
        )

    print(f"\nFigures written to {assets}/")
    print(f"Evaluation summary: {out_dir / f'{name}_evaluation.json'}")


if __name__ == "__main__":
    main()
