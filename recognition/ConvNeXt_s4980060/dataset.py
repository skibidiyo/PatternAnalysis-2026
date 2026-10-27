"""
ADNI data loading, patient-level splitting and augmentation.

The ADNI directory is laid out as AD_NC/{train,test}/{AD,NC}/<scan>_<slice>.jpeg.
The numeric prefix before the underscore (e.g. '218391' in '218391_78.jpeg') is
an ADNI *scan* identifier, not a patient identifier: a single patient can have
several scans taken at different visits, each scan contributing 20 slices. Using
the scan id as if it were a patient id makes the provided train/test folders look
patient-disjoint (they are not) and would let the model memorise one patient's
anatomy rather than learn the pathology.

The real patient identifier (standard ADNI format 'SITE_S_NUMBER', e.g.
'068_S_0473') is recovered from meta_data_with_label.json, shipped alongside
AD_NC/ on Rangpur, which maps each scan id to the raw NIfTI path the scan was
derived from. With real patient ids, the provided train/test folders turn out to
share 216 patients - the leakage figure reported in the README - which this
module rebuilds the split to eliminate.

Note: the JSON's own 'label' field is a 3-way ADNI diagnostic code (CN/MCI/AD),
not the AD/NC binary label used here, so it is used only for patient identity;
class labels still come from the AD/NC folder each slice lives in.

Author: s4980060
"""

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Label encoding used throughout the project. CN is the negative class (0) so that
# "positive" in precision/recall and AUROC consistently means "Alzheimer's".
CLASS_TO_IDX = {"NC": 0, "AD": 1}
IDX_TO_CLASS = {0: "CN", 1: "AD"}

# ADNI slices are 256x240 single-channel JPEGs; ConvNeXt's patchify stem expects
# a square input divisible by 32 (four 2x downsampling stages after the 4x4 stem).
DEFAULT_IMAGE_SIZE = 224

# Matches the ADNI subject id embedded in the raw NIfTI filename, e.g.
# '...ADNI_068_S_0473_MR_Accelerated...' -> '068_S_0473'.
_PATIENT_ID_PATTERN = re.compile(r"ADNI_(\d{3}_S_\d{4})_")

# Default location of the metadata file relative to the AD_NC image root.
DEFAULT_METADATA_FILENAME = "meta_data_with_label.json"

# The metadata file's own 'label' field is a 3-way ADNI diagnostic code covering
# the full cohort (CN / MCI / AD). Only CN and AD subjects were exported into the
# AD_NC image folders, so MCI never appears here; the mapping below is used purely
# to cross-validate the folder-derived labels against an independent source.
JSON_DIAGNOSIS_TO_IDX = {0: CLASS_TO_IDX["NC"], 2: CLASS_TO_IDX["AD"]}  # 1 = MCI, excluded


@dataclass(frozen=True)
class SliceRecord:
    """One 2D slice: where it lives, what it is, and who it came from.

    `scan_id` is the identifier embedded in the slice filename (one ADNI scan,
    20 slices). `patient_id` is the real ADNI subject id, resolved via metadata,
    and is what splitting and leakage checks operate on.
    """

    path: Path
    label: int
    scan_id: str
    patient_id: str
    slice_index: int


def scan_id_from_path(path):
    """Extract the scan identifier from a slice filename.

    '218391_78.jpeg' -> '218391'. This is a scan id, not a patient id - see
    module docstring.
    """
    stem = Path(path).stem
    scan_id, _, _ = stem.rpartition("_")
    if not scan_id:
        raise ValueError(f"Unexpected ADNI filename (no underscore): {path}")
    return scan_id


def slice_index_from_path(path):
    """Extract the slice number within the scan. '218391_78.jpeg' -> 78."""
    stem = Path(path).stem
    _, _, slice_str = stem.rpartition("_")
    try:
        return int(slice_str)
    except ValueError as exc:
        raise ValueError(f"Unexpected ADNI filename (no slice index): {path}") from exc


def load_scan_to_patient_map(metadata_path):
    """Build {scan_id: real_patient_id} from meta_data_with_label.json.

    Each metadata entry's 'raw' field is a path like
    '.../ADNI_068_S_0473_MR_Accelerated_..._I932015.nii', where the leading
    group is the real patient id and the training image filename '932015_*.jpeg'
    uses the trailing 'I932015' as its scan id.
    """
    metadata_path = Path(metadata_path)
    with open(metadata_path) as f:
        meta = json.load(f)

    scan_to_patient = {}
    unmatched = []
    for scan_id, entry in meta.items():
        match = _PATIENT_ID_PATTERN.search(entry.get("raw", ""))
        if match:
            scan_to_patient[scan_id] = match.group(1)
        else:
            unmatched.append(scan_id)

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} metadata entries did not match the expected ADNI "
            f"patient-id pattern, e.g. {unmatched[:3]}"
        )
    return scan_to_patient


def index_all_slices(root, metadata_path=None):
    """Walk AD_NC/{train,test}/{AD,NC} and return a list of SliceRecord.

    Both provided folders are pooled deliberately: the split is rebuilt from
    scratch in patient_level_split() rather than trusting the given one.

    `metadata_path` defaults to 'meta_data_with_label.json' next to the AD_NC
    root (i.e. root.parent / DEFAULT_METADATA_FILENAME), matching the Rangpur
    layout. Every scan id found on disk must resolve to a real patient id;
    an unresolved scan id means a leaked/unverified split, so it fails loudly.
    """
    root = Path(root)
    if metadata_path is None:
        metadata_path = root.parent / DEFAULT_METADATA_FILENAME
    scan_to_patient = load_scan_to_patient_map(metadata_path)

    records = []
    for provided_split in ("train", "test"):
        for class_name, label in CLASS_TO_IDX.items():
            class_dir = root / provided_split / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.glob("*.jpeg")):
                scan_id = scan_id_from_path(path)
                patient_id = scan_to_patient.get(scan_id)
                if patient_id is None:
                    raise KeyError(
                        f"Scan id '{scan_id}' from {path} has no entry in "
                        f"{metadata_path}; cannot resolve its real patient id."
                    )
                records.append(
                    SliceRecord(
                        path=path,
                        label=label,
                        scan_id=scan_id,
                        patient_id=patient_id,
                        slice_index=slice_index_from_path(path),
                    )
                )
    if not records:
        raise FileNotFoundError(
            f"No ADNI slices found under {root}. Expected AD_NC/{{train,test}}/{{AD,NC}}/*.jpeg"
        )
    return records


def audit_provided_split(root, metadata_path=None):
    """Quantify leakage in the dataset's own train/test division.

    Returns a dict with per-folder real-patient counts and the set of patients
    appearing in both, which is the evidence reported in the README (216
    shared patients). Computed on real patient ids (via metadata), not on the
    scan ids embedded in filenames, which would show zero overlap by
    construction and hide the leakage entirely.
    """
    records = index_all_slices(root, metadata_path=metadata_path)

    patients_by_split = {"train": set(), "test": set()}
    for record in records:
        provided_split = "train" if "train" in str(record.path.parent.parent.name) else "test"
        patients_by_split[provided_split].add(record.patient_id)

    train_patients = patients_by_split["train"]
    test_patients = patients_by_split["test"]
    overlap = train_patients & test_patients

    return {
        "train_patients": len(train_patients),
        "test_patients": len(test_patients),
        "overlapping_patients": len(overlap),
        "overlap_ids": overlap,
    }


def _group_by_patient(records):
    """Group SliceRecords by (real) patient_id."""
    by_patient = defaultdict(list)
    for record in records:
        by_patient[record.patient_id].append(record)
    return by_patient


def patient_level_split(records, val_frac=0.15, test_frac=0.15, seed=42):
    """Partition real patients (never slices, never scans) into train/val/test.

    Splitting on patients is what prevents leakage: slices from two scans of the
    same patient are near-duplicates, so splitting by scan id (as the provided
    folders effectively do) lets the model recognise the patient rather than
    learn the pathology. Stratified by patient label to keep the AD/CN balance
    comparable across splits. The seed is fixed for reproducibility.

    Returns a dict {"train": [...], "val": [...], "test": [...]} of SliceRecord
    lists.
    """
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    by_patient = _group_by_patient(records)

    # Each patient must carry a single label to stratify the split by class;
    # verify_split_disjoint() re-checks this invariant after splitting too.
    patient_label = {}
    for patient_id, patient_records in by_patient.items():
        labels = {r.label for r in patient_records}
        if len(labels) != 1:
            raise ValueError(
                f"Patient {patient_id} has inconsistent labels across slices: {labels}"
            )
        patient_label[patient_id] = labels.pop()

    patients_by_class = defaultdict(list)
    for patient_id, label in patient_label.items():
        patients_by_class[label].append(patient_id)

    rng = random.Random(seed)
    split_patients = {"train": [], "val": [], "test": []}

    for label, patients in patients_by_class.items():
        patients = sorted(patients)  # deterministic order before shuffling
        rng.shuffle(patients)
        n = len(patients)
        n_val = round(n * val_frac)
        n_test = round(n * test_frac)
        n_train = n - n_val - n_test

        split_patients["train"].extend(patients[:n_train])
        split_patients["val"].extend(patients[n_train : n_train + n_val])
        split_patients["test"].extend(patients[n_train + n_val :])

    splits = {}
    for split_name, patient_ids in split_patients.items():
        patient_id_set = set(patient_ids)
        splits[split_name] = [r for r in records if r.patient_id in patient_id_set]

    return splits


def verify_split_disjoint(splits):
    """Assert no (real) patient appears in two splits and none carries both labels.

    Called before training rather than trusted implicitly - this is the
    guarantee the headline accuracy figure rests on. Raises AssertionError on
    violation.
    """
    patients_by_split = {
        split_name: {r.patient_id for r in records} for split_name, records in splits.items()
    }

    split_names = list(patients_by_split)
    for i, name_a in enumerate(split_names):
        for name_b in split_names[i + 1 :]:
            overlap = patients_by_split[name_a] & patients_by_split[name_b]
            assert not overlap, (
                f"Patient leakage: {len(overlap)} patient(s) appear in both "
                f"'{name_a}' and '{name_b}': {sorted(overlap)[:5]}..."
            )

    all_records = [r for records in splits.values() for r in records]
    by_patient = _group_by_patient(all_records)
    for patient_id, records in by_patient.items():
        labels = {r.label for r in records}
        assert len(labels) == 1, (
            f"Patient {patient_id} carries inconsistent labels across slices: {labels}"
        )

    return True


def build_transforms(
    train, image_size=DEFAULT_IMAGE_SIZE, mean=None, std=None, augment="light"
):
    """Compose the preprocessing pipeline.

    Augmentation is applied to the training split only; val and test stay
    deterministic so metrics are comparable across epochs and between models.

    Two strengths are available. "light" is a horizontal flip and a small
    rotation. "strong" adds random resized cropping and brightness/contrast
    jitter, which matter once a high-capacity model is used: ConvNeXt-T reaches
    99% training accuracy under light augmentation while validation accuracy
    stalls, so the training set needs to be harder to memorise.

    Rotation and flipping are anatomically safe here - brains are roughly
    bilaterally symmetric and the slices are already coarsely aligned - while
    scale and intensity jitter mimic scanner and acquisition variation.
    """
    mean = [0.5] if mean is None else mean
    std = [0.5] if std is None else std
    if augment not in {"light", "strong"}:
        raise ValueError(f"augment must be 'light' or 'strong', got {augment!r}")

    common = [transforms.Grayscale(num_output_channels=1)]

    if train and augment == "strong":
        common += [
            transforms.RandomResizedCrop(
                image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    elif train:
        common += [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]
    else:
        common += [transforms.Resize((image_size, image_size))]

    common += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(common)


def compute_channel_stats(records, image_size=DEFAULT_IMAGE_SIZE):
    """Compute mean/std over the training split only.

    Deriving normalisation statistics from val or test would leak information
    about the evaluation data into training. Returns ([mean], [std]) for a
    single-channel image.
    """
    to_tensor = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0
    for record in records:
        with Image.open(record.path) as img:
            tensor = to_tensor(img)
        total_sum += tensor.sum().item()
        total_sq_sum += (tensor**2).sum().item()
        total_pixels += tensor.numel()

    mean = total_sum / total_pixels
    variance = (total_sq_sum / total_pixels) - mean**2
    std = max(variance, 1e-12) ** 0.5
    return [mean], [std]


class ADNISliceDataset(Dataset):
    """Dataset over SliceRecords, yielding (image, label, patient_id).

    The patient id is carried through to inference so predictions can be
    aggregated per scan/patient and failure cases traced back to a specific
    person rather than just a slice index.
    """

    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        with Image.open(record.path) as img:
            img = img.convert("L")
            if self.transform is not None:
                image = self.transform(img)
            else:
                image = transforms.functional.to_tensor(img)
        label = torch.tensor(record.label, dtype=torch.long)
        return image, label, record.patient_id


def get_dataloaders(
    root,
    batch_size=32,
    image_size=DEFAULT_IMAGE_SIZE,
    seed=42,
    num_workers=4,
    val_frac=0.15,
    test_frac=0.15,
    metadata_path=None,
    augment="light",
):
    """End-to-end: index (with real patient ids), split, verify, wrap in
    DataLoaders.

    Returns (train_loader, val_loader, test_loader, info) where info carries the
    split metadata (patient counts, normalisation stats) needed for reporting.
    """
    records = index_all_slices(root, metadata_path=metadata_path)
    splits = patient_level_split(records, val_frac=val_frac, test_frac=test_frac, seed=seed)
    verify_split_disjoint(splits)

    mean, std = compute_channel_stats(splits["train"], image_size=image_size)

    train_transform = build_transforms(
        train=True, image_size=image_size, mean=mean, std=std, augment=augment
    )
    eval_transform = build_transforms(train=False, image_size=image_size, mean=mean, std=std)

    train_ds = ADNISliceDataset(splits["train"], transform=train_transform)
    val_ds = ADNISliceDataset(splits["val"], transform=eval_transform)
    test_ds = ADNISliceDataset(splits["test"], transform=eval_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    info = {
        "mean": mean,
        "std": std,
        "seed": seed,
        "augment": augment,
        "n_patients": {name: len({r.patient_id for r in recs}) for name, recs in splits.items()},
        "n_slices": {name: len(recs) for name, recs in splits.items()},
    }
    return train_loader, val_loader, test_loader, info


def cross_validate_labels(records, metadata_path):
    """Check folder-derived AD/NC labels against the metadata diagnostic code.

    The label of each slice comes from the AD/NC directory it sits in. The
    metadata file carries an independent 3-way diagnosis per scan, so agreement
    between the two confirms the folder layout has not been mislabelled or
    reorganised. Returns the number of scans checked; raises on any disagreement.
    """
    with open(Path(metadata_path)) as f:
        meta = json.load(f)

    checked = 0
    disagreements = []
    for scan_id, label in {r.scan_id: r.label for r in records}.items():
        entry = meta.get(scan_id)
        if entry is None:
            continue
        expected = JSON_DIAGNOSIS_TO_IDX.get(entry["label"])
        if expected is None:
            disagreements.append((scan_id, "MCI/unknown diagnosis in AD_NC set"))
        elif expected != label:
            disagreements.append((scan_id, f"folder={label} metadata={expected}"))
        checked += 1

    if disagreements:
        raise AssertionError(
            f"{len(disagreements)} scan(s) disagree with the metadata diagnosis, "
            f"e.g. {disagreements[:3]}"
        )
    return checked
