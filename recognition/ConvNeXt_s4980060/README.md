# ConvNeXt for Alzheimer's (AD vs CN) Classification on ADNI

> COMP3710 Pattern Recognition Report — Project 2.1 (Classification), **ConvNeXt [Hard Difficulty]**
> Author: s4980060 · Branch: `topic-recognition`


## Table of Contents
1. [Problem Description](#1-problem-description)
2. [How ConvNeXt Works](#2-how-convnext-works)
3. [Feasibility Review](#3-feasibility-review)
4. [Repository Structure](#4-repository-structure)

---

## 1. Problem Description

Alzheimer's Disease (AD) causes irreversible neurodegeneration, visible on structural T1-weighted MRI as
hippocampal and cortical atrophy and ventricular enlargement. This project builds an automated classifier
that labels 2D brain MRI slices from the ADNI dataset as **AD** or **Cognitively Normal (CN)**, intended as a
triage assistant: confident predictions are automated, while uncertain ones are routed to a specialist.
The target is **≥ 0.8 test accuracy** on a patient-disjoint test set, but the main question is whether the
model's confidence can be trusted.

## 2. How ConvNeXt Works

ConvNeXt (Liu et al., 2022) is a pure convolutional network "modernised" with design choices borrowed from
Vision Transformers:

- **Patchify stem:** a 4×4 convolution with stride 4 replaces the ResNet 7×7 conv + max-pool stem.
- **Stage layout:** four stages with block ratio 3:3:9:3 (ConvNeXt-T), with downsampling layers
  (LayerNorm + 2×2 stride-2 conv) between stages.
- **ConvNeXt block:** 7×7 depthwise conv → LayerNorm → 1×1 conv expanding channels 4× → GELU →
  1×1 conv projecting back, with a residual connection, layer scale and stochastic depth (drop path).
- **Fewer normalisation/activation layers** than ResNet, and LayerNorm instead of BatchNorm.

The large depthwise kernel gives a wide receptive field (similar in spirit to attention windows) while keeping
the inductive biases of convolutions, which matters for a small medical dataset.

## 3. Feasibility Review

*The one-page review presented at the midpoint check-off.*

### 3.1 User Need, Scope and Acceptance Criteria

This prototype is for a doctor or screening reviewer who needs brain MRI scans sorted into Alzheimer's (AD)
or healthy (CN), with any scan the model is unsure about sent to a specialist. It labels 2D brain slices and
also reports how sure it is, and that confidence number has to be trustworthy. The project asks two questions:
when the model says it is confident, is it actually right, and is ConvNeXt accurate enough to justify its extra
cost compared with a simple CNN?

**Acceptance criteria** (all measured on a patient-disjoint test set):

| # | Criterion | Target |
|---|-----------|--------|
| 1 | Accuracy | Test accuracy ≥ 0.80 for AD vs CN |
| 2 | Calibration | Expected Calibration Error (ECE) ≤ 0.10 after calibration, and lower than the baseline's ECE |
| 3 | Resources | Peak training VRAM ≤ 16 GB and one training run ≤ 2 h on one Rangpur GPU |

### 3.2 Model Choice and Course Concepts

I use ADNI brain MRI slices labelled AD or CN, split by patient so no person appears in both training and test
data. The main model is ConvNeXt (Hard difficulty), compared against a simple CNN baseline on the exact same
split. ConvNeXt is a modern CNN that builds on our course lab material and suits small medical datasets, and it
matches newer transformer models in accuracy at a similar cost. I judge both models on accuracy, precision,
recall, F1, AUROC, and calibration (ECE), which shows whether the model's confidence matches how often it is
right.

### 3.3 Preliminary Feasibility Evidence

**Data audit.** The dataset contains **30,520 image slices** (14,860 AD and 15,660 CN), drawn from
**1,526 scans** (743 AD and 783 CN) at exactly 20 slices per scan, belonging to **680 unique patients**.

The patient identifier is *not* the number in the filename: `218391_78.jpeg` identifies a scan, and one
patient may be scanned at several visits. Real ADNI subject ids (e.g. `068_S_0473`) are recovered from
`meta_data_with_label.json`, which maps each scan to the raw NIfTI it came from. Resolving them matters,
because the class balance looks very different at the two levels:

| Level | AD | CN | Balance |
|-------|----|----|---------|
| Slices | 14,860 | 15,660 | 49 / 51 |
| Patients | 221 | 459 | 32 / 68 |

The slice-level balance is an artefact: AD patients average 3.36 scans each against 1.71 for CN, so the two
classes even out only after a patient is counted once per scan. Training operates on slices and is therefore
effectively balanced, but **any patient-level metric faces a 68% majority-class baseline** — predicting CN
for everyone would score 0.68. Per-class recall, macro-F1 and AUROC are reported for this reason, rather than
aggregate accuracy alone.

**Key finding:** once real subject ids are resolved, the dataset's provided `train` and `test` folders
**share 216 of the same patients**, which would make test results look better than they really are. (Judged by
the scan number in the filename the folders appear perfectly disjoint, which is exactly why the leakage is easy
to miss.) So I combine all the images and split them again by patient (70% train, 15% validation, 15% test,
stratified so each split keeps a similar AD/CN mix), then assert that no patient appears in more than one set.

**Smoke test (RTX 4070 laptop GPU).** The rebuilt split assigns all 680 patients as train 476 / val 102 /
test 102 (21,560 / 4,580 / 4,380 slices) with zero overlap, and no patient carries both labels. Folder-derived
labels were additionally cross-checked against the diagnostic code in `meta_data_with_label.json`: all 1,526
scans agree.

| Model | Parameters | Measurement | Result |
|-------|-----------|-------------|--------|
| Tiny baseline CNN | 0.02M | ~8 s/epoch, 0.29 GB GPU memory | Still predicted a single class after 2 epochs (val acc 0.51) → needs input normalisation and longer training |
| ConvNeXt-Tiny | 27.8M | 3.67 GB GPU memory for one training step, batch 32, 224 px | Fits comfortably on available GPUs |

### 3.4 Risks, Budget and Fallback

- **Risks:** The given train and test folders share some of the same patients, so I make my own split by
  patient. Accuracy will likely be lower, but it will be honest. The model may also be very confident even when
  it is wrong on unclear scans, which is the main question this project investigates. Because the dataset is
  small, the model may memorise instead of learning, so I use augmentation such as flips and small rotations,
  plus other regularisation. Finally, Rangpur's GPUs are shared, so jobs may wait in the queue.
- **Budget** (initial estimate, to be updated after the first full training run): about 15 hours of GPU time —
  ~1 h for the baseline, ~10 h for ConvNeXt, ~2 h for analysis, and spare time for reruns. Short interactive
  sessions for testing and batch jobs for long training.
- **Next experiment:** train the simple CNN on the patient split, then train ConvNeXt on the same split, and
  compare their accuracy and how honest their confidence is.
- **Fallback:** if ConvNeXt doesn't reach 0.8 accuracy, try a smaller ConvNeXt with more augmentation. If it is
  still short, report the real result and explain why.

## 4. Repository Structure

```
ConvNeXt_s4980060/
├── modules.py    # ConvNeXt and baseline model components (PyTorch)
├── dataset.py    # ADNI loading, patient-level splits, augmentation
├── train.py      # training, validation, testing, checkpointing, loss/metric plots
├── predict.py    # loads trained weights, runs inference, visualises predictions
├── assets/       # figures used in this README
└── README.md
```
