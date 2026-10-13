# Model card: EchoSeg-RT U-Net ResNet34

## Model

| | |
|---|---|
| Architecture | U-Net with ResNet34 encoder (ImageNet initialised), 4 sigmoid outputs |
| Input | 256x256 greyscale ultrasound frame, optionally with previous frames as extra channels |
| Output | Per-pixel probability for `lv_endo`, `myocardium`, `lv_epi`, `left_atrium`, thresholded at 0.5 |
| Checkpoint | `runs/unet_r34_ctx0/best.pt`, config hash `664a086661c1`, best validation epoch 4 of 0-7 |
| Code version | git commit `4c5754e` |
| Training | 8 epochs, batch 16, AdamW, mixed precision, one Tesla T4 |

Each checkpoint stores its full config, config hash and the exact patient ids of its train, validation and test splits, so any reported number can be traced to the data and settings that produced it.

## Intended use

Research and engineering demonstration of real-time ultrasound video segmentation. It is not a medical device and must not be used for diagnosis or to guide a procedure.

## Data

- Source: CAMUS public dataset, 500 patients, apical two-chamber and four-chamber views, from a single centre (University Hospital of St Etienne, France), acquired on GE Vivid E95 scanners.
- Annotation: expert cardiologist contours on every frame from end-diastole to end-systole.
- Split: by patient, 70% train, 10% validation, 20% test, seed 42. Both views of a patient always fall in the same split.
- Preprocessing: resize to 256x256, intensities scaled to [0, 1]. Training augmentation is random gamma and gain.

## Performance on the held-out test patients

100 patients, 200 sequences, 3,828 frames. Mean Dice 0.908, mean IoU 0.837. Source: [results/unet_r34_ctx0/report_test.md](results/unet_r34_ctx0/report_test.md); the patient ids of each split are in `split.json` beside it.

| Label | Dice | IoU | Sequences with a sustained failure event |
|---|---|---|---|
| lv_endo | 0.931 | 0.872 | 1 of 200 |
| myocardium | 0.859 | 0.755 | 4 of 200 |
| lv_epi | 0.952 | 0.908 | 0 of 200 |
| left_atrium | 0.891 | 0.811 | 14 of 200 |

Frame-level presence precision and recall are 1.000 for every label, with no false-alarm events. This is expected on CAMUS, where every structure is in view in every frame, and says little about behaviour when a structure enters or leaves the view.

## Known limitations and failure modes

- Single centre and single scanner vendor; performance on other equipment is unknown.
- Dice by image quality grade: Good 0.912, Medium 0.904, Poor 0.900.
- The left atrium is the least reliable label: it has the most sustained failures (3 or more consecutive frames below 0.7 Dice).
- The myocardium, a thin wall, has the lowest Dice and its predicted area flickers about three times more between frames than the annotation does (0.016 against 0.005).
- No comparison against inter-observer variability was made, so these scores are not calibrated against how much two clinicians disagree.
- Patient demographics are not balanced or analysed here.

## Before any clinical use, this would need

- Validation on multi-centre, multi-vendor data with a locked test set.
- Reader review of predictions by clinicians, with acceptance criteria agreed in advance.
- Versioned data and model lineage, change control for retraining, and post-deployment monitoring.
