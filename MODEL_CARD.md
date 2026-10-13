# Model card: EchoSeg-RT U-Net ResNet34

## Model

| | |
|---|---|
| Architecture | U-Net with ResNet34 encoder (ImageNet initialised), 4 sigmoid outputs |
| Input | 256x256 greyscale ultrasound frame, optionally with previous frames as extra channels |
| Output | Per-pixel probability for `lv_endo`, `myocardium`, `lv_epi`, `left_atrium`, thresholded at 0.5 |
| Checkpoint | `runs/TODO/best.pt`, config hash `TODO`, epoch TODO |
| Code version | git commit `TODO` |

Each checkpoint stores its full config, config hash and the exact patient ids of its train, validation and test splits, so any reported number can be traced to the data and settings that produced it.

## Intended use

Research and engineering demonstration of real-time ultrasound video segmentation. It is not a medical device and must not be used for diagnosis or to guide a procedure.

## Data

- Source: CAMUS public dataset, 500 patients, apical two-chamber and four-chamber views, from a single centre (University Hospital of St Etienne, France), acquired on GE Vivid E95 scanners.
- Annotation: expert cardiologist contours on every frame from end-diastole to end-systole.
- Split: by patient, 70% train, 10% validation, 20% test, seed 42. Both views of a patient always fall in the same split.
- Preprocessing: resize to 256x256, intensities scaled to [0, 1]. Training augmentation is random gamma and gain.

## Performance on the held-out test patients

Paste `runs/<run>/report_test.md` here.

TODO

## Known limitations and failure modes

- Single centre and single scanner vendor; performance on other equipment is unknown.
- Dice by image quality grade (from the report): Good TODO, Medium TODO, Poor TODO.
- TODO: describe one or two failure cases seen in the overlay GIFs, for example the atrium boundary in poor-quality four-chamber views.
- Patient demographics are not balanced or analysed here.

## Before any clinical use, this would need

- Validation on multi-centre, multi-vendor data with a locked test set.
- Reader review of predictions by clinicians, with acceptance criteria agreed in advance.
- Versioned data and model lineage, change control for retraining, and post-deployment monitoring.
