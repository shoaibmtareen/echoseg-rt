### unet_r34_ctx2 (test split)

100 patients, 200 sequences, 3828 frames, 1.15 min of video. Config `ab85f4e0004d`, epoch 4.

**Mean Dice 0.911, mean IoU 0.842**

| label | dice | iou | frame_precision | frame_recall | gt_events | event_recall | false_alarms_per_min | sustained_failures_per_sequence | area_jitter_pred | area_jitter_gt |
|---|---|---|---|---|---|---|---|---|---|---|
| lv_endo | 0.934 | 0.878 | 1.000 | 1.000 | 200 | 1.000 | 0.000 | 0.000 | 0.026 | 0.025 |
| myocardium | 0.866 | 0.766 | 1.000 | 1.000 | 200 | 1.000 | 0.000 | 0.015 | 0.012 | 0.005 |
| lv_epi | 0.954 | 0.913 | 1.000 | 1.000 | 200 | 1.000 | 0.000 | 0.000 | 0.014 | 0.013 |
| left_atrium | 0.890 | 0.811 | 1.000 | 1.000 | 200 | 1.000 | 0.000 | 0.065 | 0.030 | 0.020 |

Mean Dice by image quality: Good: 0.916, Medium: 0.906, Poor: 0.899
