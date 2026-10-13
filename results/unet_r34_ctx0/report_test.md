### unet_r34_ctx0 (test split)

100 patients, 200 sequences, 3828 frames, 1.15 min of video. Config `664a086661c1`, epoch 4.

**Mean Dice 0.908, mean IoU 0.837**

| label | dice | iou | frame_precision | frame_recall | gt_events | event_recall | false_alarms_per_min | sustained_failures_per_sequence | area_jitter_pred | area_jitter_gt |
|---|---|---|---|---|---|---|---|---|---|---|
| lv_endo | 0.931 | 0.872 | 1.000 | 1.000 | 200 | 1.000 | 0.000 | 0.005 | 0.028 | 0.025 |
| myocardium | 0.859 | 0.755 | 1.000 | 1.000 | 200 | 1.000 | 0.000 | 0.020 | 0.016 | 0.005 |
| lv_epi | 0.952 | 0.908 | 1.000 | 1.000 | 200 | 1.000 | 0.000 | 0.000 | 0.016 | 0.013 |
| left_atrium | 0.891 | 0.811 | 1.000 | 1.000 | 200 | 1.000 | 0.000 | 0.070 | 0.036 | 0.020 |

Mean Dice by image quality: Good: 0.912, Medium: 0.904, Poor: 0.900
