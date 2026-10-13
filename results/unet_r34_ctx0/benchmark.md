Inference benchmark on Tesla T4, input 256x256.

| variant | batch | ms / batch | ms / frame | frames / s | max source fps at stride 2 |
|---|---|---|---|---|---|
| fp32 | 1 | 7.3 | 7.26 | 138 | 275 |
| fp32 | 4 | 18.6 | 4.66 | 215 | 429 |
| fp32 | 8 | 35.9 | 4.49 | 223 | 445 |
| fp32 + channels_last | 1 | 8.0 | 7.98 | 125 | 251 |
| fp32 + channels_last | 4 | 22.2 | 5.55 | 180 | 360 |
| fp32 + channels_last | 8 | 43.0 | 5.38 | 186 | 372 |
| mixed precision | 1 | 8.2 | 8.16 | 123 | 245 |
| mixed precision | 4 | 10.6 | 2.64 | 379 | 757 |
| mixed precision | 8 | 18.8 | 2.35 | 426 | 851 |
| mixed precision + channels_last | 1 | 7.2 | 7.22 | 138 | 277 |
| mixed precision + channels_last | 4 | 9.0 | 2.24 | 446 | 893 |
| mixed precision + channels_last | 8 | 15.8 | 1.98 | 506 | 1011 |
| mixed precision + channels_last + torch.compile | 1 | 3.3 | 3.29 | 304 | 607 |
| mixed precision + channels_last + torch.compile | 4 | 6.3 | 1.58 | 632 | 1264 |
| mixed precision + channels_last + torch.compile | 8 | 11.0 | 1.37 | 730 | 1459 |
