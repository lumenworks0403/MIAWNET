# Implementation details

## Method mapping

| Manuscript component | Source |
| --- | --- |
| Swin-B and BERT | `src/miawnet/models/network.py` |
| Spatial cross-modal alignment | `CrossModalAttention` in `models/modules.py` |
| MFIE, equations (1)–(4) | `MFIE` in `models/modules.py` |
| AFW, equations (5)–(7) | `AFW` in `models/modules.py` |
| BCE + Dice, equations (8)–(10) | `src/miawnet/losses.py` |
| Training and checkpoint selection | `src/miawnet/train.py` |
| oIoU, mIoU and P@X | `src/miawnet/metrics.py` |

MFIE uses full-resolution queries and average-pooled keys and values. Scaled dot-product attention uses the channel dimension as the scale. The local branch applies separate depthwise convolutions to Q, K and V. Query–key products pass through pointwise projection, SiLU (Swish), a second pointwise projection, and Tanh after division by the square root of the channel dimension. Local and global results are concatenated and projected.

AFW applies two independent `1×1 Conv → GELU → 1×1 Conv → Tanh` mappings. Its weights can be negative and are not normalized or constrained to sum to one.

The output is one foreground logit per pixel. BCE uses `binary_cross_entropy_with_logits` for numerical stability. Dice uses continuous sigmoid probabilities. Both losses are calculated per sample and then averaged over the batch, with equal weights and no auxiliary terms.

## Choices not specified by the manuscript

| Detail | Release default |
| --- | --- |
| BERT checkpoint | `bert-base-uncased` |
| Encoder updates | End-to-end optimization of Swin-B and BERT |
| Cross-modal attention | Single-head projected visual queries and text keys/values; BERT padding is masked |
| Cross-modal feature width | 256 channels at all four stages |
| MFIE input projection | Pointwise linear map followed by a joint QKV map, following Fig. 3 |
| Pool kernel and stride | 8, 4, 2, 1 at stages 1–4; stride equals kernel |
| Pool border handling | Valid average pooling; incomplete bottom/right windows are omitted |
| Local convolution | 3×3 depthwise convolution with padding 1 |
| Decoder | Per-stage projection of concatenated visual/semantic features; bilinear resize to stage 1; concatenate four stages; 1×1 and 3×3 refinement with GroupNorm/GELU; single-channel output |
| Decoder width | 256 channels |
| Resize and normalization | Direct square bilinear RGB resize, nearest-neighbor mask resize, ImageNet mean/std |
| Maximum text length | 64 BERT tokens, including special tokens |
| Dice epsilon | 1e-6 in numerator and denominator |
| Cosine update | One learning rate per epoch; first/last training epochs use the stated maximum/minimum |
| Threshold and metric resolution | Probability ≥ 0.5; original-size masks |
| Empty-union IoU | 1 when both masks are empty |
| AFW-only ablation | Two independent gates applied to the same cross-modal features |
| Seed and precision | Seed 42; float32 by default |

These choices make the implementation concrete and configurable. They are not recoverable experimental settings established by the PDF. To reproduce a previous experiment exactly, reconcile them with its configuration and checkpoint.

The manuscript refers to DroneRIS validation in the experimental setup and ablation text, while Table 2 labels the DroneRIS column as Test. Evaluation therefore requires an explicit manifest; test scores are not computed during checkpoint selection. The checkpoint is always selected using the configured validation split.

Random generators are seeded and saved for resumption at epoch boundaries. GPU kernels and changes in software versions can still affect numerical reproducibility. This release does not fabricate splits, weights, or benchmark results.
