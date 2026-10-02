# Verification

The release was checked locally with Python 3.12.14, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, Transformers 4.38.2, and NumPy 1.26.4 on Windows.

| Check | Result |
| --- | --- |
| `pytest -q` | 24 passed |
| `ruff check src tests train.py evaluate.py predict.py tools` | Passed |
| `ruff format --check src tests train.py evaluate.py predict.py tools` | Passed |
| Python source compilation | Passed |
| Complete Swin-B/BERT model, 480 × 480 RGB input | Finite output with shape `[1, 1, 480, 480]` |
| Baseline / MFIE / AFW / full variants | Forward and backward checks passed |
| Epoch-boundary resume | Parameters matched uninterrupted CPU training exactly |
| Offline checkpoint restoration | Saved BERT configuration restored; outputs matched |
| Offline prediction/evaluation commands | Native-size mask, overlay and metric JSON written |
| Official-format RefCOCO conversion | Expressions, instance mask and split membership preserved |
| Repository-root command entry points | Training, evaluation, prediction, data audit and conversion help commands passed |
| README and documentation assets | Relative links and figure files checked |

The complete architecture check used randomly initialized encoders with pretrained downloads disabled. Integration tests used synthetic data and compact injected encoders to exercise the same model modules and command paths.

The manuscript environment is specified in `requirements.txt`. That exact Python 3.10/PyTorch 2.0 environment and CUDA training were not run locally. No DroneRIS or RefCOCO benchmark training was performed, and no trained checkpoints or claimed benchmark results are bundled.
