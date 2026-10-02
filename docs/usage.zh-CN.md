# MIAWNet

MIAWNet 的 PyTorch 实现，用于根据自然语言描述分割低空无人机图像中的目标。模型包括 Swin-B、BERT、跨模态注意力、MFIE 和 AFW，使用等权重 BCE + Dice 监督最终掩码。

本项目依据论文中的方法和实验设置编写，包含训练、断点续训、评估、预测、数据转换及消融配置。实际数据和训练权重需要另行准备。这里没有通过训练获得论文中的结果；池化尺寸、解码器结构等论文未明确的细节采用可配置默认值，详见 [实现说明](implementation.md)。

## 安装

论文环境为 Python 3.10、PyTorch 2.0.0、Ubuntu 22.04 和 RTX 4090。先创建虚拟环境，再安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[dev,refcoco]"
```

Windows 使用 `.venv\Scripts\Activate.ps1` 激活环境。PyTorch 应选择与 CUDA 匹配的版本；Python 3.12 需要使用兼容的新版本 PyTorch/torchvision，不能直接安装论文版本锁定文件。

## 准备数据

在 `data/droneris/annotations/` 下放置 `train.jsonl`、`val.jsonl` 和 `test.jsonl`，每行对应一个图像—描述—目标掩码三元组：

```json
{"id":"video03_frame001_car02","image":"images/video03/frame001.jpg","mask":"masks/video03/frame001_car02.png","text":"The white car next to the bus.","video_id":"video03"}
```

掩码只包含描述指定的实例，背景为 0，前景为 1 或 255。图像与掩码保持相同原始尺寸。DroneRIS 必须按源视频隔离训练、验证和测试数据，不能只随机划分三元组。论文给出的约 7:1:2 比例无法恢复原始划分，应使用实际发布的划分清单。

```bash
miawnet-check-data --root data/droneris \
  --manifests annotations/train.jsonl annotations/val.jsonl annotations/test.jsonl \
  --require-video-ids
```

RefCOCO/RefCOCO+ 的官方标注可用 `miawnet-convert-refcoco` 转换，步骤见 [数据说明](data.md)。

## 训练和评估

```bash
miawnet-train --config configs/droneris.yaml
miawnet-train --config configs/droneris.yaml --resume runs/droneris/last.pt
miawnet-evaluate --checkpoint runs/droneris/best.pt \
  --manifest annotations/test.jsonl --output runs/droneris/test.json
```

默认设置为 480×480、50 个 epoch、batch size 8、AdamW、权重衰减 0.05，学习率从 3e-5 余弦衰减至 1e-6，不使用随机数据增强。以验证集 mIoU 选择 `best.pt`。评估报告 oIoU、mIoU 和 P@0.5/0.7/0.9，单位均为百分比；P@X 按 IoU 严格大于 X 统计。

默认将预测 logit 上采样回原始掩码尺寸再计算指标。若采用 480×480 评估，可设置 `--resolution resized`，并在实验记录中写明，避免不同评估尺寸影响比较。

## 单图预测

```bash
miawnet-predict --checkpoint runs/droneris/best.pt \
  --image scene.jpg --text "The white car next to the bus." \
  --output runs/demo/mask.png --overlay runs/demo/overlay.png
```

## 上传 GitHub

将整个项目目录作为仓库内容上传即可；保留 `src/`、`configs/`、`tests/`、文档及工作流。数据、运行日志和权重已通过 `.gitignore` 排除。`CITATION.cff` 使用论文提供的作者名单，发表信息明确后可以补充 DOI 和正式文献信息。开源许可证需由作者按所属机构要求确定，本项目没有预设授权条款。

完整命令、消融配置和技术说明见 [English README](../README.md)。
