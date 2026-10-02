# GitHub 上传说明

## 仓库设置

建议仓库名称为 `MIAWNet`，简介为：

```text
Multi-scale interaction and adaptive weighting for referring image segmentation in low-altitude UAV imagery.
```

可填写的 Topics：

```text
referring-image-segmentation uav computer-vision pytorch swin-transformer bert multimodal-learning
```

## 网页上传

1. 解压 `MIAWNet_GitHub_upload.zip`。
2. 在 GitHub 创建仓库。新建空仓库时，不必另外生成 README。
3. 进入仓库，选择 **Add file → Upload files**。
4. 打开解压后的 `MIAWNet` 目录，将里面的文件和子目录上传到仓库根目录。不要将 ZIP 本身当作仓库内容上传，也不要再套一层 `MIAWNet/`。
5. 确认根目录中直接出现 `README.md`、`train.py`、`src/`、`configs/` 和 `assets/`，再提交。

上传时保留 `.github/`、`.gitignore` 和 `.gitattributes`。文件浏览器可能隐藏以点开头的文件；若网页上传未包含它们，使用下面的 Git 命令可以完整保留。

## 使用 Git 上传

在解压后的 `MIAWNet` 目录中打开终端，执行：

```bash
git init
git add .
git commit -m "Add MIAWNet implementation and paper assets"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/MIAWNet.git
git push -u origin main
```

将 `YOUR_USERNAME` 替换成你的 GitHub 用户名。如果仓库已有文件或提交，先克隆仓库，再将上传包内容复制进克隆目录后提交；不要对已有仓库强制推送。

## 首页内容

仓库只保留一个英文 `README.md` 作为首页，以及一个 `requirements.txt` 作为依赖文件。首页包括论文作者、任务徽章、数据集制作图、整体框架图、论文结果表、定性对比和消融可视化，以及安装、训练、评估和代码目录说明。所有论文图片均保存在仓库内，使用相对路径。

`assets/paper/MIAWNet.pdf` 是你提供的论文稿，`assets/figures/framework.pdf` 是你提供的框架原图。论文更新后，可替换同名文件。

## 数据和权重

上传包没有实际训练数据和训练权重。首页没有虚构下载链接；取得正式的 DroneRIS 发布地址或模型下载地址后，可在首页的对应说明中加入链接。

论文结果表已注明来自论文。仓库的运行验证记录在 `docs/verification.md`，实现选择记录在 `docs/implementation.md`，两者不代替论文实验。

`CITATION.cff` 已填写论文提供的作者名单和标题。正式发表后，可以补充 DOI 和出版信息。许可证尚未指定，需由作者确定授权方式后再添加。
