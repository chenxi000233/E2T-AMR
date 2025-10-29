# 🛰️ E2T-AMR: Enhanced End-to-End Automatic Modulation Recognition Framework

> **Authors:** Xi Chen, Xu Chen, Lin Lei, et al.  
> **Affiliation:** Wuhan University  
> **Last Updated:** 2025-10-29  
> **License:** MIT  

---

## 🌐 1.概述（Overview）

在复杂多变的无线通信环境中，**调制识别**（Automatic Modulation Recognition, *AMR*）是实现智能信号感知与频谱管理的核心环节。  
然而，传统深度学习方法多依赖单一模态的时域或频域特征，易受 **相位旋转、时间平移及低信噪比干扰** 影响，导致特征错位和识别性能退化。  
现有多模态融合方法虽在一定程度上缓解了该问题，但普遍 **忽略信号的物理一致性与模态能量差异**，易在融合过程中放大噪声模态的影响；同时，模型预测的置信度缺乏有效校准，降低了其在复杂信道下的可靠性。

为解决上述挑战，本文提出了一种 **等变与能量引导的多模态调制识别框架——E²T-AMR**（*Equivariant & Energy-guided Transformer for Multimodal Modulation Recognition*）。  
该框架由三大核心模块组成：

- 🔹 **C-GET 模块**：基于复数群等变卷积与可旋可移位置编码，实现信号在 **相位旋转与时间平移下的等变表征**，确保特征一致性。  
- 🔹 **E-Mixer 模块**：引入 **能量引导的动态加权机制**，依据模态能量自适应调整融合权重，有效抑制低信噪模态干扰。  
- 🔹 **U-Trans 模块**：结合 **不确定性感知与置信度自校准机制**，提升模型预测的可信性与稳健性。  

整体而言，E²T-AMR 在理论上融合了 **几何等变性、能量自适应性与不确定性建模**，  
在实践中显著提升了多模态融合的鲁棒性与可信性，尤其在 **低信噪比与复杂信道环境下** 展现出更优异的识别精度。

---

## ⚙️ 2.安装说明（Installation）

### 环境依赖
- Python ≥ 3.8  
- PyTorch ≥ 2.0.1
- NumPy、PyYAML、Matplotlib、scikit-learn

### 安装步骤
```bash
git clone https://github.com/yourname/E2T-AMR.git
cd E2T-AMR
pip install -r requirements.txt
export PYTHONPATH=$(pwd)/src
```
---

## 📚 3.数据集准备（Dataset Preparation）

### RML2016.10a 数据集

本项目采用 **DeepSig RML2016.10a** 数据集作为主要实验基准。  
每条样本信号均表示为一个复数基带序列 `(2×128)`，并附带以下标签信息：

- **Modulation Type（调制类型）**  
- **SNR（信噪比）**：从 -20 dB 到 +18 dB，间隔 2 dB  

该数据集被广泛用于无线通信调制识别研究，是验证 AMR 模型性能的标准基准。

---

### 📦 数据下载与放置

1. 从 [DeepSig 官方网站](https://www.deepsig.io/datasets) 下载原始文件 `RML2016.10a_dict.pkl`。  
2. 将文件放入以下路径. (按 "类别×信噪比" 进行 8:2 分层划分好并且处理后的数据我们提供了网盘链接：通过网盘分享的文件：data2md
链接: https://pan.baidu.com/s/1KsKHF2cCP_5PotXz9Q6F-A?pwd=cxcx 提取码: cxcx)：
```bash
data/processed/rml2016_10a/
├── RML2016.10a_dict.pkl # 原始数据
├── train.pkl # 训练集（8:2划分生成）
├── val.pkl # 验证集（8:2划分生成）
...
✅ Classes: ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
Split sizes -> train: 176000, val: 22000

最终的目录结构为：
E2T-AMR/
├── data/
│   ├── processed/
│   │   └── rml2016_10a/
│   │       ├── RML2016.10a_dict.pkl
│   │       ├── train.pkl
│   │       └── val.pkl        
│   └── raw/
└── scripts/
    └── split_rml2016.py             # 数据划分脚本
    ...
```

## 🚀 4.模型训练与评估（Training & Evaluation）

### 🧠 训练主流程

E2T-AMR 的训练采用统一入口脚本 `train.py`，支持多模态信号输入与自定义配置。  
默认配置位于 `configs/config.yaml`，可通过命令行参数修改。

#### 启动训练
```bash
python train.py --config configs/config.yaml
```
训练过程中将自动执行以下步骤：

(1).加载数据配置文件：configs/data/rml2016_10a.yaml

(2).构建多模态信号编码器（IQ、Spectral、Time-Frequency）

(3).启动 E-Mixer 模块融合多模态特征

(4).按照设定的 SNR Curriculum 策略逐步提升难度

(5).自动保存最佳模型权重、训练日志保存至：
```bash
outputs/checkpoints/best.pt
outputs/logs/
outputs/figures/
```
#### 评估指标
> 我们采用边训练边评估，训练配置5轮评估一次(可配置)，会输出在验证集上的：
整体准确率 (Overall Accuracy）、 低信噪比区间准确率 (Low-SNR ≤ 0 dB)、 高信噪比区间准确率 (High-SNR > 0 dB)、
按信噪比分组的详细性能表格示例：

**Overall:** **Val Acc = 0.6192**
**SNR 分段：** Low (≤ 0 dB) = **0.4314** ｜ High (> 0 dB) = **0.8620**

 说明：我们采用“边训练边评估”，每 5 个 epoch 验证一次并打印如下统计。

<details>
<summary><b>🔎 展开查看：按 SNR × 类别 的详细准确率表</b></summary>

> 单位：%（每格为该 SNR 下对应调制类别的准确率；最右为该行均值）

| SNR(dB) | 8PSK | AM-DSB | AM-SSB | BPSK | CPFSK | GFSK | PAM4 | QAM16 | QAM64 | QPSK | WBFM | **Mean** |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| -20 | 4.5 | 3.0 | 91.5 | 14.5 | 4.0 | 4.5 | 3.0 | 0.5 | 5.5 | 1.0 | 2.0 | **12.2** |
| -18 | 3.5 | 4.5 | 85.5 | 9.5 | 5.5 | 2.0 | 5.0 | 5.0 | 8.5 | 3.0 | 1.5 | **12.1** |
| -16 | 3.5 | 7.0 | 89.5 | 13.0 | 5.5 | 4.5 | 2.5 | 6.0 | 30.0 | 2.5 | 4.0 | **15.3** |
| -14 | 3.0 | 8.0 | 92.5 | 10.0 | 8.0 | 5.5 | 8.5 | 18.5 | 54.5 | 6.5 | 6.0 | **20.1** |
| -12 | 5.0 | 20.0 | 89.0 | 13.0 | 5.5 | 8.5 | 21.5 | 36.0 | 56.0 | 4.5 | 21.0 | **25.5** |
| -10 | 11.0 | 44.0 | 87.0 | 18.0 | 16.5 | 21.0 | 60.0 | 45.0 | 44.5 | 13.0 | 35.5 | **36.0** |
| -8 | 25.0 | 64.0 | 86.0 | 48.0 | 43.0 | 51.5 | 77.0 | 41.0 | 53.0 | 31.0 | 54.5 | **52.2** |
| -6 | 58.0 | 65.5 | 89.5 | 73.5 | 78.0 | 84.0 | 85.0 | 54.5 | 52.5 | 64.5 | 62.5 | **69.8** |
| -4 | 67.5 | 66.0 | 87.5 | 87.5 | 93.5 | 95.0 | 95.0 | 56.0 | 60.5 | 69.5 | 55.0 | **75.7** |
| -2 | 62.5 | 75.5 | 89.5 | 91.5 | 99.0 | 97.5 | 98.5 | 48.0 | 60.0 | 66.0 | 52.5 | **76.4** |
| 0 | 75.5 | 75.0 | 88.5 | 95.5 | 99.0 | 97.0 | 98.5 | 49.5 | 58.0 | 73.0 | 63.0 | **79.3** |
| 2 | 89.0 | 76.5 | 90.0 | 99.5 | 100.0 | 99.5 | 99.0 | 56.0 | 59.5 | 84.0 | 57.5 | **82.8** |
| 4 | 92.0 | 74.5 | 88.5 | 99.5 | 99.5 | 99.5 | 98.5 | 49.0 | 60.5 | 93.0 | 55.0 | **82.7** |
| 6 | 92.5 | 93.5 | 86.5 | 98.5 | 100.0 | 99.5 | 98.0 | 50.5 | 63.0 | 96.5 | 53.5 | **84.7** |
| 8 | 90.5 | 95.5 | 89.5 | 98.0 | 100.0 | 100.0 | 97.5 | 60.0 | 55.0 | 95.0 | 49.0 | **84.5** |
| 10 | 94.5 | 98.5 | 84.0 | 99.0 | 100.0 | 100.0 | 96.5 | 52.0 | 62.5 | 95.5 | 44.5 | **84.3** |
| 12 | 95.5 | 99.5 | 91.5 | 99.5 | 100.0 | 99.5 | 97.5 | 58.0 | 59.0 | 96.0 | 43.5 | **85.4** |
| 14 | 94.5 | 99.5 | 90.0 | 98.5 | 100.0 | 100.0 | 98.0 | 55.5 | 35.0 | 95.5 | 45.0 | **82.9** |
| 16 | 93.0 | 98.0 | 87.0 | 99.0 | 100.0 | 100.0 | 97.5 | 65.5 | 88.0 | 97.5 | 48.5 | **88.5** |
| 18 | 92.0 | 99.5 | 87.5 | 98.5 | 100.0 | 99.5 | 97.5 | 55.0 | 99.5 | 96.5 | 43.0 | **88.0** |

</details>

---

## 🔧 6. 模型微调（Model Fine-Tuning）

E2T-AMR 支持基于已有模型权重进行快速迁移学习或微调（Fine-Tuning）。  
通过加载已训练好的 `best.pt` 权重，可在新的数据分布或扩展数据集上快速收敛，  
以提升模型在特定信噪比或特定调制类型下的识别性能。

---

### ⚙️ 微调命令示例（Fine-Tuning Command）

以下命令以 `outputs/checkpoints/best.pt` 作为初始化参数，  
在原始训练集或新的样本集上继续训练 50 个 epoch：

```bash
python -m e2tamr.entrypoints.finetune \
  --config configs/config.yaml \
  --data-cfg configs/data/rml2016_10a.yaml \
  --model-cfg configs/model/e2t_amr.yaml \
  --train-pkl data/processed/rml2016_10a/train_s.pkl \
  --val-pkl   data/processed/rml2016_10a/val_s.pkl \
  --finetune-ckpt outputs/checkpoints/best.pt \
  --epochs 20 \
  --epochs-freeze 3 \
  --freeze-body \
  --lr-body 1e-4 \
  --lr-head 5e-4 \
  --batch-size 512 \
  --device cuda:0
  
  模型保存至：
  outputs/checkpoints/best_finetune.pt

```

---


## 🚀 7. 模型推理（Model Inference）

E2T-AMR 支持在训练完成后对任意数据进行快速推理与结果可视化。  
推理阶段同样支持多模态输入（IQ、频谱、时频特征），可在 GPU/CPU 环境中灵活切换。  

---

### 🧠 单样本推理（Single Sample Inference）

以下命令将加载训练好的模型(网盘给出了一个我们训练好的模型，放置在这个路径下即可执行推理：outputs/checkpoints/)，并对指定信噪比下的单个样本进行调制类型预测与可视化展示：

```bash
python scripts/infer.py \
  --data-cfg configs/data/rml2016_10a.yaml \
  --model-cfg configs/model/e2t_amr.yaml \
  --ckpt outputs/checkpoints/best.pt \
  --train-pkl data/processed/rml2016_10a/train.pkl \
  --index 12345 \
  --device cuda:0
#### 结果示例
========================================================================
[BATCH] shapes:
    iq: shape=(1, 2, 128) dtype=torch.float32
  spec: shape=(1, 129) dtype=torch.float32
    tf: shape=(1, 33, 5) dtype=torch.float32
     y: shape=(1,) dtype=torch.int64
   snr: shape=(1,) dtype=torch.int64
  -> True Label (after pipeline): id=1, name=AM-DSB
========================================================================
[info] loaded ckpt: missing=0, unexpected=0
[INFER] predicted:
  class_id=1, class_name=AM-DSB, confidence=0.9076
========================================================================
```
---
## 8. 实验结果
### 📊 模型性能对比（Performance Comparison on RadioML2016.10a）

我们将 **E2T-AMR** 与代表性调制识别模型进行对比，包括 PET-CGDNN、MCLDNN、AMC-NET、FEA-T 和 IQFormer 等。  
对比维度包括最高识别率（Highest Accuracy）、低信噪比性能（Low-SNR ≤ 0dB）、高信噪比性能（High-SNR > 0dB）及总体平均性能（OverAll）。

| Model | Parameters | Highest Acc (%) | Low-SNR (≤0dB) | High-SNR (≥10dB) | OverAll (%) |
|:------|:-----------:|:----------------:|:----------------:|:----------------:|:------------:|
| PET-CGDNN | **0.07M** | 90.77 | 36.21 | 89.45 | 60.38 |
| MCLDNN | 0.41M | 92.86 | 37.27 | 91.68 | 61.93 |
| AMC-NET | 0.47M | 92.82 | 38.56 | 91.32 | 62.40 |
| FEA-T | 0.17M | 90.09 | 37.31 | 88.54 | 60.55 |
| IQFormer | 0.35M | **93.91** | 40.33 | 93.15 | 64.19 |
| **E2T-AMR (ours)** | **0.42M** | 🟩 **96.11** | 🟩 **58.11** | 🟩 **94.71** | 🟩 **76.41** |

> **说明：** OverAll 为低/高信噪比准确率的算术平均值。  
> E2T-AMR 在低信噪比场景下显著优于同类模型，体现出多模态融合架构在噪声环境中的稳健性。

---

### 📈 可视化结果（Accuracy Comparison）

| 模型 | ![bar](https://img.shields.io/badge/-🟩-brightgreen) Highest | ![bar](https://img.shields.io/badge/-🟨-yellow) Low-SNR | ![bar](https://img.shields.io/badge/-🟦-blue) High-SNR | ![bar](https://img.shields.io/badge/-⚫-lightgrey) OverAll |
|------|:----------------------:|:--------------------:|:-------------------:|:----------------:|
| PET-CGDNN | ████████░░░ 90.8 | ██░░░░░░░░░░ 36.2 | ████████░░░░ 89.4 | ████░░░░░░░░ 60.4 |
| MCLDNN | ████████░░░░ 92.9 | ██░░░░░░░░░░ 37.3 | ████████░░░░ 91.7 | ████░░░░░░░░ 61.9 |
| AMC-NET | ████████░░░░ 92.8 | ██░░░░░░░░░░ 38.6 | ████████░░░░ 91.3 | ████░░░░░░░░ 62.4 |
| FEA-T | ███████░░░░░ 90.1 | ██░░░░░░░░░░ 37.3 | ███████░░░░░ 88.5 | ████░░░░░░░░ 60.5 |
| IQFormer | ████████░░░░ 93.9 | ██░░░░░░░░░░ 40.3 | ████████░░░░ 93.1 | █████░░░░░░░ 64.2 |
| **E2T-AMR (ours)** | 🟩█████████░ 96.1 | 🟩█████░░░░░ 58.1 | 🟩████████░░ 94.7 | 🟩███████░░░ 76.4 |


---

### 🧾 9. 引用信息（Citation）


```markdown
## 🧾 引用（Citation）

如果您在研究中使用了本项目，请引用如下论文：

```bibtex
@article{chen2025e2tamr,
  title={E2T-AMR: Enhanced End-to-End Multi-Modal Framework for Automatic Modulation Recognition},
  author={Chen, Xi and Chen, Xu and Lei, Lin},
  journal={-},
  year={2025}
}
