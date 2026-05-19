# Flower Recognition: Fine-tuning CNNs on the 102 Category Flower Dataset

## 1. Introduction

This report presents a study and an analysis of CNN fine-tuning for the purpose of flower classification from the Oxford 102 Category Flower Dataset, using Kaggle as the development environment. The impact and effect of ImageNet pretraining on the dataset was examined, perform hyperparameter analysis, conduct an ablation study, and extend the baseline model using attention mechanisms.

---

## 2. Dataset

The Oxford 102 Category Flower Dataset consists of 8,189 images across 102 flower
categories commonly found in the United Kingdom. Each class contains between 40 and
258 images. The official data splits are used throughout all experiments:

| Split      | Samples |
|------------|---------|
| Train      | 1,020   |
| Validation | 1,020   |
| Test       | 6,149   |

Images are resized to 224×224 pixels. Augmentations for training include random resized 
crop, random horizontal flip, and color jitter (brightness, contrast, saturation with ±0.2). 
Center crop is applied on validation and test sets only. Images are normalized based on 
ImageNet mean and standard deviation.

---

## 3. Baseline Model

### 3.1 Architecture

I used ResNet-34 pretrained on ImageNet as the baseline. The final fully connected
layer is replaced with a new linear layer mapping 512 features to 102 classes.

### 3.2 Training Strategy

Training proceeds in two phases:

- **Phase 1 (epochs 1–5):** Only the new classification head is trained. All backbone
  parameters are frozen. This allows the head to adapt to the new task before the
  backbone is modified.
- **Phase 2 (epochs 6–20):** All layers are unfrozen. The backbone is fine-tuned with
  a small learning rate to preserve pretrained features while adapting to flowers.

| Setting          | Value   |
|------------------|---------|
| Optimizer        | Adam    |
| Head LR          | 1e-3    |
| Backbone LR      | 1e-4    |
| Epochs           | 20      |
| Batch size       | 32      |
| Input size       | 224×224 |

### 3.3 Results

| Metric       | Value  |
|--------------|--------|
| Best val acc | 92.25% |
| Test acc     | 88.73% |

---

## 4. Hyperparameter Analysis

A grid search was conducted over head learning rate, backbone (finetune) learning rate,
and number of training epochs. All other settings were held constant.

| head_lr | finetune_lr | epochs | Val Acc | Test Acc |
|---------|-------------|--------|---------|----------|
| 1e-3    | 1e-4        | 20     | 92.25%  | 88.73%   |
| 1e-3    | 1e-4        | 15     | 93.24%  | 89.54%   |
| 1e-3    | 1e-4        | 25     | 92.65%  | 90.42%   |
| **1e-3**| **5e-5**    | **20** | **94.31%** | **92.08%** |
| 5e-4    | 1e-4        | 20     | 93.33%  | 89.38%   |
| 1e-3    | 2e-4        | 20     | 89.80%  | 85.69%   |
| 2e-3    | 1e-4        | 20     | 90.69%  | 87.72%   |

The best configuration uses `head_lr=1e-3` and `finetune_lr=5e-5` for 20 epochs,
achieving **92.08% test accuracy**. Key observations:

- The finetune LR reduction from 1e-4 to 5e-5 improved the accuracy on test data
  by 3.35%. Smaller backbone LR ensures better preservation of ImageNet pretrained data.
- The increase in the finetune LR to 2e-4 produced the biggest decrease (85.69%),
  which proves that aggressive finetuning leads to loss of pretrained knowledge.
- Finetuning for 25 epochs using the initial LR also resulted in a higher performance
  than with 20 epochs used as the base.

---

## 5. Ablation Study: Effect of Pretraining

To quantify the contribution of ImageNet pretraining, the same ResNet-34
architecture was trained from random initialization on the flower dataset only.

| Model                        | Val Acc | Test Acc |
|------------------------------|---------|----------|
| ResNet-34 (random init)      | 6.27%   | 6.29%    |
| ResNet-34 (ImageNet pretrained) | 94.31% | 92.08%  |
| **Improvement**              | +88.04% | +85.79%  |

Training from scratch on 1,020 images across 102 classes (≈10 images per class) is
insufficient to learn meaningful representations in a 21M parameter network. The model
fails to converge, achieving only 6.29% test accuracy — barely above random chance
(0.98% for 102 classes). ImageNet pretraining provides essential low-level and
mid-level feature representations that transfer effectively to flower classification.

---

## 6. Attention Mechanism: SE-ResNet-34

### 6.1 Architecture

Squeeze-and-Excitation (SE) blocks are inserted after each of the four residual
stages of ResNet-34. An SE block performs channel-wise attention in two steps:

1. **Squeeze:** Global average pooling compresses each feature map to a single value,
   producing a channel descriptor of shape (C,).
2. **Excitation:** Two fully connected layers (with ReLU and Sigmoid) produce a
   channel-wise weight vector. The reduction ratio is set to 16.
3. **Scale:** The original feature map is multiplied by the learned channel weights,
   allowing the network to emphasize informative channels and suppress less useful ones.

The SE blocks add approximately 47,000 parameters to the 21.3M parameter ResNet-34.

### 6.2 Training

The same best configuration is used: `head_lr=1e-3`, `finetune_lr=5e-5`, 20 epochs,
with the same two-phase training strategy.

### 6.3 Results

| Model                           | Val Acc | Test Acc |
|---------------------------------|---------|----------|
| ResNet-34 (pretrained, best LR) | 94.31%  | 92.08%   |
| ResNet-34 + SE blocks           | 94.80%  | 90.91%   |

The SE model achieves the highest validation accuracy (94.80%) but slightly lower
test accuracy (90.91%) compared to the best baseline. The gap between val and test
accuracy suggests mild overfitting, likely due to the limited training set size.
The SE blocks provide measurable benefit on the validation set, indicating that
channel attention helps the model focus on discriminative flower features.

---

## 7. Summary and Comparison

| Model                           | Val Acc | Test Acc |
|---------------------------------|---------|----------|
| ResNet-34 (random init)         | 6.27%   | 6.29%    |
| ResNet-34 (pretrained, baseline)| 92.25%  | 88.73%   |
| ResNet-34 (pretrained, best LR) | 94.31%  | 92.08%   |
| ResNet-34 + SE blocks           | 94.80%  | 90.91%   |

The best overall test accuracy of **92.08%** is achieved by ResNet-34 with ImageNet
pretraining and a carefully tuned fine-tuning learning rate of 5e-5. The key findings are:

1. ImageNet pretraining is critical — it provides an +85.79% absolute improvement
   over random initialization on this small dataset.
2. The backbone fine-tuning learning rate has a larger impact than the number of
   epochs. Too large a finetune LR destroys pretrained features.
3. SE blocks improve validation accuracy but do not consistently improve test accuracy
   on this dataset size, suggesting attention mechanisms benefit more from larger
   training sets.

