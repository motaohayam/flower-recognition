# Flower Recognition using 102 Category Flower Dataset

Fine-tuning pretrained CNNs for 102-class flower classification with hyperparameter
analysis, ablation study, and attention mechanisms.

## Results Summary

| Model                            | Val Acc | Test Acc |
|----------------------------------|---------|----------|
| ResNet-34 (random init)          | 6.27%   | 6.29%    |
| ResNet-34 (pretrained, baseline) | 92.25%  | 88.73%   |
| ResNet-34 (pretrained, best LR)  | 94.31%  | 92.08%   |
| ResNet-34 + SE blocks            | 94.80%  | 90.91%   |

## Repository Structure
flower-recognition/
├── flower_recognition_main.ipynb  # Main notebook with all experiments
├── report.md                      # Full experiment report
├── logs/                          # Per-epoch training logs (CSV)
├── results/                       # Experiment result summaries (JSON)
└── figures/                       # Training curves and comparison charts

## Experiments

1. **Baseline**: ResNet-34 pretrained on ImageNet, head replaced for 102 classes,
   two-phase training (freeze then fine-tune).
2. **Hyperparameter sweep**: Grid search over learning rates and epochs.
   Best config: head_lr=1e-3, finetune_lr=5e-5, 20 epochs.
3. **Ablation study**: Same architecture trained from random initialization.
   Pretraining provides +85.79% absolute improvement in test accuracy.
4. **Attention model**: SE blocks inserted after each ResNet-34 residual stage.
   Achieves highest validation accuracy (94.80%).

## Dataset

Oxford 102 Category Flower Dataset:
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

Splits: 1,020 train / 1,020 val / 6,149 test (official splits).

## How to Run

1. Open `flower_recognition_main.ipynb` in Kaggle or Google Colab.
2. Set dataset path to point to the 102 flowers data folder.
3. Run all cells in order.

## Requirements

- Python 3.10+
- PyTorch 2.x
- torchvision
- scipy
- matplotlib
- PIL
