# Hierarchical Sports Tennis Classification

Perfect! Here’s your **full, improved, detailed README** based on the draft you gave, with professional formatting, clear sections, badges, and polished explanations while keeping it lengthy:

---

# Hierarchical Deep Learning for Sports & Tennis Action Classification 🎾🏋️‍♂️

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-1.13-red)

This project implements a **two-stage hierarchical deep learning pipeline**:

1. **Sport Classification** → A ResNet-50 model trained on the [Sports Classification Dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification) with **100 sport categories**.
2. **Tennis Action Classification** → If the predicted sport is *Tennis*, a second ResNet-50 model predicts fine-grained tennis actions such as forehand, backhand, serve, volley, etc.

This hierarchical approach ensures that general sports are classified first, and only if the sport is Tennis, a more detailed action classification is performed.

---

## Features

* **Two-stage hierarchical pipeline** for efficient multi-level classification
* **ResNet-50 backbone** with transfer learning for both stages
* **Modular training and evaluation scripts**
* **Inference with softmax confidence scores**
* **Training history visualization** (loss & accuracy curves)
* **Flexible dataset integration** (easily swap in your own datasets)

---

## Project Structure

```
📦 hierarchical-sports-tennis-classification
┣ 📜 train_eval.py              # Main training & evaluation script
┣ 📜 requirements.txt           # Python dependencies
┣ 📜 README.md                  # Project documentation
┣ 📂 archive/                   # Sports classification dataset (download from Kaggle)
┣ 📂 tennis-action-dataset/     # Tennis action dataset (custom dataset)
┗ 📂 Notebook/                  # Optional Jupyter notebooks for experiments
```

* **Phase 1:** Sport Classification → predicts general sport category (100 classes)
* **Phase 2:** Tennis Action Classification → fine-grained tennis action prediction if sport is Tennis
* **Phase 3:** Combined inference function → chains both models for end-to-end prediction

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/Jayantsrikarp/hierarchical-sports-tennis-classification.git
cd hierarchical-sports-tennis-classification
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. (Optional) Install GPU version of PyTorch for faster training: [PyTorch Installation](https://pytorch.org/get-started/locally/).

---

## 🏋️ Training

### 1. Train Sport Classifier

```bash
python train_eval.py
```

* Trains ResNet-50 on the sports classification dataset.
* Saves the best model as `best_resnet_sports_classifier.pth`.

### 2. Train Tennis Action Classifier

* Uses the `tennis-action-dataset/` for fine-grained action prediction.
* Saves the best model as `best_resnet_tennis_action_classifier.pth`.

---

## 🔎 Evaluation

The `train_eval.py` script includes:

* **Classification reports**: precision, recall, F1-score for both phases
* **Accuracy & Loss plots** for both sport and tennis action classification
* Supports both **validation and test datasets**

---

## 📊 Results

* **Sport Classification**: Validation Accuracy → \~89%
* **Tennis Action Classification**: Validation Accuracy → \~71%

The hierarchical approach improves efficiency by first narrowing down the sport category before predicting detailed tennis actions.

---

## 📌 Notes

* **Datasets are not included** due to size.
* Download the sports dataset from Kaggle and place it in `archive/`.
* Prepare your tennis dataset and place it in `tennis-action-dataset/`.
* Large model checkpoints (`.pth`) can be tracked with [Git LFS](https://git-lfs.github.com).

---

## 🔮 Future Work

* Incorporate **BiLSTM with Attention** for sequence modeling of tennis actions (video-level classification).
* Extend the pipeline to **other sports for hierarchical action classification**.
* Optimize models for **real-time inference** and deployment in sports analytics dashboards.

---

## 📚 References

* He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
* Bahdanau, D., Cho, K., & Bengio, Y. (2015). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).

