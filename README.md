# CAP-IQA

## Description

Prompt-based methods, which encode medical priors through descriptive text, have been only minimally explored for CT Image Quality Assessment (IQA).  While such prompts can embed prior knowledge about diagnostic quality, they often introduce bias by reflecting idealized definitions that may not hold under real-world degradations such as noise, motion artifacts, or scanner variability. To address this, we propose a Context-Aware Prompt-Guided Image Quality Assessment (CAP-IQA) framework that integrates text-level priors with instance-level context prompts, and applies causal debiasing to separate idealized knowledge from factual, image-specific degradations. Our framework combines a CNN–based visual encoder inspired by U-Net architecture with a domain-specific text encoder to assess diagnostic visibility, anatomical clarity, and noise perception. The model leverages radiology-style prompts and context-aware fusion to align semantic and perceptual representations. On the LDCTIQAC-2023 benchmark, CAP-IQA achieves an overall correlation score of 2.8590 (sum of PLCC, SROCC, and KROCC), surpassing the top-ranked leaderboard team (2.7427) by 4.24\%. Moreover, ablation analyses confirm that prompt-guided fusion and the simplified encoder-only design jointly enhance feature alignment and interpretability. The results demonstrate the effectiveness of CAP-IQA in terms of perceptual fidelity in CT image quality assessment.
![](figure/fig1.png)

---

## Installation

1. Clone the repository and navigate to its directory.
2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

---

## Download Pretrained Weights

1. Download the pretrained weights from the following link: [XYZ](#)
2. Create a folder named `saved_models` in the root directory of the project.
3. Place the downloaded pretrained weights in the `saved_models` folder.

---

## Running Inference

1. Open the `inference.ipynb` notebook.
2. Follow the steps provided in the notebook to test sample results.

---

## Example Usage

- After setting up the environment and downloading the pretrained weights, you can test the model's performance on your CT images.

---
