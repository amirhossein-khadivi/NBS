---
layout: default
title: About
permalink: /about/
---

---

<div class="profile-header">
  <img
    src="https://github.com/amirhossein-khadivi.png"
    alt="Amirhossein Khadivi"
    class="profile-photo"
  >

  <h1>Amirhossein Khadivi</h1>
  <p class="profile-role">Statistician & Researcher</p>
</div>

---

I am a statistician and researcher with an academic background in statistics and data science. I received my bachelor's degree in **Statistics from the University of Guilan** and my master's degree in **Data Science from the University of Tehran**.

My research interests lie at the intersection of **statistics, data science, and artificial intelligence**, with a particular interest in **computer vision** and learning-based methods for visual data analysis.

I am interested in developing intelligent and data-driven methods that combine statistical reasoning with modern machine learning and deep learning techniques, particularly for challenging problems where data are limited, heterogeneous, or difficult to annotate.

---

## Research Interests

My main research interests include:

- Statistics and Data Science
- Machine Learning and Deep Learning
- Artificial Intelligence
- Computer Vision
- Anomaly Detection
- Reinforcement Learning
- Data-driven and intelligent learning methods

---

## Master's Thesis

My master's thesis at the **University of Tehran** focused on industrial visual anomaly detection using deep reinforcement learning:

> **Neural Batch Sampling with Reinforcement Learning for Semi-Supervised Anomaly Detection**

Pixel-level anomaly detection is an important problem in industrial visual inspection, where defects can be subtle, localized, and difficult to annotate. At the same time, real-world industrial environments often provide only a limited number of defective samples.

In this work, I developed a semi-supervised deep reinforcement learning framework that combines **adaptive spatial sampling, image reconstruction, and anomaly prediction**. The framework is designed to allow an RL agent to actively select informative image patches rather than processing all image regions uniformly.

The proposed framework integrates three main components:

1. **Neural Batch Sampler**  
   An RL-based agent that sequentially selects informative image patches using visual, structural, reconstruction, and sampling-history information.

2. **Robust Autoencoder**  
   An autoencoder that produces reconstruction-error information for anomaly localization while using structured dropout to provide diverse and stable reconstructions without relying on periodic model resets.

3. **Lightweight Anomaly Predictor**  
   A fully convolutional predictor that operates directly on the reconstruction-error space to produce pixel-level anomaly predictions.

The framework also includes a theoretical analysis of the stochastic reset mechanism. The analysis studies how sudden changes in the autoencoder state can affect reward variability and training stability.

Experiments on the **MVTec AD** benchmark demonstrated the effectiveness of the proposed framework for industrial anomaly detection and localization, with an average **AUC of 0.949** and **maximum F1 score of 0.561** across the evaluated scenarios.

The complete methodology, experimental results, implementation details, and theoretical analysis are available throughout this website.

---

## Academic Advisors

I would like to acknowledge my master's thesis advisors for their guidance and support throughout this research:

- **[Dr. Abdollah Safari](https://scholar.google.com/citations?user=cuX6eCMAAAAJ&hl=en)**
- **[Dr. Firoozeh Haghighi](https://scholar.google.com/citations?hl=en&user=JO2wIwsAAAAJ)**
- **[Dr. Fatemeh Ziaeetabar](https://scholar.google.com/citations?hl=en&user=QxfijdkAAAAJ)**

---

## Research Perspective

My broader research goal is to explore how statistical thinking and modern artificial intelligence can be combined to develop more efficient, interpretable, and adaptive learning systems.

I am particularly interested in problems where intelligent models must make decisions from complex visual data while dealing with limited supervision, uncertainty, and challenging data distributions.
