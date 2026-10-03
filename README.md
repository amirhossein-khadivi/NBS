# Neural Batch Sampling (NBS)

## Neural Batch Sampling with Reinforcement Learning for Semi-Supervised Industrial Anomaly Detection

Neural Batch Sampling (NBS) is a reinforcement-learning-based framework for intelligent spatial sampling in industrial image anomaly detection.

Instead of treating image regions uniformly, NBS learns to navigate through an image and select informative patches using a neural sampling policy. The selected regions are evaluated through reconstruction and anomaly-prediction signals, while the sampling policy is optimized using a composite reward.

---

## Project Overview

Industrial anomaly detection often requires identifying subtle and spatially localized defects.

A conventional approach may process the entire image uniformly or rely on predefined sampling strategies. NBS formulates spatial patch selection as a sequential decision-making problem.

At each step, the agent:

1. Observes the current image state.
2. Incorporates reconstruction and structural information.
3. Selects one of nine spatial actions.
4. Moves the sampling window.
5. Extracts a candidate patch.
6. Evaluates the selected region using the anomaly-detection pipeline.
7. Receives a reward.
8. Updates the sampling policy.

The overall process is designed to learn which image regions are more informative for anomaly detection.

---

## Research Website

A complete research-oriented presentation of the project is available through the GitHub Pages website.

**Website:**

https://amirhossein-khadivi.github.io/NBS/

The website contains:

- Project Overview
- Method
- Results
- Experiments
- Code
- Theoretical Analysis

---

## Method

The NBS framework combines several components:

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
Autoencoder Reconstruction
     │
     ▼
Reconstruction / Structural Information
     │
     ▼
State Construction
     │
     ▼
Neural Batch Sampler
     │
     ▼
Action Selection
     │
     ▼
Patch Extraction
     │
     ▼
Anomaly Predictor
     │
     ▼
Reward Computation
     │
     ▼
Policy Update
     │
     └──────────────► Next Sampling Step
```

The sampling policy therefore learns a sequential strategy for selecting informative image regions.

---

## Neural Batch Sampler

The main sampler is initialized with:

```python
NeuralBatchSampler(
    input_size=900,
    crop_size=128,
    patch_size=64,
    move_size=24,
    action_space=9
)
```

The main spatial parameters are:

| Parameter      |     Value |
| -------------- | --------: |
| Input size     |       900 |
| Crop size      | 128 × 128 |
| Patch size     |   64 × 64 |
| Movement size  | 24 pixels |
| Action space   |         9 |
| State channels |         6 |

---

## State Representation

The NBS agent receives a multi-channel state containing visual and reconstruction-related information.

The state incorporates information such as:

- RGB image information
- Reconstruction-error information
- Structural information
- Statistical information
- Previous error information
- Historical sampling information

This multi-source representation allows the policy to make spatial decisions using both image appearance and reconstruction behavior.

---

## Action Space

The sampler uses nine spatial actions:

| Action | Movement    |
| -----: | ----------- |
|      0 | No movement |
|      1 | Left        |
|      2 | Right       |
|      3 | Up          |
|      4 | Down        |
|      5 | Up-left     |
|      6 | Down-left   |
|      7 | Up-right    |
|      8 | Down-right  |

The movement step is 24 pixels and the selected patch has a spatial size of 64 × 64 pixels.

---

## Neural Sampler Architecture

The main convolutional architecture follows:

```text
Input: 6 channels
       │
       ▼
Conv 6 → 16
       │
       ▼
Conv 16 → 32
       │
       ▼
Conv 32 → 32
       │
       ▼
Conv 32 → 64
       │
       ▼
Conv 64 → 64
       │
       ▼
Flatten
       │
       ▼
FC 64×4×4 → 256
       │
       ▼
FC 256 → 9
```

The final layer produces one output for each available spatial action.

---

## Reconstruction and Anomaly Prediction

An autoencoder is used to reconstruct the input image and generate spatial reconstruction-error information.

A separate predictor network processes the available anomaly-related information.

The predictor is configured as:

```python
Predictor(input_channels=10)
```

with the following channel progression:

```text
10 → 32 → 16 → 8 → 4 → 1
```

Dilated convolutions are used to preserve spatial information while increasing the effective receptive field.

---

## Fused Anomaly Signal

The implementation combines multiple normalized signals:

\[
F
=
0.7\,MAE_z

- 0.1\,Var_z
- 0.2\,Grad_z
  \]

where:

- \(MAE_z\) represents normalized reconstruction error,
- \(Var_z\) represents normalized local variance,
- \(Grad_z\) represents normalized gradient information.

The resulting signal is normalized using min–max normalization.

---

## Reward Function

The sampling policy is optimized using a composite reward:

\[
R*{\text{total}}
=
\beta(R*{\text{clone}}+R\_{\text{cover}})

- (1-\beta)R\_{\text{pred}}
  \]

The reward combines sampling-related and prediction-related objectives.

This allows the policy to consider both the characteristics of the selected regions and their contribution to anomaly prediction.

---

## Experimental Evaluation

The repository contains several levels of experimental analysis.

### Overall Evaluation

The overall experiments include:

- Accuracy
- AUC
- F1-score
- BCE
- Weighted BCE

Results are available under:

```text
Resultes/Overall/
```

### Object-level Evaluation

Object-oriented scenarios are analyzed separately under:

```text
Resultes/Overall/Objects/
```

### Texture-level Evaluation

Texture-oriented scenarios are analyzed separately under:

```text
Resultes/Overall/Textures/
```

### Scenario-level Evaluation

Detailed analyses for individual industrial scenarios are available under:

```text
Resultes/Senarioes/
```

These experiments include training curves, loss–metric relationships, correlation analyses, and sample-level comparisons.

---

## Experimental Analyses

Scenario-level experiments include outputs such as:

```text
autoencoder_losses_over_training.png
prediction_losses_over_training.png
metrics_over_training.png

losspred_vs_acc.png
losspred_vs_auc.png
losspred_vs_f1.png
losspred_vs_metrics.png

heapmap_between_losspred_metrics.png

top_10_comparison.png
worst_10_comparison.png
```

These analyses provide complementary views of training dynamics, prediction behavior, metric relationships, and sample-level performance.

---

## Theoretical Analysis

The repository also contains a separate stochastic simulation for studying the effect of autoencoder resets on reward variability.

The simulation is located at:

```text
reset-ae-simulation/
```

The reset process is modeled using a Bernoulli indicator:

\[
I_t\sim\operatorname{Bernoulli}(p)
\]

and the autoencoder state is represented as:

\[
\theta_t^+
=
(1-I_t)\theta_t+I_tZ_t
\]

where \(Z_t\) represents a newly initialized state.

The reset-induced reward shock is defined as:

\[
D_t=R(Z_t)-R(\theta_t)
\]

leading to:

\[
R_t^{reset}
=
R(\theta_t)+I_tD_t
\]

Under the corresponding assumptions, the variance of the reset contribution is:

\[
\operatorname{Var}(I_tD_t)
=
p\sigma_D^2+p(1-p)\mu_D^2
\]

and in the zero-mean case:

\[
\operatorname{Var}(I_tD_t)
=
p\sigma_D^2
\]

The complete variance decomposition is:

\[
\operatorname{Var}(R_t^{reset})
=
\operatorname{Var}(R_t)

- \operatorname{Var}(I_tD_t)
- 2\operatorname{Cov}(R_t,I_tD_t)
  \]

The simulation is intended as a statistical abstraction of the reset mechanism rather than a replacement for the complete neural training process.

---

## Reset-AE Simulation Structure

```text
reset-ae-simulation/
├── experiment.py
├── main.py
│
└── src/
    ├── config.py
    ├── generate_trajectory.py
    ├── generate_reset.py
    ├── calculate_variance.py
    ├── calculate_theoretical.py
    ├── plot_results.py
    └── print_results.py
```

The simulation separates trajectory generation, stochastic reset generation, empirical variance calculation, theoretical analysis, plotting, and reporting.

---

## Main Repository Structure

```text
NBS/
│
├── Code/
│   └── Step6/
│       └── main.py
│
├── Diagrams/
│   ├── overview_flowchart.jpg
│   ├── detail_flowchart.jpg
│   ├── nbs_state_schematic.jpg
│   └── image3.jpg
│
├── Resultes/
│   ├── Overall/
│   └── Senarioes/
│
├── reset-ae-simulation/
│   ├── experiment.py
│   ├── main.py
│   └── src/
│       ├── config.py
│       ├── generate_trajectory.py
│       ├── generate_reset.py
│       ├── calculate_variance.py
│       ├── calculate_theoretical.py
│       ├── plot_results.py
│       └── print_results.py
│
├── docs/
│   ├── _config.yml
│   ├── _layouts/
│   │   └── default.html
│   ├── assets/
│   │   └── css/
│   │       └── style.css
│   ├── index.md
│   ├── method.md
│   ├── results.md
│   ├── experiments.md
│   ├── code.md
│   └── theory.md
│
├── README.md
└── LICENSE
```

---

## Main Implementation

The primary implementation is:

```text
Code/Step6/main.py
```

It contains the main NBS training pipeline, including:

- Image preprocessing
- Data augmentation
- Structural filtering
- Autoencoder reconstruction
- Reconstruction-error computation
- State construction
- Neural batch sampler
- Action selection
- Patch extraction
- Anomaly prediction
- Reward computation
- Policy optimization

---

## Dataset Configuration

The current implementation contains environment-specific paths for an MVTec bottle experiment:

```text
/content/content/MyDrive/MVTec/bottle/train/good
/content/content/MyDrive/MVTec/bottle/test/good
/content/content/MyDrive/MVTec/bottle/test
/content/content/MyDrive/MVTec/bottle/ground_truth
```

The autoencoder checkpoint is configured as:

```text
/content/content/MyDrive/MVTec/bottle/autoencoder.pth
```

These paths should be modified when running the code in another environment.

---

## Dependencies

The principal Python dependencies include:

```bash
pip install torch torchvision numpy scipy matplotlib scikit-learn pillow
```

A compatible CUDA-enabled PyTorch installation can be used for GPU execution.

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/amirhossein-khadivi/NBS.git
cd NBS
```

The main implementation is located at:

```text
Code/Step6/main.py
```

Before running the implementation, configure the dataset and checkpoint paths according to the local environment.

---

## Reproducibility

For reproducible experiments, the following factors should be kept consistent:

- Dataset version
- Dataset preprocessing
- Data augmentation
- Model architecture
- Training configuration
- Random seeds
- Autoencoder checkpoint
- Evaluation procedure
- Hardware environment

The repository separates implementation code, experimental outputs, diagrams, simulation code, and documentation to make the experimental workflow easier to inspect and reproduce.

---

## Documentation

The project website provides a structured description of the framework:

| Section     | Description                                  |
| ----------- | -------------------------------------------- |
| Overview    | Research motivation and project summary      |
| Method      | NBS architecture, state, actions, and reward |
| Results     | Aggregated and scenario-level results        |
| Experiments | Experimental organization and simulation     |
| Code        | Implementation and repository structure      |
| Theory      | Mathematical analysis of stochastic resets   |

---

## Citation

If you use this implementation, methodology, or experimental materials in your research, please cite the corresponding research work.

```bibtex
@misc{khadivi_nbs,
  title  = {Neural Batch Sampling with Reinforcement Learning for Semi-Supervised Industrial Anomaly Detection},
  author = {Amirhossein Khadivi},
  year   = {2026}
}
```

---

## License

See the `LICENSE` file for the licensing terms of this repository.

---

## Contact

For questions regarding the implementation, experiments, or research framework, please open an issue in the repository.

---

## Repository

**GitHub:**  
https://github.com/amirhossein-khadivi/NBS

**Project Website:**  
https://amirhossein-khadivi.github.io/NBS/
