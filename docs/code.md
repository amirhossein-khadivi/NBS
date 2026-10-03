---
layout: default
title: Code
permalink: /code/
---

# Code

The Neural Batch Sampling (NBS) implementation is provided in the repository together with the experimental results, diagrams, and the stochastic reset simulation.

The implementation is organized to separate the main NBS training pipeline from the theoretical simulation experiments.

---

## Repository

The complete source code and experimental materials are available in the GitHub repository:

<div class="button-row">

<a
class="button primary"
href="https://github.com/amirhossein-khadivi/NBS"
target="_blank">
View Repository
</a>

</div>

---

## Repository Structure

The main repository is organized as follows:

```text
NBS/
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
│   ├── assets/
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

The primary NBS implementation is located at:

```text
Code/Step6/main.py
```

This script contains the main components required for the reinforcement-learning-based neural batch sampling process.

The implementation includes:

- Dataset loading
- Image preprocessing
- Data augmentation
- Sobel-based structural filtering
- Autoencoder reconstruction
- Reconstruction-error computation
- State construction
- Neural batch sampler
- Action selection
- Patch extraction
- Predictor network
- Reward computation
- Policy optimization

---

## Neural Batch Sampler

The main sampler is initialized as:

```python
NeuralBatchSampler(
    input_size=900,
    crop_size=128,
    patch_size=64,
    move_size=24,
    action_space=9
)
```

The sampler operates on a multi-channel state representation.

The main spatial parameters are:

| Parameter         |     Value |
| ----------------- | --------: |
| Input size        |       900 |
| Crop size         | 128 × 128 |
| Patch size        |   64 × 64 |
| Movement size     | 24 pixels |
| Number of actions |         9 |
| State channels    |         6 |

---

## Neural Sampler Architecture

The convolutional feature extractor uses the following channel progression:

```text
Input
  │
  ▼
Conv: 6 → 16
  │
  ▼
Conv: 16 → 32
  │
  ▼
Conv: 32 → 32
  │
  ▼
Conv: 32 → 64
  │
  ▼
Conv: 64 → 64
  │
  ▼
Flatten
  │
  ▼
FC: 64×4×4 → 256
  │
  ▼
FC: 256 → 9
```

Batch normalization and nonlinear transformations are used throughout the feature-extraction process.

The final layer produces one output for each available spatial action.

---

## Action Space

The sampler uses a nine-action spatial movement space.

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

The movement size is:

\[
m=24
\]

pixels.

The selected patch is extracted from the current crop using a patch size of:

\[
64\times64
\]

pixels.

---

## State Representation

The NBS agent receives a multi-channel state containing visual and reconstruction-related information.

The state integrates:

- RGB image information
- Structural or reconstruction-error information
- Statistical information
- Previous reconstruction error
- Historical information

The resulting state is used as the input to the neural batch sampler.

The state representation is designed to provide the agent with both the visual context of the current region and information related to previously observed reconstruction behavior.

---

## Image Preprocessing

The main image preprocessing pipeline includes resizing and tensor conversion.

The images are resized to:

```text
256 × 256
```

and converted into PyTorch tensors.

Data augmentation includes:

- Random horizontal flipping
- Random vertical flipping
- Random rotation up to approximately ±30°
- Random translation
- Random shearing

These transformations increase variation in the training samples.

---

## Structural Filtering

A Sobel-based filtering operation is used to obtain structural information from the input image.

The filtering procedure includes:

1. Conversion to grayscale
2. Gaussian smoothing
3. Sobel filtering in the horizontal direction
4. Sobel filtering in the vertical direction

The horizontal and vertical responses are used to capture local structural information.

This information is incorporated into the state representation used by the NBS agent.

---

## Autoencoder

The implementation includes an autoencoder for reconstruction-error generation.

The autoencoder contains convolutional and transposed-convolutional components together with a skip connection.

A representative feature flow is:

```text
Input
  │
  ▼
Encoder
  │
  ▼
Bottleneck
  │
  ▼
Decoder
  │
  ├───────────────┐
  │               │
  ▼               │
Decoder Features  │
  │               │
  └──── Skip ─────┘
          │
          ▼
      Reconstruction
```

The reconstruction is compared with the input image to generate spatial error information.

---

## Reconstruction Error

The implementation computes a patch-wise reconstruction error between the original input and the autoencoder reconstruction.

The resulting error map is used as part of the information available to the sampling framework.

The error map provides a spatial representation of reconstruction quality and can therefore highlight regions whose appearance differs from what is reconstructed by the autoencoder.

---

## Predictor Network

The anomaly predictor is implemented as:

```python
Predictor(input_channels=10)
```

The predictor uses a sequence of dilated convolutional layers.

The channel progression is:

```text
10
 ↓
32
 ↓
16
 ↓
8
 ↓
4
 ↓
1
```

A sigmoid output is used to produce the final prediction map.

The predictor therefore transforms the combined input information into a spatial anomaly-related prediction.

---

## Fused Anomaly Signal

The implementation combines several normalized signals into a fused representation.

The current formulation is:

\[
F
=
0.7\,MAE_z

- 0.1\,Var_z
- 0.2\,Grad_z
  \]

where:

- \(MAE_z\) represents the normalized reconstruction-error signal,
- \(Var_z\) represents normalized local variance information,
- \(Grad_z\) represents normalized gradient information.

The resulting signal is subsequently normalized using min–max normalization.

This fused representation provides the predictor with complementary information about reconstruction error, local variation, and image structure.

---

## Reward Function

The reinforcement-learning reward is composed of several terms.

The current formulation is:

\[
R*{\text{total}}
=
\beta(R*{\text{clone}}+R\_{\text{cover}})

- (1-\beta)R\_{\text{pred}}
  \]

where:

- \(R\_{\text{clone}}\) represents the cloning-related reward component,
- \(R\_{\text{cover}}\) represents the coverage-related reward component,
- \(R\_{\text{pred}}\) represents the prediction-related reward component,
- \(\beta\) controls the relative contribution of the first two components.

The resulting reward is used to optimize the sampling policy.

---

## Training Process

The main training process follows the general sequence:

```text
Input Image
    │
    ▼
Preprocessing
    │
    ▼
Autoencoder Reconstruction
    │
    ▼
Reconstruction Error
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
Predictor
    │
    ▼
Reward Computation
    │
    ▼
Policy Update
    │
    └──────────────► Next Sampling Step
```

This process allows the sampling policy to progressively select informative image regions according to the reward signal.

---

## Dataset Configuration

The current implementation contains dataset paths configured for an MVTec bottle experiment.

The paths in the current script include:

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

These paths are environment-specific and may need to be changed when running the code outside the original execution environment.

---

## Dependencies

The main implementation relies on common Python scientific-computing and deep-learning libraries.

The principal dependencies include:

```bash
pip install torch torchvision numpy scipy matplotlib scikit-learn pillow
```

A compatible CUDA-enabled PyTorch installation can be used when GPU acceleration is available.

---

## Running the Code

After cloning the repository:

```bash
git clone https://github.com/amirhossein-khadivi/NBS.git
cd NBS
```

The main implementation can be found at:

```text
Code/Step6/main.py
```

Before execution, dataset paths and checkpoint paths should be adjusted to match the local environment.

---

## Computational Device

The implementation automatically selects CUDA when an available GPU is detected.

The general device-selection logic is:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

This allows the same implementation to operate on either GPU or CPU environments.

---

## Experimental Outputs

The generated experimental outputs are stored separately from the source code.

The main result directories are:

```text
Resultes/Overall/
Resultes/Senarioes/
```

The results include:

- Accuracy distributions
- AUC distributions
- F1-score distributions
- BCE
- Weighted BCE
- Training curves
- Loss–metric relationships
- Sample-level comparisons
- Scenario-specific analyses

Detailed descriptions of these experiments are provided in the [Results]({{ '/results/' | relative_url }}) and [Experiments]({{ '/experiments/' | relative_url }}) sections.

---

## Reset-AE Simulation Code

The stochastic reset simulation is implemented independently from the main NBS training code.

Its source files are located at:

```text
reset-ae-simulation/
```

The simulation contains separate modules for:

- Configuration
- Trajectory generation
- Reset generation
- Variance calculation
- Theoretical calculation
- Plot generation
- Result reporting

The theoretical motivation and mathematical formulation are described in the [Theory]({{ '/theory/' | relative_url }}) section.

---

## Reproducibility

For reproducibility, the repository separates the main implementation, experimental outputs, diagrams, and simulation code.

When reproducing the experiments, the following factors should be kept consistent:

- Dataset version
- Dataset paths
- Image preprocessing
- Data augmentation
- Model architecture
- Training configuration
- Random seeds
- Hardware environment
- Autoencoder checkpoint
- Evaluation procedure

Because the current implementation contains environment-specific paths, these paths should be configured before running the code in a new environment.

---

## Implementation Notes

The current repository represents the experimental implementation associated with the NBS research framework.

In particular, the code and the theoretical simulation should be considered as two related but distinct components:

| Component              | Role                                                |
| ---------------------- | --------------------------------------------------- |
| `Code/Step6/main.py`   | Main NBS training and sampling implementation       |
| `Resultes/`            | Experimental results and visualizations             |
| `Diagrams/`            | Method and architecture diagrams                    |
| `reset-ae-simulation/` | Statistical simulation of reset-induced variability |
| `docs/`                | Research project documentation                      |

The separation makes it possible to inspect the implementation, experimental evidence, and theoretical analysis independently.

---

## Repository Access

The complete implementation and experimental materials are available on GitHub.

<div class="button-row">

<a
class="button primary"
href="https://github.com/amirhossein-khadivi/NBS"
target="_blank">
Open NBS Repository
</a>

</div>
