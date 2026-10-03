# Neural Batch Sampling (NBS)

## Neural Batch Sampling with Reinforcement Learning for Semi-Supervised Industrial Anomaly Detection

Neural Batch Sampling (NBS) is a reinforcement-learning-based framework for intelligent spatial sampling in industrial image anomaly detection.

Instead of processing image regions uniformly, NBS learns a sequential sampling policy that navigates through an image and selects informative patches using visual, structural, and reconstruction-related information.

---

## Project Website

The complete research presentation, including the methodology, experimental results, theoretical analysis, and implementation details, is available on the project website:

### [NBS Research Website](https://amirhossein-khadivi.github.io/NBS/)

The website includes:

- **Overview** — Project motivation and framework summary
- **Method** — NBS architecture, state representation, action space, and reward
- **Results** — Overall, object-level, texture-level, and scenario-level results
- **Experiments** — Experimental organization and reset simulation
- **Code** — Implementation details and repository structure
- **Theory** — Mathematical analysis of stochastic autoencoder resets

---

## Framework Overview

The general NBS pipeline is:

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
Anomaly Prediction
     │
     ▼
Reward
     │
     ▼
Policy Update
     │
     └──────────────► Next Sampling Step
```

The sampling agent learns which spatial regions are informative for anomaly detection through sequential interaction with the image and reward-driven policy optimization.

---

## Main Components

The current implementation contains:

- Neural Batch Sampler
- Autoencoder-based reconstruction
- Reconstruction-error analysis
- Structural image information
- Anomaly prediction network
- Reinforcement-learning-based spatial sampling
- Composite reward formulation
- Scenario-level experimental analysis
- Reset-AE stochastic simulation

Detailed descriptions of these components are provided on the [project website](https://amirhossein-khadivi.github.io/NBS/).

---

## Repository Structure

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

## Code

The main implementation is located at:

```text
Code/Step6/main.py
```

The repository also contains the separate stochastic reset simulation under:

```text
reset-ae-simulation/
```

For implementation details and instructions, see the [Code section of the project website](https://amirhossein-khadivi.github.io/NBS/code/).

---

## Experiments

Experimental results are organized at several levels:

- Overall performance
- Object-oriented scenarios
- Texture-oriented scenarios
- Individual industrial scenarios
- Training dynamics
- Loss–metric relationships
- Sample-level analysis
- Reset-AE simulation

Detailed figures and analyses are available in the [Results](https://amirhossein-khadivi.github.io/NBS/results/) and [Experiments](https://amirhossein-khadivi.github.io/NBS/experiments/) sections of the website.

---

## Theoretical Analysis

A separate stochastic simulation investigates the effect of autoencoder resets on reward variability.

The theoretical formulation models the reset process as:

\[
\theta_t^+
=
(1-I_t)\theta_t+I_tZ_t
\]

with

\[
I_t\sim\operatorname{Bernoulli}(p)
\]

The resulting reward-shock formulation and variance analysis are described in detail on the [Theory page](https://amirhossein-khadivi.github.io/NBS/theory/).

---

## Citation

If you use this implementation or research framework, please cite the corresponding work:

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

## Links

- **Research Website:** https://amirhossein-khadivi.github.io/NBS/
- **GitHub Repository:** https://github.com/amirhossein-khadivi/NBS