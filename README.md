# Neural Batch Sampling (NBS)

## Neural Batch Sampling with Reinforcement Learning for Semi-Supervised Industrial Anomaly Detection

Neural Batch Sampling (NBS) is a reinforcement-learning-based framework for intelligent spatial sampling in industrial image anomaly detection.

Instead of processing image regions uniformly, NBS learns a sequential sampling policy to identify informative image patches using visual, structural, and reconstruction-related information.

---

## Project Website

The complete research presentation, including the methodology, results, experiments, implementation details, and theoretical analysis, is available on the project website:

### [NBS Research Website](https://amirhossein-khadivi.github.io/NBS/)

---

## Framework Overview

<p align="center">
  <img src="Diagrams/overview_flowchart.jpg" alt="NBS Framework Overview" width="850">
</p>

NBS combines image reconstruction, sequential patch sampling, anomaly prediction, and reinforcement learning into an integrated sampling framework.

---

## Main Components

- **Neural Batch Sampler** — learns a sequential spatial sampling policy.
- **Autoencoder** — provides reconstruction-based information for state construction.
- **Anomaly Predictor** — estimates anomaly-related information from sampled patches.
- **Reinforcement Learning** — optimizes patch selection through a composite reward.

---

## Repository Structure

```text
NBS/
│
├── Code/                  # Main implementation
├── Diagrams/              # Framework and architecture diagrams
├── Resultes/              # Experimental results
├── reset-ae-simulation/   # Stochastic reset simulation
├── docs/                  # GitHub Pages website
├── README.md
└── LICENSE
```

---

## Code

The main implementation is available in:

```text
Code/Step6/main.py
```

Detailed implementation information is available in the [Code section](https://amirhossein-khadivi.github.io/NBS/code/) of the project website.

---

## Experiments

The repository contains overall, object-level, texture-level, and scenario-level experimental results, together with training-dynamics and stochastic reset analyses.

See the [Results](https://amirhossein-khadivi.github.io/NBS/results/) and [Experiments](https://amirhossein-khadivi.github.io/NBS/experiments/) pages for details.

---

## Theoretical Analysis

The repository includes a stochastic simulation for analyzing the effect of autoencoder resets on reward variability.

The complete mathematical formulation is provided in the [Theory](https://amirhossein-khadivi.github.io/NBS/theory/) section of the project website.

---

## Citation

---

## License

See the `LICENSE` file for the licensing terms of this repository.

---

## Links

- **Research Website:** https://amirhossein-khadivi.github.io/NBS/
- **GitHub Repository:** https://github.com/amirhossein-khadivi/NBS
