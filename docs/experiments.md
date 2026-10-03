---
layout: default
title: Experiments
permalink: /experiments/
---

# Experiments

The experimental study of Neural Batch Sampling is organized across multiple levels to evaluate the behavior of the proposed sampling framework under different industrial anomaly-detection scenarios.

The experiments are structured into overall, category-level, scenario-level, and simulation-based analyses.

---

## Experimental Organization

The experimental results are organized as follows:

```text
Experiments
│
├── Overall Experiments
│
├── Object-level Experiments
│
├── Texture-level Experiments
│
├── Scenario-level Experiments
│   ├── Training Dynamics
│   ├── Loss–Metric Relationships
│   ├── Sample-level Analysis
│   └── Metric Relationships
│
└── Reset-AE Simulation
    ├── Stochastic Reset Model
    ├── Empirical Analysis
    └── Theoretical Analysis
```

This organization separates the evaluation of the complete system from more detailed investigations of individual components and experimental hypotheses.

---

## Overall Experiments

The overall experiments aggregate the performance of Neural Batch Sampling across the evaluated industrial scenarios.

The main evaluation metrics include:

- Accuracy
- Area Under the ROC Curve (AUC)
- F1-score
- Binary Cross-Entropy (BCE)
- Weighted Binary Cross-Entropy

The corresponding visualizations are available in:

```text
Resultes/Overall/
```

The overall experiments are intended to provide a consolidated view of the behavior of the proposed sampling framework across the complete set of evaluated scenarios.

---

## Object-level Experiments

A separate analysis is performed for object-oriented industrial scenarios.

These experiments investigate the behavior of the framework on scenarios in which anomalies occur on distinguishable objects or object-like structures.

The corresponding results are stored in:

```text
Resultes/Overall/Objects/
```

The object-level analysis includes:

- Accuracy distributions
- AUC distributions
- F1-score distributions
- BCE
- Weighted BCE
- Multi-metric performance comparisons

These experiments allow the behavior of the sampling framework to be examined separately from texture-oriented scenarios.

---

## Texture-level Experiments

Texture-oriented scenarios are evaluated separately from object-oriented scenarios.

The corresponding results are stored in:

```text
Resultes/Overall/Textures/
```

The texture-level experiments include:

- Accuracy distributions
- AUC distributions
- F1-score distributions
- BCE
- Weighted BCE
- Multi-metric performance comparisons

This separation provides an additional level of analysis for understanding how the sampling strategy behaves across different types of industrial visual patterns.

---

## Scenario-level Experiments

In addition to aggregated analyses, detailed experiments are performed for individual industrial scenarios.

The scenario-level results are organized under:

```text
Resultes/Senarioes/
```

Representative scenarios include:

```text
Bottle
Cable
Capsule
Carpet
Grid
Hazelnut
...
```

Each scenario contains multiple analyses that examine different aspects of training and prediction behavior.

---

## Training Dynamics

The training-dynamics experiments track the evolution of the main losses during optimization.

Typical outputs include:

```text
autoencoder_losses_over_training.png
prediction_losses_over_training.png
metrics_over_training.png
```

### Autoencoder Loss

The autoencoder loss provides information about the reconstruction behavior during training.

Its evolution can be inspected to identify the general optimization trend and changes in reconstruction quality over the training process.

### Prediction Loss

The prediction loss describes the behavior of the anomaly prediction component throughout training.

Monitoring this loss provides an additional view of the learning process and its relationship to the final anomaly-detection metrics.

### Evaluation Metrics

The scenario-level experiments also record the evolution of evaluation metrics during training.

These results allow the relationship between optimization dynamics and downstream performance to be examined.

---

## Loss–Metric Relationships

The experiments additionally investigate relationships between prediction loss and the final evaluation metrics.

Typical outputs include:

```text
losspred_vs_acc.png
losspred_vs_auc.png
losspred_vs_f1.png
losspred_vs_metrics.png
```

These analyses examine whether changes in the prediction loss are associated with corresponding changes in Accuracy, AUC, and F1-score.

Rather than considering the final metrics independently, this analysis provides a more detailed view of the relationship between the learned anomaly signal and evaluation performance.

---

## Metric Relationship Analysis

Additional correlation and relationship analyses are available for individual scenarios.

Typical files include:

```text
heapmap_between_losspred_metrics.png
Relationship_between_truelossprediction_metrics.png
Relationship_between_truelossprediction_metrics(Separately).png
```

These visualizations provide complementary perspectives on the relationships between prediction behavior and evaluation metrics.

The separate relationship analysis can also be used to inspect metric behavior independently rather than only through an aggregated representation.

---

## Sample-level Analysis

The repository includes sample-level comparisons for selected scenarios.

Typical outputs include:

```text
top_10_comparison.png
worst_10_comparison.png
```

These experiments provide a qualitative view of samples associated with different prediction outcomes.

The purpose is to complement aggregate metrics with direct inspection of individual samples and their corresponding model behavior.

---

## Reset-AE Simulation

A separate simulation study investigates the effect of periodic autoencoder resets on reward variability.

The simulation is implemented under:

```text
reset-ae-simulation/
```

The simulation is designed as a simplified stochastic abstraction of the autoencoder-reset mechanism rather than as a complete neural-network training experiment.

Its purpose is to study the statistical behavior of reward trajectories under reset and no-reset conditions.

---

## Simulation Structure

The simulation contains the following main components:

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

The modules separate configuration, trajectory generation, stochastic reset generation, variance calculation, theoretical analysis, visualization, and result reporting.

---

## Simulation Configuration

The main simulation parameters are defined in:

```text
reset-ae-simulation/src/config.py
```

The configuration includes:

| Parameter                          |   Value |
| ---------------------------------- | ------: |
| Number of time steps \(T\)         |    1000 |
| Reset interval \(K\)               |      50 |
| Number of trajectories             |     100 |
| Initial AE loss                    |     1.0 |
| Minimum AE loss                    |    0.01 |
| Learning rate                      |   0.005 |
| AE noise standard deviation        |    0.01 |
| Predictor noise standard deviation |    0.01 |
| \(\beta\)                          |     0.8 |
| Stabilization window               |      50 |
| Stabilization epsilon              |    0.01 |
| Random seed                        |      42 |
| Reset probability \(p\)            | \(1/K\) |

The reset probability is selected to correspond to the average reset frequency associated with an interval of \(K\) updates.

---

## Stochastic Reset Model

The reset mechanism is modeled using a Bernoulli indicator:

\[
I_t \sim \mathrm{Bernoulli}(p)
\]

where \(I_t=1\) indicates that a reset occurs at time \(t\).

The parameter \(p\) represents the probability of a reset at a given time step.

The reset process can therefore be represented as:

\[
\theta_t^+
=
(1-I_t)\theta_t + I_t Z_t
\]

where:

- \(\theta_t\) is the current autoencoder state,
- \(Z_t\) is a newly initialized state,
- \(I_t\) determines whether the reset occurs.

This formulation provides a stochastic representation of sudden changes in the reconstruction process.

---

## Reward Shock

The simulation models the effect of a reset on the reward using a reward-shock term.

Let the reward difference caused by a reset be:

\[
D_t = R(Z_t)-R(\theta_t)
\]

The reward under the reset mechanism can then be represented as:

\[
R_t^{reset}
=
R(\theta_t)+I_tD_t
\]

Thus, the reset contribution is:

\[
I_tD_t
\]

This term represents the additional variability introduced by stochastic resets.

---

## Variance Analysis

The simulation compares reward variability between trajectories with and without resets.

For the reset contribution, the variance can be expressed as:

\[
\operatorname{Var}(I_tD_t)
=
p\sigma_D^2

- p(1-p)\mu_D^2
  \]

where:

\[
\mu_D=E[D_t]
\]

and

\[
\sigma_D^2=\operatorname{Var}(D_t)
\]

In the special case where the shock has zero mean,

\[
\mu_D=0
\]

the expression simplifies to:

\[
\operatorname{Var}(I_tD_t)
=
p\sigma_D^2
\]

This provides the theoretical basis for analyzing the additional variability associated with the reset mechanism.

---

## Total Reward Variance

For the complete reward expression,

\[
R_t^{reset}
=
R_t+I_tD_t
\]

the general variance decomposition is:

\[
\operatorname{Var}(R_t^{reset})
=
\operatorname{Var}(R_t)

- \operatorname{Var}(I_tD_t)
- 2\operatorname{Cov}(R_t,I_tD_t)
  \]

Therefore, the total reward variance depends not only on the variance of the reset shock but also on its covariance with the baseline reward process.

This distinction is important when interpreting the simulation results.

---

## Empirical Analysis

The simulation generates multiple trajectories for both reset and no-reset conditions.

The trajectories are used to estimate:

- Reward variance
- Mean reward behavior
- Stabilization behavior
- Differences between reset and no-reset conditions

The simulation uses a shared-noise setup for the paired comparison so that the two conditions can be compared under the same underlying random perturbations.

The resulting empirical statistics are then compared with the corresponding theoretical expressions.

---

## Theoretical Analysis

The theoretical analysis is implemented separately from the trajectory-generation process.

The simulation evaluates the empirical variance and compares it with the variance predicted by the stochastic reset model.

The analysis is based on:

\[
I_t\sim\mathrm{Bernoulli}(p)
\]

and

\[
D_t=R(Z_t)-R(\theta_t)
\]

with the reset contribution:

\[
I_tD_t
\]

The theoretical and empirical results provide two complementary views:

1. The empirical results describe the behavior observed in simulated trajectories.
2. The theoretical results describe the expected statistical contribution of the reset process under the stated assumptions.

---

## Important Interpretation

The reset simulation should be interpreted as a statistical abstraction of the reset mechanism.

It does not reproduce the full dynamics of training the neural autoencoder.

Instead, it isolates the stochastic component associated with sudden changes in the autoencoder state and examines how these changes can affect reward variability.

This distinction is important when relating the simulation to the complete Neural Batch Sampling implementation.

---

## Relation to the Main NBS Implementation

The main Neural Batch Sampling implementation is located under:

```text
Code/Step6/
```

with the primary training script:

```text
Code/Step6/main.py
```

The main implementation contains the neural sampler, autoencoder, predictor, state construction, action selection, reward computation, and training loop.

The reset simulation is maintained separately because it serves a theoretical and statistical investigation rather than representing the complete NBS training pipeline.

The two components therefore have different purposes:

| Component              | Purpose                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `Code/Step6/`          | Main Neural Batch Sampling implementation                  |
| `reset-ae-simulation/` | Statistical simulation of reset-induced reward variability |

---

## Experimental Summary

The complete experimental framework combines several complementary analyses:

### 1. Overall Evaluation

Aggregated performance across the evaluated industrial scenarios.

### 2. Category-level Evaluation

Separate analysis of object-oriented and texture-oriented scenarios.

### 3. Scenario-level Evaluation

Detailed analysis of individual industrial scenarios, including training dynamics, loss relationships, metric relationships, and sample-level comparisons.

### 4. Reset Simulation

A controlled stochastic simulation for analyzing the relationship between periodic resets and reward variability.

Together, these experiments provide both empirical evaluation of the NBS framework and a separate theoretical investigation of the stochastic behavior associated with autoencoder resets.
