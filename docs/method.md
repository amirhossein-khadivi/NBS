---
layout: default
title: Method
permalink: /method/
---

# Method

## Neural Batch Sampling

Neural Batch Sampling formulates informative patch selection
as a sequential decision-making problem.

Instead of sampling image regions independently, the agent
observes the current state and selects one of nine spatial
actions to move the sampling window.

<div class="figure">

<img
src="{{ '/Diagrams/detail_flowchart.jpg' | relative_url }}"
alt="Detailed NBS workflow">

<div class="figure-caption">
Detailed workflow of Neural Batch Sampling.
</div>

</div>

## Architecture

The current NBS sampler uses a convolutional neural network
to map the six-channel state representation to a probability
distribution over nine actions.

| Component              | Configuration              |
| ---------------------- | -------------------------- |
| Input size             | 900                        |
| Crop size              | 128 × 128                  |
| Patch size             | 64 × 64                    |
| Movement size          | 24 pixels                  |
| Input channels         | 6                          |
| Action space           | 9                          |
| Convolutional channels | 6 → 16 → 32 → 32 → 64 → 64 |
| Fully connected layers | 1024 → 256 → 9             |
| Normalization          | Batch Normalization        |
| Output                 | Softmax over actions       |

## State Representation

The agent receives a multi-channel state containing visual
information together with information derived from anomaly
reconstruction, structural characteristics, and previous
sampling history.

<div class="figure">

<img
src="{{ '/Diagrams/nbs_state_schematic.jpg' | relative_url }}"
alt="NBS state representation">

<div class="figure-caption">
Schematic representation of the NBS state.
</div>

</div>

## Action Space

The agent selects one of nine movements.

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

Each movement changes the location of the sampling window by
the predefined movement size.

## Reward Design

The total reward is formulated as

$$
R_{\mathrm{total}}
=
\beta
\left(
R_{\mathrm{clone}} + R_{\mathrm{cover}}
\right)
+
(1-\beta)R_{\mathrm{pred}}.
$$

The formulation combines three complementary signals:

- **Structural information:** encourages attention to
  informative image structures.
- **Coverage:** rewards exploration of previously covered
  regions.
- **Prediction feedback:** incorporates feedback from the
  anomaly prediction component.

## Fused Information

The anomaly/structural signal is constructed using normalized
reconstruction and structural statistics:

$$
F =
0.7\,MAE_z
+
0.1\,Var_z
+
0.2\,Grad_z.
$$

The resulting signal is subsequently normalized using
min-max normalization before being incorporated into the
sampling process.
