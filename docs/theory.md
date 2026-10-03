---
layout: default
title: Theory
permalink: /theory/
---

# Theoretical Analysis

This section presents the theoretical motivation behind the analysis of periodic autoencoder resets and their effect on reward variability in the Neural Batch Sampling framework.

The main objective is to formulate the reset mechanism as a stochastic process and characterize the additional variance introduced into the reward signal.

---

## Motivation

In the original training mechanism, the autoencoder can be periodically reinitialized after a fixed number of updates.

The purpose of this mechanism is to introduce diversity into the reconstruction-error maps used by the sampling agent.

However, a complete reinitialization can produce a sudden change in the reconstruction behavior of the autoencoder.

Since the reconstruction error contributes to the state and reward computation of the Neural Batch Sampling agent, such changes can propagate to the reward signal.

The theoretical analysis therefore investigates the reset mechanism from a stochastic perspective.

---

## Stochastic Representation of the Reset

Let \(\theta_t\) denote the autoencoder parameters at time \(t\).

A reset replaces the current parameters with a newly initialized parameter state \(Z_t\).

The reset mechanism can be represented as:

\[
\theta_t^+
=
(1-I_t)\theta_t+I_tZ_t
\]

where

\[
I_t\sim\operatorname{Bernoulli}(p)
\]

and:

- \(I_t=1\) indicates that a reset occurs.
- \(I_t=0\) indicates that training continues without a reset.
- \(p\) is the probability of a reset at a given update.
- \(Z_t\) represents a newly initialized autoencoder state.

For a periodic reset with interval \(K\), the corresponding average reset rate is:

\[
p=\frac{1}{K}
\]

This formulation allows the periodic mechanism to be analyzed through an equivalent average stochastic reset rate.

---

## Assumptions

The theoretical analysis is based on the following assumptions.

### Assumption 1 — Independent Initialization

The newly initialized state \(Z_t\) is assumed to be independent of the current autoencoder state and its previous trajectory.

Formally,

\[
Z_t\perp\!\!\!\perp\theta_t
\]

This represents the effect of a fresh random initialization.

---

### Assumption 2 — Non-zero Reset Shock Variance

Define the reward difference caused by a reset as:

\[
D_t=R(Z_t)-R(\theta_t)
\]

The analysis assumes that:

\[
\sigma_D^2
=
\operatorname{Var}(D_t)>0
\]

Thus, a reset produces a non-degenerate stochastic change in the reward.

---

### Assumption 3 — Finite Variance

The relevant reward and reset-shock variables are assumed to have finite variance.

In particular,

\[
\operatorname{Var}(R_t)<\infty
\]

and

\[
\operatorname{Var}(D_t)<\infty
\]

This ensures that the variance expressions used in the analysis are well-defined.

---

## Lemma 1 — Stochastic Reset Process

Consider the parameter update:

\[
\theta_t^+
=
(1-I_t)\theta_t+I_tZ_t
\]

with

\[
I_t\sim\operatorname{Bernoulli}(p)
\]

Then:

- when \(I_t=0\),

\[
\theta_t^+=\theta_t
\]

- when \(I_t=1\),

\[
\theta_t^+=Z_t
\]

Therefore, the reset mechanism can be interpreted as a stochastic jump process in parameter space.

The expected reset indicator is:

\[
E[I_t]=p
\]

and consequently the expected frequency of reset events is controlled by \(p\).

For a periodic reset interval \(K\), the corresponding average rate is:

\[
p=\frac{1}{K}
\]

---

## Lemma 2 — Reward Decomposition

Let \(R(\theta_t)\) denote the reward associated with the current autoencoder state.

After introducing the reset indicator, the reward can be written as:

\[
R_t^{reset}
=
R(\theta_t)+I_tD_t
\]

where:

\[
D_t
=
R(Z_t)-R(\theta_t)
\]

The term

\[
I_tD_t
\]

is the stochastic contribution introduced by the reset.

Therefore, the reset mechanism decomposes the reward into two components:

\[
\boxed{
R*t^{reset}
=
\underbrace{R(\theta_t)}*{\text{baseline reward}}

- \underbrace{I*tD_t}*{\text{reset shock}}
  }
  \]

This decomposition is the central representation used in the variance analysis.

---

## Lemma 3 — Variance of the Reset Shock

Let

\[
I_t\sim\operatorname{Bernoulli}(p)
\]

and define

\[
D_t=R(Z_t)-R(\theta_t)
\]

Assuming the reset indicator is independent of the shock variable, the variance of the reset contribution is:

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

### Proof

Since

\[
I_t^2=I_t
\]

for a Bernoulli random variable,

\[
E[I_tD_t]
=
E[I_t]E[D_t]
=
p\mu_D
\]

Also,

\[
E[(I_tD_t)^2]
=
E[I_tD_t^2]
=
pE[D_t^2]
\]

Using

\[
E[D_t^2]
=
\sigma_D^2+\mu_D^2
\]

we obtain:

\[
E[(I_tD_t)^2]
=
p(\sigma_D^2+\mu_D^2)
\]

Therefore,

\[
\begin{aligned}
\operatorname{Var}(I_tD_t)
&=
E[(I_tD_t)^2]

- E[I_tD_t]^2
  \\
  &=
  p(\sigma_D^2+\mu_D^2)
- p^2\mu_D^2
  \\
  &=
  p\sigma_D^2

* p(1-p)\mu_D^2
  \end{aligned}
  \]

Hence,

\[
\boxed{
\operatorname{Var}(I_tD_t)
=
p\sigma_D^2

- p(1-p)\mu_D^2
  }
  \]

  ***

## Zero-Mean Reset Shock

If the reset shock has zero mean,

\[
\mu_D=0
\]

then the previous expression simplifies to:

\[
\boxed{
\operatorname{Var}(I_tD_t)
=
p\sigma_D^2
}
\]

This special case isolates the contribution of the stochastic variability of the reset itself.

Since

\[
p>0
\]

and

\[
\sigma_D^2>0
\]

we obtain:

\[
\operatorname{Var}(I_tD_t)>0
\]

Thus, under these assumptions, the reset process introduces a strictly positive variance component into the reward signal.

---

## Total Reward Variance

The complete reward is:

\[
R_t^{reset}
=
R_t+I_tD_t
\]

Therefore, its variance is:

\[
\boxed{
\operatorname{Var}(R_t^{reset})
=
\operatorname{Var}(R_t)

- \operatorname{Var}(I_tD_t)
- 2\operatorname{Cov}(R_t,I_tD_t)
  }
  \]

Substituting the reset-shock variance gives:

\[
\begin{aligned}
\operatorname{Var}(R_t^{reset})
=
&
\operatorname{Var}(R_t)

- p\sigma_D^2
- p(1-p)\mu_D^2
  \\
  &
- 2\operatorname{Cov}(R_t,I_tD_t)
  \end{aligned}
  \]

This is the general variance decomposition.

---

## Important Covariance Term

The covariance term

\[
2\operatorname{Cov}(R_t,I_tD_t)
\]

should not be omitted without an additional assumption.

In particular, the existence of a positive reset-shock variance alone does not mathematically imply that the total reward variance must always increase.

If an additional assumption gives:

\[
\operatorname{Cov}(R_t,I_tD_t)=0
\]

then:

\[
\operatorname{Var}(R_t^{reset})
=
\operatorname{Var}(R_t)

- p\sigma_D^2
- p(1-p)\mu_D^2
  \]

and therefore:

\[
\operatorname{Var}(R_t^{reset})
\geq
\operatorname{Var}(R_t)
\]

because:

\[
p\sigma_D^2\geq0
\]

and

\[
p(1-p)\mu_D^2\geq0
\]

Under the zero-mean condition \(\mu_D=0\), this further reduces to:

\[
\boxed{
\operatorname{Var}(R_t^{reset})
=
\operatorname{Var}(R_t)

- p\sigma_D^2
  }
  \]

  ***

## Interpretation

The theoretical formulation separates the reward variability into:

1. The natural variability of the baseline training process.
2. The variability introduced by stochastic reset events.
3. The covariance between the baseline reward and the reset-induced shock.

The reset mechanism therefore acts as an additional stochastic source in the reward process.

The magnitude of its direct variance contribution is controlled by:

\[
p\sigma_D^2
\]

in the zero-mean case.

Consequently, increasing either the frequency of reset events or the variability of their reward effect increases the direct variance contribution of the reset component.

---

## Periodic Reset vs. Controlled Diversity

The theoretical analysis motivates investigating alternatives to complete periodic reinitialization.

A complete reset changes the entire autoencoder state:

\[
\theta_t\rightarrow Z_t
\]

and can consequently produce a large change in the reconstruction-error map.

An alternative mechanism can introduce controlled perturbations into the representation without replacing the complete model state.

In the thesis formulation, this motivation is associated with replacing periodic autoencoder reinitialization by random deletion in the bottleneck representation.

The goal is to preserve controlled diversity in reconstruction-error maps while avoiding complete parameter reinitialization.

---

## Relation to Neural Batch Sampling

The Neural Batch Sampling framework uses reconstruction-related information as part of the state available to the reinforcement-learning agent.

The state contains image information together with structural and reconstruction-error information.

Changes in the autoencoder therefore affect the information observed by the sampling policy.

A sudden reset can modify the reconstruction-error map and consequently alter the state and reward experienced by the agent.

The stochastic formulation presented above provides a mathematical abstraction for analyzing this effect.

---

## Theoretical Summary

The main theoretical relationships are:

### Reset Process

\[
\theta_t^+
=
(1-I_t)\theta_t+I_tZ_t
\]

with

\[
I_t\sim\operatorname{Bernoulli}(p)
\]

### Reward Shock

\[
D_t=R(Z_t)-R(\theta_t)
\]

### Reward Decomposition

\[
R_t^{reset}
=
R(\theta_t)+I_tD_t
\]

### Reset-shock Variance

\[
\operatorname{Var}(I_tD_t)
=
p\sigma_D^2+p(1-p)\mu_D^2
\]

### Zero-mean Case

\[
\operatorname{Var}(I_tD_t)
=
p\sigma_D^2
\]

### General Total Variance

\[
\operatorname{Var}(R_t^{reset})
=
\operatorname{Var}(R_t)

- \operatorname{Var}(I_tD_t)
- 2\operatorname{Cov}(R_t,I_tD_t)
  \]

These equations provide the theoretical basis for the reset simulation presented in the Experiments section.
