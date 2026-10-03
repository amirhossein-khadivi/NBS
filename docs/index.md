---
layout: default
title: Overview
---

<section class="hero">

  <div class="hero-kicker">
    Research Project · Computer Vision · Reinforcement Learning
  </div>

  <h1>
    Neural Batch <span>Sampling</span>
  </h1>

  <p>
    A reinforcement-learning framework for intelligent patch
    selection in semi-supervised industrial anomaly detection.
    NBS learns where to inspect next instead of relying on
    uniformly sampled image regions.
  </p>

  <div class="button-row">

    <a class="button"
       href="{{ '/method/' | relative_url }}">
      Explore the Method
    </a>

    <a class="button secondary"
       href="https://github.com/amirhossein-khadivi/NBS"
       target="_blank"
       rel="noopener">
      View on GitHub ↗
    </a>

  </div>

</section>

<div class="figure">

<img
    src="{{ '/Diagrams/overview_flowchart.jpg' | relative_url }}"
    alt="NBS overview flowchart">

  <div class="figure-caption">
    Overview of the Neural Batch Sampling framework.
  </div>

</div>

<section class="section">

  <div class="section-header">

    <h2>Overview</h2>

    <p>
      NBS formulates image-region selection as a sequential
      decision-making problem.
    </p>

  </div>

  <div class="card-grid">

    <div class="card">

      <h3>Intelligent Sampling</h3>

      <p>
        The agent learns to move across an image and select
        informative patches rather than treating all regions
        equally.
      </p>

    </div>

    <div class="card">

      <h3>Multi-source State</h3>

      <p>
        The state combines visual information with anomaly,
        structural, and historical information.
      </p>

    </div>

    <div class="card">

      <h3>Reward-driven Selection</h3>

      <p>
        Sampling decisions are guided by structural information,
        region coverage, and prediction feedback.
      </p>

    </div>

  </div>

</section>

<section class="section">

  <div class="section-header">

    <h2>Research Map</h2>

    <p>
      Explore the project through the following sections.
    </p>

  </div>

  <div class="research-map">

    <a class="map-card"
       href="{{ '/method/' | relative_url }}">

      <h3>01 · Method</h3>

      <p>
        Architecture, state representation, action space,
        reward design, and the complete NBS pipeline.
      </p>

    </a>


    <a class="map-card"
       href="{{ '/results/' | relative_url }}">

      <h3>02 · Results</h3>

      <p>
        Overall, object-level, texture-level, and
        scenario-specific experimental visualizations.
      </p>

    </a>


    <a class="map-card"
       href="{{ '/experiments/' | relative_url }}">

      <h3>03 · Experiments</h3>

      <p>
        Experimental scenarios, training dynamics,
        and the reset simulation study.
      </p>

    </a>


    <a class="map-card"
       href="{{ '/code/' | relative_url }}">

      <h3>04 · Code</h3>

      <p>
        Implementation structure, model components,
        configuration, and reproducibility information.
      </p>

    </a>


    <a class="map-card"
       href="{{ '/theory/' | relative_url }}">

      <h3>05 · Theory</h3>

      <p>
        Mathematical formulation of the stochastic reset
        process and reward-variance analysis.
      </p>

    </a>


    <a class="map-card"
       href="https://github.com/amirhossein-khadivi/NBS"
       target="_blank"
       rel="noopener">

      <h3>06 · Repository</h3>

      <p>
        Source code, diagrams, results, and research artifacts
        are available in the GitHub repository.
      </p>

    </a>

  </div>

</section>
