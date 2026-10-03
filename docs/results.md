---
layout: default
title: Results
permalink: /results/
---

# Results

The experimental results of Neural Batch Sampling are organized at multiple levels, including overall performance, object categories, texture categories, and individual industrial scenarios.

---

## Overall Performance

The overall analysis provides a consolidated view of the experimental results across the evaluated scenarios.

### Accuracy

<div class="figure">

<img
src="{{ '/Resultes/Overall/ACC_Distribution_Across_overall.png' | relative_url }}"
alt="Accuracy distribution across overall experiments">

<div class="figure-caption">
Accuracy distribution across the overall experimental scenarios.
</div>

</div>

### AUC

<div class="figure">

<img
src="{{ '/Resultes/Overall/AUC_Distribution_Across_overall.png' | relative_url }}"
alt="AUC distribution across overall experiments">

<div class="figure-caption">
AUC distribution across the overall experimental scenarios.
</div>

</div>

### F1 Score

<div class="figure">

<img
src="{{ '/Resultes/Overall/F1_Distribution_Across_overall.png' | relative_url }}"
alt="F1 distribution across overall experiments">

<div class="figure-caption">
F1-score distribution across the overall experimental scenarios.
</div>

</div>

---

## Loss Analysis

In addition to evaluation metrics, the repository provides loss-based analyses for examining the behavior of the anomaly prediction component.

### BCE

<div class="figure">

<img
src="{{ '/Resultes/Overall/BCE_Across_overall.png' | relative_url }}"
alt="BCE across overall experiments">

<div class="figure-caption">
Binary cross-entropy distribution across the overall experiments.
</div>

</div>

### Weighted BCE

<div class="figure">

<img
src="{{ '/Resultes/Overall/Weighted_BCE_Across_overall.png' | relative_url }}"
alt="Weighted BCE across overall experiments">

<div class="figure-caption">
Weighted BCE distribution across the overall experiments.
</div>

</div>

---

## Object vs. Texture

The results are additionally organized according to the characteristics of the evaluated industrial scenarios.

This separation allows the behavior of the method to be examined across object-oriented and texture-oriented categories.

### BCE Comparison

<div class="figure">

<img
src="{{ '/Resultes/Overall/BCE_Across_obj_vs_texture.png' | relative_url }}"
alt="BCE comparison between object and texture categories">

<div class="figure-caption">
BCE comparison between object and texture categories.
</div>

</div>

### Weighted BCE Comparison

<div class="figure">

<img
src="{{ '/Resultes/Overall/Weighted_BCE_Across_obj_vs_texture.png' | relative_url }}"
alt="Weighted BCE comparison between object and texture categories">

<div class="figure-caption">
Weighted BCE comparison between object and texture categories.
</div>

</div>

---

## Object-level Results

Object-specific results are available in:

```text
Resultes/Overall/Objects/
```

### Accuracy

<div class="figure">

<img
src="{{ '/Resultes/Overall/Objects/ACC_Distribution_Across_Object.png' | relative_url }}"
alt="Accuracy distribution across object scenarios">

<div class="figure-caption">
Accuracy distribution across the evaluated object-oriented scenarios.
</div>

</div>

### AUC

<div class="figure">

<img
src="{{ '/Resultes/Overall/Objects/AUC_Distribution_Across_Object.png' | relative_url }}"
alt="AUC distribution across object scenarios">

<div class="figure-caption">
AUC distribution across the evaluated object-oriented scenarios.
</div>

</div>

### F1 Score

<div class="figure">

<img
src="{{ '/Resultes/Overall/Objects/f1_Distribution_Across_Object.png' | relative_url }}"
alt="F1 distribution across object scenarios">

<div class="figure-caption">
F1-score distribution across the evaluated object-oriented scenarios.
</div>

</div>

### BCE

<div class="figure">

<img
src="{{ '/Resultes/Overall/Objects/BCE_Across_object.png' | relative_url }}"
alt="BCE across object scenarios">

<div class="figure-caption">
Binary cross-entropy results across object-oriented scenarios.
</div>

</div>

### Weighted BCE

<div class="figure">

<img
src="{{ '/Resultes/Overall/Objects/Weighted_BCE_Across_object.png' | relative_url }}"
alt="Weighted BCE across object scenarios">

<div class="figure-caption">
Weighted BCE results across object-oriented scenarios.
</div>

</div>

### Multi-metric Performance

<div class="figure">

<img
src="{{ '/Resultes/Overall/Objects/Multi-metric_Performance_Across_object.png' | relative_url }}"
alt="Multi-metric performance across object scenarios">

<div class="figure-caption">
Joint comparison of the main evaluation metrics across object-oriented scenarios.
</div>

</div>

---

## Texture-level Results

Texture-specific results are available in:

```text
Resultes/Overall/Textures/
```

### Accuracy

<div class="figure">

<img
src="{{ '/Resultes/Overall/Textures/ACC_Distribution_Across_texture.png' | relative_url }}"
alt="Accuracy distribution across texture scenarios">

<div class="figure-caption">
Accuracy distribution across the evaluated texture-oriented scenarios.
</div>

</div>

### AUC

<div class="figure">

<img
src="{{ '/Resultes/Overall/Textures/AUC_Distribution_Across_texture.png' | relative_url }}"
alt="AUC distribution across texture scenarios">

<div class="figure-caption">
AUC distribution across the evaluated texture-oriented scenarios.
</div>

</div>

### F1 Score

<div class="figure">

<img
src="{{ '/Resultes/Overall/Textures/f1_Distribution_Across_texture.png' | relative_url }}"
alt="F1 distribution across texture scenarios">

<div class="figure-caption">
F1-score distribution across the evaluated texture-oriented scenarios.
</div>

</div>

### BCE

<div class="figure">

<img
src="{{ '/Resultes/Overall/Textures/BCE_Across_Texture.png' | relative_url }}"
alt="BCE across texture scenarios">

<div class="figure-caption">
Binary cross-entropy results across texture-oriented scenarios.
</div>

</div>

### Weighted BCE

<div class="figure">

<img
src="{{ '/Resultes/Overall/Textures/Weighted_BCE_Across_Texture.png' | relative_url }}"
alt="Weighted BCE across texture scenarios">

<div class="figure-caption">
Weighted BCE results across texture-oriented scenarios.
</div>

</div>

### Multi-metric Performance

<div class="figure">

<img
src="{{ '/Resultes/Overall/Textures/Multi-metric_Performance_Across_Texture.png' | relative_url }}"
alt="Multi-metric performance across texture scenarios">

<div class="figure-caption">
Joint comparison of the main evaluation metrics across texture-oriented scenarios.
</div>

</div>

---

## Scenario-level Results

Beyond aggregated object and texture analyses, the repository contains detailed results for individual industrial scenarios.

The scenario-level results are organized under:

```text
Resultes/Senarioes/
```

Each scenario directory contains detailed training and evaluation analyses.

Typical scenario-level outputs include:

- Autoencoder loss during training
- Prediction loss during training
- Accuracy, AUC, and F1-score trends
- Prediction-loss versus metric relationships
- Correlation heatmaps
- True prediction-loss versus evaluation metrics
- Top-10 sample comparisons
- Worst-10 sample comparisons

This organization allows the behavior of Neural Batch Sampling to be inspected at the individual scenario level rather than only through aggregated statistics.

---

## Scenario-specific Analysis

The scenario-level experiments provide a more detailed view of the relationship between the reconstruction process, anomaly prediction, and the final evaluation metrics.

### Training Dynamics

The repository records the evolution of the autoencoder and predictor losses throughout training. These curves provide information about the optimization dynamics of the two components involved in the anomaly-detection pipeline.

### Loss–Metric Relationships

Several plots examine the relationship between prediction loss and evaluation metrics such as:

- Accuracy
- AUC
- F1-score

These analyses help characterize how changes in the learned anomaly signal correspond to downstream performance.

### Sample-level Analysis

The repository also includes comparisons of the top-performing and lowest-performing samples:

```text
top_10_comparison.png
worst_10_comparison.png
```

These visualizations provide a sample-level perspective on the behavior of the learned sampling strategy.

---

## Result Organization

The complete result structure can be summarized as:

```text
Resultes/
├── Overall/
│   ├── ACC_Distribution_Across_overall.png
│   ├── AUC_Distribution_Across_overall.png
│   ├── F1_Distribution_Across_overall.png
│   ├── BCE_Across_overall.png
│   ├── Weighted_BCE_Across_overall.png
│   ├── BCE_Across_obj_vs_texture.png
│   ├── Weighted_BCE_Across_obj_vs_texture.png
│   │
│   ├── Objects/
│   │   ├── ACC_Distribution_Across_Object.png
│   │   ├── AUC_Distribution_Across_Object.png
│   │   ├── BCE_Across_object.png
│   │   ├── Weighted_BCE_Across_object.png
│   │   ├── f1_Distribution_Across_Object.png
│   │   └── Multi-metric_Performance_Across_object.png
│   │
│   └── Textures/
│       ├── ACC_Distribution_Across_texture.png
│       ├── AUC_Distribution_Across_texture.png
│       ├── BCE_Across_Texture.png
│       ├── Weighted_BCE_Across_Texture.png
│       ├── f1_Distribution_Across_texture.png
│       └── Multi-metric_Performance_Across_Texture.png
│
└── Senarioes/
    ├── Bottle/
    ├── Cable/
    ├── Capsule/
    ├── Carpet/
    ├── Grid/
    ├── Hazelnut/
    └── ...
```

---

## Summary

The Results section provides three complementary levels of analysis:

1. **Overall analysis** for consolidated performance across scenarios.
2. **Category-level analysis** separating object-oriented and texture-oriented scenarios.
3. **Scenario-level analysis** for detailed inspection of individual industrial cases.

Together, these result groups provide a structured view of the performance and behavior of the Neural Batch Sampling framework across different experimental settings.
