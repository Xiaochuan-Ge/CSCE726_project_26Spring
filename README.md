# Accelerated Compositional Entropic Risk Minimization for Extreme Classification

Course project repository for **CSCE 726 / Optimization for ML**.  
This repo studies practical modifications of **SCENT** for extreme classification on the **TreeOfLife-10M subset** using the provided starter code and pre-extracted features.

## Model Download

We keep the model from the last epoch for every experiment. You can [download our model for experiments 1-5 here](https://drive.google.com/file/d/11bWb_-FRdTiVW-QHUKEACDxXuT46zpXd/view?usp=sharing), [Model 6 here](https://drive.google.com/file/d/1U_Y2QsRHPqXNVoJh-SWhmNoQQyXbFRfQ/view?usp=share_link)


## Project goal

The starter baseline is a linear classifier trained with the **SCENT** algorithm on a large extreme-classification problem.  
The goal of this project is to test whether several optimization and stabilization modifications can improve convergence and classification performance over the SCENT baseline.

In this repo, the experiments are organized into **six notebooks**:

- `exp_1_regularization.ipynb`
- `exp_2_storm_primal.ipynb`
- `exp_3_msvr_dual.ipynb`
- `exp_4_taxonomic_sampling.ipynb`
- `exp_5_final_selection.ipynb`
- `exp_6_final_selection.ipynb`

Outputs for each notebook are saved in separate output folders.

---

## Repository structure

    .
    ├── exp_1_regularization.ipynb
    ├── exp_2_storm_primal.ipynb
    ├── exp_3_msvr_dual.ipynb
    ├── exp_4_taxonomic_sampling.ipynb
    ├── exp_5_final_selection.ipynb
    ├── exp_6_final_selection.ipynb
    ├── outputs/                    # exp_1 outputs
    ├── outputs_exp2_storm/         # exp_2 outputs
    ├── outputs_exp3_msvr/          # exp_3 outputs
    ├── outputs_exp4_tis/           # exp_4 outputs
    ├── outputs_exp5_final/         # exp_5 outputs
    └── README.md

---

## Setup

### Data

The experiments use the **TreeOfLife-10M subset** with pre-extracted features from the provided course starter notebook.

Expected structure:

    ./features/treeoflife10m_subset/
        train/
            features.pt
            labels.pt
        val/
            features.pt
            labels.pt

A local `test/` split was not available during these notebook runs, so most conclusions in this repo are based on **validation results**.

### Environment

The notebooks were built from the provided starter notebook and the LibAUC-based SCENT training pipeline.

Typical setup:

1. Download the feature dataset.
2. Clone the required LibAUC branch used by the starter notebook.
3. Install the library.
4. Run the notebooks on GPU.

---

## Experiment summary

## Experiment 1 — `exp_1_regularization.ipynb`

### Purpose

This notebook implements the main **factorial ablation study** from the proposal.

### What was tested

Ten SCENT/SOX-based configurations:

- SOX baseline
- SCENT baseline
- SCENT + ℓ2 regularization
- SCENT + LA-CERM
- SCENT + warmup
- pairwise combinations
- unified combination
- post-hoc logit adjustment

### Main idea

This notebook focuses on the **stabilization/debiasing** side:

- ℓ2 regularization
- logit adjustment
- linear warmup

### Result

This was the most useful experiment block.

**Main finding:**  
`SCENT + warmup` was the strongest and most reliable improvement over the plain SCENT baseline.

Other findings:

- ℓ2 alone did not help enough
- LA-CERM alone was not the best variant
- the full "all tricks together" setting was not good in this implementation
- post-hoc logit adjustment could improve loss, but not necessarily accuracy

### Takeaway

**Warmup was the clearest positive result.**

---

## Experiment 2 — `exp_2_storm_primal.ipynb`

### Purpose

Test a **STORM-style recursive estimator** for the primal update.

### What was changed

The standard SCENT primal update was replaced with a STORM-inspired recursive update.

### Result

This branch was a **negative result**.

The STORM implementation ran, but compared with the SCENT reference it was substantially worse in:

- training cross-entropy
- validation loss
- validation accuracy

### Takeaway

**STORM was not included in the final method.**

---

## Experiment 3 — `exp_3_msvr_dual.ipynb`

### Purpose

Test an **MSVR-inspired dual tracking** modification.

### What was changed

An auxiliary estimate of the inner exponential quantity was maintained and used to modify the dual variable update.

### Result

This branch was a **non-winning result**.

The implementation ran and produced meaningful curves, but it did not beat the strongest reference from earlier experiments. In particular, it did not outperform the best warmup-based SCENT reference on validation accuracy.

### Takeaway

**The original MSVR dual-tracking variant was explored successfully, but it was not selected for the final method.**

---

## Experiment 4 — `exp_4_taxonomic_sampling.ipynb`

### Purpose

Test a **taxonomy-inspired importance sampling** strategy for hard negatives.

### What was changed

A GPU-friendly proxy for taxonomic importance sampling was implemented by reweighting sampled negative classes according to a hardness-based proposal.

### Result

This branch was a **negative result**.

The implementation itself ran correctly, but active TIS settings consistently hurt validation accuracy. Some variants reduced loss, but they still underperformed the SCENT references in top-1 accuracy.

### Takeaway

**The TIS proxy was excluded from the final method.**

---

## Experiment 5 — `exp_5_final_selection.ipynb`

### Purpose

Make a final selection among the methods that actually worked.

### What was tested

A small final comparison among:

- plain SCENT
- SCENT + warmup
- SCENT + warmup + post-hoc evaluation adjustment

### Result

This notebook confirmed the final choice.

**Main finding:**  
`SCENT + warmup` remained the best overall practical method.

The post-hoc evaluation adjustment could change validation loss, but it did not beat the warmup model in the metric that mattered most for final selection: **validation accuracy**.

### Takeaway

**Final selected method: SCENT + linear warmup.**

---

## Experiment 6 — `exp_6_final_selection.ipynb`

### Purpose

This notebook further tests a refined variant of the MSVR-inspired dual-tracking method from Experiment 3.

### What was tested

The goal was to evaluate whether a modified version of the MSVR dual-tracking idea, referred to here as **MSVR II**, could provide better practical performance than the earlier MSVR implementation.

The comparison focused on:

- plain SCENT
- SCENT + warmup
- the refined MSVR II variant

### Result

This experiment showed that the refined MSVR II variant can improve over vanilla SCENT in some settings.

However, based on the current validation results, MSVR II was not as consistently reliable as the warmup-based SCENT variant. Therefore, we view MSVR II as a promising but not final-selected optimization-side modification.

### Takeaway

**MSVR II showed partial improvement over vanilla SCENT, but SCENT + linear warmup remained the most reliable final choice.**

---

## Final conclusion

Across the six experiments, the most reliable and practically successful modification was:

- **SCENT + linear warmup**

The simpler stabilization-side improvement consistently gave the strongest validation-accuracy performance among the tested methods.

The heavier optimization-side modifications had mixed outcomes:

- the STORM primal update did not improve performance;
- the original MSVR dual-tracking variant did not beat the strongest warmup-based reference;
- the refined MSVR II variant showed some improvement over vanilla SCENT, but was not as consistently strong as SCENT + warmup;
- the taxonomy-inspired importance sampling proxy reduced loss in some cases but hurt validation accuracy.

Therefore, the final conclusion of this project is:

> A simple stabilization-side improvement, **linear warmup**, was the most robust and reliable modification in our experiments. The refined MSVR II variant showed some promise over vanilla SCENT, but the final selected method remains **SCENT + linear warmup**.

---

## High-level result summary

| Experiment | Focus | Outcome |
|---|---|---|
| `exp_1` | Main ablation study | **Positive** — warmup performed best |
| `exp_2` | STORM primal update | Negative |
| `exp_3` | Original MSVR dual tracking | Negative / not selected |
| `exp_4` | Taxonomy-inspired importance sampling proxy | Negative |
| `exp_5` | Final model selection | **Final winner: SCENT + warmup** |
| `exp_6` | Refined MSVR II variant | Partially positive — improved over vanilla SCENT in some settings, but not final-selected |

---

## How to use this repo

### Run order

Recommended order:

1. Run `exp_1_regularization.ipynb`
2. Run `exp_2_storm_primal.ipynb`
3. Run `exp_3_msvr_dual.ipynb`
4. Run `exp_4_taxonomic_sampling.ipynb`
5. Run `exp_5_final_selection.ipynb`
6. Run `exp_6_final_selection.ipynb`

### Re-running experiments

Each notebook writes JSONL logs and checkpoint files.  
Before rerunning the same experiment names, it is recommended to clear old outputs or ensure the notebook removes old metric files first.

### Plotting

Each notebook contains plotting utilities for:

- training cross-entropy
- validation loss
- validation accuracy

---

## Limitations

- The reported conclusions here are primarily based on **validation results**, because a local test split was not available during development.
- The taxonomic sampling branch used a **proxy implementation**, not a true taxonomy file / hierarchy-aware sampler.
- The optimization-side methods were explored honestly, but the final winner came from the simpler stabilization branch.
- The refined MSVR II variant showed partial improvement over vanilla SCENT, but it requires further tuning and validation before it can be considered a robust final method.

---

## Suggested citation / context

This project is based on the course starter materials and proposal built around:

- SCENT for compositional entropic risk minimization
- TreeOfLife-10M subset for extreme classification
- practical ablations on stabilization and optimization modifications

---

## Final recommendation

If you only want to reproduce the final best result from this repo, run:

- `exp_5_final_selection.ipynb`

and use:

- **`exp5_cfg02_scent_warmup_final`**

as the final selected model.

If you want to inspect the refined MSVR II variant, run:

- `exp_6_final_selection.ipynb`

This variant showed some improvement over vanilla SCENT, but it was not chosen as the final recommended method.
