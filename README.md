# Accelerated Compositional Entropic Risk Minimization for Extreme Classification

Course project repository for **CSCE 726 / Optimization for ML**.  
This repo studies practical modifications of **SCENT** for extreme classification on the **TreeOfLife-10M subset** using the provided starter code and pre-extracted features.

## Model Download

We keep the model from the last epoch for every experiment. You can [download our model here](https://drive.google.com/file/d/11bWb_-FRdTiVW-QHUKEACDxXuT46zpXd/view?usp=sharing).

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

```text
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
