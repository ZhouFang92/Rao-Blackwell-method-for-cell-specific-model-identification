# Rao-Blackwell-method-for-cell-specific-model-identification

## Installation (CRN_Simulation_Inference)
```bash
pip install CRN_Simulation_Inference@git+https://github.com/ZhouFang92/Rao-Blackwell-method-for-cell-specific-model-identification.git
```

## Required packages: 
igraph, sklearn

## Packages in this file
CRN_Simulation: a package for defining chemical reaction networks (CRNs), simulating the associated continuous-time Markov chain, and solving the chemical master equation by the finite state projection method

CRN_CountinuousTimeFiltering: a package for solving the filtering equation for CRNs under the setting of noise-free continuous-time observations. 
More details about this filtering problem are given in the literature: 
<br />
_D'Ambrosio, Elena Sofia, Zhou Fang, Ankit Gupta, and Mustafa Khammash. "Filtered finite state projection method for the analysis and estimation of stochastic biochemical reaction networks." bioRxiv (2022): 2022-10._ <br />
This package solves the filtering equation by applying the Krylov subspace method to the unnormalized filtering equation.

RB_method_for_model_identification: 
This package applies a Rao-Blackwell particle filter to cell-specific model identification. The method is introduced in this paper:<br />
_Fang, Zhou, Ankit Gupta, and Mustafa Khammash. "A divide-and-conquer approach for analyzing high-dimensional reaction network systems." bioRxiv 2023._

## Usage
The files MI_example_4.py, MI_example_4_two_gene_state.py, real_data_3_gene_state_model_larger_space.py, and real_data_2_gene_state_model.py show
examples about how to infer a transcription dynamics in yeast cells. The details about the model are given in the literature <br />
_Fang, Zhou, Ankit Gupta, and Mustafa Khammash. "A divide-and-conquer approach for analyzing high-dimensional noisy gene expression networks." bioRxiv 2023._ <br />
Before running the Python files, one needs to extract the two zip files in "Example_of_model_identification/results_in_paper". 
Many identification results have been given in the same repository. 
Also, these results can be visualized using the jupyter notebooks in the folder "Example_of_model_identification". 

## Examples

### Example 1: A numerical example to infer a yeast transcription system.

MI_example_4.py: This code generates observation data for a three-gene-state system and applies our method to the system. The results are plotted in "Example_of_model_identification/model_identification_example_4.ipynb".

MI_example_4_other_parameter_sets.py: This code generates the observation trajectories with several other sets of model parameters and applies our method to these generated trajectories. The results are plotted in "Example_of_model_identification/model_identification_example_4 (other parameter set).ipynb".

MI_example_4_sigma_uncertainty_general_version.py: This code infers the system when the observation noise intensity is unknown. The results are plotted in "Example_of_model_identification/model_identification_example_4(sigma_uncertainty).ipynb". 

MI_example_4_two_gene_state.py: This code generates observation data for a two-gene-state system and tests whether our method can accurately identify this system. The results are plotted in "model_identification_example_4_(2-gene-state).ipynb".

### Example 2: Identification of a yeast transcription system from experimental data

real_data_3_gene_state_model_larger_space.py: This code infers the system by assuming at most three gene states exist. The results are plotted in "Example_of_model_identification/Inference_of_real_data_3_gene_state_model.ipynb".

real_data_3_gene_state_model_small_parameter_region.py: This code infers the system with a grid refinement strategy. The results are plotted in "Example_of_model_identification/Inference_of_real_data_3_gene_state_model_small_space.ipynb".

real_data_3_gene_state_sigma_uncertainty_cell_78.py: This code infers the observation noise intensity sigma together with other parameters. The results are plotted in "Example_of_model_identification/Inference_of_real_data_3_gene_state_model_(sigma_uncertainty).ipynb"

real_data_2_gene_state_model.py: This code infers the system by assuming at most two gene states exist. The results are plotted in "Example_of_model_identification/Inference_of_real_data_2_gene_state_model.ipynb".

noise_analysis.py: This code performs noise analysis for the yeast cells based on the obtained inference results. The results are plotted in "Example_of_model_identification/noise_analysis.ipynb".

