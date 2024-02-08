# Rao-Blackwell-method-for-cell-specific-model-identification

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

## Example 

### Example 1: A numerical example to infer a yeast transcription system.

MI_example_4.py <-- This file generates observation data for a three-gene-state system and applies our method to the system. 

Example_of_model_identification/model_identification_example_4.ipynb <-- This file plots the result of the inference result for the three-gene-state model. 






A numerical example is presented in Example_of_model_identification/model_identification_example_4.ipynb, introducing how to use the code.  
