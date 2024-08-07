<img src="/Resources/D2S_logo.webp?raw=true" alt="logo" width="150" height="150">

# Drug2State
Drug2State is a computational ensemble under development for drug target discovery and repurposing by integrating multi-omic data with embeddings from foundation models like BioBERT and ChemBERTa, leveraging a Graph Attention Convolutional Network (GACN) for nuanced mechanistic and structural targetting of drug clusters.

## Installation
To install directly from github run below in command line (Not implemented)

```pip install git+https://github.com/Issacgoh/Drug2Sate.git (Not implemented)```

To clone and install:

```git clone https://github.com/Issacgoh/Drug2Sate.git (Not implemented)```

```cd ./Drug2Sate (Not implemented)```

```pip install . (Not implemented)```

## About
Drug2State, currently in development, represents an ensemble strategy aimed at enhancing the understanding of drug mechanisms through the integration of multi-omic data with sophisticated computational models. Central to Drug2State is its utilisation of drug2cell outputs, enriched with embeddings from BioBERT and ChemBERTa that have been meticulously fine-tuned using the extensive CHEMBL database. This method distinctively incorporates semantic embeddings from foundation models and structural embeddings from chemical databases, embedding a layer of causal semantics within the analysis of drug-target interactions. Employing a Graph Attention Convolutional Network (GACN), Drug2State skillfully learns attention weights among various modality embeddings, enabling the effective integration of diverse data sources. This method is aimed at the identification of novel drug mechanistic targets and streamline the processes of drug repurposing by clustering drugs based on their mechanistic activity targetted at perturbed states.

## Rationale
The rationale behind the development of the Drug2State plugin is anchored in the imperative to effectively navigate and leverage the extensive data reservoirs encapsulated within drug-target interaction databases and the expansive realms of multi-omic datasets, notably those emanating from cutting-edge single-cell RNA sequencing technologies. Conventional methodologies for predicting drug-target interactions often culminate in the generation of voluminous and unwieldy outputs, presenting researchers with a daunting array of potential gene targets for individual drugs. This proliferation of data poses significant challenges in distilling actionable insights and pinpointing drugs that share mechanistic actions.

Drug2State emerges as a solution to these challenges by orchestrating an integrated framework that integrates multiple data kernels. This integration encompasses drug-gene predictions derived from single-cell RNA expression data (as outputted by drug2cell), similarities in drug-predicted gene overlaps, semantic embeddings extracted from pre-trained language models (LLMs), and molecular embeddings from SMILES strings using ChemBERTa. This multifaceted approach aims to construct a comprehensive landscape that not only delineates drug similarities but also deepens the understanding of their underlying mechanisms.

At the heart of Drug2State's methodology is the strategic clustering of drugs, achieved by parsing their semantic, structural, and functional nuances. This process fosters the emergence of a smooth manifold where drugs are aggregated into clusters, each represented with distinct semantic and causal connotations. These clusters serve to guide towards groups of drugs that exhibit congruent mechanistic action properties or embody similar biological programs. Such a clustering mechanism is pivotal in identifying drugs with analogous modes of action or therapeutic potentials, thereby streamlining the path towards drug repurposing and novel mechanistic targets.

Drug2State lays the groundwork for a systematic and coherent analysis of drug-target interactions. The platform's reliance on transfer learning from pre-trained LLMs, coupled with sophisticated similarity analysis techniques, empowers researchers to mine actionable insights from the labyrinth of complex multi-omic datasets.

## Project team
<p>Issac Goh, Newcastle University; Sanger institute (https://haniffalab.com/team/issac-goh.html) <br></p>

## Team responsibilities
- Issac Goh is writing all analytical framework modules

### Contact
Issac Goh, (ig7@sanger.ac.uk)

## Method Overview: 
Drug2State is crafted to tackle the intricate task of unraveling drug mechanisms amid the complexity of extensive biological datasets. By amalgamating drug-gene interaction data with state-of-the-art embeddings, the method offers a comprehensive view of drug actions, integrating molecular structures and semantic, causally-inferred descriptions of drug mechanisms. The employment of GACN facilitates the synthesis of information across different data modalities, with attention mechanisms strategically highlighting crucial interactions. This approach is set to provide profound insights into drug repurposing opportunities and the unveiling of new drug targets, especially in the realm of personalized medicine and targeted therapies.

### Rationale for Creating Joint Embeddings

1. **Enhanced Interpretability**: By mapping data from different sources into a shared embedding space, it becomes easier to understand how various aspects of the data relate to one another, facilitating clearer interpretations of how drugs influence cellular mechanisms.

2. **Improved Prediction Accuracy**: Integrating embeddings from different methodologies can leverage complementary information, potentially leading to more accurate predictions of drug efficacy and side effects.

3. **Identification of Novel Relationships**: Joint embeddings can reveal new relationships between drugs and cellular responses that are not apparent when analyzing datasets in isolation.

4. **Robustness to Data Sparsity and Noise**: Combining multiple data sources can help mitigate issues related to data sparsity and noise in individual datasets, as the integration process can emphasize shared signal over noise.

### How MOFA Achieves Integration

MOFA (Multi-Omics Factor Analysis) is a statistical framework designed to integrate and decompose multiple datasets into a set of latent factors that capture the underlying sources of variability. The process by which MOFA+ achieves this can be broken down into several steps:

1. **Input Data**: MOFA+ accepts multiple input data matrices, each corresponding to a different type of data or feature set. In your case, these matrices might represent embeddings from language models and drug interaction tests.

2. **Factorization**: MOFA+ employs a group factor analysis approach to decompose each dataset into a set of latent factors and feature loadings:
$$
\mathbf{X}_k = \mathbf{Z} \mathbf{W}_k^T + \mathbf{E}_k
$$

Where:
- $( \mathbf{X}_k )$ represents the data matrix for the $( k )$-th data type.
- $( \mathbf{Z} )$ is the matrix of latent factors common across all data types.
- $( \mathbf{W}_k )$ is the loading matrix specific to the $( k )$-th data type, indicating how much each feature contributes to each factor.
- $( \mathbf{E}_k )$ is the residual noise matrix for the $( k )$-th data type.

This formulation allows for the extraction and integration of the most significant modes of variation across different types of data, yielding a joint embedding space that reflects shared and unique aspects of each dataset.
3. **Variational Inference**: MOFA+ uses a Bayesian framework with variational inference to estimate the factors and loadings. This approach allows the model to handle different types of data (continuous, binary, count, etc.) and to incorporate prior knowledge.

4. **Regularization**: The model includes options for sparsity-inducing priors on the loadings, which encourages each factor to use only a subset of the features from each data type, simplifying interpretation and enhancing model robustness.

5. **Output**: The output of MOFA+ is a set of factors that describe the main modes of variation across the datasets. These factors constitute the joint embedding space where relationships across data types can be explored.

### Implementation in the Context of Drug Discovery

In the context of drug discovery, MOFA+ can integrate drug response data, genetic data, and embeddings derived from text data about drug mechanisms. This integrated approach provides a comprehensive view that can enhance the understanding of drug effects at the cellular level, potentially leading to discoveries related to drug mechanisms, interactions, and effects on disease pathways.

The joint embedding space created by MOFA+ allows for clustering drugs based on their mechanisms and predicted effects, supporting the identification of novel drug candidates and therapeutic opportunities. The enhanced interpretability and robustness of the model also contribute to more accurate predictions and insights into complex biological systems.

By leveraging the strengths of specialized tools at each stage—from feature extraction using advanced NLP models to integration and analysis using MOFA+—your pipeline creates a powerful platform for advancing drug discovery and understanding complex biomedical relationships.

### Summary

The integration process via MOFA+ allows for a robust and interpretable model that can combine diverse data types into a coherent framework, enhancing drug discovery and research by providing a deeper understanding of drug mechanisms. The formulaic representation of MOFA+ captures the essence of how data from different sources are synthesized to uncover latent structures, making it a powerful tool in multi-omics and drug prediction studies.

## Potential and Future Directions:
Drug2State's potential lies in its innovative use of foundation models and extensive chemical databases such as CHEMBERTA, thus providing a vast repository of information for the analysis of drug mechanisms. As the development progresses, future versions are expected to integrate targeted cancer RISPR screen data from DePMap, further enriching the drug discovery landscape with essential genomic insights. This future development is anticipated to enhance Drug2State's ability to identify and characterize drugs with specific mechanistic actions, thus supporting targeted treatments for cancer and other complex diseases.

In essence, Drug2State is a promising exploration into the integration of multi-omic data and advanced computational techniques for in-depth drug mechanism analysis. Although still under development, its innovative approach holds significant promise for transforming drug discovery and repurposing, especially by leveraging foundation models and strategically utilizing large-scale databases and genomic screening information, all while embedding causal semantics within the analysis framework.

## Built With
- [Scanpy](https://scanpy.readthedocs.io/en/stable/) - An analysis environment for single-cell genomics.
- [Drug2Cell](https://www.sanger.ac.uk/technology/drug2cell/) - computational pipeline that can predict drug targets as well as drug side effects. 
- [Pandas](https://pandas.pydata.org/) - A fast, powerful, flexible and easy to use open source data analysis and manipulation tool.
- [NumPy](https://numpy.org/) - The fundamental package for scientific computing with Python.
- [SciPy](https://www.scipy.org/) - Open-source software for mathematics, science, and engineering.
- [Matplotlib](https://matplotlib.org/) - A comprehensive library for creating static, animated, and interactive visualizations in Python.
- [Seaborn](https://seaborn.pydata.org/) - A Python visualization library based on matplotlib that provides a high-level interface for drawing attractive statistical graphics.
- [scikit-learn](https://scikit-learn.org/stable/) - Simple and efficient tools for predictive data analysis.
- [MyGene](https://pypi.org/project/mygene/) - A Python wrapper to access MyGene.info services.
- [GSEApy](https://gseapy.readthedocs.io/en/latest/) - Gene Set Enrichment Analysis in Python.
- [Anndata](https://anndata.readthedocs.io/en/latest/) - An annotated data structure for matrices where rows are observations (e.g., cells) and columns are variables (e.g., genes).
- [PyMC3](https://docs.pymc.io/) - A Python package for Bayesian statistical modeling and probabilistic machine learning.
- [Joblib](https://joblib.readthedocs.io/en/latest/) - A set of tools to provide lightweight pipelining in Python.
- [tqdm](https://tqdm.github.io/) - A fast, extensible progress bar for Python and CLI.
- [Requests](https://requests.readthedocs.io/en/master/) - An elegant and simple HTTP library for Python.
- [Psutil](https://psutil.readthedocs.io/en/latest/) - A cross-platform library for retrieving information on running processes and system utilization in Python.


## Getting Started
This package takes as input:
  - An anndata object
  - A categorical data variable containing labels/states
  - A 2D array containing some XY dimensionality-reduced coordinates (PCA, VAE, etc...)
  - A sparse 2D matrix (CSR) containing cell-cell weighted connectivities (KNN, etc...)
  - CHEMBL database

### Running Locally
This package was designed to run within a Jupyter notebook to utilise fully the interactive display interfaces. Functions can also be run locally via a python script. 
Please see the example notebook under "/example_notebooks/"

### Production
To deploy this package for large data submitted to schedulers on HPCs or VMs, please see the example given in "/example_notebooks/". 

## Example outs
- D2S provides a framework for leveraging and optimising pre-trained foundational large language models trained on large data corpuses of biomedical, drug, ontological, and chemical data. It seeks to forge an integrated manifold of these modalities for efficient mapping and harmonisation of drug-cell targetting data. 

![Alt text](/Resources/D2S_structure.png?raw=true "D2S_structure")

- D2S provides a utility to customise semantic feature extractions from corresponding foundational model layers to create seperate syntactic and semantic embeddings
![Alt text](/Resources/D2S_semantic_feature_extraction.png?raw=true "D2S_semantic_feature_extraction")

- D2S outputs a series of attention-weighted graph manifolds which are clustered and semantically decoded for core mechanistic, structural, and targetting effects
![Alt text](/Resources/D2S_mechanistic_semantic_clusters.png?raw=true "D2S_mechanistic_semantic_clusters")

- D2S allows users to identify condition-specific mechanistic target perturbations and highlight condition-specifc perturbed gene programs
![Alt text](/Resources/D2S_multi_condition_perturbation_targetting.png?raw=true "D2S_multi_condition_perturbation_targetting")

![](/Resources/D2S_condition_specific_drugs.png?raw=true "D2S_condition_specific_drugs")


- D2S additionally allows cell-state specific targetting of perturbed gene-expression programs
![](/Resources/D2S_cell_state_specific_targetted_drug_clusters.png?raw=true "D2S_cell_state_specific_targetted_drug_cluster")

- D2S allows users to quickly quantify drug mechanistic and structural overlaps for quick identification of novel target mechanisms
![](/Resources/D2S_drug_target_mechanistic_overlap_hull.png?raw=true "D2S_drug_target_mechanistic_overlap_hull")


## Workflow
To be updated
