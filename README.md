<img src="/Resources/D2S_logo.webp?raw=true" alt="logo" width="150" height="150">

# Drug2State
Drug2State is a computational ensemble under development for drug target discovery and repurposing by integrating multi-omic data with embeddings from foundation models like BioBERT and ChemBERTa, leveraging a Graph Attention Convolutional Network (GACN) for nuanced mechanistic clustering of drugs.

## Installation
To install directly from github run below in command line (Not implemented)

```pip install git+https://github.com/Issacgoh/Drug2Sate.git (Not implemented)```

To clone and install:

```git clone https://github.com/Issacgoh/Drug2Sate.git (Not implemented)```

```cd ./Drug2Sate (Not implemented)```

```pip install . (Not implemented)```

## About
Drug2State, currently in development, represents an ensemble strategy aimed at enhancing the understanding of drug mechanisms through the integration of multi-omic data with sophisticated computational models. Central to Drug2State is its utilization of drug2cell outputs, enriched with embeddings from BioBERT and ChemBERTa that have been meticulously fine-tuned using the extensive CHEMBL database. This method distinctively incorporates semantic embeddings from foundation models and structural embeddings from chemical databases, embedding a layer of causal semantics within the analysis of drug-target interactions. Employing a Graph Attention Convolutional Network (GACN), Drug2State skillfully learns attention weights among various modality embeddings, enabling the effective integration of diverse data sources. This method is poised to advance the identification of novel drug targets and streamline the processes of drug discovery and repurposing by clustering drugs based on their causal mechanistic similarities.

## Rationale
The rationale behind the development of the Drug2State plugin is anchored in the imperative to effectively navigate and leverage the extensive data reservoirs encapsulated within drug-target interaction databases and the expansive realms of multi-omic datasets, notably those emanating from cutting-edge single-cell RNA sequencing technologies. Conventional methodologies for predicting drug-target interactions often culminate in the generation of voluminous and unwieldy outputs, presenting researchers with a daunting array of potential gene targets for individual drugs. This proliferation of data poses significant challenges in distilling actionable insights and pinpointing drugs that share mechanistic actions.

Drug2State emerges as a solution to these challenges by orchestrating an integrated framework that synergizes multiple data kernels. This integration encompasses drug-gene predictions derived from single-cell RNA expression data (as outputted by drug2cell), similarities in drug-predicted gene overlaps, semantic embeddings extracted from pre-trained language models (LLMs), and molecular embeddings from SMILES strings using ChemBERTa. This multifaceted approach aims to construct a comprehensive landscape that not only delineates drug similarities but also deepens the understanding of their underlying mechanisms.

At the heart of Drug2State's methodology is the strategic clustering of drugs, achieved by parsing their semantic, structural, and functional nuances. This process fosters the emergence of a smooth manifold where drugs are aggregated into clusters, each imbued with distinct semantic and causal connotations. These clusters serve as a beacon, guiding researchers towards groups of drugs that exhibit congruent mechanistic action properties or embody similar biological programs. Such a clustering mechanism is pivotal in identifying drugs with analogous modes of action or therapeutic potentials, thereby streamlining the path towards drug repurposing and the unveiling of novel drug candidates.

Drug2State lays the groundwork for a systematic and coherent analysis of drug-target interactions. The platform's reliance on transfer learning from pre-trained LLMs, coupled with sophisticated similarity analysis techniques, empowers researchers to mine actionable insights from the labyrinth of complex multi-omic datasets. Consequently, Drug2State stands at the vanguard of efforts to harness the full potential of contemporary bioinformatics in drug discovery, promising a new horizon in the identification of therapeutic opportunities and the strategic repurposing of existing drugs to address specific biological perturbations.

## Project team
<p>Issac Goh, Newcastle University; Sanger institute (https://haniffalab.com/team/issac-goh.html) <br></p>

## Team responsibilities
- Issac Goh is writing all analytical framework modules

### Contact
Issac Goh, (ig7@sanger.ac.uk)

## Method Overview: 
Drug2State is crafted to tackle the intricate task of unraveling drug mechanisms amid the complexity of extensive biological datasets. By amalgamating drug-gene interaction data with state-of-the-art embeddings, the method offers a comprehensive view of drug actions, integrating molecular structures and semantic, causally-inferred descriptions of drug mechanisms. The employment of GACN facilitates the synthesis of information across different data modalities, with attention mechanisms strategically highlighting crucial interactions. This approach is set to provide profound insights into drug repurposing opportunities and the unveiling of new drug targets, especially in the realm of personalized medicine and targeted therapies.

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

![Alt text](/Resources/D2S_structure.png")

- D2S provides a utility to customise semantic feature extractions from corresponding foundational model layers to create seperate syntactic and semantic embeddings
![Alt text](/Resources/D2S_semantic_feature_extraction.png")

- D2S outputs a series of attention-weighted graph manifolds which are clustered and semantically decoded for core mechanistic, structural, and targetting effects
![Alt text](/Resources/D2S_mechanistic_semantic_clusters.png")

- D2S allows users to identify condition-specific mechanistic target perturbations and highlight condition-specifc perturbed gene programs
![Alt text](/Resources/D2S_multi_condition_perturbation_targetting.png")
![Alt text](/Resources/D2S_condition_specific_drugs")


- D2S additionally allows cell-state specific targetting of perturbed gene-expression programs
![Alt text](/Resources/D2S_cell_state_specific_targetted_drug_clusters.png")

- D2S allows users to quickly quantify drug mechanistic and structural overlaps for quick identification of novel target mechanisms
![Alt text](/Resources/D2S_drug_target_mechanistic_overlap_hull")


## Workflow
To be updated
