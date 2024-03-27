# Drug2Sate
Drug2State is a computational ensemble under development for drug target discovery and repurposing by integrating multi-omic data with embeddings from foundation models like BioBERT and ChemBERTa, leveraging a Graph Attention Convolutional Network (GACN) for nuanced mechanistic clustering of drugs.

## Installation
(Not implemented) To install directly from github run below in command line

(Not implemented)```pip install git+https://github.com/Issacgoh/Drug2Sate.git```

To clone and install:

(Not implemented)```git clone https://github.com/Issacgoh/Drug2Sate.git```

(Not implemented)```cd ./Drug2Sate```

(Not implemented)```pip install .```

## About
Drug2State, currently in development, represents an ensemble strategy aimed at enhancing the understanding of drug mechanisms through the integration of multi-omic data with sophisticated computational models. Central to Drug2State is its utilization of drug2cell outputs, enriched with embeddings from BioBERT and ChemBERTa that have been meticulously fine-tuned using the extensive CHEMBL database. This method distinctively incorporates semantic embeddings from foundation models and structural embeddings from chemical databases, embedding a layer of causal semantics within the analysis of drug-target interactions. Employing a Graph Attention Convolutional Network (GACN), Drug2State skillfully learns attention weights among various modality embeddings, enabling the effective integration of diverse data sources. This method is poised to advance the identification of novel drug targets and streamline the processes of drug discovery and repurposing by clustering drugs based on their causal mechanistic similarities.

## Method Overview: 
Drug2State is crafted to tackle the intricate task of unraveling drug mechanisms amid the complexity of extensive biological datasets. By amalgamating drug-gene interaction data with state-of-the-art embeddings, the method offers a comprehensive view of drug actions, integrating molecular structures and semantic, causally-inferred descriptions of drug mechanisms. The employment of GACN facilitates the synthesis of information across different data modalities, with attention mechanisms strategically highlighting crucial interactions. This approach is set to provide profound insights into drug repurposing opportunities and the unveiling of new drug targets, especially in the realm of personalized medicine and targeted therapies.

## Potential and Future Directions:
Drug2State's potential lies in its innovative use of foundation models and extensive chemical databases such as CHEMBERTA, thus providing a vast repository of information for the analysis of drug mechanisms. As the development progresses, future versions are expected to integrate targeted cancer RISPR screen data from DePMap, further enriching the drug discovery landscape with essential genomic insights. This future development is anticipated to enhance Drug2State's ability to identify and characterize drugs with specific mechanistic actions, thus supporting targeted treatments for cancer and other complex diseases.

In essence, Drug2State is a promising exploration into the integration of multi-omic data and advanced computational techniques for in-depth drug mechanism analysis. Although still under development, its innovative approach holds significant promise for transforming drug discovery and repurposing, especially by leveraging foundation models and strategically utilizing large-scale databases and genomic screening information, all while embedding causal semantics within the analysis framework.
