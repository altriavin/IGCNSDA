# IGCNSDA: a interpretable graph convolutional network approach for predicting potential snoRNA-disease associations

The IGCNSDA utilizes an interpretable graph convolutional network approach to predict potential snoRNA-disease associations. Initially, we apply a graph convolutional network algorithm on the bipartite graph representing snoRNA-disease associations to obtain first-order embeddings. We introduce a novel subgraph generation algorithm that clusters similar snoRNAs into subgraphs. Subsequently, we iteratively apply the Graph Convolutional Network (GCN) on each subgraph to update the embeddings of snoRNAs and diseases, yielding higher-order embeddings. Finally, a layer aggregation algorithm is employed to derive the ultimate embeddings for snoRNAs and diseases.

# Requirements
```
torch 1.8.1
python 3.7.16
numpy 1.21.6
pandas 1.3.5
scikit-learn 1.0.2
scipy 1.7.3
```

# Dataset
In this study, the benchmark dataset was sourced from MNDR v4.0\cite{ning2021mndr}, a comprehensive repository that aggregates experimentally validated and predicted ncRNA-disease associations extracted from literature and other reputable resources. The original snoRNA-disease associations were extracted by querying the snoRNA records within MNDR dataset. Subsequently, entries with redundant or incomplete information were meticulously eliminated, resulting in the curation of a benchmark dataset consisting of 471 snoRNAs, 84 diseases, and a total of 1106 well-documented associations between these entities. Moreover, we collect 584 pairs of associations between 82 diseases and 13 snoRNAs were extracted from the ncRPheo dataset for independent testing.

# Run the demo
```
cd code && python main.py
```
