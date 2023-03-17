---
title: Applying NCEM and MISTy on MIBI-TOF colorrectal cancer data as a primer for spatial cell-cell communication benchmarking
description: Stegle-lab internship report
image: https://i.imgur.com/J57ANsD.jpg
---


# Applying NCEM and MISTy on MIBI-TOF colorrectal cancer data as a primer for spatial cell-cell communication benchmarking

![](https://i.imgur.com/J57ANsD.jpg =400x400)

# Table of contents
[toc]

# Abstract
- [ ] Reduce length to fit a single page
    - 2-3 paragraphs less (striked through text)
---

Prior to the emergence of spatial-omics technologies, cell-cell communication (CCC) methods relied on prior knowledge about interacting features between cells. The measure of correlation between putative ligand-receptor interactions are a common approach to this end. Novel spatially-resolved omics technologies like MIBI-TOF and MERFISH enable the measurement and characterization of feature patterns in space. Novel CCC methods followed which attempt to leverage the spatial dimention. These methods face four main challenges: (1) the lack of a consensus on how to define and approach CCC, (2) the abundance of spatial-omics technologies with different output types and properties, (3) the high level of artifacts and uncertainty in upstream steps and (4) the lack of ground truth for validation.

In the present study, the aim was to set the groundwork towards benchmarking spatially-resolved CCC methods by applying recent spatial CCC methods NCEM and MISTy to a colorrectal tumour dataset. Hence, we analyse their potential weaknesses and attempt to reach a comparable output.

The linear NCEM model was applied to the dataset and the interaction term tensor was summarised via two different transformations to achieve a cell-type sender-receiver effect matrix. Furthermore, erroneous cell-type annotation due to upstream effects was simulated via shuffling the annotations of varying fractions of randomly selected cells. The results showed a high sensitivity towards cell-type annotation as shuffling 10% of the cells' cell-types reached similar explained variance ($R^2$) values to 100% annotation randomization.

MISTy was applied by setting the "intra-view" and a "para-view". ~~The latter was adefined as the distance weighted sum by feature considering all cells within a 35 µm radius.~~ As opposed to NCEM, MISTy doesn't model batch and conditions directly; it is run on an individual sample basis and then summarized via the mean and standard deviation. When examining the improvement of explained variance ($R^2$) attributed to the "para-view", high standard deviation for all features was observed. This shows the necessity of better sample integration or the assessment of experimental conditions for intermediate step interpretability.

~~Due to lack of ground truth, neither linear NCEM's nor MISTy's CCC output could be validated.~~ MISTy was customized to yield a similar output to NCEM, ~~a cell-type predictor-target importance matrix.~~ However, no positive importances were found on the summarized statistic for all images. Only on the individual sample level predictor target importances were obtained. Additionally, the gian in explained variance due to extrinsic "para-view" information once again showed high standard deviation.

~~The statistical exploration of the collective dataset on a spatial- and non-spatial level recovered the major immune lineage profiles via the mean by cell-type. However, PCA variance nor clustering and UMAP embedding showed a clear separation of the cell-types, indicating the need for a better sample integration method or erroneous cell-type annotation.~~

These findings highlight the present challenges of the spatial CCC field and set the ground-work for the application and benchmarking of further spatial CCC methods to establish better putative protocols.


# Introduction ! In  process !

Spatially-resolved omics technologies harness the information found in the spatial dimention to give insight into tissue properties, structure and processe which cannot be reliably captured without the spatial layer of information. These insights are essential to understanding biological processes like embryonic development and disease. Cell-cell interactions underly many of these biological processes and are hence one of the desired downstream ouputs.

### CCC
- Jin & Ramos
    - Due to the finite spatial diffusivity of the soluble ligands in paracrine signaling and the physical contact between adjacent cells in juxtacrine signaling [3], spatial information is vital to study cell–cell communication. This spatial information of biological tissue is lost in scRNA-seq, but preserved in imaging-based technologies. 
    - Incorporating spatial information will likely reduce false-positive inferred signaling links, because cells only communicate directly over a limited spatial distance. Computational methods that do not use spatial information may fail to detect certain expected ligand–receptor interactions [43]. Here, we discuss some emerging approaches that have been developed to infer cell–cell communication using spatial information 
Often times a "niche" radius must previously be defined to
- Spatial museum also poses CCC as prob of ct-colocalization based on L-R expre, or p(L-R) based on co-localization.

> Communication between cells is often mediated by various types of soluble and membrane-bound factors, such as ligands, receptors, extracellular matrix (ECM), integrins and junction proteins [1]. - Suoqin Jin
-  (Jin & Ramos)
    -  Sender receiver definitions

### CCC challenges (papers?)

- Upstream data analysis converts raw data into forms  more amenable to biological interpretation and is dependent on the  data-collection technology
- CCC challegnes
Downside: plethora of technologies & IO.

Another challenge in the field of cell-cell communication is the plethora of different spatial-omics technologies which output fundamentally different data. To list a few spatial transcriptomics technologies categorized by their approachs: LCM[] and Tomo-seq[] select a region of interest (ROI); seqFISH[] and MERFISH[] are based on fluorescence in-situ hibridization (FISH) of mRNA molecules; FISSEQ[] uses in-situ sequencing (ISS); Slide-seqV2[], spatial transcriptomics (ST)[], VISIUM[] and Xenium][] apply next generation sequencing (NGS) with spatial barcoding on arrays; and DNA microscopy[] reconstructs spatial locations based on cDNA proximity. Other omics technologies include spatial proteomics like MIBI-TOF[] and seqFISH+[]. There are currently no established upstream protocols or workflows that account for this variation in spatial data types as their preprocessing is mostly adapted to the technologies' specific experimental nature and  intricacies.

<!-- As a general trend, a trade-off between resolution and multiplexity (feature coverage) is observed. 


Existing spatial omics technologies take different approaches towards capturing the location of omics features. Spatial transcriptomics approaches range from region of interest (ROI) selection (e.g. LCM)[], fluorescence in-situ hibridization (FISH) (e.g. MERFISH)[] and in-situ sequencing (ISS) (e.g. FISSEQ[] to next generation (NG) barcoding on arrays (e.g. Slide-seqV2, VISIUM and Xenium)[] and DNA microscopy []. Other omics technologies include spatial proteomics like MIBI-TOF and seqFISH+. As a general trend, a trade-off between resolution and multiplexity (feature coverage) is observed.-->






### CCC methods
Often times a "niche" radius must previously be defined to

-  (Jin & Ramos)
    - R-L based are biased towards DB.
    - R-L 1 vs complexes
    - FP and FN

- Spatial museum 2
    - CCI for supercell res, mainly Deconvolution, then colocalization but other types also possible e.g. NCEM
        - For ST and Visium, we can use one of the cell type deconvolution methods to find the number and proportion of cell types per unit area in each tissue region and cell type colocalization. When two cell types colocalize, they might interact with secreted ligands or ligands and receptors bound to the membrane. Expression of ligand-receptor (L-R) pairs in neighboring cells is often used to identify cell-cell interaction in spatial data, and the CellPhoneDB (Efremova et al. 2020) database of ligands, receptors, and their interactions is often used to identify such L-R pairs.

- Mention that subcell resolution not always required thanks to deconvolution techniques or specialized variants deconcem


Context

(linear or higher order, unsupervised, co-expr) 
| METHod            | Spatinf | R?egt | Prkn | A   | Niches  | Prdctv | Coreconcept                             | LoGoCV | L    |
| ----------------- |:-------:|:-----:| ---- | --- |:-------:|:------:| --------------------------------------- |:------:| ---- |
| MISTy             |    Y    |   N   | N    | N   | depends |   Y    | Non-linear ensembl ML                   |   Y    | R    |
| NCEM (linear)     |    Y    |   N   | N    | Y   |    Y    |   Y    | Linear regression                       |   N    | P    |
| SVCA              |    Y    |   N   | N    | N   |   Y?    |   Y    | Probabilistic modlling                  |   ?    | mus2 |
| DIALOGUE          |         |       |      |     |         |        |                                         |        | R    |
| COMMOT            |    Y    |   Y   | R-L  | N   |    N    |   N    | Collective optimal transport            |   N    |      |
| GCNG              |    Y    |       |      |     |         |   Y    | Graph convolutional neural networks     |        | P    |
| SpaTalk           |         |       |      |     |         |        |                                         |        |      |
| SpaCET            |         |       |      |     |         |        |                                         |        |      |
| stLearn           |    Y    |   Y   | Y    | Y   |   N?    |   N    | R-L Co-expression related to ct         |        | P    |
| CellPhoneDB v3    |    Y    |   Y   | L-R  | Y   |    N    |   N    | Stat sign interactions via v rand       |        |      |
| CellPhoneDB v2    |    N    |   Y   | L-R  | Y   |    N    |   N    | Stat sign interactions via v rand       |        |      |
| CellChat          |   N*    |   Y   | L-R  | Y   |   N?    |   N?   | ariba(Comparison across cond compl comp | matr.f |      |
| ICELLNET?         |    N    |   Y   | L-R  | Y?  |         |        |                                         |        |      |
| SoptSC            |    N    |   Y   | L-R  | N   |         |        | (Individual cell-based method)          |        |      |
| NicheNet          |    N    |   Y   |      |     |         |        |                                         |        |      |
| CytoTalk          |    N    |   Y   |      |     |         |        |                                         |        |      |
| scTensor          |    N    |       |      |     |         |        | Hypergraphs                             |        |      |
| Giotto            |    Y    |   Y   | Y?   | ?   |    N    |   N    | R-L interaction in  graph               |        |      |
| Neigh en Squid?   |    Y    |   N   | N    | Y   |    N    |   N    | Counting neighbours in graph            |        | P    |
| CellTalk? DB      |         |   Y   | L-R  |     |         |        |                                         |        |      |
|                   |         |       |      |     |         |        |                                         |        |      |
| Cell2Cell         |    Y    |       |      |     |         |        |                                         |        |      |
| SpaOTsc           |    Y    |       |      |     |         |        |                                         |        | P    |
| MESSI             |    Y    |       |      |     |         |        |                                         |        |      |
| STRISH            |    Y    |       |      |     |         |        |                                         |        |      |
|                   |         |       |      |     |         |        |                                         |        |      |
| iTALK             |    N    |       |      |     |         |        | DEA (Comparison across conditions)      |        |      |
| Connectome        |    N    |       |      |     |         |        | DEA (Comparison across conditions)      |        |      |
| NATMI             |    N    |       |      |     |         |        |                                         |        |      |
| SingleCellSignalR |    N    |       |      |     |         |        |                                         |        |      |
| PyMINEr           |    N    |       |      |     |         |        |                                         |        |      |

- Table properties
- Table challenges
 Jin and ramos shows some procs

### CCC relevance in bio and cancer

### MISTy
- Metamodel
	- Input
		- Main
		- Secondary
	- Output
	- Coersion
	- Experience

### NCEM

### Aim and intentions



# Materials and methods

**Dataset.** The publically available MIBI-TOF colorectal cancer dataset from [[Hartmann-2021]] spans 58 images of 4 donors. 40 samples belong to 2 donors which were diagnosed with the colorectal carcinoma. Both cancerous colon and healthy adjacent tissue was included. The remaining 18 samples belonged to 2 donors with healthy colon. The images measure 36 protein levels across $400 \mu m^2$ and 1024 bins^2^. 63747 cells were measured across all images. The data had been previously preprocessed by the authors. It had undergone noise removal, cell-size normalization, arcsinh transformation and percentile normalization. The authors also provided the segmentation masks, which were calculated via a watershed-coupled convolutional neural network trained on different cancer types.The provided cell-type annotation was obtained via FlowSOM clustering and manual annotation based on main cell-lineage markers. Included cell types were: endothelial, epithelial cells, fibroblasts, other CD45+ immune cells, CD68+ myeloid cells, CD68+ myeloid cells, CD4+ T-cells and CD8+ T-cells. Major lineage markers were defined by the authors: CD11c, CD14, CD3, CD31, CD4, CD45, CD68, CD8, smooth muscle actin (SMA), epithelial cadherin (Ecad), cytokeratin (CK) and vimentin.

**Visualization.** Segmentation and raw expression images were generated using the `Squidpy` module and loading the expression data, metadata and image channels into an `anndata.AnnData()` class from the `Anndata` module. PCA and UMAP embedding were generated using the `Scanpy` module. The cell-type frequency distribution plots were generated with the `matplotlib.pyplot` module.

**Unsupervised analysis.** PCA dimentionality reduction and neighbourhood graph were calculated using the `Scanpy` module with default settings. UMAP embedding was also calculated via the `Scanpy` module and the top 10 ranked  principal components. The latter number was chosen based on the elbow plots of accounted variance of ranked principal components. Highly variable genes were calculated using the `scanpy.pp.highly_variable_genes()` function.

**Spatial connectivity graph.** The spatial graph was generated using the `Squidpy` package. The coordinate type was set to "generic" (coord_type) and the neighbourhood was set to include all cells within a 35 px (~14 µm) radius.

**Spatial distribution statistics of cell-type annotation.** The Ripley's L statistic, clustering coefficients (centrality score statistic), neighbourhood enrichment analysis and co-occurence analysis were obtained using the `Squidpy` package using default parameters.

**Spatial feature distribution statistic.** The Moran's L statistic was calculated using the `Squidpy` package. The mode was set to "Moran" and  permutations (n_perms) were set to 100.

**Linear NCEM.**  Linear NCEM model was applied using its publically available python module. The linear NCEM model is a simple linear regression: $\hat{Y} = X^D \beta$ where $\hat{Y} \in \mathbb{R}^{N \times J}$ where N are the number of cells and J is the number of features. The design matrix  $X^D \in \mathbb{R}^{N \times (L + L^2 + C)}$, where C is the batch assignment, is constructed using the concatenation of the one-hot encoding of the index cell type, the one hot encoded presence (0 or 1) of each cell type in the radius-based constsructed neighbourhood graph and the batch assignment (interaction terms as combinatorial and directional cell-type labels). The covariate matrix $\beta \in \mathbb{R}^{(L + L^2 + C) \times J}$ is learned via ordinary least squares loss function. To train, validate and test the model, the tutorial provided by the author's was used [https://github.com/theislab/ncem_tutorials/](https://github.com/theislab/ncem_tutorials). The same parameters were with exception of dataset specific arguments. The data was loaded using the `DataLoaderHartmann()` class via setting the `data_origin='hartmann` parameter when running the `get_data()` function on the `ncem.estimators.EstimatorInteractions` class. The `radius` was set to 35 px which the author had shown to yield the best predicitve performance. The `batch_size` was set to 58. The dataset was split to 80% training, 10% validation and 10% testing sets. Evaluation outputs several statistics including the loss, gaussian reconstruction loss and $R^2$ value.


**Linear NCEM sender-receiver effects.** The linear interactions tutorial from the author's tutorial github repository [https://github.com/theislab/ncem_tutorials/](https://github.com/theislab/ncem_tutorials) was followed. The data was loaded using the `DataLoaderHartmann()` class via setting the `data_origin='hartmann` parameter when running the `get_data()` function on the `ncem.estimators.EstimatorInteractions` class. The radius parameter was set to 35 px which was observed to yield the best predictive performance in the grid search. Observation data was split into training (90%), validation (10%) and test (10%) sets. 10 nodes per graph were used for hyperparameter tuning. After applying the `ncem.estimators.EstimatorInteractions.get_sender_receiver_effects()` function, NCEM output two relevant tensors (1) the interaction term values tensor $I \in \mathbb{R}^{L \times L \times J}$ where L are the cell-type labels and J the features (dimentions: receiver-cell-type x sender-cell-type x features ) and (2) the p-values tensor of the Wald test applied to the interaction terms, FDR-corrected via the Benjamini-Hochberg correction (q-values). The former represents the effect of individual features of a sender cell-type on a receiver cell-type. Next, the the first summary statistic was calculated inspired by the `type_coupling_analysis()` approach of the package authors. The number of statistically significant features were quantified per sender-receiver label pairs via the L1-norm. Alpha-level value 0.05 was used to set the significance threshold. The second summary statistic approach, deviating from the author's approach, calculated the L2-norm of the feature axis per sender-receiver cell-type pairs. Both summary statistic matrics are of (sender-cell-type x receiver-cell-type) dimentionality. A wrapper was written to make the application these approaches easier on the end user. The wrapper allows to set the radius, number of evaluated nodes, alpha-level values for FDR corrected signficance threshold and end-value-level thresholds for both approaches. The former parameter can also be set using a quantile threshold for the second approach. Exclusively for the second approach, a range-threshold for the raw interaction terms can be set (hard range or standard-deviation-based).

**NCEM cell-type annotation shuffling analysis.** To simulate upstream artifacts like imperfect segmentation or wrong cell-type annotation, the provided cell-type annotations were shuffled in different fractions (0%, 0.1%. 1%, 10%, 50% and 100%) of randomly selected cells. Afterwards, the linear NCEM model was applied as described in (Methods: linear NCEM). The process was repeated 15 times for every fraction percentage to generate a distribution of explained variance values ($R^2$).

**MISTy.** `MISTy` was applied using the public R package on the author's github repository. MISTy is a flexible machine learning framework which tries to identify constrained intercellular feature interactions. It establishes a metamodel that incorporates different "views". These are spatial areas which contribute a user-defined aggregation statistic across the selected cells to the model. The intrinsic "para-view" is defined as the prediction of individual features of the index cell using the remaining features. It aims to model the individual cells' feature state. Additional views can be defined by the user. These are used to complement the "intra-view" in an attempt to improve prediction performance. In the case of this study, a single "para-view" was defined as the weighted sum of the feature values across all cells within a 35 px `radius`. The weighting was set to a gaussian kernel (`family='gaussian`) and the learnable function (`model.function`) is left as per default to random forest. The input were the spatial positions and feature expression. MISTy outputs three types of data (1) the improvement of explained variance ($R^2$) via contrasting the application of the model with solely the "intra-view" and the application of the model using both the "multi-view" model ("intra-" and "para-view" considered) on a feature-level, (2) the learnable weight parameters of the metamodel which represent the "view-effect" and (3) the interactions or importances that each predictor feature contributes to the target feature via leave-one-out procedure. The latter output is calculated on a per-view basis which is then substracted to create the importances contrast heatmap. Image points 1, 4, 10, 13, 17, 18, 19, 20, 30, 32, 34, 50 and 52 were excluded.

**Customized MISTy workflow.** The standard MISTy workflow (Methods: MISTy) was modified to use as input the one-hot encoding of the index cell-type instead of the feature space. The "intra-view" was bypassed. The "para-view" definition was changed to use a constant kernel.

**Hardware and OS.** All analysis and code was run on a an XPS 17 9710 laptop equipped with an 11th Gen Intel i7-11800H CPU (16 threads @ 4.600GHz), an NVIDIA GeForce RTX 3050 Mobile GPU with 2x8 GiB DDR4 RAM (3200 MHz). OS: Pop!_OS 22.04 LTS x86_64 on kernel "6.1.11-76060111-generic"

**Table 3: List of general software.**
| Software         | Version       |
| ---------------- | ------------- |
| Conda            | 22.9.0        |
| Jupyter notebook | 6.5.2              |
| Python           | 3.8.16        |
| R                | 4.22          |
| R-Studio         | 2022.12.0+353 |


**Table 4: List of used python packages.**
| Package    | Version |
|:---------- |:------- |
| matplotlib | 3.6.3   |
| ncem       | 0.1.4   |
| numpy      | 1.22.4  |
| pandas     | 1.5.3   |
| scanpy     | 1.9.1   |
| scipy      | 1.10.0  |
| seaborn    | 0.11.2  |
| tensorflow | 2.11.0  |

**Table 5: List of used R packages.**
| Package              | Version |
|:-------------------- |:------- |
| Biobase              | 2.56.0  |
| BiocGenerics         | 0.42.0  |
| GenomeInfoDb         | 1.32.4  |
| GenomicRanges        | 1.48.0  |
| IRanges              | 2.30.1  |
| MatrixGenerics       | 1.8.1   |
| S4Vectors            | 0.34.0  |
| SingleCellExperiment | 1.18.1  |
| SummarizedExperiment | 1.26.1  |
| base                 | 4.2.2   |
| datasets             | 4.2.2   |
| distances            | 0.1.9   |
| dplyr                | 1.1.0   |
| forcats              | 1.0.0   |
| future               | 1.31.0  |
| ggplot2              | 3.4.0   |
| grDevices            | 4.2.2   |
| graphics             | 4.2.2   |
| igraph               | 1.3.5   |
| matrixStats          | 0.63.0  |
| methods              | 4.2.2   |
| mistyR               | 1.6.0   |
| purrr                | 1.0.1   |
| readr                | 2.1.3   |
| stats                | 4.2.2   |
| stats4               | 4.2.2   |
| stringr              | 1.5.0   |
| tibble               | 3.1.8   |
| tidyr                | 1.3.0   |
| tidyverse            | 1.3.2   |
| utils                | 4.2.2   |
| zellkonverter        | 1.6.5   |




# Results

## Collective dataset characterization via the analysis of cell-type frequency distribution, feature variation analysis and immune lineage marker expression patterns

In order to apply CCC methods, the publicly available [[Hartmann-2021]]  proteomics dataset of human colon  tissue acquired via MIBI-TOF dataset was chosen and analytically explored. This dataset included a total of 58 images or fields of view (FOV) pertaining to healthy (n=2) patients and patients with colorectal carcinoma (n=2) . With an image size of 400 $\mu m^2$  spanning 1024 pixels x 1024 pixels, the resolution approximated 400 $nm^2$. The dataset measured the signal of 36 lineage and metabolic (phosphorilated) proteins (Supplementary fig. 1, 2). [[Hartmann-2021]] also provide the segmentation masks, cell clustering and annotation into 8 distinct cell-types and haematopoietic lineages (Supplementary fig. 3). Investigation of the cell-type frequency distribution showed high variability not only across samples and donors, but also between conditions (Fig. 1).

![](https://i.imgur.com/m1j9ndn.png)
> Figure 1: Annotated cell-type frequency distributions. (A) Proportion of cell-types across all samples, donors and conditions. (B) Cell-type counts grouped by condition and tumour-immune border presence as defined by [[Hartmann-2021]]. (C) Proportion of cell-types across all samples.

Next, the collective dataset (pooled single-cell data across all samples, conditions and donors) was analysed for general variance, clustering or expression patterns. Mean feature expression was calculated for each feature yielding unique expression patterns for each of the annotated cell-types (Fig. 2A).

Next, variation sources were examined via PCA. The first two principal components of PCA on the pooled single-cell data showed no distinct clustering of cells by sample, condition or donor whereas cells of the same cell-type did cluster (Fig. 2B). However, endothelial, myeloid CD68, myeloid CD11c and fibroblast cell-types did not cluster as clearly as CD4+, CD8+ T cells, endothelial and 'other immune cells'. Since the cell-type accounted for most of the variation, the original pooled feature space of the single-cell data was used for further collective dataset analysis like NCEM. Additionally, performing UMAP on the neighbourhood graph, based on the first 10 principal components of the PCA, showed clustering of the [[Hartmann-2021]] annotated cell-types, where again CD4+, CD8+ T cells, endothelial and 'other immune cells' clustered more clearly than endothelial, myeloid CD68, myeloid CD11c cells and fibroblasts.

![](https://i.imgur.com/tqoSNcm.jpg)

> **Figure 2: Data exploration of the [[Hartmann-2021]] dataset on pooled single-cell data across samples, donors and conditions.** (A) Feature expression mean was calculated by cell-type and lineage marker across all samples, donors and conditions. (B) Yuxtaposed 2-dimentional representation of PCA and UMAP dimentionality reduction coloured by cell-type, donor, sample and condition. The neighbourhood graph and UMAP  were calculated using the first 10 principal components.

The specific expression profiles of major immune cell lineage markers from the original paper were recovered via the mean expression per cell-type (Fig. 3A). `! include some marker examples !` On the UMAP space, it was expected that the expression of the lineage markers would enrich or deplete according to their cell-type. This was observed with the exception of markers CD11c, CD68 and CD31, corresponding to cell-types whose clustering was less clear in the UMAP space. Furthermore a set of 12 highly variable features were obtained that were not part of the lineage-marker set (Supplementary table 1).


![](https://i.imgur.com/dfeckf6.jpg =700x)
> **Figure 3: Lineage specific expression profiles of provided cell-type annotation by [[Hartmann-2021]].** (A) The expression mean of major immune lineage specific markers was calculated by cell-type and lineage marker across all samples, donors and conditions. (B) The neighbourhood graph and consecutive UMAP embedding was performed using the first 10 principal components and then coloured by cell-type and (C) by the major immune lineage markers.

## Exploration and characterization of colorectal carinoma and healthy samples

Next, two arbitrary samples were chosen for an exemplary analysis on an image-level basis.  "Point23" was chosen from the colorectal carcinoma samples with a tumour-immune border (defined as per [[Hartmann-2021]] ). "Point49" was chosen from the healthy colon samples.

PCA and UMAP were performed on the cancer sample to examine how much cell-type accounts for variation. The first two principal components only accounted for epithelial cell clustering. UMAP clustered CD4+, CD8+ and 'immune other' cells in addition to epithelial cells (Fig. 4A). Plotting the expression of major immune lineage markers under the cell segmentation area revealed the expected spatial distribution and enrichment linked to their cell-type (Fig 4B). In addition, highly variable genes of the sample which differed from the lineage markers were calculated (Supplementary table 1) and the expression under the segmentation mask of CD98, Ki67 and NaKATPase was plotted (Fig 5C). CD98 was expressed towards the cancerous epithelial cells with depletion in the epithelial cells. Ki67 was sparsely expressed in apparently arbitrary cell-types. NaKATPase was found to be enriched towards the cancerous epithelial cells.


![](https://i.imgur.com/LgIk40p.jpg =700x)
> **Figure 4: Exploratory data analysis of the colorectal cancer sample "Point23".** (A) Segmentation mask, PCA and UMAP coloured by cell type. UMAP was performed on the neighbourhood graph of the first ten principal components. (B) Major immune lineage marker expression under the segmentation mask. (C) Expression of highly variable features were are not lineage markers.

To contrast the characterization of the colorectal carcinoma sample, PCA and UMAP were performed on the healthy "Point49" sample (Fig. 5A). Similar to the colorectal carcinoma sample, the first two components' variation only accounted for the clustering and distinction of the epithelial cell type. Upon performing UMAP on the neighbourhood graph of the first 10 principal components, the 'immune other' cell type clustered and separated from the remaining cell-types. The latter cell-types clustered but didn't separate.


![](https://i.imgur.com/8UN6VoC.jpg =700x)
> **Figure 5: Exploratory data analysis of the healthy sample "Point49"**. (A) The segmentation mask coloured by cell-type and the first two dimentions of PCA and UMAP. (B) The segmentation mask of coloured by the expression levels of the major immune lineage markers. (C) Highly variable features not found within the lineage marker feature set.

Highly variable genes in the collective dataset (pooled single-cell data) were found in either the highly variable genes set of the colorectal carcinoma sample "Point 23" or the healthy sample "Point49". Accordingly, plotting the expression under the segmentation mask wasn't necessary.

## Spatial statistical analysis of cell-type distributions

In order to investigate CCC it is important to analyse the spatial context of the the image samples on a cell-type level. This can validate or uncover tissue and niche structures as well as hint towards cell signalling on different spatial organisational levels like yuxtacrine and paracrine signalling between cell types.

First the connectivity graphs of both the colorectal carcinoma sample "Point23" and the healthy sample "Point49" were calculated (Fig. 6 A) by setting the neighbourhood size to a radius of 35 px (~14µm), as was calculated by [[Fischer-2022]] to yield the best predictive performance on this dataset (explained variance, R2) as a hole and by cell-type.

Given the anatomy of the colon, we expected a clear epithelial wall to be present in both carcinoma and healthy samples and therefore the epithelial cells to cluster. Similarly, we expected immune cells and possibly fibroblasts to aggregate close to the epithelium forming the lamina propria. Ripley's L statistic was computed to analyse the relative distribution of cells of the same annotated cell type (Fig. 6B). Indeed in both the carcinoma and healthy samples, CD4+ T cells and other CD45+ immune cells showed the highest score indicating higher relative clustering. In the carcinoma sample the CD4+ T cells had the highest score the followed by other CD45+ T cells. In the healthy sample the order was reversed. In both samples epithelial cells also showed a relatively high Ripley's L score ranked 3rd in both conditions. Furthermore, the clustering coefficient centrality scores were computed, which measure the degree of node clustering in the spatial graph for a cell-type annotations. The obtained scores in the colorectal carninoma sample showed higher clustering scores in immune cells compared to non-immune cells (endothelial, epithelial and fibroblasts) (Fig. 6C). Clustering coefficients in the healthy sample showed fibroblasts to be less clustered than the rest of the cells.

Once the clustering patterns were analysed, relative spatial proximity between cell-types was measured via neighbourhood enrichment analysis and co-occurence. Neighbourhood enrichment yielded in both carcinoma and healthy conditions higher values in the diagonal than other cell-type pairs (Fig. 6D). In the colorectal carcinoma sample epithelial and other CD45+ cells showed noticeably intra-cell-type high enrichment scores indicating their respective clustering. Similarly in the healthy sample showed high enrichment of between same cell-type pairs CD4+, CD8+ T cells and CD11c+, CD68+ myeloid cells. Additionally CD4+ and CD8+ T cells showed to be mutually enriched. CD11c+ and CD68+ myeloid cells pair also showed high enrichment scores. With respect to cell-type proximity to the epithelial wall, the co-occurence score was computed for all cell-types given the presence of epithelial cells for increasing distance (Fig 6 E). Applied on the colorectal carcinoma sample, it was shown that fibroblasts were more likely to be found near epithelial cells at close distance (400 px ≈ 156 µm) than at further distance (600 px ≈ 234 µm). Epithelial cell co-occurence showed high vales in this interval which declined with distance, further showing epithelial cell clustering. CD4+ and CD8+ T cells showed a constant co-occurence score while other CD45+ cells co-occurence increase with distance to epithelial cells. Co-occurence scores on the healthy colon sample showed high values in the close range (100 px ≈ 39 µm) for epithelial, endothelial, CD11c+ and CD68+ myeloid cells. These scores quickly declined to a plateau at (400 px ≈ 156 µm). Epithelial cells had noticably higher co-occurrence scores in at 39 µm radius indicating clustered behaviour.

![](https://i.imgur.com/t8r3k3s.jpg =700x)
![](https://i.imgur.com/Pe8bJnV.jpg =600x)
> **Figure 6: Spatial statistical analysis of annotated cell-type distribution on the spatial connectivity graph.** Colorectal carcinoma sample "Point23" localed on the left and the healthy sample "Point 49" on the right. (A) Spatial connectivity graph calculated using a radius threshold of 35 px (~14 µm). (B) Ripley's L spatial distribution statistic (C) Clustering coefficient scores computed via Squidpy (C) Neighbourhood enrichment analysis (E) Co-occurence score


## Spatially variable features

In order to obtain a fuller context on a spatial level, the analysis of spatial cell-type distribution was complemented with analysis of spatially variable features on the colorectal carcinoma sample "Point23" and healthy colon sample "Point49". This was achieved via the [[Morans I]] autocorrelation, which describes the level of dispersion and clustering of a feature across space. It was predicted that the major immune lineage markers would score a Moran's I value closer to 1 since they should enrich in their in their corresponding cell-types according to their unique profile (Fig. 3A). In addition, highly variable genes found previously were hypothesized to show some spatial patterning. Indeed, in the colorectal cancer sample, four of the ranked features by Moran's I score were from the lineage marker set (descending order: CK, CD45, CD11c, CD4) (Fig. 3A, 4B). Furthermore, two of the top ranked features pertained to the previously calculated highly variable features (CD98, NaKATPase) (Fig. 4C, Supplementary table 1) and four novel features were obtained (GLUT1, HK1, LDHA, PKM2) (Fig. 7). From the top ten ranked features by Moran's I score on the healthy colon sample, 7 pertained to the major immune lineage marker set (descending order: CK, CD45, CD11c, vimentin, Ecadherin, CD14, CD3) (Fig. 3A, 5B), 3 formed part of the preiously calculated highly variable features (CD98, Ki67, NaKATPase) (Fig. 5C, Supplementary table 1) and no novel features with distinct expression in space were found.

![](https://i.imgur.com/j2Ruu4l.jpg)
> **Figure 7: Novel top ranked features by Moran's I score on colorectal carcinoma sample "Point23"**. Segmentation mask coloured of "Point23" coloured by the expression of features scoring top 10 in the Moran's I analysis and were not part of either the previously investigated immune lineage markers or the highly variable features calculated for the present sample.


## Cell-type-based sender-receiver effects calculation via NCEM interaction-terms

Once the dataset had been characterized on the spatial and non-spatial level, the CCC methods were applied.

First, the NCEM linear model was applied to the entire dataset to obtain the interaction term coefficients, also referred to as sender-receiver effects. They are obtained asymmetrically for each sender-receiver cell-type pair and are specific to the directional interaction (dimensions: receiver cell-type x sender cell-type x features). Alongside the interaction term coefficients, FDR-corrected significance values for each interaction term were obtained.

To summarize the three-dimensional tensor, two approaches were followed. First, [[Fischer-2022]]s approach was followed, in which the number of significant interaction terms after performing a Wald-test are quantified via the L1-norm of the FDR-corrected p-value tensor. The user can set the significance threshold and an additional threshold for minimum number of obtained significant features (Fig. 8A). Sender epithelial cells showed noticeably high interaction values on receiving CD4+ T cells. Sender CD4+ T cells showed high values when paired with CD11c, CD68 myeloid cells and other CD45+ immune cells. Other CD45+ immune cells showed high valued-interactions both as sender and receiver cells.

The second approach was aimed to be less reliant on discrete values based on a significance threshold and allow for the value of the interaction term to shape the summary statistic. This was achieved via the application of the L2-norm on the significance filtered interaction term tensor. The user can define a range- or standar-deviation-based threshold value for the interaction terms as well as a threshold for the final L2-norm obtained values. The latter threshold a high-pass threshold or a percentile-based threshold. Similar to the previous significance-based approach, this one showed a high interaction term value for the sender epithelial and receiver CD4+ cells. In contrast to the previous method, for the sender epithelial cells, a high value for the interaction with CD11c+ myeloid cells was observed. Another dissimilarity was the high obtained value for the endothelial-epithelial cell-types pair. When observing the sender effect of CD4+ T cells, their interaction coefficients on all other cell-types were not as high as in the previous approach relative to all the distribution of values. In other words, the relative effect of CD4+ T cells on all other cells was decreased in the L2-norm approach in comparison to the significance based approach. The same relative effect reduction could also be observed on both the sender effect of other CD45+ immune cells on all other cell-types and the receiver effect of other CD45+ immune cells by all other immune cells. The only noticeably high values that were constant in both approaches were the other CD45+ immune cells effect on CD11c+ immune cells and the receiving effect of other CD45+ immune cells by CD8+ T cells.

Since the linear NCEM model is reliant on correct cell-type annotation it was hypothesized that simulating false cell-type annotation due to upstream artifacts or processing steps would greatly impact the output of the model. Hence, NCEM was applied on shuffled cell-type annotations in varying fractions of random cells (fractions: 0%, 0.1%, 1%, 10%, 50% and 100%; shuffling repetitions for each fraction: n=1) (Fig. 8B). As expected, the explained variance ($R^2$) already decreased in the interval between none and 1% fraction shuffling. The decrease of $R^2$  between the shuffling of 1% of the cells and 10% accounted for most of the explained variance by the model, as 50% shuffling reached the same level of $R^2$ as completely randomized cell-type annotation.

![](https://i.imgur.com/OvW65Rj.jpg =600x)
> **Figure 8: Application of linear NCEM.** (A) Sender-receiver effects obtained via two approaches. Two interaction term summary approaches (1) Quantifying the significant features per cell-type pair. The FDR-corrected p-value tensor was filtered by a significance threshold of 0.05 and the consecutively the L1-norm was calculated (Left) (2) Applying the L2-norm on the interaction terms coefficients tensor, previously filtered by a significance threshold of 0.05 (Right). (B) NCEM application on shuffled cell-type annotations by different fractions of random cells whose annotations were randomized. For each box, the centerline defines the median of the explained variance ($R^2$), the height of the box represents the interquartile range (IQR) and the whiskers denote the 1.5 * IQR.


## Analysis of spatial dependencies via MISTy

The next step was to apply MISTy to the dataset. MISTy is a predictive machine learning framework which defines "views" as spatial contexts that are to be used to predict the feature expression of the index cells. In our specific case, to not deviate too much from the NCEM concept of nighbourhoods, only one view was defined apart from the standard "intra-view". This "para-view", was constructed by adding the feature expression of cells within a certain radius 35 px (~14µm) of the index cell in a weighted manner (gaussian kernel). MISTy was run using an ensemble random forest algorithm for F().  Three types of data were collected (1) the gain of explained variance ($R^2$) achieved by comparing the $R^2$ of the model with solely the "intra-view" and the $R^2$ of the model that includes  both the "intra-view" and the "para-view" (Fig. 9A) (2) the contribution of each view to the metamodel in form of the learnable weight parameters (Fig. 9B) and (3) the improvement of the prediction of target features in the index cell via a leave-one-out procedure of predictor features in the "para-view" (Fig. 9C). MISTy was run on an individual sample level and the results where aggregated via the mean and standard deviation summary statistics.

The gain in $R^2$ of the aggregated results showed very high standard deviation, often accounting for its entire effect. SMA, PD1 and H3 were the top ranked proteins that contributed to the gain in $R^2$ by ~2.5 fold increase. Next, the results on the individual colorectal carcinoma sample "Point23" was examined yielding CD8, H3 and CD11c to be the top ranked proteins improving $R^2$ by ~3 fold. When examining the healthy colon sample "Point49", PD1, CD11c and Ki67 were the proteins that improved the $R^2$ the most by over a 4 fold increase.

The contributions of each view to the metamodel for each protein showed, as expected, that most of the contribution (consistently >50%) comes from the "intra-view" or from the index cell feature state when regarding the aggregated results. Some of the proteins that showed the highest contribution of "para-view" were CD11c, CD8 and HIFA. In the colorectal carcinoma sample HIFA and KI67 showed a higher contribution of the "para-view" than the "intra-view". In the healthy colon sample, only CD8 showed more than half of the contribution via the "para-view". 

The feature importance results of MISTy showed sparse results in the aggregated, carcinoma sample and healthy sample results. CK was a predictor with noticeably many of the highest values across the three latter categories. Predictor CD11c showed many high values in the colorectal carinoma and healthy samples.

![](https://i.imgur.com/QlbNoLu.jpg =500x)
![](https://i.imgur.com/U82vkVZ.jpg =800x)
> **Figure 9: MISTy analysis of spatial dependencies by defining the "intra-" and "para-view"**. The results are shown for three categories (1) the aggregated results over all samples (Top), (2) the results on the individual colorectal carcinoma sample "Point23" (Middle) and (3) the results on the individual healthy colon sample "Point49" (Bottom). (A) Gain in explanation of variance when comparing the MISTy model solely with the "intra-view" and the MISTy model including both the "intra-" and the "para-view". (B) The learnable parameters of the metamodel for each view for every feature. (C) The importances contrast heatmap. The importance of each feature to the prediction of the individual targets from the "intra-view" are substracted from the importances calculated for the "intra-view".

The obtained results via MISTy were not suitable for comparison with the sender-receiver effects calculated via NCEM due to format disparity. Since the overarching goal was to to lay the groundwork for future benchmarking and evaluation of CCC methods, an approach was devised to adapt the MISTy workflow to yield comparable output. The approach consisted of transforming the input of MISTy from the feature expression matrix to the one-hot encoding matrix of the own index cells, yielding a comparable output (Fig. 10). In essence, the prediction approach changed to predicting the index cell-type based on the neighbourhood cell-type weighted abundance. The "intra-view" was bypassed to avoid illogical prediction predictions of cell-type within the index cell. The "para-view" was additionally modified to use a constant kernel (sum of neighbourhood cell-types across cells).

The variance explained $R^2$ was not a contrast in this case, since there was no "intra-view" to substract. Nevertheless, it showed very high standard deviation when considering the aggregation of this measure across all samples (Fig. 10A). In all three conditions, aggregated samples, the colorectal carcinoma and healthy sample, "other immune CD45+ cells" showed the highest $R^2$ with CD4+ T-cells being ranked second in the cancer and healthy samples.

Upon examining the contributions of the views for each cell type, as expected, all the contribution was attributed to the "para-view" (Fig. 10B), since there was no learnable contribution weight for the "intra-view".

The importances output showed no positive importance for any cell-type pair for the aggregated statistic (Fig. 10C). For the colorectal carcinoma sample, predictor endothelial cell-type showed interaction values for targets "other immune CD45+ cell", fibroblasts and CD4+ T cells.  In contrast, when examining the results on the healthy colon sample, predictor CD11c+ myeloid cells had a positive interaction value for epithelial, "other immune CD45+", and CD4+ T cells. Also, predictor cell-type CD4+ T cells showed positive interactions with target CD11c+ myeloid, endothelial and fibroblast cells.

![](https://i.imgur.com/8zIafEu.jpg =500x)
![](https://i.imgur.com/whZqGmY.jpg =800x)
![](https://i.imgur.com/hxlSO20.jpg =500x)
> **Figure 10: MISTy workflow modified to input cell-types.** Three different result categories were shown: (1) aggregated results across all samples (Top), (2) application of MISTy to colorectal carcinoma sample "Point23" (Middle) and (3) MISTy applied to healthy colon sample "Point49" (Bottom). (A) Gain in variance explained ($R^2$) when the MISTy model includes the "para-view" in comparison to only the "intra-view". (C) The importances heatmap of the "para-view". Importances are obtained for each target and predictor pair by a 

## MISTy $R^2$ at different shuffling fractions

## (Training NCEM and MISTy on all data and testing on all and by condition in a combinatorial fashion)



# Discussion

In the present study spatial cell-cell communication methods NCEM and MISTy were applied and evaluated for bechmarking on the [[Hartmann-2021]] colorectal cancer MIBI-TOF dataset. Applying the linear NCEM model output the interactin terms' coefficients, also referenced as sender-receiver effects. These were further summarized via two approaches: (1) quantifying the amount of significant features per sender-receiver cell-type pair and (2) via calculating the L2-norm of interaction terms matrix which had previously been filtered by a significance threshold. There were some distinct interactions found in both summarizing statistics, for example the interaction between sender epithelial and receiver CD4+ T cells. However, there were also dissimilarities present as the interaction between epithelial and CD11c myeloid cells was not as distinct in the significant in approach 1 compared to approach 2. Also, in the first approach, other CD45+ immune cells showed high interactions with most cell-types both as sender and receiver whilst in approach 2, most of these interactions were diluted. None of these interactions can be validated due to the lack of ground truth. Common upstream errors and artifacts are one of the major challenges in the spatial-omics field. For example incorrect feature detection attribution to its corresponding cell due to imprecise cell-segmentation can lead to incorrect cell-type labeling. We analyzed the sensitivity of the linear NCEM model towards cell-type labels by shuffling the cell-type of varying fractions of randomly selected cells. The high sensitivity even at low shuffling fractions like 10%, which is common in spatial-omics processing, indicates that variation in the preprocessing steps will have a great effect on linar NCEM output and that both the spatial technology used to obtain the data and the pre-processing steps effectuated before NCEM application must be assessed carefully to account for false annotation.

Next, MISTy was applied. The flexibility of of MISTy to define the views and indpendence of cell-type annotation was the key reason for its application. Defining the "para-view" as the weighted sum of the neighbourhood features promised independence of cell-type annotation. However, a different set of problems were encountered upon application. First, MISTy natively runs on on individual samples and aggregates their results via summary statistics like the mean and standard deviation. The standard deviation obtained for the increase of explained variance ($R^2$) was very high and virtually always in the range of the own mean value. Also, the top ranked features for this result varied greatly when inspecting individual samples (e.g. colorectal carcinoma and healthy colon). The metamodel learnable parameters for each view, also references as contributions, showed continuously higher contributions from the "intra-view" than the "para-view" with exceptions in the individual sample resolution. The higher HIFA and Ki67 contributions to the "para-view" effect found in the colorectal carcinoma sample "Point 23" as compared to the healthy colon sample "Point49" and the aggregated results could be a hint toward the effect of the cancer. However, making a statement would be speculative due to the lack of ground truth. Similar conclusions were arrived at while inspecting the feature importances output of MISTy.

The second challenge was the lack of comparable output to the output of linear NCEM. To this aim, the MISTy workflow was adapted to use the one-hot encoding of the index cell-types as input and bypass the intraview. As such, the model would predict the index cell-type based on the count of neighbouring cell-types. Despite the variance explanation ($R^2$) not being a contrast of the "intra-view" against the "para-view", the top 3 ranked cell-types showed higher robustness in top ranked cell-types. The output in comparable format to NCEM was the importances between the cell-type pairs. For the aggregated results, no importances were found, while the individual examples showed that endothelial was a predominant predictor in the colorectal carcioma sample and CD4+ T-cells and CD11c+ myeloid cells were predominant in the healthy colon sample.

The application of MISTy showed the need for a better way to integrate different samples, donors and conditions. Alternatively, the user must have a precise comprehension about the biology and experimental setup to apply MISTy in a tailored manner and make any statements based on MISTy's output. MISTy's flexibility allowed for shaping of the output to our specific purposes. This came at the cost of low interpretability. Again, validation of the results was impossible due to the lack of ground truth.

In contrast to MISTy, NCEM models the sample or batch directly into the linear model. This could be practical to integrate samples. However, accounting for conditions in this way could influence the interaction terms and the final output. Similarly to MISTy, it is evident that the user must have a good understanding of the experimental setup to judge if all samples and conditions are to be included in the application of linear NCEM.

In order to critically assess these methods, the dataset was also statistically characterized on the spatial and single-cell level. Cell-type frequency distribution showed a high variability in cell-type frequencies across all samples and also between conditions which may lead to unbalanced model training in NCEM and MISTy, especially the latter since it is applied on an individual sample basis. By summarizing the mean feature expression of cell-types across all samples, the major immune lineage profiles were recovered based on specific markers. However, this wasn't further PCA analysis, clustering and UMAP embedding did not separate these cell-types clearly, indicating possible upstream annotations errors or the need for better sample integration. Spatial statistics verified the clustering of epithelial cells and immune cells as per colon tissue structure. In addition, Moran's I autocorrelation analysis uncovered  feature with clustered distribution patterns which differed from the lineage markers: GLUT1, HK1, PKM2 and LDHA in the colorectal carcinoma sample "Point23". GLUT1, pyruvate kinase M2 (PKM2) and lactate dehydrogenase A (LDHA),  were enriched in most cells except cancerous epithelial cells. They are key proteins in channels and enzymes in the energy metabolism, possibly indicating the activation of immune cells near to the tumour-immune border.

In the big picture, the study highlights the lack of a consensus on the approach to quantify of cell-cell communication effects which needs urgent reevaluation to adapt to the latest approaches and technologies. Dependence of spatial CCC methods from upstream steps and artifacts was also a repeated challenge throughout the study which was shown to contribute to unaccounted variation. The high number of uncertainty sources leads to the necessity of the establishment of new protocols, redefinition of concepts  and the development of new methods that take the latter factors into account. The lack of ground truth will also continue to curse spatial single-cell benchmarking and evaluation.



To continue the investigation, the oulook includes the application of other spatial CCC methods with different approaches. Non-predictive approaches like DIALOGUE, COMMOT and possibly other non-spatial methods (e.g. CellChat) could leverage other cell properties. Methods that integrate prior knowledge like ligand-receptor interactions have great potential (e.g. CellPhoneDB v3). The comparison and benchmarking of CCC methods and the effect of upstream steps would remain the main focus. In-silico simulated spatial data could be used to bechmark the methods. Also CCC methods could be applied to more datasets of different technologies and omics to evaluate the robustness and flexibility of the CCC methods towards different technologies and omics. Also, applying NCEM and MITy on the PCA space could be analysed for predicitve performance improvement. 

#### Cross condition testing



# Code availability

All the analysis, code, conda environment data, figures, input and output data are publically available at [https://github.com/idf-io/SpatialPipe](https://github.com/idf-io/SpatialPipe).



# Data availability

The colorectal carcinoma MIBI-TOF dataset (Methods) is publicably available at [https://zenodo.org/record/3951613](https://zenodo.org/record/3951613).



# Supplementary information

![](https://i.imgur.com/H6wjCLg.jpg =700x)
> **Supplementary figure 1: MIBI-TOF image of colorectal cancer sample "Point23".** All 36 detected protein channels using an inverted gray-scale colour map for the 255 bit intensities. For visualization purposes, the images were log1p transformed.

![](https://i.imgur.com/RxdMQtT.jpg =700x)
> **Supplementary figure 2: MIBI-TOF image of the healthy colon sample "Point49".** Represented are the images of all 36 channels showing the signal of the respective proteins. The images were log1p transformed and represen

> **Supplementary figure 3:** Segmentation masks of the MIBI-TOF [[Hartmann-2021]] dataset. 


> **Supplementary figure 4: UMAP of lineage markers per sample condition** 


**Table 1: Highly variable genes on pooled data and subsampled data by condition**

| | Global | Colorectal carcinoma | Healthy samples|
|:----------|---------:|---------:|----------:|
| ASCT2 |  |  |  |
| ATP5A |  |  |  |
| CD11c |  |  |  |
| CD14 | X |  | X |
| CD3 | X | X | X |
| CD31 |  |  |  |
| CD36 | X |  | X |
| CD39 |  |  |  |
| CD4 |  |  | X |
| CD45 | X | X | X |
| CD57 |  |  |  |
| CD68 |  |  |  |
| CD8 |  | X |  |
| CD98 | X | X | X |
| CK | X |  |  |
| CPT1A |  |  |  |
| CS |  |  |  |
| Ecad |  |  |  |
| G6PD | X | X | X |
| GLUT1 |  |  |  |
| H3 | X | X | X |
| HIF1A | X | X | X |
| HK1 | X |  | X |
| IDH2 |  |  |  |
| Ki67 | X | X | X |
| LDHA |  |  | X |
| NRF2p | X | X |  |
| NaKATPase | X | X | X |
| PD1 |  |  |  |
| PKM2 | X |  | X |
| S6p | X | X | X |
| SDHA |  | X |  |
| SMA |  | X |  |
| VDAC1 | X | X |  |
| XBP1 |  | X |  |
| vimentin |  | X | X |


**Table 2: Moran I scores and their respective FDR corrected p-values under the assumption of normality applied to colorecatal cancer sample "Point23" and healhty colon sample "Point49"**

| | I_cancer | pval_norm_fdr_bh_cancer | I_heathy | pval_norm_healthy_fdr_bh |
|:----------|-----------:|--------------------------:|-----------:|---------------------------:|
| ASCT2 | 0.434129 | 0 | 0.198777 | 0 |
| ATP5A | 0.258313 | 0 | 0.352868 | 0 |
| CD11c | 0.764473 | 0 | 0.664476 | 0 |
| CD14 | 0.451225 | 0 | 0.593698 | 0 |
| CD3 | 0.46889 | 0 | 0.575716 | 0 |
| CD31 | 0.378775 | 0 | 0.237861 | 0 |
| CD36 | 0.487321 | 0 | 0.185019 | 0 |
| CD39 | 0.552447 | 0 | 0.332658 | 0 |
| CD4 | 0.601358 | 0 | 0.511698 | 0 |
| CD45 | 0.781277 | 0 | 0.755632 | 0 |
| CD57 | 0.0119118 | 0.240685 | 0.096923 | 5.7088e-11 |
| CD68 | 0.390981 | 0 | 0.363916 | 0 |
| CD8 | 0.303137 | 0 | 0.188665 | 0 |
| CD98 | 0.586363 | 0 | 0.591238 | 0 |
| CK | 0.861324 | 0 | 0.870824 | 0 |
| CPT1A | 0.387363 | 0 | 0.439394 | 0 |
| CS | 0.429202 | 0 | 0.446503 | 0 |
| Ecad | 0.450744 | 0 | 0.617346 | 0 |
| G6PD | 0.300406 | 0 | 0.120117 | 7.05318e-16 |
| GLUT1 | 0.591208 | 0 | 0.557849 | 0 |
| H3 | 0.418025 | 0 | 0.516858 | 0 |
| HIF1A | 0.0297889 | 0.0451517 | 0.386171 | 0 |
| HK1 | 0.616579 | 0 | 0.404228 | 0 |
| IDH2 | 0.405139 | 0 | 0.300103 | 0 |
| Ki67 | 0.145286 | 3.33067e-16 | 0.572123 | 0 |
| LDHA | 0.715174 | 0 | 0.528124 | 0 |
| NRF2p | 0.27139 | 0 | 0.254227 | 0 |
| NaKATPase | 0.599137 | 0 | 0.704375 | 0 |
| PD1 | 0.428382 | 0 | 0.400004 | 0 |
| PKM2 | 0.670421 | 0 | 0.561082 | 0 |
| S6p | 0.477971 | 0 | 0.510066 | 0 |
| SDHA | 0.254028 | 0 | 0.320846 | 0 |
| SMA | 0.419539 | 0 | 0.444856 | 0 |
| VDAC1 | 0.323745 | 0 | 0.137777 | 0 |
| XBP1 | 0.159283 | 0 | 0.0620587 | 1.69041e-05 |
| vimentin | 0.507407 | 0 | 0.