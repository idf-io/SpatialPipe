# Application and evaluation of spatially-resolved cell-cell communication methods NCEM and MISTy on MIBI-TOF colorrectal cancer data
> [!info]- Title alternatives
> - SpatialPipe: Systematic standardization and framework integration of public spatial-omics datasets for cell-cell communication analysis
> - Application and evaluation of NCEM and MISTy on spatially-resolved single-cell data
> - A comprehensive and sistematic integration of spatial-omics data and evalutaion of spatially resolved [[cell-cell communication]] methods
### [[Report notes - Internship Stegle| Report notes]]

AS IF IM TEACHING OF MY SUBJECT


![[spatialpipe_logo.png| 500]] 

## Abstract

Prior to the emergence of spatial-omics technologies, [[cell-cell communication]] methods relied on prior knowledge about interacting features between cells. The measure of correlation between putative ligand-receptor interactions are a common approach to this end. Novel spatial-omics technologies like MIBI-TOF and MERFISH leverage the spatial-dimension to measure and characterize emerging feature patterns along the aforementioned dimension. Often times a "niche" radius must previously be defined to 

In the present study, novel [[Machine learning|machine learning]] based [[cell-cell communication]] methods NCEM and MISTy are applied to a colorrectal tumour dataset and evaluated in their applicability on [[cell-cell communication]] profiling.

The sources of uncertainty are still many which leads to the necessity of the establishment of protocols, redefinition of concepts (e.g. [[cell-cell communication]]) and the development of new methods that take the latter factors into account.


```toc
```

Focus on upstream
## Introduction










- Spatial omics technology emergence an`d evolution in context of single-cell technologies.
	- Very short 1 paragraph, directing to CCC. SO helps us because... maybe with some examples WNT signalling HARTMANN findings
	- Include a technology comparison
		- Resolution
		- Multiplexity
		- Omics type
		- Scope
		- Nr spots/cells
		- Sensitivity
	- Main pro's and cons
	- [[Svensson-2018]]
	- [Giesen2014]( paperhttps://www.nature.com/articles/nmeth.2869) Imaging Mass Spectrometry for spatial data first
	 - [Jackson2020](https://www.nature.com/articles/s41586-019-1876-x) recent Imaging Mass Cytometry spatial data.
	- [Slideseq2](https://www.nature.com/articles/s41587-020-0739-1)

- Cell-cell communication technologies with their pro's and con's
	- Include a methods comparison table with pros and cons
	- Categorize on the (non-)spatial levela and inclusion/exclusion of prior knowledge
	- Core assumptions and problem
		- Major: no spatial dimension/information, so only partial interactions are measured and probably many FP
- [[cell-cell communication]] importance
		- [[Severvekova-2023]]

- Spice it up with CCC bio cancer, 
	- Read TME paper elyas

- Introduce NCEM
	- Pro's
	- Con's
	- Assumptions
	- Parameters

- Introduce MISTY
	- Same as NCEM

- Explain intention (research question) and core objective of the project
	- In the present study, the advantages and the limitations of NCEM and MISTy were explored in the context of spatially-resolved [[cell-cell communication]] profiling methods. They were chosen as cutting-edge representatives of the [[cell-cell communication]] field due to .(e.g. no need of prior knowledge except cell type).. NCEM and MISTy were applied to MIBI-TOF cell-level resolution proteomics dataset of humancolorrectal cancer tissue [[Hartmann-2021 DS| Hartmann-2021]]. 


## Materials and methods

### Datasets
[[Hartmann-2021]]
List of markers

### Single-cell pre-processing
- Scaling
- Transformation
- Integration
- Clustering and data visualisation
	- UMAP[[Hartmann-2021]]
- Image manipulation

- Segmentation and clustering was provided via FlowSOM
### [[Fischer-2022|NCEM]]
- Linear method
	- Model
	- Input
	- Output
	- Experience
- Wrapper

### [[Tanevski-2020|MISTy]] 
- Metamodel
	- Input
		- Main
		- Secondary
	- Output
	- Coersion
	- Experience
- Wrapper

### Statistical processes
e.g. significance, wald test and correction etc

### Software
- R, Python, Conda, Jupyter-lab
- Respective base packages
- Other necessary imported packages like numpy
- MISTy, NCEM, Scanorama, Squidpy, Tensorflow, Scanpy




## Results

### Collective dataset characterization via the analysis of cell-type frequency distribution, feature variation analysis and immune lineage marker expression patterns

In order to apply CCC methods, the publicly available [[Hartmann-2021]]  proteomics dataset of human colon  tissue acquired via MIBI-TOF dataset was chosen and analytically explored. This dataset included a total of 58 images or fields of view (FOV) pertaining to healthy (n=2) patients and patients with colorectal carcinoma (n=2) . With an image size of 400 $\mu m^2$  spanning 1024 pixels x 1024 pixels, the resolution approximated 400 $nm^2$. The dataset measured the signal of 36 lineage and metabolic (phosphorilated) proteins (Supplementary fig. A). [[Hartmann-2021]] also provide the segmentation masks, cell clustering and annotation into 8 distinct cell-types and haematopoietic lineages (Supplementary fig. a). **Investigation of the cell-type frequency distribution showed high variability not only across samples and donors, but also between conditions (Fig. 1).**

![[cell-type_frequency_distributions 1.png]]
> Figure 1: Annotated cell-type frequency distributions. (A) Proportion of cell-types across all samples, donors and conditions. (B) Cell-type counts grouped by condition and tumour-immune border presence as defined by [[Hartmann-2021]]. (C) Proportion of cell-types across all samples.

Next, the collective dataset (pooled single-cell data across all samples, conditions and donors) was analysed for general variance, clustering or expression patterns. Mean feature expression was calculated for each feature yielding unique expression patterns for each of the annotated cell-types (Fig. 2A).

Next, variation sources were examined via PCA. The first two principal components of PCA on the pooled single-cell data showed no distinct clustering of cells by sample, condition or donor whereas cells of the same cell-type did cluster (Fig. 2B). However, endothelial, myeloid CD68, myeloid CD11c and fibroblast cell-types did not cluster as clearly as CD4+, CD8+ T cells, endothelial and 'other immune cells'. Since the cell-type accounted for most of the variation, the original pooled feature space of the single-cell data was used for further collective dataset analysis like NCEM. Additionally, performing UMAP on the neighbourhood graph, based on the first 10 principal components of the PCA, showed clustering of the [[Hartmann-2021]] annotated cell-types, where again CD4+, CD8+ T cells, endothelial and 'other immune cells' clustered more clearly than endothelial, myeloid CD68, myeloid CD11c cells and fibroblasts.

![[global_expression_pca_umap.png]]
> **Figure 2: Data exploration of the [[Hartmann-2021]] dataset on pooled single-cell data across samples, donors and conditions.** (A) Feature expression mean was calculated by cell-type and lineage marker across all samples, donors and conditions. (B) Yuxtaposed 2-dimentional representation of PCA and UMAP dimentionality reduction coloured by cell-type, donor, sample and condition. The neighbourhood graph and UMAP  were calculated using the first 10 principal components.

The specific expression profiles of major immune cell lineage markers from the original paper were recovered via the mean expression per cell-type (Fig. 3A). On the UMAP space, it was expected that the expression of the lineage markers would enrich or deplete according to their cell-type. This was observed with the exception of markers CD11c, CD68 and CD31, corresponding to cell-types whose clustering was less clear in the UMAP space. Furthermore a set of 12 highly variable features were obtained that were not part of the lineage-marker set (Supplementary table 1).

![[lineage_profile_recovery.png]]
> **Figure 3: Lineage specific expression profiles of provided cell-type annotation by [[Hartmann-2021]].** (A) The expression mean of major immune lineage specific markers was calculated by cell-type and lineage marker across all samples, donors and conditions. (B) The neighbourhood graph and consecutive UMAP embedding was performed using the first 10 principal components and then coloured by cell-type and (C) by the major immune lineage markers.

### Exploration and characterization of colorectal carinoma and healthy samples

Next, two arbitrary samples were chosen for an exemplary analysis on an image-level basis.  "Point23" was chosen from the colorectal carcinoma samples with a tumour-immune border (defined as per [[Hartmann-2021]] ). "Point49" was chosen from the healthy colon samples.

PCA and UMAP were performed on the cancer sample to examine how much cell-type accounts for variation. The first two principal components only accounted for epithelial cell clustering. UMAP clustered CD4+, CD8+ and 'immune other' cells in addition to epithelial cells (Fig. 4A). Plotting the expression of major immune lineage markers under the cell segmentation area revealed the expected spatial distribution and enrichment linked to their cell-type (Fig 4B). In addition, highly variable genes of the sample which differed from the lineage markers were calculated (Supplementary table 1) and the expression under the segmentation mask of CD98, Ki67 and NaKATPase was plotted (Fig 5C). CD98 was expressed towards the cancerous epithelial cells with depletion in the epithelial cells. Ki67 was sparsely expressed in apparently arbitrary cell-types. NaKATPase was found to be enriched towards the cancerous epithelial cells.

![[cancer_sample_exploration.png]]
> Figure 4: Exploratory data analysis of the colorectal cancer sample "Point23". (A) Segmentation mask, PCA and UMAP coloured by cell type. UMAP was performed on the neighbourhood graph of the first ten principal components. (B) Major immune lineage marker expression under the segmentation mask. (C) Expression of highly variable features were are not lineage markers.

To contrast the characterization of the colorectal carcinoma sample, PCA and UMAP were performed on the healthy "Point49" sample (Fig. 5A). Similar to the colorectal carcinoma sample, the first two components' variation only accounted for the clustering and distinction of the epithelial cell type. Upon performing UMAP on the neighbourhood graph of the first 10 principal components, the 'immune other' cell type clustered and separated from the remaining cell-types. The latter cell-types clustered but didn't separate.


![[healthy_sample_exploration.png]]
> **Figure 5: Exploratory data analysis of the healthy sample "Point49"**. (A) The segmentation mask coloured by cell-type and the first two dimentions of PCA and UMAP. (B) The segmentation mask of coloured by the expression levels of the major immune lineage markers. (C) Highly variable features not found within the lineage marker feature set.

Highly variable genes in the collective dataset (pooled single-cell data) were found in either the highly variable genes set of the colorectal carcinoma sample "Point 23" or the healthy sample "Point49". Accordingly, plotting the expression under the segmentation mask wasn't necessary.

### Spatial statistical analysis of cell-type distributions

In order to investigate CCC it is important to analyse the spatial context of the the image samples on a cell-type level. This can validate or uncover tissue and niche structures as well as hint towards cell signalling on different spatial organisational levels like yuxtacrine and paracrine signalling between cell types.

First the connectivity graphs of both the colorectal carcinoma sample "Point23" and the healthy sample "Point49" were calculated (Fig. 6 A) by setting the inclusion radius parameter to 35 px (~14µm), as was calculated by [[Fischer-2022]] to yield the best predictive performance on this dataset.

Given the anatomy of the colon, we expected a clear epithelial wall to be present in both carcinoma and healthy samples and therefore the epithelial cells to cluster. Similarly, we expected immune cells and possibly fibroblasts to aggregate close to the epithelium forming the lamina propria. Ripley's L statistic was computed to analyse the relative distribution of cells of the same annotated cell type (Fig. 6B). Indeed in both the carcinoma and healthy samples, CD4+ T cells and other CD45+ immune cells showed the highest score indicating higher relative clustering. In the carcinoma sample the CD4+ T cells had the highest score the followed by other CD45+ T cells. In the healthy sample the order was reversed. In both samples epithelial cells also showed a relatively high Ripley's L score ranked 3rd in both conditions. Furthermore, computing the clustering coefficient centrality scores of the cell-types in the colorectal carninoma sample showed higher clustering scores in immune cells compared to non-immune cells (endothelial, epithelial and fibroblasts) (Fig. 6C). Clustering coefficients in the healthy sample showed fibroblasts to be less clustered than the rest of the cells.

Once the clustering patterns were analysed, relative spatial proximity between cell-types was measured via neighbourhood enrichment analysis and co-occurence. Neighbourhood enrichment yielded in both carcinoma and healthy conditions higher values in the diagonal than other cell-type pairs (Fig. 6D). In the colorectal carcinoma sample epithelial and other CD45+ cells showed noticeably intra-cell-type high enrichment scores indicating their respective clustering. Similarly in the healthy sample showed high enrichment of between same cell-type pairs CD4+, CD8+ T cells and CD11c+, CD68+ myeloid cells. Additionally CD4+ and CD8+ T cells showed to be mutually enriched. CD11c+ and CD68+ myeloid cells pair also showed high enrichment scores. With respect to cell-type proximity to the epithelial wall, the co-occurence score was computed for all cell-types given the presence of epithelial cells for increasing distance (Fig 6 E). Applied on the colorectal carcinoma sample, it was shown that fibroblasts were more likely to be found near epithelial cells at close distance (400 px ≈ 156 µm) than at further distance (600 px ≈ 234 µm). Epithelial cell co-occurence showed high vales in this interval which declined with distance, further showing epithelial cell clustering. CD4+ and CD8+ T cells showed a constant co-occurence score while other CD45+ cells co-occurence increase with distance to epithelial cells. Co-occurence scores on the healthy colon sample showed high values in the close range (100 px ≈ 39 µm) for epithelial, endothelial, CD11c+ and CD68+ myeloid cells. These scores quickly declined to a plateau at (400 px ≈ 156 µm). Epithelial cells had noticably higher co-occurrence scores in at 39 µm radius indicating clustered behaviour.

![[cancer_vs_healhty_spatial-statistics_1 2.jpg]]

![[cancer_vs_healhty_spatial-statistics_2 2.jpg]]
**Figure 6: Spatial statistical analysis of annotated cell-type distribution on the spatial connectivity graph.** Colorectal carcinoma sample "Point23" localed on the left and the healthy sample "Point 49" on the right. (A) Spatial connectivity graph calculated using a radius threshold of 35 px (~14 µm). (B) Ripley's L spatial distribution statistic (C) Clustering coefficient scores computed via Squidpy (C) Neighbourhood enrichment analysis (E) Co-occurence score

### Spatial statistical analysis of feature expression

In order to obtain a fuller context on a spatial level, the analysis of spatial cell-type distribution was complemented with analysis of spatial feature distribution . This was achieved via the [[Morans I]] autocorrelation, which describes the level of dispersion and clustering of a feature across space. It was predicted that the major immune lineage markers would score a Moran's I value closer to 1 since they should enrich in their in their corresponding cell-types according to their unique profile (Fig. 3A). In addition, highly variable genes found previously were hypothesized to show some spatial patterning. Indeed, in the colorectal cancer sample, four of the ranked features by Moran's I score were from the lineage marker set (descending order: CK, CD45, CD11c, CD4) (Fig. 3A, 4B). Furthermore, two of the top ranked features pertained to the previously calculated highly variable features (CD98, NaKATPase) (Fig. 4C, Supplementary table 1) and four novel features were obtained (GLUT1, HK1, LDHA, PKM2) (Fig. 7). From the top ten ranked features by Moran's I score on the healthy colon sample, 7 pertained to the major immune lineage marker set (descending order: CK, CD45, CD11c, vimentin, Ecadherin, CD14, CD3) (Fig. 3A, 5B), 3 formed part of the preiously calculated highly variable features (CD98, Ki67, NaKATPase) (Fig. 5C, Supplementary table 1) and no novel features with distinct expression in space were found.

![[Pasted image 20230301031403.png]]
**Figure 7: Novel top ranked features by Moran's I score on colorectal carcinoma sample "Point23"**. Segmentation mask coloured of "Point23" coloured by the expression of features scoring top 10 in the Moran's I analysis and were not part of either the previously investigated immune lineage markers or the highly variable features calculated for the present sample.

### Cell-type-based sender-receiver effects calculation via NCEM interaction-terms matrix

Once the dataset had been characterized on the spatial and non-spatial level, the CCC methods were applied.




We hypothesized that... we predicted... we expected...

The NCEM interaction terms matrix was calculated in two different fashions.

The first approach was based on the [[Fischer-2022]] interaction_matrix() function output and considers only the significance ([[p-value]] < 0.05) of the interaction terms after performing a Wald test on the interaction terms. The [[L1 norm]] of the indicator function of the significance matrix is calculated and used as a proxy of.

FIGURE 2

The second approach consisted of calculating the feature-level [[L2 norm]] of the significance-filtered (p < 0.05) interaction terms matrix. The interaction terms were additionally filtered by their value (-0.1 < x_i > 0.1).

FIGURE 3

### Spatial feature distribution analysis
- [ ] [[Spatial proximity enrichment analysis|Context-dependent spatial enrichment analysis]]
"nriched  around other GLUT1high cells, independent of cell lineage. This analysis also revealed spatial enrichments of enzymes within the same  metabolic pathway (for example, GLUT1high cells enriched around  PKM2high cells), suggesting the existence of environmental niches  that enable or drive certain cellular metabolic behavior. We found  such spatial enrichment for glycolysis, respiratory and amino acid  pathways in contrast to fatty acid metabolism where FAT/CD36, but  not CPT1A, was enriched on endothelial cells, potentially indicating their role in tissue uptake of fatty acids but not their oxidation41" [[Hartmann-2021]]
![[Pasted image 20230210114351.png]]
	
### MISTy cell-type based feature importance matrix
- Include importance significance

### MISTy coefficients and performance improvements

### Pearson correlation between previous two

- Do I recover anything from the [[Hartmann-2021]] paper?
# Discussion

SMA: [[Tanevski-2020]]for marker In particular, the gain for markers CD68, ki67, and SMA were found  to be the highest, suggesting that proliferation, presence, or absence of CD68 and  changes in vascularization in different grades and clinical subtypes are significantly  affected by the change in regulation as a result of intercellular interactions. Collectively, these results support the importance of the tissue structure for the expression  of proteins at the single-cell level and overall overlap with results from SVCA and the  initial performed single-cell analysis. 

- [[MKI67|KI67]] as marker of proliferation and t cell activation
- Tumour-associated T-cells were rich in exhaustion associated molecules PD1 and CD39, decreased metabolic regulator expression and decreased mitochondrial capacity [[Hartmann-2021#^a4ce5c]]

- Summary of findings? 
	- Comparable output of NCEM and MISTy was achieved
	- Pearson correlation
 
- Interpretation of findings
	- Find some specific examples of explanation of the influences
	- The definition of [[cell-cell communication]] is still very vague and needs reevaluation


- Put in context of other methods
	- DIALOGUE
	- COMMOT
	- Non-spatial

- Sources of variability AKA explain unexpected results
	- Lack of ground truth
	- Unreliable results even with synthetic data [[Tanevski-2020#^dc98dd|MISTy]]

- Put in context of current literature and latest findings
	- Mention other CCC technologies that weren't explored

- Outlook
	- Better benchmarking
		- in-silico data
		- vs R-L based methods
	- Other methods
	- Better types of analysis than used here?
	- More DS
	- In general: new method necessary that accounts for high variability, sources of variability (e.g. segmentation and stuff) and redefinition of [[cell-cell communication]].



Reference findings: (most in T cells)
- Of note, the tumor-associated metabolic T cell state (scMEP3)  was significantly enriched in exhaustion-associated molecules programmed death 1 (PD1) and CD39 [[Hartmann-2021]]
- In agreement with our mass cytometry analysis   (see Fig. 4), distinct cell lineages displayed lineage-specific metabolic  protein expression patterns with potentially activation-induced glycolytic expression in T cells and high metabolic protein levels in epithelial cells from colorectal carcinoma tissue (Fig. 5d and Supplementary   Fig. 12e).
- markers: "36 antibody dimensions, thus allowing us to determine cell   lineage, subset and activation status, together with metabolic   characteristics (Fig. 5b)." [[Hartmann-2021]] 
- Increased expression of CD98 and ASCT2 towards tumour-immune border
	- CD98 and ASCT2 have shown  prognostic value in human cancer43,44, and it will be of great interest  to see whether integrating their expression with tissue features or  multi-dimensional co-expression of other metabolic proteins will  further improve diagnostic power. [[Hartmann-2021]]
 - Increased CD39/PD1 expression towards border, best correlated [[Hartmann-2021]]
 - Correlation of CD39/PD1 and metabolic feature expression to (inverse) distance to tumor-imm border. Close T cells were metabolically active as opposed to distant cells. [[Hartmann-2021]]
- Higher metabolic function near tumor-immune border, which contradicts current literature which says that PD1/CD39 AND lower metabolic function are markers for exhausted CD8+ T-cells
	- Indicates more nuanced function, activation and exclusion of CD8+ T-cells around colorectal carcinoma tumor-immune border.
 
# Supplementary information

![[Pasted image 20230227155512.png]]
> Supplementary figure B: 

![[Pasted image 20230227203153.png]]
> Supplementary figure C: 

![[Pasted image 20230215003746.png|(Supplementary fig. )]]
![[Pasted image 20230215004415.png]]
> **Supplementary figure B:** Segmentation masks of the MIBI-TOF [[Hartmann-2021]] dataset. 


![[Pasted image 20230227171524.png]]


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
| vimentin | 0.507407 | 0 | 0.661713 | 0 |