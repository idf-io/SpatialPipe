# Application and evaluation of spatially-resolved cell-cell communication methods NCEM and MISTy on MIBI-TOF colorrectal cancer data
> [!info]- Title alternatives
> - Application and evaluation of NCEM and MISTy on spatially-resolved single-cell data
> - A comprehensive and sistematic integration of spatial-omics data and evalutaion of spatially resolved [[cell-cell communication]] methods
### [[Report notes - Internship Stegle| Report notes]]

![[spatialpipe_logo.png]] 

## Abstract

Prior to the emergence of spatial-omics technologies, [[cell-cell communication]] methods relied on prior knowledge about interacting features between cells. The measure of correlation between putative ligand-receptor interactions are a common approach to this end. Novel spatial-omics technologies like MIBI-TOF and MERFISH leverage the spatial-dimension to measure and characterize emerging feature patterns along the aforementioned dimension. Often times a "niche" radius must previously be defined to 

In the present study, novel [[Machine learning|machine learning]] based [[cell-cell communication]] methods NCEM and MISTy are applied to a colorrectal tumour dataset and evaluated in their applicability on [[cell-cell communication]] profiling.

The sources of uncertainty are still many which leads to the necessity of the establishment of protocols, redefinition of concepts (e.g. [[cell-cell communication]]) and the development of new methods that take the latter factors into account.


```toc
```


## Introduction

- Spatial omics technology emergence and evolution in context of single-cell technologies.
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

### Single-cell pre-processing
- Scaling
- Transformation
- Integration
- Clustering
- UMAP
- Image manipulation

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
e.g. significance, wald test etc

### Software
- R, Python, Conda, Jupyter-lab
- Respective base packages
- Other necessary imported packages like numpy
- MISTy, NCEM, Scanorama, Squidpy, Tensorflow, Scanpy




## Results

### Cell-type distributions

First, since both NCEM and MISTy are cell-type based methods, the distribution of the cell-types was calculated for all samples.

FIGURE 1

### Other data-exploration stuff

- Opt: Hierarchivcal Clustering of features
- Opt: UMAP of cell-types
- Opt: CellPhoneDB

### Cell-type-based sender-receiver effects calculation via NCEM interaction-terms matrix

The NCEM interaction terms matrix was calculated in two different fashions.

The first approach was based on the [[Fischer-2022]] interaction_matrix() function output and considers only the significance ([[p-value]] < 0.05) of the interaction terms after performing a Wald test on the interaction terms. The [[L1 norm]] of the indicator function of the significance matrix is calculated and used as a proxy of.

FIGURE 2

The second approach consisted of calculating the feature-level [[L2 norm]] of the significance-filtered (p < 0.05) interaction terms matrix. The interaction terms were additionally filtered by their value (-0.1 < x_i > 0.1).

FIGURE 3

### MISTy cell-type based feature importance matrix
- Include importance significance

### MISTy coefficients and performance improvements

### Pearson correlation between previous two


### Discussion
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

# 