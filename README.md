# SpatialPipe: Systematic standardization and framework integration of public spatial-omics datasets for cell-cell communication analysis

Preprocessing pipeline implementation of some public spatial omics datasets which will be used for a shared project we have in collaboration with Omer Bayraktar group from Sanger institute at UK. There is of course some flexibility in defining the actual project, but the bottom-line is, the project is gonna be about low-level analysis of spatial omics data

![](spatialpipe_logo.png 50x50)

**IMPORTANT**
- Objective of project: create code that will be reusable by the group
- Focus on reusability, clarity and documentation

# Datasets

## Colorrectal carcinoma: Hartmann-2021
[Hartmann, FJ et al. (2021). Single-cell metabolic profiling of human cytotoxic T cells. Nat Biotechnol 39, 186–197](https://www.nature.com/articles/s41587-020-0651-8)

[Download link](https://zenodo.org/record/3951613)

Focused on MIBI-TOF multiplexed images and segmented single-cell data of colorectal carcinoma and healthy adjacent colon tissue.

- MIBI-TOF images have undergone noise removal as described in Keren et al. (2018)
- Cell Segmentation masks for MIBI-TOF data contain large non-cellular regions that need to be removed during downstream processing (Not corrected in this script)
- MIBI-TOF derived single-cell data is cell size normalized, arcsinh transformed and percentile normalized and contains manually annotated FlowSOM clustering results