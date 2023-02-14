# Sanity check
def sender_receiver_status():
    print("sender_receiver_custom module is loaded!")
sender_receiver_status()


# Custom sender-receiver effects wrapper function

import ncem
import numpy as np
from matplotlib import pyplot as plt

def ncem_sender_receiver_interactions_Hartmann(
    data_loader: str=None,
    data_path: str=None,
    radius: int=35,
    n_eval_nodes: int=10,
    qval_thresh: float=0.05, 
    l1_thresh: int=0,
    fc_thresh: tuple=[0.00, 0.00],
    l2_thresh = 0,
    plot: bool=True
    ):
    ''''
    Calculates a custom sender-receiver graph based on interpreter.sender_effect()
     affected gene numbers per sender-receiver pair.

    Prerequisites:
     - import ncem
     - know what data loader you want to use
     - spatial data directories and files structured after the data loaders structure

    Input
     - Spatial features data
     - Parameters:
        - qval_thresh: FDR corrected p-value to be used as an upper-bound significance threshold
           (both L1 and L2 arrays)
        - l1_thresh: Minimum number of significant genes threshold for the L1 array.
        - fc_thresh:
            - Option 1: a list of two int or float numbers to be used as an upper-bound (first number)
               and lower-bound (second number)
               thresholds for the original coefficient values (only applied to L2 array).
            - Option 2: an int or float number. Assuming a normal distribution of coefficient values,
               the number indicates the standard deviation to be used as a cutoff. Only values above and
               below the mean +/- the standard deviation will be included (only on L2 norm array).
        - l2_thresh:
            - Option 1: Lower-bound threshold for computed L2 norm values (L2 array only).
            - Option 2: Use a specific percentile as a lower-bound threshold for computed
               L2 norm values (L2 array only).
               Notation str('AXX'), where 'A' is short for 'Above' and the 'XX' are strictly
               two numbers (int) to be considered as the lower-bound percentile threshold.

    Output:
     - List of 2 numpy arrays:
        - 1. Type-coupling (Sender-receiver effect) interaction matrix
            L1 norm of the number of significant interaction terms per cell
        - 2. Sender-receiver magnitude effect interaction matrix
            L2 norm of interacton term coefficients of genes per cell
        - dim(cell-types x cell-types)
            - [receiver, sender]
    '''

    # Load in the data using a specific data loader
    interpreter = ncem.interpretation.interpreter.InterpreterInteraction()
    interpreter.get_data(
        data_origin=data_loader, # Reference to specfic DataLoader (see notes in Dropbox paper)
        data_path=data_path,
        radius=radius,
        node_label_space_id='type',
        node_feature_space_id='standard',
    )

    # Extracting sender-receiver effects with NCEM
    interpreter.split_data_node(0.1, 0.1)
    interpreter.n_eval_nodes_per_graph = n_eval_nodes
    interpreter.cell_names = list(interpreter.data.celldata.uns['node_type_names'].values())

    # Compute sender receiver effects
    interpreter.get_sender_receiver_effects()


    # Some relevant variables
    clusters = interpreter.cell_names
    num_clusters = len(clusters)
    vars = interpreter.data.var_names
    num_vars = len(vars)
    qvals = interpreter.qvalues

    # interpreter.fold_change
    #  - numpy array of shape [8, 8 , 36] = [receiver clusters, sender clusters, features]
    #  - dtype=float32



    # L1-signigicance interaction matrix

    l1_effects = interpreter.fold_change

    # Filter out genes under corrected qval
    l1_effects = l1_effects * (qvals < qval_thresh)

    # L1 norm AKA Count genes > 0 per sender-receiver pair
    l1_effects = np.sum(l1_effects != 0, axis=2)

    # Filter out sender-receiver effects A.K.A node edges > nr_genes_thresh
    l1_effects = np.where(l1_effects > l1_thresh, l1_effects, np.zeros([num_clusters, num_clusters]))

    np.fill_diagonal(l1_effects, 0)



    # L2-coefficient interaction matrix

    l2_effects = interpreter.fold_change

    # Filter out genes under/over fc_thresh

    try:

        if isinstance(fc_thresh, list) & (len(fc_thresh) == 2) & isinstance(fc_thresh[0], (int, float)) & isinstance(fc_thresh[1], (int, float)):

            l2_effects = np.where((l2_effects < fc_thresh[0]) | (l2_effects > (fc_thresh[1])), l2_effects, np.zeros([num_clusters, num_clusters, num_vars]))

    except: 

        if isinstance(fc_thresh, (int, float)):
            
            mean = np.mean(l2_effects)
            std = np.std(l2_effects)

            l2_effects = np.where((l2_effects < (mean - (fc_thresh * std))) | (l2_effects > (mean + (fc_thresh * std))), l2_effects, np.zeros([num_clusters, num_clusters, num_vars]))
        
        else:
            raise Exception(f'{fc_thresh} doesnt comply with possible parameter types.')

    # BUG in next step! features aren't filtered correctly.
    # Probably some data-type related problem

    # Filter out genes under corrected qval
    l2_effects = l2_effects * (qvals < qval_thresh)

    # L2 norm
    l2_effects = np.sqrt(np.sum(l2_effects ** 2, axis=-1))

    # Filter out sender-receiver effects A.K.A L2 score > l2_thresh
    if isinstance(l2_thresh, (int, float)):

        l2_effects = np.where(l2_effects > l2_thresh, l2_effects, np.zeros([num_clusters, num_clusters]))
    
    elif isinstance(l2_thresh, str) & (len(l2_thresh) == 3):
        
        try:

            _ = l2_thresh[0] == 'A'
            int_number = int(l2_thresh[1:3])

            percentile =  np.percentile(l2_effects, int_number)

            l2_effects = np.where(l2_effects > percentile, l2_effects, np.zeros([num_clusters, num_clusters]))


        except ValueError:

            print('l2_thresh doesnt comply with the parameter possibilities.')
    
    else:

        print('l2_thresh doesnt comply with the parameter possibilities.')

    np.fill_diagonal(l2_effects, 0)


    # Heatmap
    if plot == True:

        fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True)

        im0 = ax0.imshow(l1_effects, cmap='gray_r')
        im1 = ax1.imshow(l2_effects, cmap='gray_r')
        ax0.set_title('Sender-receiver effects:\nL1(significance)', y=1.1)
        ax1.set_title("Sender-receiver effects:\nL2(interaction terms)", y=1.1)

        for ax in fig.get_axes():
            
            ax.set_xlabel('Senders')
            ax.set_xticks([i for i in range(0,8)])
            ax.set_xticklabels(interpreter.cell_names, rotation=90)

            ax.set_ylabel('Receivers')
            ax.set_yticks([i for i in range(0,8)])
            ax.set_yticklabels(interpreter.cell_names)

            ax.label_outer()

        fig.colorbar(im0, ax=ax0, shrink=0.49)
        fig.colorbar(im1, ax=ax1, shrink=0.49)

        plt.show()

    return [interpreter.fold_change, l1_effects, l2_effects]



def shuffle(dataframe, col = None, new_name: str = 'shuffled', frac: float = 1.0):

    if col == None:
        raise Exception("Argument `col` must be specified!")

    df = dataframe[[col]].copy()

    rand_index = np.random.choice(df.shape[0], int(df.shape[0] * frac), replace=False)
    rand_df = np.random.choice(np.unique(df[col]), int(df.shape[0] * frac), replace=True)

    df['old'] = df[col]
    df = df.drop(col, axis=1)
    df[new_name] = df['old']

    df[new_name].iloc[rand_index, ] = rand_df

    return df[new_name]


####################################################################################################

# class DataLoaderHartmannCustom(DataLoader):

#     """DataLoaderHartmann class. Inherits all functions from DataLoader."""

#     cell_type_merge_dict = {
#         "Imm_other": "Other immune cells",
#         "Epithelial": "Epithelial",
#         "Tcell_CD4": "CD4 T cells",
#         "Myeloid_CD68": "CD68 Myeloid",
#         "Fibroblast": "Fibroblast",
#         "Tcell_CD8": "CD8 T cells",
#         "Endothelial": "Endothelial",
#         "Myeloid_CD11c": "CD11c Myeloid",
#     }

#     def neww():
#         print('Working')

#     def _register_celldata(self, n_top_genes: Optional[int] = None):
#         """Load AnnData object of complete dataset."""
#         metadata = {
#             "lateral_resolution": 400 / 1024,
#             "fn": ["scMEP_MIBI_singlecell/scMEP_MIBI_singlecell.csv", "scMEP_sample_description.xlsx"],
#             "image_col": "point",
#             "pos_cols": ["center_colcoord", "center_rowcoord"],
#             "cluster_col": "Cluster",
#             "cluster_col_preprocessed": "Cluster_preprocessed",
#             "patient_col": "donor",
#         }
#         celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"][0]))
#         celldata_df["point"] = [f"scMEP_point_{str(x)}" for x in celldata_df["point"]]
#         celldata_df = celldata_df.fillna(0)
#         # celldata_df = celldata_df.dropna(inplace=False).reset_index()
#         feature_cols = [
#             "H3",
#             "vimentin",
#             "SMA",
#             "CD98",
#             "NRF2p",
#             "CD4",
#             "CD14",
#             "CD45",
#             "PD1",
#             "CD31",
#             "SDHA",
#             "Ki67",
#             "CS",
#             "S6p",
#             "CD11c",
#             "CD68",
#             "CD36",
#             "ATP5A",
#             "CD3",
#             "CD39",
#             "VDAC1",
#             "G6PD",
#             "XBP1",
#             "PKM2",
#             "ASCT2",
#             "GLUT1",
#             "CD8",
#             "CD57",
#             "LDHA",
#             "IDH2",
#             "HK1",
#             "Ecad",
#             "CPT1A",
#             "CK",
#             "NaKATPase",
#             "HIF1A",
#             # "X1",
#             # "cell_size",
#             # "category",
#             # "donor",
#             # "Cluster",
#         ]
#         var_names = [
#             'H3-4', 
#             'VIM', 
#             'SMN1', 
#             'SLC3A2', 
#             'NFE2L2', 
#             'CD4', 
#             'CD14', 
#             'PTPRC', 
#             'PDCD1',
#             'PECAM1', 
#             'SDHA', 
#             'MKI67', 
#             'CS', 
#             'RPS6', 
#             'ITGAX', 
#             'CD68', 
#             'CD36', 
#             'ATP5F1A',
#             'CD247', 
#             'ENTPD1', 
#             'VDAC1', 
#             'G6PD', 
#             'XBP1', 
#             'PKM', 
#             'SLC1A5', 
#             'SLC2A1', 
#             'CD8A',
#             'B3GAT1', 
#             'LDHA', 
#             'IDH2', 
#             'HK1', 
#             'CDH1', 
#             'CPT1A', 
#             'CKM', 
#             'ATP1A1',
#             'HIF1A'
#         ]

#         celldata = AnnData(
#             X=pd.DataFrame(np.array(celldata_df[feature_cols]), columns=var_names), obs=celldata_df[
#                 ["point", "cell_id", "cell_size", "donor", "Cluster"]
#             ].astype("category"),
#         )

#         celldata.uns["metadata"] = metadata
#         img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
#         celldata.uns["img_keys"] = img_keys

#         # register x and y coordinates into obsm
#         celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

#         img_to_patient_dict = {
#             str(x): celldata_df[metadata["patient_col"]].values[i]
#             for i, x in enumerate(celldata_df[metadata["image_col"]].values)
#         }
#         # img_to_patient_dict = {k: "p_1" for k in img_keys}
#         celldata.uns["img_to_patient_dict"] = img_to_patient_dict
#         self.img_to_patient_dict = img_to_patient_dict

#         # add clean cluster column which removes regular expression from cluster_col
#         celldata.obs[metadata["cluster_col_preprocessed"]] = list(
#             pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="category").map(self.cell_type_merge_dict)
#         )
#         celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype(
#             "category"
#         )

#         # register node type names
#         node_type_names = list(np.unique(celldata.obs[metadata["cluster_col_preprocessed"]]))
#         celldata.uns["node_type_names"] = {x: x for x in node_type_names}
#         node_types = np.zeros((celldata.shape[0], len(node_type_names)))
#         node_type_idx = np.array(
#             [
#                 node_type_names.index(x) for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
#             ]  # index in encoding vector
#         )
#         node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
#         celldata.obsm["node_types"] = node_types

#         self.celldata = celldata

#     def _register_img_celldata(self):
#         """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
#         image_col = self.celldata.uns["metadata"]["image_col"]
#         img_celldata = {}
#         for k in self.celldata.uns["img_keys"]:
#             img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
#         self.img_celldata = img_celldata

#     def _register_graph_features(self, label_selection):
#         """Load graph level covariates.
#         Parameters
#         ----------
#         label_selection
#             Label selection.
#         """
#         # DEFINE COLUMN NAMES FOR TABULAR DATA.
#         # Define column names to extract from patient-wise tabular data:
#         patient_col = "ID"
#         # These are required to assign the image to dieased and non-diseased:
#         disease_features = {"Diagnosis": "categorical"}
#         patient_features = {"ID": "categorical", "Age": "continuous", "Sex": "categorical"}
#         label_cols = {}
#         label_cols.update(disease_features)
#         label_cols.update(patient_features)

#         if label_selection is None:
#             label_selection = set(label_cols.keys())
#         else:
#             label_selection = set(label_selection)
#         label_cols_toread = list(label_selection.intersection(set(list(label_cols.keys()))))
#         usecols = label_cols_toread + [patient_col]

#         tissue_meta_data = read_excel(os.path.join(self.data_path, "scMEP_sample_description.xlsx"), usecols=usecols)
#         # BUILD LABEL VECTORS FROM LABEL COLUMNS
#         # The columns contain unprocessed numeric and categorical entries that are now processed to prediction-ready
#         # numeric tensors. Here we first generate a dictionary of tensors for each label (label_tensors). We then
#         # transform this to have as output of this section dictionary by image with a dictionary by labels as values
#         # which can be easily queried by image in a data generator.
#         # Subset labels and label types:
#         label_cols = {label: nt for label, nt in label_cols.items() if label in label_selection}
#         label_tensors = {}
#         label_names = {}  # Names of individual variables in each label vector (eg. categories in onehot-encoding).
#         # 1. Standardize continuous labels to z-scores:
#         continuous_mean = {
#             feature: tissue_meta_data[feature].mean(skipna=True)
#             for feature in list(label_cols.keys())
#             if label_cols[feature] == "continuous"
#         }
#         continuous_std = {
#             feature: tissue_meta_data[feature].std(skipna=True)
#             for feature in list(label_cols.keys())
#             if label_cols[feature] == "continuous"
#         }
#         for feature in list(label_cols.keys()):
#             if label_cols[feature] == "continuous":
#                 label_tensors[feature] = (tissue_meta_data[feature].values - continuous_mean[feature]) / continuous_std[
#                     feature
#                 ]
#                 label_names[feature] = [feature]
#         # 2. One-hot encode categorical columns
#         # Force all entries in categorical columns to be string so that GLM-like formula processing can be performed.
#         for feature in list(label_cols.keys()):
#             if label_cols[feature] == "categorical":
#                 tissue_meta_data[feature] = tissue_meta_data[feature].astype("str")
#         # One-hot encode each string label vector:
#         for i, feature in enumerate(list(label_cols.keys())):
#             if label_cols[feature] == "categorical":
#                 oh = pd.get_dummies(tissue_meta_data[feature], prefix=feature, prefix_sep=">", drop_first=False)
#                 # Change all entries of corresponding observation to np.nan instead.
#                 idx_nan_col = np.array([i for i, x in enumerate(oh.columns) if x.endswith(">nan")])
#                 if len(idx_nan_col) > 0:
#                     assert len(idx_nan_col) == 1, "fatal processing error"
#                     nan_rows = np.where(oh.iloc[:, idx_nan_col[0]].values == 1.0)[0]
#                     oh.loc[nan_rows, :] = np.nan
#                 # Drop nan element column.
#                 oh = oh.loc[:, [x for x in oh.columns if not x.endswith(">nan")]]
#                 label_tensors[feature] = oh.values
#                 label_names[feature] = oh.columns
#         # Make sure all tensors are 2D for indexing:
#         for feature in list(label_tensors.keys()):
#             if len(label_tensors[feature].shape) == 1:
#                 label_tensors[feature] = np.expand_dims(label_tensors[feature], axis=1)
#         # The dictionary of tensor is nested in slices in a dictionary by image which is easier to query with a
#         # generator.
#         tissue_meta_data_patients = tissue_meta_data[patient_col].values.tolist()
#         label_tensors = {
#             img: {
#                 feature_name: np.array(features[tissue_meta_data_patients.index(patient), :], ndmin=1)
#                 for feature_name, features in label_tensors.items()
#             }
#             if patient in tissue_meta_data_patients
#             else None
#             for img, patient in self.celldata.uns["img_to_patient_dict"].items()
#         }
#         # Reduce to observed patients:
#         label_tensors = dict([(k, v) for k, v in label_tensors.items() if v is not None])

#         # Save processed data to attributes.
#         for k, adata in self.img_celldata.items():
#             graph_covariates = {
#                 "label_names": label_names,
#                 "label_tensors": label_tensors[k],
#                 "label_selection": list(label_cols.keys()),
#                 "continuous_mean": continuous_mean,
#                 "continuous_std": continuous_std,
#                 "label_data_types": label_cols,
#             }
#             adata.uns["graph_covariates"] = graph_covariates

#         graph_covariates = {
#             "label_names": label_names,
#             "label_selection": list(label_cols.keys()),
#             "continuous_mean": continuous_mean,
#             "continuous_std": continuous_std,
#             "label_data_types": label_cols,
#         }
#         self.celldata.uns["graph_covariates"] = graph_covariates
#         self.celldata.uns["worked"] = {"Yes": True}

#         # self.ref_img_keys = {k: [] for k, v in self.nodes_by_image.items()}
