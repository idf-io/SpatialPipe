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


    # Filter out genes under corrected qval
    l2_effects = l2_effects * (qvals < qval_thresh)

    # L2 norm
    l2_effects = np.sqrt(np.sum(l2_effects ** 2, axis=-1))

    # Filter out sender-receiver effects A.K.A L2 score > l2_thresh
    if isinstance(l2_thresh, (int, float)):

        l2_effects = np.where(l2_effects > l2_thresh, l2_effects, np.zeros([num_clusters, num_clusters]))
    
    elif isinstance(l2_thresh, str) & len(l2_thresh) == 3:
        
        try:

            _ = l2_thresh[0] == 'A'
            int_number = int(l2_thresh[1, 3])

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
        ax0.set_title('Sender-receiver effects:\nL1(significance)')
        ax1.set_title("Sender-receiver effects:\nL2(interaction terms)")

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