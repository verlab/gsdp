import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import h5py
import math


#      Global Descriptor
# --------------------------------------------------------------------------------------------------
# 0 --- Feature Extraction -> features
# 1 --- Compute imput vector ( features, std)
# 2 --- Dimensionality reduction
#       2.1 Compute angles 
#       2.2 Compute_signature_unitary_M (sum vector + plot option)
# 3 -  global_prttype_descriptor
#       3.0 Compute signature and plot options
#       3.1 category_signature
#            | semantic_value | std category|
#       3.2 prototype_signature
#            | semantic_value | 000000000000|
# 4 -  global_features_descriptor
#            | semantic_value member | member fetaure - prototype category|
# 5 -  plot asignatures as histograms
# ##################################################################################################


def compute_input_vector(model, class_idx, features, semantic=True):
    '''
    Compute the input vector of dimensionality reduction function
    :param model: Extractor model instance
    :param class_idx: category index (categorization result)
    :param features: extracted features
    :param semantic: if True compute
    :return: input vector
    '''

    # compute  semantic meaning
    if semantic:
        # b = b/n in n pos
        b_n = np.empty(len(features))
        b_n.fill(model.b[class_idx] / len(features))
        # z_n = w*a + b/n
        input_vector = np.multiply(features, model.W[:, class_idx]) + b_n  # np.multiply(f,w)+b_m

    # compute semantic difference
    else:
        # z_n = |w|*a
        input_vector = np.multiply(features, np.absolute(model.W[:, class_idx]))  # np.multiply(f,w)
    return input_vector

# --------------------------------------------------------------------------------------------------
# 2 --- Dimensionality reduction
# --------------------------------------------------------------------------------------------------


def dimensionality_reduction(features, aux_matrix_dim=None, scale=None,
                             plot=True,
                             plot_color='g',
                             save_file=None,
                             save_format='svg'):
    '''
        Simple reduction of input feature dimension. (Use the function two times for graphical plot of descriptor)
    :param features: array of semantic feature
    :param aux_matrix_dim: dimension of auxiliary_matrix used in mapping process
    :param scale: plotting scale
    :param plot: True to plot a descriptor, False otherwise
    :param plot_color: color of signature plotting
    :param save_file: save graph representation of signature
    :param save_format: file output format
    :return: a low dimensional signature nparray (half of descriptor signature)
    '''

    # ------------------------DIMENSION CONFIGURATION-------------------------

    # 128 features (Example: MNIST features)
    if len(features) == 128:
        # reshape input feature
        features = np.reshape(features, (8, 16))
        # r-dimension of auxiliary matrix (X_r_x_r)
        aux_matrix_dim = 8 if aux_matrix_dim is None else aux_matrix_dim
        # if plotting : set maximum activation value
        if scale is None:
            scale = 2 if aux_matrix_dim == 4 else 12
        #  unitary_descriptor plot size
        size = 3

    # 2048 features (Example: ResNet50 features) and 4096 features (Example: VGG16 features)
    if len(features) == 2048 or len(features) == 4096:
        # reshape input feature
        features = np.reshape(features, (32, 64)) if len(features) == 2048 else np.reshape(features, (64, 64))
        # r-dimension of auxiliary matrix (X_r_x_r)
        aux_matrix_dim = 16 if aux_matrix_dim is None else aux_matrix_dim
        # if plotting: set maximum activation value
        if scale is None:
            scale = 20 if aux_matrix_dim == 16 else 40
        #  unitary_descriptor plot size
        size = 3.6

    # ----------------------------CONFIGURATION----------------------------
    # angle matrix related to auxiliary matrix (X_r_x_r)
    angles_m = angles_matrix(aux_matrix_dim)

    # graphic signature dimensions
    rows = int(features.shape[0] / aux_matrix_dim)
    cols = int(features.shape[1] / aux_matrix_dim)

    # unitary signature count
    block_count = cols * rows
    block_size = aux_matrix_dim

    # ----------------------------MAPPING FEATURES--------------------------
    features_map = [None for _ in range(block_count)]
    count = 0
    i_init = 0
    for i in range(rows):
        j_init = 0
        for j in range(cols):
            # Mapping Features
            features_map[count] = features[i_init:i_init + block_size, j_init:j_init + block_size]
            j_init += block_size
            count += 1
        i_init += block_size

    # ---------------------------GRAPH PLOTTING -------------------------------------------
    if plot:
        signature = None
        fig = plt.figure(figsize=(size * cols, size * rows))  # (w,h)
        gs = gridspec.GridSpec(rows, cols)

        # build figure dynamically
        for i in range(block_count):
            ax = fig.add_subplot(gs[i])
            unitary = unitary_descriptor(angles_m.flatten(), features_map[i].flatten(),
                                         plot=plot, ax=ax, title=str(i), scale=scale, color=plot_color)
            signature = unitary if signature is None else np.concatenate((signature, unitary), axis=0)
        plt.tight_layout()

        if save_file is not None:
            # fig.savefig(save_file + '.png')
            fig.savefig(save_file + '.' + save_format, bbox_inches='tight')
        else:
            plt.show()
            plt.close(fig)
    # ---------------------------Concatenate unitary signatures -------------------------------------------
    else:
        signature = None
        for i in range(block_count):
            unitary = unitary_descriptor(angles_m.flatten(), features_map[i].flatten(), plot=plot)
            signature = unitary if signature is None else np.concatenate((signature, unitary), axis=0)

    return signature

# --------------------------------------------------------------------------------------------------
#    2.1 Compute angles 
# -------------------------------------------------------------------------------------------------


def angles_matrix(matrix_dim):
    '''
    Build angles matrix. Use sectors_angles() function.
        s2 | s1
        ___|___
        s3 | s4

    :param matrix_dim: r-dimension of auxiliary matrix (X_r_x_r)
    :return: Return a r-dimensional matrix of angles (in 2d coordinate)
    '''

    sector_size = int(matrix_dim / 2)
    s1, s2, s3, s4 = sectors_angles(sector_size)
    angles = np.zeros((matrix_dim, matrix_dim))
    angles[:int(sector_size), :int(sector_size)] = s2
    angles[:int(sector_size), int(sector_size):] = s1
    angles[int(sector_size):, :int(sector_size)] = s3
    angles[int(sector_size):, int(sector_size):] = s4
    return angles
# -------------------------------------------------------------------------------------------------
#    2.1.1 Build angles of each sector
# --------------------------------------------------------------------------------------------------


def sectors_angles(sector_dim):
    '''
    Build angles of each sector (4 sectors)

    :param sector_dim: sector dimension
    :return: 4 angles sector
    '''

    cell_size = 2
    # build s1 [0-90 degree]
    s1 = np.zeros((sector_dim, sector_dim))
    for j in range(sector_dim):
        for i in range(sector_dim):
            angle_tan = (i * cell_size + cell_size / 2) / (j * cell_size + cell_size / 2)
            s1[j, i] = math.degrees(math.atan(angle_tan))
    s1 = np.rot90(s1)

    # To diagonal (same angles values)
    max_less_45 = s1[1, sector_dim - 1]
    diagonal_increase = (45 - max_less_45 - 3) / (sector_dim / 2)

    for i in range(sector_dim):
        if i < int(sector_dim/2):
            s1[sector_dim - i - 1, i] = max_less_45 + (i + 1) * diagonal_increase
        else:
            s1[sector_dim - i - 1, i] = 90 - max_less_45 - (i - sector_dim / 2 + 1) * diagonal_increase

    # build s4 [270-360 degree]
    tmp_s = np.flip(s1, 0)
    s4 = -tmp_s + 360
    # build s3 [180-270 degree]
    tmp_s = np.flip(tmp_s, 1)
    s3 = tmp_s + 180
    # build s2 [90-180 degree]
    tmp_s = np.flip(s1, 1)
    s2 = 180 - tmp_s

    return s1, s2, s3, s4

# --------------------------------------------------------------------------------------------------
#       2.2 Compute_signature_unitary_M (sum and return 8 vector + plot option)
# --------------------------------------------------------------------------------------------------


def unitary_descriptor(angles, features, plot=True, ax=None, title=' ', scale=10, color='g'):
    '''
    Compute unitary_signature of input features (mapped).
    Sum the features vectors (semantic gradient) using angles counter-clock wise fashion( starting  in 45 degree).
    :param angles: angles matrix A
    :param features: features vectors
    :param plot: Boolean flag. Plot unitary signature
    :param ax: plt.ax object (used for dynamic plotting)
    :param title: title of unitary signature (index)
    :param scale: plotting scale
    :param color: plotting color
    :return: a 8-dimensional nparray (unitary_signature)
    '''
    # angles counter-clock wise start in 45 degree
    angles_desc = np.arange(45, 405, 45)
    # angles_desc[7]=0
    magnitudes = np.zeros(8)
    for angle, i in zip(angles_desc, range(8)):
        # print(angle,i,angle-45,angle)
        magnitudes[i] = np.sum(features[np.logical_and(angles > angle - 45, angles <= angle)])
    if plot:
        unitary_descriptor_plot(magnitudes, ax=ax, title=title, scale=scale, positive_color=color)

    return magnitudes


# ---------------------------Plotting options
def unitary_descriptor_plot(magnitudes, ax=None, title=' ', scale=10, positive_color='g'):
    '''
    Plotting unitary signature (dynamic plotting).
    :param magnitudes:
    :param ax: plt.ax object (used for dynamic plotting)
    :param title: title of unitary signature (index)
    :param scale: plotting scale
    :param positive_color: plotting color
    :return:
    '''

    # origin point
    origin = [0], [0]
    # vectors arrow ends (V[count,0] -> X cor, V[count,1] -> Y cor)
    vectors = np.zeros((8, 2))
    # angles between sum
    angles = np.arange(45, 405, 45)
    angles[7] = 0
    # angles magnitudes sum
    for count in range(8):
        # endx
        vectors[count, 0] = np.abs(magnitudes[count]) * math.cos(math.radians(angles[count]))
        # endy
        vectors[count, 1] = np.abs(magnitudes[count]) * math.sin(math.radians(angles[count]))
    # plotting
    if ax:  # to plot dinamicaly ()
        ax.quiver(*origin, vectors[:, 0][magnitudes >= 0], vectors[:, 1][magnitudes >= 0],
                  color=positive_color, angles='xy', scale_units='xy', scale=1, width=0.012)

        ax.quiver(vectors[:, 0][magnitudes < 0], vectors[:, 1][magnitudes < 0], -vectors[:, 0][magnitudes < 0],
                  -vectors[:, 1][magnitudes < 0], color='r', angles='xy', scale_units='xy', scale=1, width=0.012)

        ax.set_title(title)
        ax.axis([-scale, scale, -scale, scale])
    else:
        plt.quiver(*origin, vectors[:, 0], vectors[:, 1], scale=np.max(np.max(vectors[:, 0]), np.max(vectors[:, 1])),
                   scale_units='inches')
        # color=['r','b','g'],, ,scale=1
        plt.show()

# --------------------------------------------------------------------------------------------------
#       3.0 Compute signature and plot options
# --------------------------------------------------------------------------------------------------


def compute_plot_signature(model, features, category_i, aux_matrix_dim=None, save_file=None,
                           semantic=True, plotting=True, plot_ceil=True, plot_color='g'):
    '''
    Compute signature and plot options
    :param model: Keras classification model
    :param features: Keras model features
    :param category_i: category label (int index)
    :param aux_matrix_dim: Auxiliary matrix dimension
    :param save_file:
    :param semantic:
    :param plotting:
    :param plot_ceil:
    :param plot_color:
    :return:
    '''
    input_vector = compute_input_vector(model, category_i, features, semantic=semantic)
    # compute without plot
    signature = dimensionality_reduction(input_vector, aux_matrix_dim=aux_matrix_dim, plot=False)
    if plotting or save_file:
        # print(math.ceil(np.max(df)))
        # file_name to save (if save)
        file_name = save_file + '_graph' if save_file else None
        # graph plot (normalized to max (signature))
        plot_scale = math.ceil(np.max(np.abs(signature))) if plot_ceil else np.max(np.abs(signature))
        signature = dimensionality_reduction(input_vector, aux_matrix_dim=aux_matrix_dim, plot=True, scale=plot_scale,
                                             plot_color=plot_color, save_file=file_name)
        # file_name = save_file + '_hist' if save_file else None
        # title = str(class_i) if title is None else title
    return signature

# --------------------------------------------------------------------------------------------------
#       3.1,3.2 -  global_prttype_descriptor 
#             if category =True  3.1  (category_signature) -> | semantic_value | std category|                         
#             if category =False 3.2   prototype_signature  > | semantic_value | 000000000000|  
# --------------------------------------------------------------------------------------------------


def global_prototype_descriptor(model, category_i, file_prototypes,
                                category=True, plotting=True,
                                plot_ceil=True, save_file=None):
    '''
    Return the signature of  i-th category prototype
    :param model: Keras classification model
    :param category_i: category label (int index)
    :param file_prototypes: path of prototype dataset file (.h5 format)
    :param category: Boolean flag. If True: category signature, else: abstract prototype signature
    :param plotting: Boolean flag. If True: graphical plot of prototype signature
    :param plot_ceil: Boolean flag. If true, plot with ceiling of signature values max.
    :param save_file: File path where the signature plot will be saved.
    :return: prototype signature
    '''

    class_i = int(category_i)
    f_prototype = h5py.File(file_prototypes, 'r')
    prototype_mean = f_prototype['mean'][class_i]

    save_file_semantic = save_file + '_semantic' if save_file else save_file
    semantic_signature = compute_plot_signature(model, prototype_mean, class_i, plotting=plotting,
                                                plot_ceil=plot_ceil, save_file=save_file_semantic)
    if category:  # category descriptor
            prototype_std = f_prototype['std'][class_i]
            save_file_diff = save_file + '_difference' if save_file else save_file
            boundary_signature = compute_plot_signature(model, prototype_std, class_i, plotting=plotting,
                                                        plot_ceil=plot_ceil, semantic=False, save_file=save_file_diff,
                                                        plot_color='m')
    else:        # prototype_descriptor
        boundary_signature = np.full(semantic_signature.shape, 0)
    f_prototype.close()
    return np.concatenate((semantic_signature, boundary_signature))


# --------------------------------------------------------------------------------------------------
#       4 -  global_features_descriptor  (extended = False)
#            
# --------------------------------------------------------------------------------------------------
def global_features_descriptor(model, features, class_i, prototype_dif, extended=False,
                               plotting=False, plot_ceil=True, aspect_ratio=(10, 5), save_file=None):

    save_file_semantic = save_file + '_semantic' if save_file else save_file
    semantic_meaning = compute_plot_signature(model, features, class_i, plotting=plotting,
                                              plot_ceil=plot_ceil, save_file=save_file_semantic)

    if extended:  # concatenate d_base + std(respect to category_prototype)
        save_file_diff = save_file + '_difference' if save_file else save_file
        semantic_difference = compute_plot_signature(model, prototype_dif, class_i, plotting=plotting,
                                                     plot_ceil=plot_ceil, semantic=False, save_file=save_file_diff)

        return np.concatenate((semantic_meaning, semantic_difference))
    return semantic_meaning

# --------------------------------------------------------------------------------------------------
#      5 -  plot asignatures as histograms
# --------------------------------------------------------------------------------------------------


def descriptor_signature_plot_h(signature, figsize=(30, 6), title='', save_fig=None, save_format='svg'):
    fig = plt.figure(figsize=figsize)
    x = np.arange(0, len(signature), 1)
    colors = ['g' for _ in np.arange(len(signature))]
    for i in x[signature < 0]:
                colors[i] = 'r'
    plt.bar(x, signature, color=colors)
    plt.title(title)
    plt.xlabel('dimension')
    plt.ylabel('magnitude')  
    plt.tight_layout()
    plt.show()
    if save_fig is not None:
            fig.savefig(save_fig + '.' + save_format, bbox_inches='tight')
    plt.close(fig)
