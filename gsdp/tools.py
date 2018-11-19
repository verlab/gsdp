
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import numpy as np
import h5py
import math


def element_signature(model,class_idx,file_feature, file_stats, dataset,
                      category, element_index, plotting=False, save_file=None):
    ff = h5py.File(file_feature, 'r')
    sf = h5py.File(file_stats, 'r')
    # match = load_file_index(file_index)
    if model.config.model_name == 'MNIST':
        features = ff['features/' + dataset]
        f_block = features[dataset + str(class_idx)]
    else:
        f_block = ff[str(class_idx)]

    prttype_mean = sf['mean'][class_idx]
    prttype_std = sf['std'][class_idx]

    data_block = f_block[element_index:element_index + 1]
    labels = np.full(data_block.shape[0], int(category))
    signatures = compute_block_of_descriptors(model, data_block, labels, file_stats,
                                              extended=True, plotting=plotting,
                                              type_plot=1, save_file=save_file)
    # X_cor1,Y_cor1,P_x1,P_y1 = compute_distances('Z',data_block,w,b,prttype_mean,False)
    # X_cor,Y_cor,P_x,P_y= compute_distances_in_signatures(signatures,prototype_signature)
    # Y_L1 = np.asarray(list(map(lambda x: simpleL1(prototype_signature,signatures[x]),range(signatures.shape[0]))))
    ff.close
    sf.close
    return signatures


# import Metrics
from .utils.metrics import semantic_v,semantic_vN,semantic_av,semantic_avN,absoluteWxL1,absoluteWxL1_N,simpleL1
from .extractors.extractor import SimpleExtractor, ImageNetExtractor
from .extractors.config import ExtractorConfig

# 0 ---
#--------------------------------------------------------------------------------------------------
def init_model(model_name,models_path,dataset_path):
    '''

    :param model_name: Keras CNN-Model Name
    :param models_path: path of models folder
    :param dataset_path: path of datasets forlder
    :return: Extractor model
    '''
    model_config = ExtractorConfig(root_path=models_path,model_name = model_name)
    #update dataset_np path
    model_config.dataset_path = dataset_path
    #build_extractor
    if model_name=='MNIST':
            model = SimpleExtractor(config=model_config)
    else:
            model = ImageNetExtractor(config=model_config)
    return model  

def block_of_data(model,dataset,categories,file_feature,file_stats,size =None,descriptors_out = True,proto_category=True):
#class_idx_s = [5,6,3,0,8,1,7]
        ff = h5py.File(file_feature, 'r')
        sf = h5py.File(file_stats, 'r')
        if model.config.model_name =='MNIST':
               features = ff['features/' + dataset]
        data = None
        labels = None
        p_data = []

        for class_idx in categories:
            #print(class_idx)
            f_block  = features[dataset + str(class_idx)]  if model.config.model_name =='MNIST' else ff[str(class_idx)] 
            prttype_mean = sf['mean'][class_idx]
            w = model.W[:, class_idx]
            b = model.b[class_idx]
           # print(f_block.shape)
            #print(prttype_mean.shape)

            data_block = f_block if size is None else f_block[:size]
            #print(data_block.shape)
            labels_i = np.full(data_block.shape[0],int(class_idx))
            if descriptors_out: # if descriptors output
                signatures_i = compute_block_of_descriptors(model,data_block,labels_i,file_stats,extended=True)
                p_signature_i = global_prttype_descriptor(model,class_idx,file_stats,plotting = False,category=proto_category)
                #X_cor1,Y_cor1,P_x1,P_y1 = compute_distances('Z',data_block,w,b,prttype_mean,False)
                #X_cor,Y_cor,P_x,P_y= compute_distances_in_signatures(signatures,prototype_signature)
                #Y_L1 = np.asarray(list(map(lambda x: simpleL1(prototype_signature,signatures[x]),range(signatures.shape[0]))))
                
                # prototype
                p_data.append(p_signature_i)
                data = np.concatenate((data,signatures_i)) if not data is None else signatures_i
               
            else:  # if feature output
                p_data.append(prttype_mean)
                data = np.concatenate((data,data_block)) if not data is None else data_block
            
            labels = np.concatenate((labels,labels_i)) if not labels is None else labels_i
            #print(data.shape)   
                
        ff.close
        sf.close 
        p_data = np.row_stack(p_data)
        
        return data,labels,p_data  

def compute_distances_in_signatures(signatures,prttype_signature,metric ='L1'):
    
        half=int(signatures.shape[1]/2)
        # X_cor = np.asarray(list(map(lambda x: simpleL1(prttype_signature[:half],signatures[x][:half]),range(signatures.shape[0]))))
        if metric =='L1':
                # compute semantic_value in signatures
                X_cor  = np.asarray(list(map(lambda x: np.sum(signatures[x][:half]),range(signatures.shape[0]))))
                # compute prototype distance in signatures
                Y_cor  = np.asarray(list(map(lambda x: simpleL1(prttype_signature[half:],
                                                              signatures[x][half:]),range(signatures.shape[0]))))
        # compute values of prototype        
        P_x = np.sum(prttype_signature[:half])  # semantic value of category        
        P_y = np.sum(prttype_signature[half:])  # distance of prototype (0 in this case)
        
        return X_cor, Y_cor, P_x, P_y              
###############################################################################################
#                            prototype_organization
#----------------------------------------------------------------------------------------------
def compute_distances(metric,features,weight,bias,prototype,norm =False):
    # '''Compute all distance of category elements to prototype
    #    features -----> features block of category
    #    weigth,bias  - > weight learned in model
    #    prototype  ----> category prototype
    #    metric + norm -------> metric used to distance'''
    
    if norm:
        # Z
        if metric =='Z':
            X_cor = np.asarray(list(map(lambda x: semantic_vN(weight,features[x],bias),range(features.shape[0]))))
            P_x = semantic_vN(weight,prototype,bias)
        # aZ    
        if metric =='aZ': 
            X_cor = np.asarray(list(map(lambda x: semantic_avN(weight,features[x],bias),range(features.shape[0]))))
            P_x = semantic_avN(weight,prototype,bias)
            
        # compute Y
        Y_cor = np.asarray(list(map(lambda x: absoluteWxL1_N(prototype,features[x],weight),range(features.shape[0]))))
        P_y = absoluteWxL1_N(prototype,prototype,weight)
        
    else:
        # Z
        if metric =='Z':
            X_cor = np.asarray(list(map(lambda x: semantic_v(weight,features[x],bias),range(features.shape[0]))))
            P_x = semantic_v(weight,prototype,bias)
        # aZ    
        if metric =='aZ': 
            X_cor = np.asarray(list(map(lambda x: semantic_av(weight,features[x],bias),range(features.shape[0]))))
            P_x = semantic_av(weight,prototype,bias)
            
        # compute Y
        Y_cor = np.asarray(list(map(lambda x: absoluteWxL1(prototype,features[x],weight),range(features.shape[0]))))
        P_y = absoluteWxL1(prototype,prototype,weight)
        
    return X_cor, Y_cor, P_x, P_y    
 
#---------------------------------------------------------------------------------------------------- 
def top_n_prottype_distance(top_n,Y_data,X_data,plot=True):

        # compute top_min
        min_idx = Y_data.argsort()[:top_n]
        min_y = np.asarray(list(map(lambda x:Y_data[x],min_idx)))
        min_x = np.asarray(list(map(lambda x:X_data[x],min_idx)))

        # compute top max
        max_idx = Y_data.argsort()[-top_n:][::-1]
        max_y = np.asarray(list(map(lambda x:Y_data[x],max_idx)))
        max_x = np.asarray(list(map(lambda x:X_data[x],max_idx)))
        if plot:
            print('min_distance:',min_y,' index:',min_idx)
            print('max_distance:',max_y,' index:',max_idx)
            print('semantic_values_for_min:',min_x)
            print('semantic_values_for_max:',max_x)
        return (min_idx,min_y,min_x),(max_idx,max_y,max_x)
# ----------------------------------------------------------------------------------------------------
def prototype_organization_2D(top_n,Y_data,X_data,top_min_tuple,top_max_tuple,prottype_cor,
                              title ='2D_prototype_organization', aspect_ratio = (8,6)):
        
         # top N coordenate
        (min_idx,min_y,min_x),(max_idx,max_y,max_x) = top_min_tuple,top_max_tuple
         # prototype coordenate
        (P_x,P_y) = prottype_cor
        # Generate Data
        
        all_data = pd.DataFrame(dict(x=X_data, y=Y_data, label='other_members'))     # label=labels
        # more data
        min_d = pd.DataFrame(dict(x=min_x, y=min_y, label='top_' + str(top_n) + '_closest'))  # label=labels
        max_d = pd.DataFrame(dict(x=max_x, y=max_y, label='top_' + str(top_n) + '_furthest'))  # label=labels
        proto = pd.DataFrame(dict(x=[P_x], y=[P_y], label='prototype'))  # label=labels
        #join data
        df = pd.concat([all_data, min_d,max_d,proto])
        groups = df.groupby('label')

        # Plot
        fig, ax = plt.subplots(figsize=aspect_ratio)
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
        for name, group in groups:
            print(name)
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)
        ax.legend()
        plt.xlabel('semantic_value')
        plt.ylabel('prototype_distance')
        plt.title(title)
        plt.grid(True)
        plt.show()
# ------------------------------------------------------------------------------------------------------
def prototype_distance_ft_L1(top_n,Y_position,X_position,reference,data_range = (0,200),
                             title ='distance_behavior', x_label ='category_samples', aspect_ratio = (20, 4)):
    #  
    # Y_position -> prototype_distance to reference
    # X_position -> semantic_values
    
    # reference  -> (semantic_value,distancetoreference) = (semantic_value,0)  

    # compute distance in 2D space
    (P_x,P_y) = reference
    Px_n = np.full(Y_position.shape,P_x)
    Py_n = np.full(Y_position.shape,P_y)
    L1_dist = np.absolute(X_position - Px_n) + np.absolute(Y_position - Py_n)

    # plot distance
    fig, ax = plt.subplots(figsize=aspect_ratio)

    #fig  = plt.figure(figsize=(size*cols, size))
    x = np.arange(Y_position[data_range[0]:data_range[1]].shape[0])
    ax.plot(x, Y_position[data_range[0]:data_range[1]], label='prototype_distance')
    ax.plot(x, L1_dist[data_range[0]:data_range[1]], label='L1_distance')
    ax.plot(x, Y_position[data_range[0]:data_range[1]]*2,label='2_x_prototype_distance')

    plt.xlabel(x_label)
    plt.ylabel('distance_values')
    plt.title(title)
    ax.legend()
    plt.grid(True)
    plt.show()

###################################################################################################        
#                        Global Descriptor construction
#--------------------------------------------------------------------------------------------------
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
###################################################################################################
# 1 --- Compute imput vector ( features, std)
# --------------------------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------------------------    
# 2 --- Dimensionality reduction
# --------------------------------------------------------------------------------------------------
def dimensionality_reduction(features, aux_matrix_dim=None, scale=None,
                             plot=True,
                             plot_color='g',
                             save_file=None,
                             save_format='svg'):
    '''
        Simple reduction of input feature dimension.
        Note:    Use the function 2-times    to    plot: 1    to    descriptor and know    de    max    value    to    scale
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
    A = angles_matrix(aux_matrix_dim)

    # graphic signature dimensions
    rows = int(features.shape[0] / aux_matrix_dim)
    cols = int(features.shape[1] / aux_matrix_dim)

    # unitary signature count
    block_count = cols * rows
    block_size = aux_matrix_dim

    # ----------------------------MAPPING FEATURES--------------------------
    M = [None for _ in range(block_count)]
    count = 0
    i_init = 0
    for i in range(rows):
        j_init = 0
        for j in range(cols):
            # Mapping Features
            M[count] = features[i_init:i_init + block_size, j_init:j_init + block_size]
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
            unitary = unitary_descriptor(A.flatten(), M[i].flatten(),
                                         plot=plot, ax=ax, title=str(i), scale=scale, color=plot_color)
            signature = unitary if signature is None else np.concatenate((signature, unitary), axis=0)
        plt.tight_layout()

        if not save_file is None:
            # fig.savefig(save_file + '.png')
            fig.savefig(save_file + '.' + save_format, bbox_inches='tight')
        else:
            plt.show()
            plt.close(fig)
    # ---------------------------Concatenate unitary signatures -------------------------------------------
    else:
        signature = None
        for i in range(block_count):
            unitary = unitary_descriptor(A.flatten(), M[i].flatten(), plot=plot)
            signature = unitary if signature is None else np.concatenate((signature, unitary), axis=0)

    return signature


# --------------------------------------------------------------------------------------------------
#    2.1 Compute angles 
# -------------------------------------------------------------------------------------------------
def angles_matrix(matrix_dim):
    '''
    Build angles matrix. Use sectors_angles() function.
        C2 | C1
        ___|___
        C3 | C4

    :param matrix_dim: r-dimension of auxiliary matrix (X_r_x_r)
    :return: Return a r-dimensional matrix of angles (in 2d coordinate)
    '''
    sector_size = int(matrix_dim / 2)
    C1, C2, C3, C4 = sectors_angles(sector_size)
    A = np.zeros((matrix_dim, matrix_dim))
    A[:int(sector_size), :int(sector_size)] = C2
    A[:int(sector_size), int(sector_size):] = C1
    A[int(sector_size):, :int(sector_size)] = C3
    A[int(sector_size):, int(sector_size):] = C4
    return A
#-------------------------------------------------------------------------------------------------
#    2.1.1 Build angles of each sector
# --------------------------------------------------------------------------------------------------
def sectors_angles(sector_dim):
    '''
    Build angles of each sector (4 sectors)

    :param sector_dim: sector dimension
    :return: 4 angles sector
    '''

    cell_size = 2

    # build C1 [0-90 degree]
    C1 = np.zeros((sector_dim, sector_dim))
    for j in range(sector_dim):
        for i in range(sector_dim):
            angle_tan = (i * cell_size + cell_size / 2) / (j * cell_size + cell_size / 2)
            C1[j, i] = math.degrees(math.atan(angle_tan))
    C1 = np.rot90(C1)

    # To diagonal (same angles values)
    max_less_45 = C1[1, sector_dim - 1]
    diagonal_increase = (45 - max_less_45 - 3) / (sector_dim / 2)
    # print( max_less_45,diagonal_increase)
    for i in range(sector_dim):
        if i < sector_dim / 2:
            C1[sector_dim - i - 1, i] = max_less_45 + (i + 1) * diagonal_increase
        else:
            C1[sector_dim - i - 1, i] = 90 - max_less_45 - (i - sector_dim / 2 + 1) * diagonal_increase

    # build C4 [270-360 degree]
    tmpC = np.flip(C1, 0)
    C4 = -tmpC + 360
    # build C3 [180-270 degree]
    tmpC = np.flip(tmpC, 1)
    C3 = tmpC + 180
    # build C2 [90-180 degree]
    tmpC = np.flip(C1, 1)
    C2 = 180 - tmpC

    return C1, C2, C3, C4


# --------------------------------------------------------------------------------------------------
#       2.2 Compute_signature_unitary_M (sum and return 8 vector + plot option)
# --------------------------------------------------------------------------------------------------
def unitary_descriptor(angles, features, plot=True, ax=None, title=' ', scale=10, color='g'):
    '''
    Compute unitary_signature of input features (mapped).
    Sum the features vectors (semantic gradient) using angles counter-clock wise fashion( starting  in 45 degree).
    :param angles: angles matrix A
    :param features: features vectors
    :param plot: plot unitary signature
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
    V = np.zeros((8, 2))
    # angles between sum
    angles = np.arange(45, 405, 45)
    angles[7] = 0
    # angles magnitudes sum
    for count in range(8):
        # endx
        V[count, 0] = np.abs(magnitudes[count]) * math.cos(math.radians(angles[count]))
        # endy
        V[count, 1] = np.abs(magnitudes[count]) * math.sin(math.radians(angles[count]))
        # plotting
    if ax:  # to plot dinamicaly ()
        ax.quiver(*origin, V[:, 0][magnitudes >= 0], V[:, 1][magnitudes >= 0],
                  color=positive_color, angles='xy', scale_units='xy', scale=1, width=0.012)

        ax.quiver(V[:, 0][magnitudes < 0], V[:, 1][magnitudes < 0], -V[:, 0][magnitudes < 0], -V[:, 1][magnitudes < 0],
                  color='r', angles='xy', scale_units='xy', scale=1, width=0.012)

        ax.set_title(title)
        ax.axis([-scale, scale, -scale, scale])
    else:
        plt.quiver(*origin, V[:, 0], V[:, 1], scale=np.max(np.max(V[:, 0]), np.max(V[:, 1])), scale_units='inches')
        # color=['r','b','g'],, ,scale=1
        plt.show()

# --------------------------------------------------------------------------------------------------
#       3.0 Compute signature and plot options
# --------------------------------------------------------------------------------------------------
# mnist_compute_plot_prttype - > mnist_compute_plot_descriptor
def compute_plot_signature(model, features, class_i, title=None, save_file=None, semantic=True,
                        plotting=True, plot_ceil=True, plot_color='g'):
    input_vector = compute_input_vector(model, class_i, features, semantic=semantic)
    # compute without plot
    signature = dimensionality_reduction(input_vector, plot=False)
    if plotting or save_file:
        # print(math.ceil(np.max(df)))
        # file_name to save (if save)
        file_name = save_file + '_graph' if save_file else None
        # graph plot (normalized to max (signature))
        plot_scale = math.ceil(np.max(np.abs(signature))) if plot_ceil else np.max(np.abs(signature))
        signature = dimensionality_reduction(input_vector, plot=True, scale=plot_scale,
                                             plot_color=plot_color, save_file=file_name)
        file_name = save_file + '_hist' if save_file else None
        title = str(class_i) if title is None else title
        # histogram plot
        # descriptor_signature_plot_h(signature,aspect_ratio,title = title,save_fig=file_name)

    return signature

# --------------------------------------------------------------------------------------------------
#       3.1,3.2 -  global_prttype_descriptor 
#             if category =True  3.1  (category_signature) -> | semantic_value | std category|                         
#             if category =False 3.2   prototype_signature  > | semantic_value | 000000000000|  
# --------------------------------------------------------------------------------------------------
def global_prttype_descriptor(model,class_i,file_prototypes,
                              category=True,plotting=True,
                              plot_ceil=True,
                              save_file=None):
    # open
    class_i = int(class_i)
    fprototype = h5py.File(file_prototypes, 'r')
    prttype_mean = fprototype['mean'][class_i]

    save_file_semantic = save_file + '_semantic' if save_file else save_file
    semantic_signature = compute_plot_signature(model,prttype_mean,class_i,plotting = plotting,
                                                 plot_ceil=plot_ceil,
                                                 save_file = save_file_semantic)
    if category: # category descriptor
            prttype_std = fprototype['std'][int(class_i)]
            save_file_diff = save_file + '_difference' if save_file else save_file
            boundary_signature  = compute_plot_signature(model,prttype_std,class_i,plotting = plotting,
                                                          plot_ceil=plot_ceil, semantic=False,save_file = save_file_diff,
                                                          plot_color='m')
    else:        # prttype_descriptor
        boundary_signature = np.full(semantic_signature.shape,0)
    fprototype.close()    
    return np.concatenate((semantic_signature,boundary_signature))


# --------------------------------------------------------------------------------------------------
#       4 -  global_features_descriptor  (extended = False)
#            
# --------------------------------------------------------------------------------------------------
def global_features_descriptor(model, features, class_i, prototype_dif, extended=False,
                               plotting=False, plot_ceil=True, aspect_ratio=(10, 5), save_file=None):

    save_file_semantic = save_file + '_semantic' if save_file else save_file
    semantic_meaning = compute_plot_signature(model, features, class_i, plotting=plotting,
                                                 plot_ceil=plot_ceil,
                                                 save_file=save_file_semantic)

    if extended:  # concatenate d_base + std(respect to category_prototype)
        save_file_diff = save_file + '_difference' if save_file else save_file
        semantic_difference = compute_plot_signature(model, prototype_dif, class_i, plotting=plotting,
                                                plot_ceil=plot_ceil,
                                                semantic=False, save_file=save_file_diff)

        return np.concatenate((semantic_meaning, semantic_difference))

    return semantic_meaning
# --------------------------------------------------------------------------------------------------
#      5 -  plot asignatures as histograms
# --------------------------------------------------------------------------------------------------
# print descriptor histogram descriptor_pttype_plot_h -> descriptor_signature_plot_h
def descriptor_signature_plot_h(signature,figsize=(30, 6),title = '',save_fig = None,save_format='svg'):
    fig = plt.figure(figsize=figsize)
    x = np.arange(0,len(signature),1)
    #print(x)
    colors = ['g' for _ in np.arange(len(signature))]
    for i in x[signature<0]:
                colors[i] = 'r'
    plt.bar(x,signature,color =colors)
    plt.title(title)
    plt.xlabel('dimension')
    plt.ylabel('magnitude')  
    plt.tight_layout()
    plt.show()
    if not save_fig is None:
            fig.savefig(save_fig + '.' + save_format, bbox_inches='tight')
    plt.close(fig)
#---------------------------------------------------------------------------------------------------
#
#     Compute for model, the descriptors of features(f_block), default...extended version
#
# --------------------------------------------------------------------------------------------------
def compute_block_of_descriptors(model,f_block,label_block,file_prototype,
                                 extended = True,plotting=False,type_plot=0,save_file =None):
    # open 
    block = []
    fprototype = h5py.File(file_prototype, 'r')
    for feature,class_i in zip(f_block,label_block):
        prttype_mean= fprototype['mean'][int(class_i)]
        if type_plot:
             plot_ceil=False
             aspect_ratio =(40,5)
             
        else:
           plot_ceil=True
           aspect_ratio =(10,5)

        #descrptr = mnist_compute_plot_prttype(model,prttype_i,int(class_i),plotting=False)
        feature_signature = global_features_descriptor(model,feature,int(class_i),np.absolute(feature-prttype_mean),extended=extended,
                                                      plotting=plotting,plot_ceil=plot_ceil,aspect_ratio =aspect_ratio,
                                                      save_file=save_file)
        block.append(feature_signature)
    fprototype.close()    
    return np.row_stack(block)
# --------------------------------------------------------------------------------------------------
# plot mean,std,W,b of prototype
def plot_save_prototype_representation(dir_path,class_number,data_x,X_mean,X_SD,
                           titles = {'xlabel':'Feature Index','yMean':'Mean Feature Value','ySD':'SD Feature Value','yMeanSD':'Mean/SD Feature Value'},
                           save_format='png',verbose=True,save = True):
 
        i = class_number
        px = data_x
        file_name = 'Class_{}_Mean_SD_Feature'.format(i)
        fig = plt.figure(figsize=(18,15)) # (w,h)
        gs = gridspec.GridSpec(3, 1) 
        # Mean
        ax0 = plt.subplot(gs[0])
        ax0.set_title('Class {}'.format(i))
        ax0.set_ylabel(titles['yMean'])
        ax0.set_xlabel(titles['xlabel'])
        ax0.plot(px,X_mean, 'go-')
        # SD
        ax1 = plt.subplot(gs[1])
        ax1.set_title('Class {}'.format(i))
        ax1.set_ylabel('SD Feature Value') 
        ax1.set_xlabel(titles['xlabel'])
        ax1.plot(px,X_SD, 'ro-')
        # Mean + - SD
        ax2 = plt.subplot(gs[2])
        ax2.set_title('Class {}'.format(i))
        ax2.set_ylabel(titles['yMeanSD']) 
        ax2.set_xlabel(titles['xlabel'])
        ax2.errorbar(px,X_mean, X_SD, linestyle='None', marker='o')
        
        file_path = '{}/{}.{}'.format(dir_path,file_name,save_format)
        if(save):
            plt.tight_layout()
            plt.savefig(file_path)
        if(verbose):
            plt.show()
# --------------------------------------------------------------------------------------------------            