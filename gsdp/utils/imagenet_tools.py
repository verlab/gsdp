# utils import
import os
import pandas as pd

# ImageNet synsets tools
# -------------------------------------------------------------------------------------------------------------------
def synsets_map(root_path_dataset=None):
    '''
    Load  ImageNet meta_clsloc.mat map as csv
    :param root_path_dataset: folder path of Imagenet .csv maps
    :return: pandas
    '''
    if root_path_dataset is None:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        root_path_dataset = os.path.join(os.path.dirname(os.path.dirname(current_dir)),'data','ImageNet')

    for file in ['clsloc_class_labels.csv', 'det_class_labels.csv']:
        file_name = os.path.join(root_path_dataset, file)
        if os.path.isfile(file_name):
            class_map = pd.read_csv(file_name, names=['id', 'wnID', 'words'], header=0)
            return class_map

    raise Exception(" Maps files  are not available in %s !!!!. Update config.dataset_path " % root_path_dataset)


synsets_ImageNet = synsets_map()


# ImageNet mapping function
# -------------------------------------------------------------------------------------------------------------------
def synset_to_id(synset,synsets_map=None,to_categorize=True):
    '''
    Returns the ImageNet id corresponding to input synset
    :param synset: ImageNet synset (Ex: n02123394 )
    :param synsets_map: pandas synsets_ImageNet map (return of synsets_map() function)
    :param to_categorize: ImageNet range -> [1..1000] , categorization range -> [0..999]
                          to_categorrize is a parameter to select which range is desired
    :return: ImageNet id (Ex: for synset n02123394. id = 9 if to_categorize is True else id = 10)
    '''
    if synsets_map is None:
        synsets_map = synsets_ImageNet
    category_id = synsets_map[synsets_map['wnID'] == synset]['id'].to_string(index=False)
    return int(category_id)-1 if to_categorize else int(category_id)
# -------------------------------------------------------------------------------------------------------------------
def synset_to_words(synset,synsets_map=None):
    '''
    Returns the ImageNet category name corresponding to input synset
    :param synset: ImageNet synset (Ex: n02123394 )
    :param synsets_map: pandas synsets_ImageNet map (return of synsets_map() function)
    :return: ImageNet category name (Ex: or synset n02123394, return Persian_cat)
    '''
    if synsets_map is None:
        synsets_map = synsets_ImageNet
    category_name = synsets_map[synsets_map['wnID'] == synset]['words'].to_string(index=False)
    return category_name

# -------------------------------------------------------------------------------------------------------------------
def id_to_synset(id_,synsets_map=None,to_categorize=True):
    '''
    Returns the ImageNet synset corresponding to input ImageNet id
    :param id_: ImageNet id
    :param synsets_map: pandas synsets_ImageNet map (return of synsets_map() function)
    :param to_categorize: ImageNet range -> [1..1000] , categorization range -> [0..999]
                          to_categorrize is a parameter to select which range is desired
    :return: ImageNet synset
    '''
    if synsets_map is None:
        synsets_map = synsets_ImageNet
    id_ = int(id_) + 1 if to_categorize else int(id_)
    return synsets_map[synsets_map['id'] == id_]['wnID'].to_string(index=False)

# -------------------------------------------------------------------------------------------------------------------
def id_to_words(id_,synsets_map=None,to_categorize=True):
    '''
    Returns the ImageNet category name corresponding to input ImageNet id
    :param id_: ImageNet id
    :param synsets_map: pandas synsets_ImageNet map (return of synsets_map() function)
    :param to_categorize: ImageNet range -> [1..1000] , categorization range -> [0..999]
                          to_categorrize is a parameter to select which range is desired
    :return: ImageNet category name
    '''
    if synsets_map is None:
        synsets_map = synsets_ImageNet
    id_ = int(id_) + 1 if to_categorize else int(id_)
    return synsets_map[synsets_map['id'] == id_]['words'].to_string(index=False)

# -------------------------------------------------------------------------------------------------------------------
# def map_wnID2_x(wnID, map_values= ['id','words'],classes_map =None):  # map_values = ['id','words']
#     '''
#     ImageNet category mapping from WordNet category ID(wnID) to 'id','words'
#     :param wnID: WordNet ID
#     :param map_values: values to return
#     :param classes_map: ImageNet Category mapping (pandas format)
#     :return: classes_map[wnID]
#     '''
#     if classes_map is None:
#         classes_map = synsets_ImageNet
#     return_values = []
#     for value in map_values:
#         if value not in ['id', 'words']:
#             raise Exception("Parameter  %s don't exist in map file!!!! " % value)
#         class_value = classes_map[classes_map['wnID'] == wnID][value].to_string(index=False)
#         return_values.append(class_value)
#     return return_values
#
# # -------------------------------------------------------------------------------------------------------------------
#
# def map_ID2_x(id, map_values=['wnID','words'],classes_map =None):  # map_values = ['wnID','words']
#     '''
#     ImageNet category mapping from category number (id) to 'wnID','words'
#     :param id: category index (number)
#     :param map_values: values to return
#     :param classes_map: ImageNet Category mapping (pandas format)
#     :return: classes_map[id]
#     '''
#     if classes_map is None:
#         classes_map = synsets_ImageNet
#
#     return_values = []
#     for value in map_values:
#         if value not in ['wnID', 'words']:
#             raise Exception("Parameter  %s don't exist in map file!!!! " % value)
#         class_value = classes_map[classes_map['id'] == id][value].to_string(index=False)
#         return_values.append(class_value)
#     return return_values

# -------------------------------------------------------------------------------------------------------------------
# def labels_map(root_path_dataset=None):
#     if root_path_dataset is None:
#         current_dir = os.path.dirname(os.path.realpath(__file__))
#         root_path_dataset = os.path.join(os.path.dirname(os.path.dirname(current_dir)),'data','ImageNet')
#
#     for file in ['clsloc_class_labels.csv', 'det_class_labels.csv']:
#         file_name = os.path.join(root_path_dataset, file)
#         if os.path.isfile(file_name):
#             class_map = pd.read_csv(file_name, names=['id', 'wnID', 'words'], header=0)
#             return class_map
#
#     raise Exception(" Maps files  are not available in %s !!!!. Update config.dataset_path " % root_path_dataset)
