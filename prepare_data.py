from gsdp.utils.dump_tools import make_path_exists
import os
import wget

# init  -----------------------------------------------------------------------------
_CUSTOM_MODELS = ['MNIST']
_KERAS_MODELS = ['VGG16', 'ResNet50']
prototypes_relative_path = '/data/*.h5'
models_relative_path = '/model/*.json'
weights_relative_path = '/model/*.h5'
current_dir2 = os.path.dirname(os.path.realpath(__file__))
url_datasets = "https://www.verlab.dcc.ufmg.br/gsdp/"

# local procedures -------------------------------------------------------------------
def download_gsdp_data():
    _models = _CUSTOM_MODELS + _KERAS_MODELS
    models_path_name = 'models'
    for model in _models:
        print('\n'+ '#'*12 + '   Downloading data for Keras-' + model + ' pre-trained model  ' + '#'*50)
        prototypes_filename = os.path.join(models_path_name,model,'data','prototypes.h5')
        output_dir = os.path.join(models_path_name,model,'data')
        make_path_exists(output_dir)
        prototypes_url = url_datasets + prototypes_filename
        print('\n  ---> Downloading computed prototypes from: ' + prototypes_url+ ')')
        #wget.download(prototypes_url, out=output_dir)

        if model in _CUSTOM_MODELS:
            print('\n  ---> Downloading custom model files:')
            for file in ['model.json','weights.h5']:
                        model_filename = os.path.join(models_path_name, model, 'model',file )
                        output_dir = os.path.join(models_path_name, model, 'model')
                        make_path_exists(output_dir)
                        file_url = url_datasets + model_filename
                        print('\n       -> Downloading Keras-' +  model+ ' ' + file + ' from: ' + file_url + ')')
                        #wget.download(file_url, out=output_dir)

def models_resources():
    _models = _CUSTOM_MODELS + _KERAS_MODELS
    resources = []
    for model in _models:
        resources.append(model + prototypes_relative_path)
        if model in _CUSTOM_MODELS:
            resources.append(model + models_relative_path)
            resources.append(model + weights_relative_path)
    return resources

# MAIN
if __name__ == "__main__":
          # download prototypes and models weights
          download_gsdp_data()