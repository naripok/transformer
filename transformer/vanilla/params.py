import os
import datetime

if os.environ.get('IS_COLAB', False):
    model_path = "/content/drive/My Drive/saved_model/vanilla"
else:
    model_path = "./saved_model/vanilla"

if not os.path.exists(model_path):
    os.makedirs(model_path)

model_weights_path = model_path + '/weights.h5'
tokenizer_path = model_path + '/saved_tokenizer.pickle'
model_config_path = model_path + '/model_config.pickle'
dataset_config_path = model_path + '/dataset_config.pickle'
train_config_path = model_path + '/train_config.pickle'
log_dir = model_path + '/logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
