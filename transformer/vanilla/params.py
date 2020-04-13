import os
import datetime

IS_COLAB = False

# tokenizer params
TARGET_VOCAB_SIZE = 2**13

# Maximum number of samples to preprocess
MAX_LENGTH = 16
MAX_SAMPLES = 9999999

# Hyper-parameters
NUM_LAYERS = 2
D_MODEL = 128
NUM_HEADS = 8
UNITS = 128
DROPOUT = 0.1


if IS_COLAB:
    model_path = "/content/drive/My Drive/discordbot/saved_model/vanilla"  #@param {type:"string"}
else:
    model_path = "./saved_model/vanilla"  #@param {type:"string"}

if not os.path.exists(model_path):
    os.makedirs(model_path)

model_weights_path = model_path + '/weights.h5'
tokenizer_path = model_path + '/saved_tokenizer.pickle'
model_config_path = model_path + '/model_config.pickle'
dataset_config_path = model_path + '/dataset_config.pickle'
train_config_path = model_path + '/train_config.pickle'
log_dir = model_path + '/logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
