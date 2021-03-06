# Conversational Transformer Models Collection

This repository contains a collection of Conversational Transformer models.
It uses tensorflow and stanford's convokit for data fetching and
preprocessing.


## Installation

- On a python 3.7+ virtual environment:
```pip install -r requirements.txt```


## Training and Running

After installing the package, you can run the training with the commands:

```python -m src.biconditional.train --new --train-model --train-tokenizer```  
```python -m src.vanilla.train --new --train-model --train-tokenizer```

If on colab environment, you can leverage the TPU cluster for distributed training.
Install the packages:

```
!pip install convokit
!python3 -m spacy download en
!pip install git+https://github.com/naripok/transformer.git@release-0.0.5
```

Then, activate the TPU environment on the notebook settings
and set IS_COLAB env var to True before running the trainin stript, like so:

```!export IS_COLAB=True && python -m src.biconditional.train --new --train-model --train-tokenizer```  
```!export IS_COLAB=True && python -m src.vanilla.train --new --train-model --train-tokenizer```


## Interacting with the models

After training, you can interact with the models on a interactive
question-answer session with the commands:

```python -m src.biconditional.interact```  
```python -m src.vanilla.interact```


## Reference

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Google’s Neural Machine Translation System: Bridging the Gapbetween Human and Machine Translation](https://arxiv.org/abs/1609.08144)
- [Transformer-XL paper](https://arxiv.org/pdf/1901.02860.pdf)
- [Convokit docs](https://convokit.cornell.edu/)
- [Colab Transformer Model Chatbot](https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb#scrollTo=rHMPkA2eQrpT)
- [Tensorflow v2.2.0rc2 API docs](https://www.tensorflow.org/versions/r2.2/api_docs/python/tf)
