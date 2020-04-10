from flask import Flask
# import transformers
import torch
# from pytorch_pretrained_bert import BertTokenizer

app = Flask(__name__)

MAX_LEN = 512
BERT_PATH = "BERT_PyTorch/"
#BERT_PATH = "assets.tar.gz/"

# BERT_CONFIG = BERT_PATH+'/bert_config.json'
# BERT_MODEL = BERT_PATH+'pytorch_model.bin'
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH,do_lower_case=True)
# LABEL_LIST = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
