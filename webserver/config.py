from flask import Flask
import transformers
import torch

app = Flask(__name__)

MAX_LEN = 512
BERT_PATH = "model/language/bert-base-uncased/"
MODEL_PATH = "model/language/model2.bin"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
