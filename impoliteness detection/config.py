import transformers
import torch

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "model/bert-base-uncased/"
MODEL_PATH = "model/model2.bin"
TRAINING_FILE = "data/dataset.csv"
PROCESSED_FILE = "data/processed_dataset.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH,do_lower_case=True)
