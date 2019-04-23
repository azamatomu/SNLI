import torchtext
from torchtext import data, datasets
import nltk
from nltk import word_tokenize
nltk.download('punkt')

def load_glove_model(glovedir):
    print("Loading Glove Model")
    model = torchtext.vocab.Vectors(glovedir)
    print("Done.")
    return model

def load_data():
    #all_data = {"train": None, "dev": None, "test": None}
    x_field = data.Field(lower=True,
                 tokenize=word_tokenize,
                 include_lengths=True)
    y_field = data.Field(sequential=False,
                 pad_token=None,
                 unk_token=None,
                 is_target=True)
    all_data = {}
    all_data["train"], all_data["dev"], all_data["test"]= datasets.SNLI.splits(x_field, y_field)
    return all_data, x_field, y_field
