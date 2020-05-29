import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import logging
logging.basicConfig(level=logging.INFO)
import pandas as pd
import visualize_explanations


USE_GPU = 0
MAX_SEQ_LEN = 40 # poljubno do max 512
def word2tokens(word, tokenized):
    if word in tokenized:
        return [tokenized.index(word)]
    else:
        for t in range(len(tokenized)):
            if tokenized[t] in word and '##' in tokenized[t+1]:
                isword = tokenized[t]
                for j in range(t+1, len(tokenized)):
                    isword+=tokenized[j][2:]
                    if word == isword:
                        return([i for i in range(t,j+1)])
                        break

class SCForShap(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
    #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        output = super().forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels)
        return output[0]

# Device configuration
device = torch.device('cuda' if (torch.cuda.is_available() and USE_GPU) else 'cpu')

# Load pre-trained model tokenizer (vocabulary)
#pretrained_model = 'twitter_results/transformers_result_sl_shebert-f128'
pretrained_model = './model_bert_crosloengual/'
tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=False)
#model = BertForSequenceClassification.from_pretrained(pretrained_model)
model = SCForShap.from_pretrained(pretrained_model)


text1 = "Why is it that 'fire sauce' isn't made with any real fire? Seems like false advertising." #neutral
text2 = "Being offensive isn't illegal you idiot." #negative
text3 = "Loving the first days of summer! <3" #positive
texts = [text1, text2, text3]
num_inputs = len(texts)

train_data = pd.read_csv("./model_bert_crosloengual/English_tweet_label.csv")
train_data = list(train_data["Tweet text"])

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
bag_of_words = set([xx for x in train_data for xx in tknzr.tokenize(x)])

import IME_for_text
explainer = IME_for_text.Explainer(model, data=train_data, n_iter=50, err=0.05, tokenizer=tokenizer, bag_of_words=bag_of_words)
contribution_values, tokenized_text, predictions = explainer.explain(texts)
print("done\n", contribution_values)

real_preds = [1, 0, 3]

for i in range(len(tokenized_text)):
    text = tokenized_text[i]
    pred1 = real_preds[i]
    pred2 = int(np.argmax(predictions[i]))

    values = contribution_values[i][:, pred2]
    class_to_explain = ["Negative", "Neutral", "Positive"][pred2]
    visualize_explanations.plot_explanation(text, values, class_to_explain)
    visualize_explanations.text_box_explanation(text, values, class_to_explain)
