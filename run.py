import pandas as pd
from nltk.tokenize import TweetTokenizer

# Bag of words
train_data = pd.read_csv("./models/English_tweet_label.csv")
train_data = list(train_data["Tweet text"])
tknzr = TweetTokenizer()
bag_of_words = set([xx for x in train_data for xx in tknzr.tokenize(x)])

from transformers import BertTokenizer, BertForSequenceClassification

class SCForShap(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        output = super().forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels)
        return output[0]

pretrained_model = "./models/en_balanced_bertbase-g512-epochs3/"
tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=False)
model = SCForShap.from_pretrained(pretrained_model)

# Test texts
t1 = "Why is it that 'fire sauce' isn't made with any real fire? Seems like false advertising." #neutral
t2 = "Being offensive isn't illegal you idiot." #negative
t3 = "Loving the first days of summer! <3" #positive
t4 = "I hate when people put lol when we are having a serious talk ."   #negative
t5 = "People are complaining about lack of talent. Takes a lot of talent to ignore objectification and degradation #MissAmerica" #neutral
t6 = "Shit has been way too fucked up, for way too long. What the fuck is happening" #negative
t7 = "@Lukc5SOS bc you're super awesome😉" #positive
t8 = "RT @JakeBoys: This show is amazing! @teenhoot family are insane 😍" #positive
t9 = "So kiss me and smile for me 😊💗 http://t.co/VsRs8KUmOP"
# t6 = "RT @deerminseok: YOU'RE SO BEAUTIFUL. I MISS YOU SO MUCH. http://t.co/VATdCVypqC" #positive
# t7 = "😄😃😄 love is in the air ❤️❤️❤️ http://t.co/y1RDP5EdwG" #positive
texts = [t1, t2, t3, t4, t5, t6, t7, t8, t9]

import shap
import random
import numpy as np
import logging
import matplotlib.pyplot as plt
from explainers.SHAP_for_text import SHAPexplainer
from explainers import visualize_explanations
logging.getLogger("shap").setLevel(logging.WARNING)
shap.initjs()

words_dict = {0: None}
words_dict_reverse = {None: 0}
for h, hh in enumerate(bag_of_words):
    words_dict[h + 1] = hh
    words_dict_reverse[hh] = h + 1

predictor = SHAPexplainer(model, tokenizer, words_dict, words_dict_reverse)
# rand_id = random.sample([*range(len(train_data))], 500)
# train_dt = np.array([predictor.split_string(x) for x in np.array(train_data)[rand_id]])
train_dt = np.array([predictor.split_string(x) for x in np.array(train_data)])
idx_train_data, max_seq_len = predictor.dt_to_idx(train_dt)

explainer = shap.KernelExplainer(model=predictor.predict, data=shap.kmeans(idx_train_data, k=1000)) #idx_train_data)

texts_ = [predictor.split_string(x) for x in texts][1:4]
idx_texts, _ = predictor.dt_to_idx(texts_, max_seq_len=max_seq_len)

for ii in range(len(idx_texts)):
    t = idx_texts[ii]
    to_use = t.reshape(1, -1)
    f = predictor.predict(to_use)
    pred_f = np.argmax(f[0])

    shap_values = explainer.shap_values(X=to_use, l1_reg="aic", nsamples="auto") #nsamples=64

    # shap.force_plot(explainer.expected_value[m], shap_values[m][0, :len_], texts_[ii])

    visualize_explanations.joint_visualization(texts_[ii], shap_values[pred_f][0, :len(texts_[ii])],
                                               ["Positive", "Neutral", "Negative"][int(pred_f)], f[0][pred_f], ii)
