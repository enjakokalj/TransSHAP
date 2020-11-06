import numpy as np
import torch
from nltk.tokenize import TweetTokenizer
import logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)


class LIMExplainer:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cpu"
        self.tweet_tokenizer = TweetTokenizer()

    def predict(self, data):
        self.model.to(self.device)

        ref = self.tweet_tokenizer.tokenize(data[0])
        data_temp = [ref]
        for x in data[1:]:
            ref_temp = ref.copy()
            new = ["" for _ in range(len(ref))]
            x = self.tweet_tokenizer.tokenize(x)
            #if ".   ." in x:
            #    x = [xx if xx != ".   ." else [xxx for xxx in xx.split(" ") if xxx] for xx in x]
            #    x = [xx for xxx in x for xx in xxx]
            for w in x:
                id = ref_temp.index(w)
                new[id] = w
                ref_temp[id] = ""
            data_temp.append(new)

        data_temp = [["[PAD]" if xx == "" else xx for xx in x] for x in data_temp]
        data_temp = [" ".join(x) for x in data_temp]

        tokenized_nopad = [self.tokenizer.tokenize(text) for text in data_temp]
        MAX_SEQ_LEN = max(len(x) for x in tokenized_nopad)
        tokenized_text = [["[PAD]", ] * MAX_SEQ_LEN for _ in range(len(data))]
        for i in range(len(data)):
            tokenized_text[i][0:len(tokenized_nopad[i])] = tokenized_nopad[i][0:MAX_SEQ_LEN]
        indexed_tokens = [self.tokenizer.convert_tokens_to_ids(tt) for tt in tokenized_text]
        tokens_tensor = torch.tensor(indexed_tokens)
        tokens_tensor = tokens_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=tokens_tensor)
            # logits = outputs[0]
            predictions = outputs.detach().cpu().numpy()
        final = [self.softmax(x) for x in predictions]
        return np.array(final)

    def softmax(self, it):
        exps = np.exp(np.array(it))
        return exps / np.sum(exps)

    def split_string(self, string):
        data_raw = self.tweet_tokenizer.tokenize(string)
        data_raw = [x for x in data_raw if x not in ".,:;'"]
        return data_raw
