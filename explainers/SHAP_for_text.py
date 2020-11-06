import numpy as np
import torch
from nltk.tokenize import TweetTokenizer
import logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)


class SHAPexplainer:

    def __init__(self, model, tokenizer, words_dict, words_dict_reverse):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cpu"
        self.tweet_tokenizer = TweetTokenizer()
        self.words_dict = words_dict
        self.words_dict_reverse = words_dict_reverse

    def predict(self, indexed_words):
        # self.model.to(self.device)

        sentences = [[self.words_dict[xx] if xx != 0 else "" for xx in x] for x in indexed_words]
        indexed_tokens, _, _ = self.tknz_to_idx(sentences)

        # ref = self.tweet_tokenizer.tokenize(data[0])
        # data_temp = [ref]
        # for x in data[1:]:
        #     ref_temp = ref.copy()
        #     new = ["" for _ in range(len(ref))]
        #     x = self.tweet_tokenizer.tokenize(x)
        #     for w in x:
        #         id = ref_temp.index(w)
        #         new[id] = w
        #         ref_temp[id] = ""
        #     data_temp.append(new)
        #
        # data_temp = [["[PAD]" if xx == "" else xx for xx in x] for x in data_temp]
        # data_temp = [" ".join(x) for x in data_temp]
        #
        # tokenized_nopad = [self.tokenizer.tokenize(text) for text in data_temp]
        # MAX_SEQ_LEN = max(len(x) for x in tokenized_nopad)
        # tokenized_text = [["[PAD]", ] * MAX_SEQ_LEN for _ in range(len(data))]
        # for i in range(len(data)):
        #     tokenized_text[i][0:len(tokenized_nopad[i])] = tokenized_nopad[i][0:MAX_SEQ_LEN]
        # indexed_tokens = [self.tokenizer.convert_tokens_to_ids(tt) for tt in tokenized_text]

        tokens_tensor = torch.tensor(indexed_tokens)
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

    def tknz_to_idx(self, train_data, MAX_SEQ_LEN=None):
        tokenized_nopad = [self.tokenizer.tokenize(" ".join(text)) for text in train_data]
        if not MAX_SEQ_LEN:
            MAX_SEQ_LEN = min(max(len(x) for x in train_data), 512)
        tokenized_text = [['[PAD]', ] * MAX_SEQ_LEN for _ in range(len(tokenized_nopad))]
        for i in range(len(tokenized_nopad)):
            tokenized_text[i][0:len(tokenized_nopad[i])] = tokenized_nopad[i][0:MAX_SEQ_LEN]
        indexed_tokens = np.array([np.array(self.tokenizer.convert_tokens_to_ids(tt)) for tt in tokenized_text])
        return indexed_tokens, tokenized_text, MAX_SEQ_LEN

    def dt_to_idx(self, data, max_seq_len=None):
        idx_dt = [[self.words_dict_reverse[xx] for xx in x] for x in data]
        if not max_seq_len:
            max_seq_len = min(max(len(x) for x in idx_dt), 512)
        for i, x in enumerate(idx_dt):
            if len(x) < max_seq_len:
                idx_dt[i] = x + [0] * (max_seq_len - len(x))
        return np.array(idx_dt), max_seq_len
