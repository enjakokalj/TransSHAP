import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
import random
from nltk.tokenize import TweetTokenizer
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


class IMExplainer:

    def __init__(self, model, data, size_bg_data, n_iter, err, tokenizer, bag_of_words):
        self.model = model
        self.data = data if type(data) == np.ndarray else np.array(data)
        self.n_iter = n_iter
        self.err = err
        self.nX = self.data.shape[0]
        self.tokenizer = tokenizer
        self.bag_of_words = bag_of_words
        self.device = "cpu"
        self.tweet_tokenizer = TweetTokenizer()
        self.tweets = []

        words_dict = {0: None}
        for h, hh in enumerate(bag_of_words):
            words_dict[h + 1] = hh

        self.words_dict = words_dict

        model.to(self.device)
        model.eval()
        all_outputs = []
        rand_id = random.sample([*range(self.nX)], size_bg_data)
        logging.info("Calculating expected value from train data")
        for i in tqdm(rand_id):
            text = self.data[i]
            tokenized_text = tokenizer.tokenize(text)
            # Convert token to vocabulary indices
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to(self.device)

            # Predict all tokens
            with torch.no_grad():
                outputs = model(input_ids=tokens_tensor)  # , token_type_ids=segments_tensors)
                # logits = outputs[0]
                predictions = outputs.detach().cpu().numpy()[0]
                all_outputs.append(self.softmax(predictions))

        self.expected_value = np.mean(all_outputs, axis=0)
        self.nC = self.expected_value.shape[0]

    def text_to_ids(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_tokens

    def softmax(self, it):
        exps = np.exp(np.array(it))
        return exps / np.sum(exps)

    def IME_text(self, data_to_explain):
        contribution_values = []
        stddevs = []
        n_iter_final = []
        # contribution_values = np.zeros((self.nC, data_to_explain.shape[0], self.nA))
        # stddevs = np.zeros((self.nC, data_to_explain.shape[0], self.nA))
        # n_iter_final = np.zeros((data_to_explain.shape[0], self.nA))
        for ii, instance in enumerate(data_to_explain): #enumerate(tqdm(data_to_explain)):  # for each instance
            logging.info(f"Example {ii+1}/{len(data_to_explain)} start")
            # data_raw = self.remove_punc(instance)
            data_raw = self.tweet_tokenizer.tokenize(instance)
            data_raw = [x for x in data_raw if x not in ".,:;'"]
            self.tweets.append(data_raw)
            nA = len(data_raw)
            idx = np.arange(len(data_raw))

            words_dict_ = self.words_dict.copy()
            instance = []
            for h in data_raw:
                iid = max(words_dict_)
                words_dict_[iid + 1] = h
                instance.append(iid+1)
            instance = np.array(instance)

            n_iter_temp = []
            expl_instance = []
            stddevs_instance = []
            for a in range(nA):  # for each attribute
                print(f"feature {a+1}/{nA}")
                conv = False
                n_iter_ = 0
                expl = np.zeros(self.nC)
                stddev = np.zeros(self.nC)
                while not conv:
                    X_temp = np.zeros((self.n_iter * 2, nA))
                    n_iter_ += self.n_iter
                    for i in range(self.n_iter):
                        np.random.shuffle(idx)
                        pos = np.where(idx == a)[0][0]

                        l1 = idx[pos + 1:]
                        l2 = idx[pos:]

                        candidates = []
                        while len(candidates) < len(l2):
                            candidates.append(np.random.randint(1, len(self.words_dict)))

                        X_candidates = np.zeros(nA)
                        X_candidates[l2] = candidates
                        X_candidates = X_candidates.astype(int)

                        X_temp[i, :] = instance
                        X_temp[i, idx[pos + 1:]] = X_candidates[idx[pos + 1:]]
                        X_temp[-(i + 1), :] = instance
                        X_temp[-(i + 1), idx[pos:]] = X_candidates[idx[pos:]]

                    X_temp = X_temp.astype(int)
                    X_temp = [" ".join([words_dict_[xx] for xx in x]) for x in X_temp]

                    tokenized_nopad = [self.tokenizer.tokenize(text) for text in X_temp]
                    MAX_SEQ_LEN = max(len(x) for x in tokenized_nopad)
                    tokenized_text = [['[PAD]', ] * MAX_SEQ_LEN for _ in range(len(X_temp))]
                    for i in range(len(X_temp)):
                        tokenized_text[i][0:len(tokenized_nopad[i])] = tokenized_nopad[i][0:MAX_SEQ_LEN]
                    indexed_tokens = [self.tokenizer.convert_tokens_to_ids(tt) for tt in tokenized_text]
                    tokens_tensor = torch.tensor(indexed_tokens)

                    # Predict all tokens
                    with torch.no_grad():
                        outputs = self.model(input_ids=tokens_tensor)
                        # logits = outputs[0]
                        evals = outputs.detach().cpu().numpy()

                    evals_on = evals[:self.n_iter]
                    evals_off = evals[self.n_iter:][::-1]
                    diff = evals_on - evals_off

                    expl += np.sum(diff, axis=0)
                    stddev += np.sum(diff ** 2, axis=0)
                    v2 = stddev / n_iter_ - (expl / n_iter_) ** 2
                    # v2 = np.var(diff, axis=0)
                    z_sq = (stats.norm.ppf(self.err / 2)) ** 2
                    # needed_iter = np.ceil(self.n_iter * (z_sq * v2 / self.err ** 2))
                    needed_iter = np.ceil(z_sq * v2 / self.err ** 2)
                    print("needed", needed_iter, "n", n_iter_)
                    if all(needed_iter < n_iter_):
                        conv = True
                        n_iter_temp.append(n_iter_)

                expl /= n_iter_  # explanation for the attribute (contributions for all classes)
                stddev = np.sqrt(stddev / n_iter_ - (expl / n_iter_) ** 2)
                # diff = np.mean(diff, axis=0)
                expl_instance.append(expl)
                stddevs_instance.append(stddev)

                # for i in range(len(expl)):
                # contribution_values[i, id, a] = expl[i]
                # stddevs[i, id, a] = stddev[i]

            logging.info(f"Example {ii + 1}/{len(data_to_explain)} stop")

            n_iter_final.append(n_iter_temp)
            contribution_values.append(np.array(expl_instance))
            stddevs.append(np.array(stddevs_instance))

        self.n_iter_final = n_iter_final
        self.stddev = stddevs

        return contribution_values

    def explain(self, data_to_explain):
        contribution_values_raw = self.IME_text(data_to_explain)
        contribution_values_final = contribution_values_raw.copy()

        predictions = []
        for instance in data_to_explain:
            instance = np.array(self.text_to_ids(instance)).reshape(1, -1)
            tens = torch.tensor(instance)
            pred = self.softmax(self.model(tens).detach().numpy())[0]
            predictions.append(pred)

        diff_by_class = np.array([self.softmax(self.model(torch.tensor(np.array(self.text_to_ids(instance))
                                                                       .reshape(1, -1))).detach().cpu().numpy()[0])
                                  for instance in data_to_explain]) - self.expected_value
        sum_by_class = np.array([np.sum(contribution_values_final[i], axis=0)
                                 for i in range(len(contribution_values_final))])

        # sum_by_class & diff_by_class should be the same
        diff_ratio = diff_by_class / sum_by_class

        for ii, cc in enumerate(diff_ratio):  # ratio diff/sum for each instance (separate classes)
            for iii in range(len(cc)):
                contribution_values_final[ii][:, iii] *= cc[iii]

        sums = [np.sum(x, axis=1) for x in np.abs(contribution_values_final)]
        self.feature_importance = [x/np.sum(x) for x in sums]

        return contribution_values_final, self.tweets, predictions
