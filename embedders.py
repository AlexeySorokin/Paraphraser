import numpy as np

from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder


class TfIdfEmbedder:


    def __init__(self, word_embedder, word_counts=None, use_idf=True,
                 idf_base_count=10, min_idf_weight=0.0, log_base=np.e,
                 tag_weights=None, batch_size=32):
        self.word_embedder = word_embedder
        self.word_counts = word_counts
        self.use_idf = use_idf
        self.idf_base_count = idf_base_count
        self.min_idf_weight = min_idf_weight
        self.log_base = log_base
        self.tag_weights = tag_weights
        self.batch_size = batch_size

    def __call__(self, word_sents, tag_sents=None):
        sent_embeddings = np.zeros(shape=(len(word_sents), self.word_embedder.dim), dtype=float)
        for start in range(0, len(word_sents), self.batch_size):
            end = start + self.batch_size
            word_embeddings = self.word_embedder(word_sents[start:end])
            if self.use_idf:
                word_weights = [np.array([self._get_word_weight(word) for word in sent])
                                for sent in word_sents[start:end]]
            else:
                word_weights = [np.ones(shape=(len(sent),), dtype=float)
                                for sent in word_sents[start:end]]
            if self.tag_weights is not None:
                tag_weights =\
                    [np.array([self._get_tag_weight(*elem) for elem in zip(tag_sent, sent)])
                               for tag_sent, sent in zip(tag_sents[start:end], word_sents[start:end])]
                for curr_word_weights, curr_tag_weights in zip(word_weights, tag_weights):
                    curr_word_weights *= curr_tag_weights
            # for i, sent_embeddings in enumerate(word_embeddings):
            #     are_nonzero = (np.max(np.abs(sent_embeddings), axis=1) != 0.0).astype("int")
            #     word_weights[i] *= are_nonzero
            for i, elem in enumerate(word_weights):
                word_weights[i] /= np.sum(elem)
            # for word_sent, weight_sent in zip(word_sents, word_weights):
            #     if word_sent[0] == "цуп":
            #         weight_sent[0] /= 10.0
            # for i, elem in enumerate(word_weights):
            #     print(word_sents[start+i])
            #     print(" ".join("{:.2f}".format(x) for x in elem))
            sent_embeddings[start:end] =\
                np.array([np.sum(x * y[:,None], axis=0)
                          for x, y in zip(word_embeddings, word_weights)])
        return sent_embeddings

    def _get_word_weight(self, word):
        count = max(self.word_counts.get(word, 0), self.idf_base_count)
        log_count = np.log(count) / np.log(self.log_base)
        log_base_count = np.log(self.idf_base_count) / np.log(self.log_base)
        weight = max(1.0 / (1.0 + log_count - log_base_count), self.min_idf_weight)
        return weight

    def _get_tag_weight(self, tag, word=None):
        weight = self.tag_weights[1].get((word, tag), -1)
        if weight < 0:
            weight = self.tag_weights[0].get(tag, 0.0)
        return weight



