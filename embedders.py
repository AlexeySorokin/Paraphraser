import numpy as np

from work import SyntaxTreeGraph
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder


class BasicFreqPosEmbedder:


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

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You should implement __call__ function")

    def make_word_weights(self, sent):
        if self.use_idf:
            word_weights = np.array([self._get_word_weight(word) for word in sent])
        else:
            word_weights = np.ones(shape=(len(sent),), dtype=float)
        return word_weights

    def make_tag_weights(self, tag_sent, sent):
        return np.array([self._get_tag_weight(*elem) for elem in zip(tag_sent, sent)])

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


class FreqPosEmbedder(BasicFreqPosEmbedder):

    def __call__(self, word_sents, tag_sents=None):
        sent_embeddings = np.zeros(shape=(len(word_sents), self.word_embedder.dim), dtype=float)
        for start in range(0, len(word_sents), self.batch_size):
            end = start + self.batch_size
            word_embeddings = self.word_embedder(word_sents[start:end])
            word_weights = [self.make_word_weights(sent) for sent in word_sents[start:end]]
            if self.tag_weights is not None and tag_sents is not None:
                tag_weights = [self.make_tag_weights(*elem)
                               for elem in zip(tag_sents[start:end], word_sents[start:end])]
                for curr_word_weights, curr_tag_weights in zip(word_weights, tag_weights):
                    curr_word_weights *= curr_tag_weights
            # for i, elem in enumerate(word_weights):
                # word_weights[i] /= np.sum(elem)
            sent_embeddings[start:end] =\
                np.array([np.sum(x * y[:,None], axis=0)
                          for x, y in zip(word_embeddings, word_weights)])
        return sent_embeddings


class SVOFreqPosEmbedder(BasicFreqPosEmbedder):

    SVO_INDEXES = {"subj": 0, "verb": 1, "obj": 2}

    def __call__(self, parse_sents):
        sent_embeddings = np.zeros(shape=(len(parse_sents), 3 * self.word_embedder.dim), dtype=float)
        for start in range(0, len(parse_sents), self.batch_size):
            end = start + self.batch_size
            curr_sents = parse_sents[start:end]
            word_sents = [[elem[2] for elem in sent] for sent in curr_sents]
            word_embeddings = self.word_embedder(word_sents)
            for i, (parse, words) in enumerate(zip(curr_sents, word_sents)):
                pos_sent = [elem[3] for elem in parse]
                word_weights = self.make_word_weights(words)
                word_weights *= self.make_tag_weights(pos_sent, words)
                svo_indexes = [self.SVO_INDEXES[key] 
                               for key in SyntaxTreeGraph(parse).make_subject_types()]
                for weight, index, embedding in zip(word_weights, svo_indexes, word_embeddings[i]):
                    first_col = self.word_embedder.dim * index
                    last_col = self.word_embedder.dim * (index + 1)
                    sent_embeddings[start+i, first_col:last_col] += weight * embedding
        return sent_embeddings





