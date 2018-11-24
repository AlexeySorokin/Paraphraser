import sys, importlib
import os
import itertools
import json

import numpy as np
from scipy.spatial.distance import cosine
import statprof

import keras.layers as kl
import keras.backend as kb
import keras.callbacks as kcall
from keras.models import Model
from network import masked_mean, AttentionWeightsLayer

sys.path.append("/home/alexeysorokin/data/DeepPavlov")
MORPHO_CONFIG_PATH = "config/DeepPavlov/morpho_ru_syntagrus_pymorphy.json"
CONFIG_PATH = "config/pairwise/config_fasttext.json"

from deeppavlov.core.commands.infer import build_model
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from ufal_udpipe import Model as udModel

from read_synsets import *
from read import *
from work import MixedUDPreprocessor, SyntaxTreeGraph
from save import analyze_scores, output_errors
from main_pairwise import PairwiseScorer, read_config, read_embedders






class PairwiseTrainer:
    
    def __init__(self, dimensions_number,
                 use_pos=True, use_tag=True, use_syntax=True, syntax_depth=3,
                 nepochs=50, validation_split=0.2, batch_size=16,
                 feat_layers=2, embedding_dim=128, embedding_dropout=0.2,
                 patience=None):
        self.dimensions_number = dimensions_number
        self.use_pos = use_pos or use_tag
        self.use_tag = use_tag
        self.use_syntax = use_syntax
        self.syntax_depth = syntax_depth
        self.nepochs = nepochs
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.feat_layers = feat_layers
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout
        self.patience = patience
        
    def train(self, data, targets, model_file=None, save_file=None):
        possible_pos, possible_tags, possible_heads = set(), set(), set()
        for first, second, _, _, _ in data:
            for elem in first + second:
                possible_pos.add(elem[3])
                if elem[5] != "_":
                    feat_with_values = [x.split("=") for x in elem[5].split("|")]
                    for x in feat_with_values:
                        possible_tags.add("{}_{}_{}".format(elem[3], *x))
                possible_heads.add(elem[7])
        if self.use_pos:
            self.pos_labels_ = sorted(possible_pos)
            self.pos_codes_ = {label: code for code, label in enumerate(self.pos_labels_)}
            if self.use_tag:
                self.tags_ = sorted(possible_tags)
                self.tag_codes_ = {label: code for code, label in enumerate(self.tags_)}
        if self.use_syntax:
            self.heads_ = [None] + sorted(possible_heads)
            self.head_codes_ = {label: code for code, label in enumerate(self.heads_)}
        transformed_data = self.transform(data)
        self.build()
        self.train_model(transformed_data, targets, model_file=model_file)
        if model_file is not None:
            self.model.load_weights(model_file)
        return self
    
    def _transform_sent(self, data, indexes):
        answer, start = [[] for _ in data], 0
        if self.use_pos:
            for i, elem in enumerate(data):
                code = self.pos_codes_.get(elem[3])
                if code is not None:
                    answer[i].append(code)
            start += len(self.pos_codes_)
        if self.use_tag:
            for i, elem in enumerate(data):
                if elem[3] not in self.pos_codes_ or elem[5] == "_":
                    continue
                feat_with_values = [x.split("=") for x in elem[5].split("|")]
                feats = ["{}_{}_{}".format(elem[3], *x) for x in feat_with_values]
                for feat in feats:
                    code = self.tag_codes_.get(feat)
                    if code is not None:
                        answer[i].append(start+code)
            start += len(self.tag_codes_)
        if self.use_syntax:
            graph = SyntaxTreeGraph(data)
            branches = graph.make_branches()
            branch_codes = [[None] * self.syntax_depth for _ in branches]
            for i, branch in enumerate(branches):
                branch_codes[i][:len(branch)] = branch[:self.syntax_depth]
                branch_codes[i] = [self.head_codes_[x] for x in branch_codes[i]]
            for i, (elem, branch) in enumerate(zip(answer, branch_codes)):
                for j, code in enumerate(branch):
                    elem.append(start + j * len(self.head_codes_) + code)
        answer = [answer[i] for i in indexes]
        return answer
                
    def transform(self, data):
        """

        Attributes:
        -----------
            data: list,
                data[i] = [first_sent, second_sent, first_indexes, second_indexes, distances]

        Returns:
        --------
            answer: list
                answer[i] = [distances, first_features, second_features]
                distances: np.array(dim=2), квадратная матрица расстояний
                first_features: list[np.array[int]], список индексов активных признаков
        """
        answer = []
        for first_data, second_data, first_indexes, second_indexes, distances in data:
            first_features = self._transform_sent(first_data, first_indexes)
            second_features = self._transform_sent(second_data, second_indexes)
            # distances = distances[first_indexes, second_indexes]
            answer.append([distances, first_features, second_features])
        return answer

    def train_model(self, data, targets, model_file=None):
        m = len(data)
        indexes = np.arange(m)
        np.random.shuffle(indexes)
        level = int(m * (1.0 - self.validation_split))
        train_indexes, dev_indexes = indexes[:level], indexes[level:]
        train_data, dev_data = [data[i] for i in train_indexes], [data[i] for i in dev_indexes]
        train_targets, dev_targets = targets[train_indexes], targets[dev_indexes]
        self.train_gen = self.make_generator(train_data, train_targets, shuffle=True)
        train_steps_per_epoch = (len(train_data)+1) // self.batch_size
        self.dev_gen = self.make_generator(dev_data, dev_targets, shuffle=False)
        dev_steps_per_epoch = (len(dev_data) + 1) // self.batch_size
        callbacks = []
        if model_file is not None:
            callbacks.append(kcall.ModelCheckpoint(
                model_file, monitor="val_acc", save_weights_only=True, save_best_only=True))
        if self.patience is not None:
            callbacks.append(kcall.EarlyStopping(monitor="val_acc", patience=self.patience))
        self.model.fit_generator(self.train_gen, train_steps_per_epoch, self.nepochs,
                                 validation_data=self.dev_gen, validation_steps=dev_steps_per_epoch)
        self.evaluate()
        return self

    def evaluate(self):
        for data, targets in self.dev_gen:
            first_weights, second_weights = self.weights_func_(data)
            first_dist, second_dist = self.dist_func_(data)
            scores = self.model_func_(data)
            break

    def make_generator(self, data, targets, shuffle=False):
        indexes = np.arange(len(data))
        while True:
            if shuffle:
                np.random.shuffle(indexes)
            for start in range(0, len(data), self.batch_size):
                curr_indexes = indexes[start:start + self.batch_size]
                m = len(curr_indexes)
                max_length = max(len(x) for index in curr_indexes for x in data[index][1:3])
                batch_distances = np.zeros(shape=(m, max_length, max_length, self.dimensions_number), dtype=float)
                first_batch_features = np.zeros(shape=(m, max_length, self.feats_number), dtype=float)
                second_batch_features = np.zeros(shape=(m, max_length, self.feats_number), dtype=float)
                mask = np.zeros(shape=(m, max_length, max_length), dtype=np.uint8)
                for i, index in enumerate(curr_indexes):
                    distances, first_features, second_features = data[index]
                    r, s, _ = distances.shape
                    batch_distances[i, :r, :s], mask[i, :r, :s] = distances, 1
                    for j, elem in enumerate(first_features):
                        first_batch_features[i, j, elem] = 1
                    for j, elem in enumerate(second_features):
                        second_batch_features[i, j, elem] = 1
                curr_batch = [batch_distances, first_batch_features, second_batch_features, mask]
                curr_targets = targets[curr_indexes]
                yield (curr_batch, curr_targets)

    @property
    def pos_number(self):
        return len(self.pos_codes_)

    @property
    def tag_number(self):
        return len(self.tag_codes_)

    @property
    def heads_number(self):
        return len(self.head_codes_)

    @property
    def feats_number(self):
        answer = 0
        if self.use_pos:
            answer += self.pos_number
        if self.use_tag:
            answer += self.tag_number
        if self.use_syntax:
            answer += self.heads_number * self.syntax_depth
        return answer

    def build(self):
        scores = kl.Input(shape=(None, None, self.dimensions_number), dtype="float")
        first_feats = kl.Input(shape=(None, self.feats_number), dtype="float")
        second_feats = kl.Input(shape=(None, self.feats_number), dtype="float")
        mask = kl.Input(shape=(None, None), dtype="uint8")

        first_u, first_v = first_feats, first_feats
        second_u, second_v = second_feats, second_feats
        for i in range(self.feat_layers):
            u_matrix = kl.Dense(self.embedding_dim, activation="tanh")
            v_matrix = kl.Dense(self.embedding_dim, activation="tanh")
            first_u, first_v = u_matrix(first_u), v_matrix(first_v)
            second_u, second_v = u_matrix(second_u), v_matrix(second_v)
            if self.embedding_dropout > 0.0:
                first_u, first_v, second_u, second_v = [kl.Dropout(self.embedding_dropout)(x)
                                                        for x in [first_u, first_v, second_u, second_v]]
        weights_12 = AttentionWeightsLayer()([first_u, second_v, mask])
        weights_21 = AttentionWeightsLayer()([second_u, first_v, mask])
        weights_12 = kl.Lambda(lambda x: kb.repeat_elements(kb.expand_dims(x), self.dimensions_number, -1),
                               output_shape=lambda x: x + (1,))(weights_12)
        weights_21 = kl.Lambda(lambda x: kb.repeat_elements(kb.expand_dims(x), self.dimensions_number, -1),
                               output_shape=lambda x: x + (1,))(weights_21)
        dist_12 = kl.Multiply()([scores, weights_12]) # B * m * n
        max_dist_12 = kl.Lambda(kb.sum, arguments={"axis": 2},
                                output_shape=(None, self.dimensions_number))(dist_12)
        transposed_scores = kl.Lambda(kb.permute_dimensions, arguments={"pattern": [0, 2, 1, 3]},
                                      output_shape=(None, None))(scores)
        transposed_mask = kl.Lambda(kb.permute_dimensions, arguments={"pattern": [0, 2, 1]},
                                    output_shape=(None, None))(mask)
        dist_21 = kl.Multiply()([transposed_scores, weights_21])  # B * n * m
        max_dist_21 = kl.Lambda(kb.sum, arguments={"axis": 2},
                                output_shape=(None, self.dimensions_number))(dist_21)
        first_mean_dist = kl.Lambda(masked_mean, arguments={"axis": 1, "mask": mask},
                                    output_shape=(self.dimensions_number,))(max_dist_12)
        second_mean_dist = kl.Lambda(masked_mean, arguments={"axis": 1, "mask": transposed_mask},
                                     output_shape=(self.dimensions_number,))(max_dist_21)
        mean_dist = kl.Lambda(lambda x, y: 0.5 * (x + y), arguments={"y": second_mean_dist})(first_mean_dist)
        outputs = kl.Dense(1, activation="sigmoid")(mean_dist)
        self.model = Model([scores, first_feats, second_feats, mask], [outputs])
        self.weights_func_ = kb.Function([scores, first_feats, second_feats, mask], [weights_12, weights_21])
        self.dist_func_ = kb.Function([scores, first_feats, second_feats, mask], [max_dist_12, max_dist_21])
        self.model_func_ = kb.Function([scores, first_feats, second_feats, mask],
                                       [first_mean_dist, second_mean_dist, outputs])
        self.model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
        return self

if __name__ == "__main__":
    config = read_config(CONFIG_PATH)
    synsets, synsets_encoding, graph, derivates = make_graph(config["relations_file"])
    synsets_by_lemmas, lemmas_by_synsets = collect_synsets_for_lemmas(config["senses_file"])

    ancestors_lists, D, depths = make_ancestors_lists(graph, derivates, use_derivates=config["use_derivates"])
    embedders = read_embedders(config["embedders"])

    scorer = PairwiseScorer(ancestors_lists, D, depths, synsets_by_lemmas,
                            synsets_encoding, config["metrics"], embedders=embedders,
                            verbose=config["verbose"], **config["scorer"])
    scorer.load(config["load_file"], config["load_scores_file"])

    sents, data, targets = read_data(config["train_file"], from_parses=True)
    data_to_scorer = list(zip(data[::2], data[1::2]))
    similarity_scores, word_scores = \
        scorer.score_sents(data_to_scorer, dump_file=config.get('train_distances_file'))
    trainer = PairwiseTrainer(scorer.metrics_number, use_syntax=False, nepochs=3, feat_layers=0)
    data_to_train = [(list(elem) + list(word_scores[i][:2])) for i, elem in enumerate(data_to_scorer)]
    for i, elem in enumerate(word_scores):
        curr_scores = [[None] * len(x) for x in elem[4]]
        for j, row in enumerate(elem[4]):
            for k, value in enumerate(row):
                curr_scores[j][k] = value[0]
        data_to_train[i].append(np.array(curr_scores))
    trainer.train(data_to_train, targets)
    # trainer.build()

