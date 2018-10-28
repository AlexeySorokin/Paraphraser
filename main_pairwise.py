import sys
import os
import itertools
import json

import numpy as np
from scipy.spatial.distance import cosine
import statprof

sys.path.append("data/DeepPavlov/deeppavlov")
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.deep import find_config
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from ufal_udpipe import Model as udModel

from read_synsets import *
from read import *
from work import MixedUDPreprocessor
from save import analyze_scores, output_errors

tagger = build_model_from_config(find_config("morpho_ru_syntagrus_train_pymorphy"))
ud_model = udModel.load("russian-syntagrus-ud-2.0-170801.udpipe")
ud_processor = MixedUDPreprocessor(tagger=tagger, model=ud_model)


POS_TO_WORK = ["PROPN", "NOUN", "VERB", "ADJ"]
NO_SYNSET = -1


def get_active_indexes(sent):
    return [i for i, elem in enumerate(sent) if elem[3] in POS_TO_WORK]


def get_common_vertexes_with_depths(first_depths, second_depths):
    # first_depths = {x: d for d, elem in enumerate(first, 1) for x in elem}
    # second_depths = {x: d for d, elem in enumerate(second, 1) for x in elem}
    common = set(first_depths) & set(second_depths)
    return [(x, first_depths[x], second_depths[x]) for x in common]


class PairwiseScorer:

    def __init__(self, ancestors, D, depths, synsets_by_lemmas, synsets_encoding,
                 metrics, embedders=None, all_scores_present=False,
                 save_zero_scores=True, verbose=0):
        self.ancestors = ancestors
        self.D = D
        self.depths = depths
        self.synsets_by_lemmas = synsets_by_lemmas
        self.synsets_encoding = synsets_encoding
        self.embedders = embedders
        self.all_scores_present = all_scores_present
        self.save_zero_scores = save_zero_scores
        self.distance_map = dict()
        self.LCA_map = dict()
        self.verbose = verbose
        self._initialize_metrics(metrics)

    def _initialize_metrics(self, metrics):
        self.word_metrics, self.synset_metrics = [], []
        for metric in metrics:
            if self.is_word_metric(metric):
               self.word_metrics.append(metric)
            elif self.is_synset_metric(metric):
                self.synset_metrics.append(metric)
            else:
                raise ValueError("Undefined metric {}".format(metric))
        return

    @property
    def metrics_number(self):
        return len(self.word_metrics) + len(self.embedders) + len(self.synset_metrics)

    @property
    def synset_metrics_number(self):
        return len(self.synset_metrics)

    @property
    def embedders_number(self):
        return len(self.embedders)

    @classmethod
    def is_word_metric(cls, metric):
        return False

    @classmethod
    def is_synset_metric(cls, metric):
        return metric in ["lch", "modified_lch", "lin", "wup", "ic"]

    def load(self, lca_infile=None, scores_infile=None):
        if lca_infile is not None:
            with open(lca_infile, "r") as fin:
                for line in fin:
                    line = line.strip()
                    if line == "":
                        continue
                    splitted = [int(x) for x in  line.split()]
                    first, second = splitted[:2]
                    if len(splitted) == 3:
                        self.LCA_map[(first, second)] = (None, -1, -1)
                    else:
                        self.LCA_map[(first, second)] = tuple(splitted[2:])
        if scores_infile is not None:
            with open(scores_infile, "r") as fin:
                for line in fin:
                    line = line.strip()
                    if line == "":
                        continue
                    splitted = line.split("\t")
                    first, second = splitted[:2]
                    scores, first_synset, second_synset = splitted[2:5]
                    self.distance_map[(first, second)] =\
                        ([float(x) for x in scores.split()], first_synset, second_synset)
        return

    def save(self, lca_outfile=None, scores_outfile=None):
        if lca_outfile is not None:
            with open(lca_outfile, "w") as fout:
                for (first, second), (common, d1, d2) in self.LCA_map.items():
                    if common is None:
                        to_write = [first, second, NO_SYNSET]
                    else:
                        to_write = [first, second, common, d1, d2]
                    fout.write("\t".join(map(str, to_write)) + "\n")
        if scores_outfile is not None:
            with open(scores_outfile, "w") as fout:
                for (first, second), scores in self.distance_map.items():
                    fout.write("{}\t{}\t{}\t{}\t{}\n".format(
                        first, second, " ".join("{:.4f}".format(x) for x in scores[0]), *scores[1:]))

    def find_LCA(self, first, second):
        if first == second:
            return (first, 0, 0)
        first_ancestors = self.ancestors[first]
        second_ancestors = self.ancestors[second]
        d = first_ancestors.get(second)
        if d is not None:
            return (second, d, 0)
        d = second_ancestors.get(first)
        if d is not None:
            return (first, 0, d)
        depths = get_common_vertexes_with_depths(first_ancestors, second_ancestors)
        if len(depths) > 0:
            return min(depths, key=lambda x: x[1]+x[2])
        # не нашли общего предка (такое бывает?) бывает.
        return (None, -1, -1)

    def lch(self, first_dist, second_dist, max_dist):
        return 1.0 - np.log(first_dist + second_dist + 1) / np.log(max_dist)

    def _score_codes(self, metric, first_code, second_code):
        data = self.LCA_map.get((first_code, second_code))
        if data is None:
            common, first_dist, second_dist = self.find_LCA(first_code, second_code)
            self.LCA_map[(first_code, second_code)] = common, first_dist, second_dist
        else:
            common, first_dist, second_dist = data
        if common is None:
            return 0.0
        if metric == "lch":
            answer = self.lch(first_dist, second_dist, 2*self.D+3)
        elif metric == "modified_lch":
            first_depth = self.depths[common] + first_dist
            second_depth = self.depths[common] + first_dist
            answer = self.lch(first_dist, second_dist, first_depth+second_depth+3)
        else:
            raise NotImplementedError("Unknown metric {}".format(metric))
        return answer

    def aggregate_scores(self, scores):
        max_scores_for_first = np.max(scores, axis=0)
        max_scores_for_second = np.max(scores, axis=1)
        answer = 0.5 * (np.mean(max_scores_for_first) + np.mean(max_scores_for_second))
        # answer = np.mean(np.concatenate([max_scores_for_first, max_scores_for_second]))
        return answer

    def score_sents(self, pairs, dump_file=None):
        if self.verbose > 0 and dump_file is not None:
            # if not os.path.exists("dump"):
            #     os.makedirs("dump")
            fout = open(dump_file, "w", encoding="utf8")
        scores, pairwise_scores = [], []
        for first, second in pairs:
            curr_pairwise_scores, first_indexes, second_indexes = self.get_scores_matrix(first, second)
            similarity_score = self.aggregate_scores([[x[0] for x in elem] for elem in curr_pairwise_scores])
            first_words = [first[i][1] for i in first_indexes]
            second_words = [second[i][1] for i in second_indexes]
            if self.verbose > 0 and dump_file is not None:
                first_length = max([len(x) for x in first_words])
                column_lengths = [max(len(x), 5 * self.metrics_number) for x in second_words]
                fout.write("{:<{width}}".format("", width=first_length+2))
                for word, width in zip(second_words, column_lengths):
                    fout.write("{:<{width}}".format(word, width=width+2))
                fout.write("\n")
                for i, (word, curr_scores) in enumerate(zip(first_words, curr_pairwise_scores)):
                    fout.write("{:<{width}}".format(word, width=first_length+2))
                    for j, width in enumerate(column_lengths):
                        fout.write("{:<{width}}".format(
                            " ".join("{:.2f}".format(x) for x in curr_scores[j][0]), width=width+2))
                    fout.write("\n")
                fout.write("Aggregate: {:.2f}\n\n".format(similarity_score))
            scores.append(similarity_score)
            pairwise_scores.append((first_words, second_words, curr_pairwise_scores))
            if len(scores) % 1000 == 0 and self.verbose:
                print("{} elements processed".format(len(scores)))
        if self.verbose > 0 and dump_file is not None:
            fout.close()
        scores = 1.0 - np.array(scores)
        return scores, pairwise_scores

    @property
    def default_score(self):
        return ([0.0] * self.metrics_number, None, None)

    def _get_word_score(self, first, second):
        return 0.0

    def _collect_word_score(self, first, second):
        answer = [self._get_word_score(metric, first, second) for metric in self.word_metrics]
        for embedder in self.embedders:
            vectors = embedder([[first, second]])[0]
            score = np.nan_to_num(cosine(vectors[0], vectors[1]))
            answer.append(1.0 - score)
        return answer

    def get_scores_matrix(self, first_sent, second_sent):
        first_indexes = get_active_indexes(first_sent)
        second_indexes = get_active_indexes(second_sent)
        first_sent = [first_sent[i] for i in first_indexes]
        second_sent = [second_sent[i] for i in second_indexes]
        synsets_with_scores = [[None] * len(second_sent) for _ in first_sent]
        for i, first_elem in enumerate(first_sent):
            first_word, first_pos = first_elem[2:4]
            first_synsets = synsets_by_lemmas[first_word]
            for j, second_elem in enumerate(second_sent):
                second_word, second_pos = second_elem[2:4]
                saved_scores = self.distance_map.get((first_word, second_word), self.default_score)
                if max(saved_scores[0]) > 0.0 or self.all_scores_present:
                    synsets_with_scores[i][j] = saved_scores
                    continue
                word_scores = self._collect_word_score(first_word, second_word)
                second_synsets = synsets_by_lemmas[second_word]
                _, score = self.get_synset_score(
                    first_word, second_word, first_synsets, second_synsets)
                score = (word_scores + score[0],) + score[1:]
                if max(score[0]) > 0.0:
                    self.distance_map[(first_word, second_word)] = score
                synsets_with_scores[i][j] = score
        return synsets_with_scores, first_indexes, second_indexes

    def get_synset_score(self, first_word, second_word, first_synsets, second_synsets):
        if self.synset_metrics_number == 0:
            return 0.0, ([], None, None)
        curr_metrics, first_synset, second_synset = None, None, None
        if len(first_synsets) == 0 or len(second_synsets) == 0:
            curr_metrics = [float(first_word == second_word)] * self.synset_metrics_number
        else:
            common_synsets = [x for x in first_synsets if x in second_synsets]
            if len(common_synsets) > 0:
                curr_metrics = [1.0] * self.synset_metrics_number
                first_synset, second_synset = common_synsets[0], common_synsets[0]
        if curr_metrics is not None:
            aggr_score = curr_metrics[0]
            return aggr_score, (curr_metrics, first_synset, second_synset)
        best_score = 0.0, [0.0] * self.synset_metrics_number
        best_first_synset, best_second_synset = None, None
        for first_synset, second_synset in itertools.product(first_synsets, second_synsets):
            first_code = self.synsets_encoding.get(first_synset)
            second_code = self.synsets_encoding.get(second_synset)
            if first_code is not None and second_code is not None:
                scores = [self._score_codes(metric, first_code, second_code)
                          for metric in self.synset_metrics]
                if any(x is None for x in scores):
                    continue
                # self.distance_map[(first_synset, second_synset)] = scores
            else:
                scores = None
            # np.mean is probitively slow
            if scores is not None:
                mean_score = sum(scores) / len(scores)
                if mean_score > best_score[0]:
                    best_score = mean_score, scores
                    best_first_synset, best_second_synset = first_synset, second_synset
        return best_score[0], (best_score[1], best_first_synset, best_second_synset)

OBLIGATORY_FIELDS = ["relations_file", "senses_file", "train_file", "test_file"]
DEFAULT_FIELDS = ["scorer", "embedders", "metrics", "use_derivates", "verbose",
                  "save_file", "save_scores_file", "load_file", "load_scores_file"]
DEFAULT_VALUES = [dict(), [], ["modified_lch"], False, 0] + [None] * 4

def read_config(infile):
    with open(infile, "r", encoding="utf8") as fin:
        data = json.load(fin)
    for key in OBLIGATORY_FIELDS:
        if key not in data:
            raise KeyError("Field {} must be present in config file".format(key))
    for key, value in zip(DEFAULT_FIELDS, DEFAULT_VALUES):
        if key not in data:
            data[key] = value
    return data


def read_embedders(data):
    embedders = []
    for infile, dim, model_type in data:
        if model_type == "fasttext":
            embedder = FasttextEmbedder(infile, dim=dim)
        else:
            raise NotImplementedError("{} embedder is not implemented".format(model_type))
        print("{} embedder loaded".format(model_type))
        embedders.append(embedder)
    return embedders


if __name__ == "__main__":
    config = read_config(sys.argv[1])
    synsets, synsets_encoding, graph, derivates = make_graph(config["relations_file"])
    synsets_by_lemmas, lemmas_by_synsets = collect_synsets_for_lemmas(config["senses_file"])
    ancestors_lists, D, depths =\
        make_ancestors_lists(graph, derivates, use_derivates=config["use_derivates"])
    embedders = read_embedders(config["embedders"])

    scorer = PairwiseScorer(ancestors_lists, D, depths, synsets_by_lemmas,
                            synsets_encoding, config["metrics"], embedders=embedders,
                            verbose=config["verbose"], **config["scorer"])
    scorer.load(config["load_file"], config["load_scores_file"])
    # scorer.load("dump/lca.out", "dump/scores.out")
    # statprof.start()
    # try:
    sents, data, targets = read_data(config["train_file"], from_parses=True)
    data_to_work = [list(zip(*elem)) for elem in zip(*data)]
    data_to_scorer = list(zip(data_to_work[::2], data_to_work[1::2]))
    similarity_scores, word_scores =\
        scorer.score_sents(data_to_scorer, dump_file=config.get('train_distances_file'))
    # finally:
    #     statprof.stop()
    #     with open("log_load.out", "w") as flog:
    #         statprof.display(flog, order=statprof.DisplayOrder.CUMULATIVE)
    # test
    test_sents, test_data, test_targets = read_data(config["test_file"], from_parses=True)
    data_to_work = [list(zip(*elem)) for elem in zip(*test_data)]
    data_to_scorer = list(zip(data_to_work[::2], data_to_work[1::2]))
    test_similarity_scores, test_word_scores =\
        scorer.score_sents(data_to_scorer, dump_file=config.get('test_distances_file'))
    scorer.save(config["save_file"], config["save_scores_file"])
    # scorer.save("dump/lca.out", "dump/scores.out")
    initial_threshold = analyze_scores(similarity_scores, targets, test_similarity_scores,
                                       test_targets, verbose=config["verbose"])
    if "train_analysis" in config:
        fp_file = config["train_analysis"] + "_FP"
        output_errors(sents, targets, initial_threshold,  similarity_scores,
                      word_scores, fp_file, scorer.metrics_number)
        fn_file = config["train_analysis"] + "_FN"
        output_errors(sents, targets, initial_threshold, similarity_scores,
                      word_scores, fn_file, scorer.metrics_number, reverse=True)
    if "test_analysis" in config:
        fp_file = config["test_analysis"] + "_FP"
        output_errors(test_sents, test_targets, initial_threshold, test_similarity_scores,
                      test_word_scores, fp_file, scorer.metrics_number)
        fn_file = config["test_analysis"] + "_FN"
        output_errors(test_sents, test_targets, initial_threshold, test_similarity_scores,
                      test_word_scores, fn_file, scorer.metrics_number, reverse=True)