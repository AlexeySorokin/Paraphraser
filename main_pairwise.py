import sys
import os
import itertools
import numpy as np

import statprof

sys.path.append("data/DeepPavlov/deeppavlov")
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.deep import find_config
from ufal_udpipe import Model as udModel

from read_synsets import *
from read import *
from work import MixedUDPreprocessor
from save import analyze_scores

tagger = build_model_from_config(find_config("morpho_ru_syntagrus_train_pymorphy"))
ud_model = udModel.load("russian-syntagrus-ud-2.0-170801.udpipe")
ud_processor = MixedUDPreprocessor(tagger=tagger, model=ud_model)


POS_TO_WORK = ["PROPN", "NOUN", "VERB", "ADJ"]

def get_active_indexes(sent):
    return [i for i, elem in enumerate(sent) if elem[3] in POS_TO_WORK]


def get_common_vertexes_with_depths(first, second):
    first_depths = {x: d for d, elem in enumerate(first, 1) for x in elem}
    second_depths = {x: d for d, elem in enumerate(second, 1) for x in elem}
    common = set(first_depths) & set(second_depths)
    return [(x, first_depths[x], second_depths[x]) for x in common]


class PairwiseScorer:

    def __init__(self, ancestors, D, depths, synsets_by_lemmas, synsets_encoding, metrics,
                 verbose=0):
        self.ancestors = ancestors
        self.D = D
        self.depths = depths
        self.synsets_by_lemmas = synsets_by_lemmas
        self.synsets_encoding = synsets_encoding
        self.metrics = metrics
        self.distance_map = dict()
        self.LCA_map = dict()
        self.verbose = verbose

    @property
    def metrics_number(self):
        return len(self.metrics)

    def find_LCA(self, first, second):
        if first == second:
            return (first, 0, 0)
        first_ancestors = self.ancestors[first]
        second_ancestors = self.ancestors[second]
        for d, curr_first_ancestors in enumerate(first_ancestors, 1):
            if second in curr_first_ancestors:
                return (second, d, 0)
        for d, curr_second_ancestors in enumerate(second_ancestors, 1):
            if first in curr_second_ancestors:
                return (first, 0, d)
        # d_max = max(len(first_ancestors), len(second_ancestors))
        # depths = defaultdict(lambda: [d_max+1, d_max+1])
        depths = get_common_vertexes_with_depths(first_ancestors, second_ancestors)
        # for d, elem in enumerate(first_ancestors, 1):
        #     for x in elem:
        #         depths[x][0] = d
        # for d, elem in enumerate(second_ancestors, 1):
        #     for x in elem:
        #         depths[x][1] = d
        # depths = [(x, elem[0], elem[1]) for x, elem in depths.items()
        #           if (elem[0] <= d_max and elem[1] <= d_max)]
        if len(depths) > 0:
            return min(depths, key=lambda x: x[1]+x[2])
        # max_path_length = len(first_ancestors) + len(second_ancestors)
        # for d in range(2, max_path_length + 1):
        #     # если одна вершина --- не предок другой, то минимальное суммарное расстояние 2
        #     # минимальное расстояние от первой вершины, позволяющее сделать сумму d
        #     min_first_distance = max(1, d - len(second_ancestors))
        #     for first_distance, curr_first_ancestors in \
        #             enumerate(first_ancestors[min_first_distance - 1:d - 1], min_first_distance):
        #         second_distance = d - first_distance
        #         curr_second_ancestors = second_ancestors[second_distance - 1]
        #         for common in curr_first_ancestors:
        #             if common in curr_second_ancestors:
        #                 return (common, first_distance, second_distance)
        # не нашли общего предка (такое бывает?) бывает.
        return (None, -1, -1)

    def lch(self, first_dist, second_dist):
        return 1.0 - np.log(first_dist + second_dist + 1) / np.log(2 * self.D)

    def _score(self, metric, first_code, second_code):
        # # first_code = self.synsets_encoding.get(first)
        # # second_code = self.synsets_encoding.get(second)
        # if first_code is None or second_code is None:
        #     return None
        common, first_dist, second_dist = self.find_LCA(first_code, second_code)
        if common is None:
            return 0.0
        if metric == "lch":
            answer = self.lch(first_dist, second_dist)
            return answer

    def aggregate_scores(self, scores):
        max_scores_for_first = np.max(scores, axis=0)
        max_scores_for_second = np.max(scores, axis=1)
        return 0.5 * (np.mean(max_scores_for_first) + np.mean(max_scores_for_second))

    def score_sents(self, pairs):
        if self.verbose > 0:
            if not os.path.exists("dump"):
                os.makedirs("dump")
            fout = open("dump/log.out", "w", encoding="utf8")
        scores = []
        for first, second in pairs:
            pairwise_scores, first_indexes, second_indexes = self.get_scores_matrix(first, second)
            similarity_score = self.aggregate_scores([[x[0] for x in elem] for elem in pairwise_scores])
            first_words = [first[i][1] for i in first_indexes]
            second_words = [second[i][1] for i in second_indexes]
            first_length = max([len(x) for x in first_words])
            column_lengths = [max(len(x), 5*self.metrics_number) for x in second_words]
            if self.verbose > 0:
                fout.write("{:<{width}}".format("", width=first_length+2))
                for word, width in zip(second_words, column_lengths):
                    fout.write("{:<{width}}".format(word, width=width+2))
                fout.write("\n")
                for i, (word, curr_scores) in enumerate(zip(first_words, pairwise_scores)):
                    fout.write("{:<{width}}".format(word, width=first_length+2))
                    for j, width in enumerate(column_lengths):
                        fout.write("{:<{width}}".format(
                            " ".join("{:.2f}".format(x) for x in curr_scores[j][0]), width=width+2))
                    fout.write("\n")
                fout.write("Aggregate: {:.2f}\n\n".format(similarity_score))
            scores.append(similarity_score)
            if len(scores) % 100 == 0:
                print("{} elements processed".format(len(scores)))
        return scores

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
                second_synsets = synsets_by_lemmas[second_word]
                if len(first_synsets) == 0 or len(second_synsets) == 0:
                    curr_metrics = [float(first_word == second_word)] * self.metrics_number
                    synsets_with_scores[i][j] = (curr_metrics, None, None)
                    continue
                common_synsets = [x for x in first_synsets if x in second_synsets]
                if len(common_synsets) > 0:
                    scores, synset = [1.0] * self.metrics_number, common_synsets[0]
                    synsets_with_scores[i][j] = (scores, synset, synset)
                    continue
                best_score = 0.0, [0.0] * self.metrics_number
                best_first_synset, best_second_synset = None, None
                for first_synset, second_synset in itertools.product(first_synsets, second_synsets):
                    scores = self.distance_map.get((first_synset, second_synset))
                    if scores is None:
                        first_code = self.synsets_encoding.get(first_synset)
                        second_code = self.synsets_encoding.get(second_synset)
                        if first_code is not None and second_code is not None:
                            scores = [self._score(metric, first_code, second_code) for metric in self.metrics]
                            if any(x is None for x in scores):
                                continue
                            self.distance_map[(first_synset, second_synset)] = scores
                    # np.mean is probitively slow
                    if scores is not None:
                        mean_score = sum(scores) / len(scores)
                        if mean_score > best_score[0]:
                            best_score = mean_score, scores
                            best_first_synset, best_second_synset = first_synset, second_synset
                synsets_with_scores[i][j] = (best_score[1], best_first_synset, best_second_synset)
        return synsets_with_scores, first_indexes, second_indexes





if __name__ == "__main__":
    synsets, synsets_encoding, graph, derivates = make_graph("ruthes-lite-2/relations_all.xml")
    synsets_by_lemmas, lemmas_by_synsets = collect_synsets_for_lemmas("ruthes-lite-2/senses_all.xml")
    ancestors_lists, D, depths = make_ancestors_lists(graph, derivates)
    scorer = PairwiseScorer(ancestors_lists, D, depths,
                            synsets_by_lemmas, synsets_encoding, ["lch"], verbose=1)

    # sents, data, targets = read_data("paraphraser/paraphrases_train.xml",
    #                                  ud_processor=ud_processor, parse_syntax=False)
    # train
    statprof.start()
    try:
        sents, data, targets = read_data("paraphraser/parsed_paraphrases_train_new.xml", from_parses=True)
        data_to_work = [list(zip(*elem)) for elem in zip(*data)][:4000]
        data_to_scorer = list(zip(data_to_work[::2], data_to_work[1::2]))
        similarity_scores = scorer.score_sents(data_to_scorer)
    finally:
        statprof.stop()
        with open("log.out", "w") as flog:
            statprof.display(flog, order=statprof.DisplayOrder.CUMULATIVE)
    # test
    # test_sents, test_data, test_targets = read_data("paraphraser/parsed_paraphrases_gold_new.xml", from_parses=True)
    # data_to_work = [list(zip(*elem)) for elem in zip(*data)][:1000]
    # data_to_scorer = list(zip(data_to_work[::2], data_to_work[1::2]))
    # test_similarity_scores = scorer.score_sents(data_to_scorer)
    #
    # initial_threshold = analyze_scores(
    #     similarity_scores, targets, test_similarity_scores, test_targets, verbose=1)