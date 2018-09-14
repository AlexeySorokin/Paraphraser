import os
import ujson as json
from scipy.spatial.distance import cosine
import keras.layers as kl
import keras

from ufal_udpipe import Model
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.deep import find_config
from deeppavlov.core.commands.infer import build_model_from_config

from read import *
from work import *
from embedders import TfIdfEmbedder

PARAPHRASE_TRAIN_PATH = "paraphraser/paraphrases_train.xml"
PARAPHRASE_TEST_PATH = "paraphraser/paraphrases_gold.xml"
# EMBEDDINGS_PATH = "/home/alexeysorokin/data/Data/Fasttext/cc.ru.300.bin"
EMBEDDINGS_PATH = "/home/alexeysorokin/data/Data/DeepPavlov Embeddings/ft_native_300_ru_wiki_lenta_lower_case.bin"
# EMBEDDINGS_PATH = "/home/alexeysorokin/data/Data/DeepPavlov Embeddings/ft_native_300_ru_wiki_lenta_lemmatize.bin"
COUNTS_PATH = "/home/alexeysorokin/data/Data/frequencies/Taiga_freq_lemmas.txt"
CONFIG_PATH = [os.path.join("config", x) for x in sorted(os.listdir("config")) if x.startswith("config_")]
TAG_CONFIG_PATH = "config/pos_config.json"
FROM_PARSES = False
TRAIN_SAVE_PATH = "paraphraser/parsed_paraphrases_train.xml"
TEST_SAVE_PATH = "paraphraser/parsed_paraphrases_gold.xml"


# import tensorflow as tf
# import keras.backend.tensorflow_backend as kbt
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# kbt.set_session(tf.Session(config=config))

UD_MODES = ["all", "word", "lemma", "pos"]

def read_data(infile, from_parses=False, save_file=None):
    if from_parses:
        pairs, data, targets = read_parsed_paraphrase_file(infile)
    else:
        # pairs, targets = read_paraphrases(infile)
        # pairs = [["ЦУП установил связь с грузовым кораблем \"Прогресс\"",
        #           "Связь с космическим кораблем \"Прогресс\" восстановлена"]]
        # targets = [1]
        # targets = (np.array(targets) >= 0).astype("int")
        data = ud_processor.process(list(chain.from_iterable(pairs)), UD_MODES)
        if save_file is not None:
            save_data(save_file, pairs, targets, data)
    return pairs, data, np.array(targets)


def save_data(outfile, pairs, targets, data):
    to_save = []
    for i, ((first, second), target) in enumerate(zip(pairs, targets)):
        curr_to_save = {"first": first, "second": second, "target": int(target),
                        "first_parse": [elem[2*i] for elem in data],
                        "second_parse": [elem[2*i+1] for elem in data]}
        to_save.append(curr_to_save)
    with open(outfile, "w", encoding="utf8") as fout:
        json.dump(to_save, fout)
    return


def make_data(infile, embedder, from_parses=False, save_file=None):
    pairs,  (parses, words, lemmas, tags), targets = read_data(infile, from_parses, save_file)
    sent_embeddings = embedder(lemmas, tags)
    X = [sent_embeddings[::2], sent_embeddings[1::2]]
    return pairs, X, targets


def build_network(dim, layers=2, activation="tanh"):
    first = kl.Input(shape=(dim,))
    second = kl.Input(shape=(dim,))
    inputs = [first, second]

    for i in range(layers):
        first = kl.Dense(dim, activation=activation)(first)
        second = kl.Dense(dim, activation=activation)(second)

    to_scorer = kl.Concatenate(axis=1)([first, second])
    score = kl.Dense(1, activation="sigmoid")(to_scorer)
    model = keras.Model(inputs, score)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(model.summary())
    return model


def dump_analysis(pairs, distances, targets):
    order = np.argsort(distances)
    with open("dump_train.out", "w", encoding="utf8") as fout:
        for i in order:
            first, second = pairs[i]
            dist, target = distances[i], targets[i]
            fout.write("{}\t{}\n{}\n{}\n".format(dist, target, first, second))


def analyze_scores(scores, targets, test_scores, test_targets):
    m = len(scores)
    order = np.argsort(scores)
    scores, targets = scores[order], targets[order]
    order = np.argsort(test_scores)
    test_scores, test_targets = test_scores[order], test_targets[order]
    score_levels = [scores[(m * i) // 100 - 1] for i in range(1, 101)]
    curr_score_level = 0
    TP, FN, FP, TN = 0, (targets == 1).astype("int").sum(), 0, (targets == 0).astype("int").sum()
    best_F1, best_index = -1, -2
    curr_test_threshold = 0
    test_TP, test_FN  = 0, (test_targets == 1).astype("int").sum()
    test_FP, test_TN = 0, (test_targets == 0).astype("int").sum()
    for index, (score, target) in enumerate(zip(scores, targets)):
        TP, FN, FP, TN = TP+target, FN-target, FP+(1-target), TN-(1-target)
        curr_F1 = TP / (TP + 0.5 * (FN + FP))
        if index == m-1 or scores[index+1] > score:
            while curr_test_threshold < len(test_scores) and test_scores[curr_test_threshold] <= score:
                test_target = test_targets[curr_test_threshold]
                test_TP, test_FN = test_TP + test_target, test_FN - test_target
                test_FP, test_TN = test_FP + (1 - test_target), test_TN - (1 - test_target)
                curr_test_threshold += 1
            curr_test_F1 = test_TP / (test_TP + 0.5 * (test_FN + test_FP))
            if curr_F1 > best_F1:
                best_F1, best_index, test_F1 = curr_F1, index, curr_test_F1
            # elif best_index == index - 1:
            #     print("Best F1: {:.2f}, test F1: {:.2f}, threshold: {:.3f}".format(
            #         100 * best_F1, 100 *test_F1, scores[index-1]))
        # if (curr_score_level < 100 and score == score_levels[curr_score_level]
        #         and (index == m-1 or scores[index+1] > score)):
        #     print("threshold: {:.3f}, F1: {:.2f}, test F1: {:.2f}".format(
        #         score, 100 * curr_F1, 100 * test_F1))
        #     curr_score_level += 1
    print("Threshold: {:.3f}, Train F1: {:.2f}, Test F1: {:.2f}".format(
        scores[best_index], 100 * best_F1, 100 * test_F1))
    return

if __name__ == "__main__":

    word_counts = read_counts(COUNTS_PATH, 1, 2)
    word_embedder = FasttextEmbedder(EMBEDDINGS_PATH, dim=300)

    # tagger = build_model_from_config(find_config("morpho_ru_syntagrus_train_pymorphy"))
    # ud_model = Model.load("russian-syntagrus-ud-2.0-170801.udpipe")
    # ud_processor = MixedUDPreprocessor(ud_model, tagger)
    for config_path in CONFIG_PATH:
        print("{:<24}".format(config_path.split("/")[-1]), end="\t")
        with open(config_path, "r", encoding="utf8") as fin:
            embedder_params = json.load(fin)
            if "tag_weights" in embedder_params:
                tag_weights = embedder_params["tag_weights"][1]
                tag_weights = {tuple(key.split("_")): value for key, value in tag_weights.items()}
        embedder = TfIdfEmbedder(word_embedder, word_counts, **embedder_params)

    # X_train, y_train = make_data(PARAPHRASE_TRAIN_PATH, embedder,
    #                              from_parses=FROM_PARSES, save_file=TRAIN_SAVE_PATH)
        pairs_train, X_train, y_train = make_data(TRAIN_SAVE_PATH, embedder, from_parses=True)
        distances = np.array([cosine(first, second) for first, second in zip(*X_train)])
        pairs_test, X_test, y_test = make_data(TEST_SAVE_PATH, embedder, from_parses=True)
        test_distances = np.array([cosine(first, second) for first, second in zip(*X_test)])
        analyze_scores(distances, y_train, test_distances, y_test)
# dump_analysis(pairs_train, distances, y_train)


# X_test, y_test = make_data(PARAPHRASE_TEST_PATH, embedder,
#                            from_parses=FROM_PARSES, save_file=TEST_SAVE_PATH)
# pairs_test, X_test, y_test = make_data(TEST_SAVE_PATH, embedder, from_parses=True)

# t_last = t
# X_train = [train_sent_embeddings[::2], train_sent_embeddings[1::2]]

# model = build_network(word_embedder.dim, layers=2)
# model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2)