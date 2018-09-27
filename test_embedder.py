import os
import getopt
import sys

import ujson as json
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
# import keras.layers as kl
from keras.callbacks import *

from ufal_udpipe import Model
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.deep import find_config
from deeppavlov.core.commands.infer import build_model_from_config

from read import *
from work import *
from network import *
from embedders import *

PARAPHRASE_TRAIN_PATH = "paraphraser/paraphrases_train.xml"
PARAPHRASE_TEST_PATH = "paraphraser/paraphrases_gold.xml"
# EMBEDDINGS_PATH = "/home/alexeysorokin/data/Data/Fasttext/cc.ru.300.bin"
EMBEDDINGS_PATH = "/home/alexeysorokin/data/Data/DeepPavlov Embeddings/ft_native_300_ru_wiki_lenta_lower_case.bin"
# EMBEDDINGS_PATH = "/cephfs/home/sorokin/data/embeddings/ft_native_300_ru_wiki_lenta_lower_case.bin"
# EMBEDDINGS_PATH = "/home/alexeysorokin/data/Data/DeepPavlov Embeddings/ft_native_300_ru_wiki_lenta_lemmatize.bin"
EMBEDDINGS_DIM = 300
COUNTS_PATH = "/home/alexeysorokin/data/Data/frequencies/Taiga_freq_lemmas.txt"
# COUNTS_PATH = "../data/frequencies/Taiga_freq_lemmas.txt"
CONFIG_PATH = ["config/config.json"]
TRAIN_SAVE_PATH = "paraphraser/parsed_paraphrases_train.xml"
TEST_SAVE_PATH = "paraphraser/parsed_paraphrases_gold.xml"


# import tensorflow as tf
# import keras.backend.tensorflow_backend as kbt
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# kbt.set_session(tf.Session(config=config))

UD_MODES = ["all", "word", "lemma", "pos"]

def read_data(infile, from_parses=False, save_file=None, ud_processor=None):
    if from_parses:
        pairs, data, targets = read_parsed_paraphrase_file(infile)
    else:
        pairs, targets = read_paraphrases(infile)
        # pairs = [["ЦУП установил связь с грузовым кораблем \"Прогресс\"",
        #           "Связь с космическим кораблем \"Прогресс\" восстановлена"]]
        # targets = [1]
        targets = (np.array(targets) >= 0).astype("int")
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


def make_data(infile, embedder, ud_processor=None,
              use_svo=False, from_parses=False, save_file=None):
    pairs, data, targets = read_data(infile, from_parses, save_file, ud_processor=ud_processor)
    parses, words, lemmas, tags = data
    sent_embeddings = embedder(parses) if use_svo else embedder(lemmas, tags)
    X = [sent_embeddings[::2], sent_embeddings[1::2]]
    return pairs, X, targets


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
        #     elif best_index == index - 1:
        #         print("Best F1: {:.2f}, test F1: {:.2f}, threshold: {:.3f}".format(
        #             100 * best_F1, 100 *test_F1, scores[index-1]))
        # if (curr_score_level < 100 and score == score_levels[curr_score_level]
        #         and (index == m-1 or scores[index+1] > score)):
        #     print("threshold: {:.3f}, F1: {:.2f}, test F1: {:.2f}".format(
        #         score, 100 * curr_F1, 100 * test_F1))
        #     print(TP, FN, FP, TN)
        #     curr_score_level += 1
    print("Threshold: {:.3f}, Train F1: {:.2f}, Test F1: {:.2f}".format(
        scores[best_index], 100 * best_F1, 100 * test_F1))
    return scores[best_index]

    
def make_train_params(params):
    if params is None:
        return dict()
    params = params.copy()
    callbacks, save_path = [], None
    if "early_stopping" in params:
        callbacks.append(EarlyStopping(**params.pop("early_stopping")))
    if "save_path" in params:
        save_path = params.pop("save_path")
        callbacks.append(ModelCheckpoint(
            save_path, save_weights_only=True, save_best_only=True))
    params["callbacks"] = callbacks
    return params, save_path
    
def measure_quality(y_true, y_pred):
    y_pred_binary = (y_pred >= 0.0)
    scores = list(precision_recall_fscore_support(y_true, y_pred_binary, average="binary"))[:3]
    scores = np.array(scores + [accuracy_score(y_true, y_pred_binary)])
    return scores


def reshape_svo_embeddings(data):
    return np.sum(np.reshape(data, (data.shape[0], 3, -1)), axis=1)
    
    
SHORT_OPTS = "S:T:pn:fo:e"
QUALITY_FORMAT_STRING = "P:{:.2f} R:{:.2f} F1:{:.2f} A:{:.2f}"

if __name__ == "__main__":

    np.set_printoptions(precision=3)
    np.random.seed(189)
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    
    vectors_save_file, targets_save_file, use_svo = None, None, False
    from_parses = False
    outfile, output_ensemble_scores = None, False
    trials = 1
    for opt, val in opts:
        if opt == "-S":
            vectors_save_file = val
        elif opt == "-T":
            targets_save_file = val
        elif opt == "-p":
            use_svo = True
        elif opt == "-n":
            trials = int(val)
        elif opt == "-f":
            from_parses = True
        elif opt == "-o":
            outfile = val
        elif opt == "-e":
            output_ensemble_scores = True

    print("Reading counts...")
    word_counts = read_counts(COUNTS_PATH, 1, 2)
    print("Reading embeddings...")
    word_embedder = FasttextEmbedder(EMBEDDINGS_PATH, dim=300)


    if from_parses:
        tagger = build_model_from_config(find_config("morpho_ru_syntagrus_train_pymorphy"))
        print("Tagger built")
        ud_model = Model.load("russian-syntagrus-ud-2.0-170801.udpipe")
        print("UD Model loaded")
        ud_processor = MixedUDPreprocessor(ud_model, tagger)
    else:
        ud_processor = None
    for config_path in CONFIG_PATH:
        # print("{:<24}".format(config_path.split("/")[-1]), end="\t")
        with open(config_path, "r", encoding="utf8") as fin:
            config_params = json.load(fin)
        embedder_params = config_params.get("embedder", dict())
        if "tag_weights" in embedder_params:
            tag_weights = embedder_params["tag_weights"][1]
            tag_weights = {tuple(key.split("_")): value for key, value in tag_weights.items()}
        Embedder = SVOFreqPosEmbedder if use_svo else FreqPosEmbedder
        embedder = Embedder(word_embedder, word_counts, **embedder_params)

        # pairs_train, X_train, y_train = make_data(PARAPHRASE_TRAIN_PATH, embedder,
                                                  # from_parses=from_parses,
                                                  # ud_processor=ud_processor,
                                                  # save_file=TRAIN_SAVE_PATH)
        pairs_train, X_train, y_train =\
            make_data(TRAIN_SAVE_PATH, embedder, use_svo=use_svo, from_parses=True)
        if vectors_save_file is not None:
            with open(vectors_save_file, "w") as fout:
                for first, second in zip(*X_train):
                    fout.write("{}\n".format(",".join("{:.2f}".format(x) for x in first)))
                    fout.write("{}\n".format(",".join("{:.2f}".format(x) for x in second)))
        if targets_save_file is not None:
            with open(targets_save_file, "w") as fout:
                fout.write("\n".join(map(str, y_train)))
        if use_svo:
            for_distances = [reshape_svo_embeddings(elem) for elem in X_train]
        else:
            for_distances = X_train
        distances = np.array([cosine(first, second) 
                              for first, second in zip(*for_distances)])
        # pairs_test, X_test, y_test = make_data(PARAPHRASE_TEST_PATH, embedder,
                                               # from_parses=FROM_PARSES, save_file=TEST_SAVE_PATH)
        pairs_test, X_test, y_test =\
            make_data(TEST_SAVE_PATH, embedder, use_svo=use_svo, from_parses=True)
        if use_svo:
            for_distances = [reshape_svo_embeddings(elem) for elem in X_test]
        else:
            for_distances = X_test
        test_distances = np.array([cosine(first, second) 
                                   for first, second in zip(*for_distances)])
        initial_threshold = analyze_scores(distances, y_train, test_distances, y_test)
        
        network_params = config_params.get("network", dict())
        
        quality = np.zeros(shape=(trials, 4), dtype="float32")
        test_scores = np.zeros(shape=(trials, len(X_test[0])), dtype="float32")
        for i in range(trials):
            indexes = np.arange(len(y_train), dtype=int)
            np.random.shuffle(indexes)
            # print(indexes[:10])
            X_train, y_train = [X_train[0][indexes], X_train[1][indexes]], np.array(y_train)[indexes]

            network_params["dim"] = network_params.get("dim", EMBEDDINGS_DIM)
            network = NetworkBuilder(use_svo=use_svo,
                                     initial_threshold=initial_threshold,
                                     **network_params).build()
            train_params, save_path = make_train_params(config_params.get("train"))
            network.fit(X_train, y_train, **train_params)
            network.load_weights(save_path)
            curr_scores = network.predict(X_test)[:,0]
            quality[i] = measure_quality(y_test, curr_scores)
            test_scores[i] = curr_scores
            print(QUALITY_FORMAT_STRING.format(*(100 * quality[i])))
        print("")
        if trials > 1:
            for i in range(trials):
                print(QUALITY_FORMAT_STRING.format(*(100 * quality[i])))
        print(("Average scores. "+ QUALITY_FORMAT_STRING).format(*(100 * quality.mean(axis=0))))
        ensemble_scores = np.mean(test_scores, axis=0)
        if trials % 2 == 0:
            median_scores = np.sort(test_scores, axis=0)[(trials // 2 - 1) : (trials // 2 + 1)].mean(axis=0)
        else:
            median_scores = np.sort(test_scores, axis=0)[trials // 2]
        ensemble_quality = np.array(measure_quality(y_test, ensemble_scores))
        median_quality = np.array(measure_quality(y_test, median_scores))
        print(("Ensemble scores. "+ QUALITY_FORMAT_STRING).format(*(100 * ensemble_quality)))
        print(("Median scores. "+ QUALITY_FORMAT_STRING).format(*(100 * median_quality)))
        if outfile is not None:
            scores_to_output = ensemble_scores if output_ensemble_scores else median_scores
            with open(outfile, "w", encoding="utf8") as fout:
                for i in range(len(scores_to_output)):
                    if int(scores_to_output[i] >= 0) != y_test[i]:
                        fout.write("{}\n{}\n".format(*pairs_test[i]))
                        basic_score = initial_threshold - test_distances[i]
                        fout.write("{}\t{:.3f}\t{:.3f}\n".format(y_test[i], scores_to_output[i], basic_score))
                        fout.write("\t".join(["{:.3f}".format(x) for x in test_scores[:,i]]) + "\n\n")




        
# dump_analysis(pairs_train, distances, y_train)


# X_test, y_test = make_data(PARAPHRASE_TEST_PATH, embedder,
#                            from_parses=FROM_PARSES, save_file=TEST_SAVE_PATH)
# pairs_test, X_test, y_test = make_data(TEST_SAVE_PATH, embedder, from_parses=True)

# t_last = t
# X_train = [train_sent_embeddings[::2], train_sent_embeddings[1::2]]

# model = build_network(word_embedder.dim, layers=2)
# model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2)