import json

import numpy as np


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


def dump_analysis(pairs, distances, targets):
    order = np.argsort(distances)
    with open("dump_train.out", "w", encoding="utf8") as fout:
        for i in order:
            first, second = pairs[i]
            dist, target = distances[i], targets[i]
            fout.write("{}\t{}\n{}\n{}\n".format(dist, target, first, second))


def analyze_scores(scores, targets, test_scores, test_targets, 
                   verbose=0, metric="F1"):
    scores, targets = np.array(scores), np.array(targets)
    test_scores, test_targets = np.array(test_scores), np.array(test_targets)
    m = len(scores)
    order = np.argsort(scores)
    scores, targets = scores[order], targets[order]
    order = np.argsort(test_scores)
    test_scores, test_targets = test_scores[order], test_targets[order]
    score_levels = [scores[(m * i) // 100 - 1] for i in range(1, 101)]
    curr_score_level = 0
    TP, FN, FP, TN = 0, (targets == 1).astype("int").sum(), 0, (targets == 0).astype("int").sum()
    best_metric, best_index = -1, -2
    curr_test_threshold = 0
    test_TP, test_FN  = 0, (test_targets == 1).astype("int").sum()
    test_FP, test_TN = 0, (test_targets == 0).astype("int").sum()
    corr, test_corr = TN , test_TN
    for index, (score, target) in enumerate(zip(scores, targets)):
        TP, FN, FP, TN, corr = TP+target, FN-target, FP+(1-target), TN-(1-target), corr + (2 * target - 1)
        curr_F1 = TP / (TP + 0.5 * (FN + FP))
        curr_acc = corr / len(targets)
        if index == m-1 or scores[index+1] > score:
            while curr_test_threshold < len(test_scores) and test_scores[curr_test_threshold] <= score:
                test_target = test_targets[curr_test_threshold]
                test_TP, test_FN = test_TP + test_target, test_FN - test_target
                test_FP, test_TN = test_FP + (1 - test_target), test_TN - (1 - test_target)
                test_corr += (2 * test_target - 1)
                curr_test_threshold += 1
            curr_test_F1 = test_TP / (test_TP + 0.5 * (test_FN + test_FP))
            curr_test_acc = test_corr / len(test_targets)
            curr_metric = curr_F1 if metric == "F1" else curr_acc
            curr_test_metric = curr_test_F1 if metric == "F1" else curr_test_acc
            if curr_metric > best_metric:
                best_metric, best_index, test_metric = curr_metric, index, curr_test_metric
            elif verbose and best_index == index - 1:
                print("Best {0}: {1:.2f}, test {0}: {2:.2f}, threshold: {3:.3f}".format(
                    metric, 100 * best_metric, 100 *curr_test_metric, scores[index-1]))
        if verbose:
            if (curr_score_level < 100 and score == score_levels[curr_score_level]
                    and (index == m-1 or scores[index+1] > score)):
                print("threshold: {1:.3f}, {0}: {2:.2f}, test {0}: {3:.2f}".format(
                    metric, score, 100 * curr_metric, 100 * curr_test_metric))
                # print(TP, FN, FP, TN)
                curr_score_level += 1
    print("Threshold: {1:.3f}, Train {0}: {2:.2f}, Test {0}: {3:.2f}".format(
        metric, scores[best_index], 100 * best_metric, 100 * test_metric))
    return scores[best_index]


def output_errors(sents, targets, threshold, scores, word_scores,
                  outfile, metrics_number, reverse=False):
    order = np.argsort(scores)
    if reverse:
        order = order[::-1]
    with open(outfile, "w", encoding="utf8") as fout:
        for index in order:
            if (scores[index] > threshold) != reverse:
                break
            if targets[index] == int(reverse):
                fout.write("\n".join(sents[index]) + "\n")
                first_words, second_words, curr_pairwise_scores = word_scores[index]
                first_length = max([len(x) for x in first_words])
                column_lengths = [max(len(x), 5 * metrics_number) for x in second_words]
                fout.write("{:<{width}}".format("", width=first_length + 2))
                for word, width in zip(second_words, column_lengths):
                    fout.write("{:<{width}}".format(word, width=width + 2))
                fout.write("\n")
                for i, (word, curr_scores) in enumerate(zip(first_words, curr_pairwise_scores)):
                    fout.write("{:<{width}}".format(word, width=first_length + 2))
                    for j, width in enumerate(column_lengths):
                        fout.write("{:<{width}}".format(
                            " ".join("{:.2f}".format(x) for x in curr_scores[j][0]), width=width + 2))
                    fout.write("\n")
                fout.write("Aggregate: {:.2f}\n\n".format(1.0 - scores[index]))

