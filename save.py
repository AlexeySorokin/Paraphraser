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


def analyze_scores(scores, targets, test_scores, test_targets, verbose=0, metric="F1"):
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
    best_F1, best_index = -1, -2
    curr_test_threshold = 0
    test_TP, test_FN  = 0, (test_targets == 1).astype("int").sum()
    test_FP, test_TN = 0, (test_targets == 0).astype("int").sum()
    # acc, test_acc = 0.0, 0.0
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
            elif verbose and best_index == index - 1:
                print("Best F1: {:.2f}, test F1: {:.2f}, threshold: {:.3f}".format(
                    100 * best_F1, 100 *curr_test_F1, scores[index-1]))
        if verbose:
            if (curr_score_level < 100 and score == score_levels[curr_score_level]
                    and (index == m-1 or scores[index+1] > score)):
                print("threshold: {:.3f}, F1: {:.2f}, test F1: {:.2f}".format(
                    score, 100 * curr_F1, 100 * curr_test_F1))
                print(TP, FN, FP, TN)
                curr_score_level += 1
    print("Threshold: {:.3f}, Train F1: {:.2f}, Test F1: {:.2f}".format(
        scores[best_index], 100 * best_F1, 100 * test_F1))
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

