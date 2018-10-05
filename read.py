from collections import defaultdict
from itertools import chain

import numpy as np
from bs4 import BeautifulSoup
import ujson as json

from save import save_data


def sanitize(text):
    text = text.replace("«", "\"")
    text = text.replace("»", "\"")
    text = text.replace("ё", "е")
    return text


def read_paraphrases(infile, n=-1):
    with open(infile, "r", encoding="utf-8") as fin:
        soup = BeautifulSoup(fin.read(), "lxml")
    paraphrases = soup.find_all("paraphrase")
    pairs, targets = [], []
    for i, elem in enumerate(paraphrases):
        if i == n:
            break
        first, second, target = elem.find_all(attrs={"name": ["text_1", "text_2", "class"]})
        first_text, second_text = sanitize(first.text), sanitize(second.text)
        pairs.append([first_text, second_text])
        targets.append(int(target.text))
    return pairs, targets


def read_parsed_paraphrase_file(infile):
    pairs, data, targets = [], [], []
    with open(infile, "r", encoding="utf8") as fin:
        from_json = json.load(fin)
    for elem in from_json:
        pairs.append((elem['first'], elem['second']))
        targets.append(elem['target'])
        data.extend([elem['first_parse'], elem['second_parse']])
    data = list(map(list, zip(*data)))
    return pairs, data, targets


def read_counts(infile, word_column=0, count_column=1, n=-1,
                to_lower=True, return_ranks=False):
    answer = defaultdict(int)
    ranks = dict()
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            if line[0] == "#":
                continue
            if len(answer) == n:
                break
            line = line.strip()
            splitted = line.split("\t")
            if len(splitted) <= max(word_column, count_column):
                continue
            word, count = splitted[word_column], int(splitted[count_column])
            if to_lower:
                word = word.lower()
            answer[word] += count
            if word not in ranks:
                ranks[word] = len(answer)
    return (answer, ranks) if return_ranks else answer


def read_data(infile, from_parses=False, save_file=None, ud_processor=None, parse_syntax=True, n=-1):
    if from_parses:
        pairs, data, targets = read_parsed_paraphrase_file(infile)
    else:
        pairs, targets = read_paraphrases(infile, n=n)
        # pairs = [["ЦУП установил связь с грузовым кораблем \"Прогресс\"",
        #           "Связь с космическим кораблем \"Прогресс\" восстановлена"]]
        # targets = [1]
        targets = (np.array(targets) >= 0).astype("int")
        modes = UD_MODES[:]
        if not parse_syntax:
            modes = modes[1:]
        data = ud_processor.process(list(chain.from_iterable(pairs)), modes)
        if save_file is not None:
            save_data(save_file, pairs, targets, data)
    return pairs, data, np.array(targets)


UD_MODES = ["all", "word", "lemma", "pos", "feats"]


def make_data(infile, embedder, return_weights=False, return_indexes=False,
              ud_processor=None, use_svo=False, from_parses=False, save_file=None):
    pairs, data, targets = read_data(infile, from_parses, save_file, ud_processor=ud_processor)
    parses, words, lemmas, tags = data
    if use_svo:
        sent_embeddings = embedder(
            parses, return_weights=return_weights, return_indexes=return_indexes)
    else:
        sent_embeddings = embedder(lemmas, tags, return_weights=return_weights)
    if isinstance(sent_embeddings, tuple):
        X = tuple([[elem[::2], elem[1::2]] for elem in sent_embeddings])
    else:
        X = [sent_embeddings[::2], sent_embeddings[1::2]]
    return pairs, X, targets