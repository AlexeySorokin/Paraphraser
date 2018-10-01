from collections import defaultdict
from bs4 import BeautifulSoup
import ujson as json


def sanitize(text):
    text = text.replace("«", "\"")
    text = text.replace("»", "\"")
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
    data =list(map(list, zip(*data)))
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