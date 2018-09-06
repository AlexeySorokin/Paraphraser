from ufal.udpipe import Model, Pipeline
from collections import defaultdict
from bs4 import BeautifulSoup

import numpy as np

def read_paraphrases(infile):
    with open(infile, "r", encoding="utf-8") as fin:
        soup = BeautifulSoup(fin.read(), "lxml")
    paraphrases = soup.find_all("paraphrase")
    pairs, targets = [], []
    for elem in paraphrases:
        first, second, target = elem.find_all(attrs={"name": ["text_1", "text_2", "class"]})
        pairs.append([first.text, second.text])
        targets.append(int(target.text))
    return pairs, targets

def make_tag(s):
    if s == "_":
        return dict()
    return dict(elem.split("=", maxsplit=1) for elem in s.split("|"))

def is_verb(pos, tag):
    return (((pos in ["VERB", "AUX"]) and ("Case" not in tag))
            or (pos == "ADV") or (pos == "ADJ" and tag.get("Variant") == "Short"))


class SyntaxTreeGraph:

    def __init__(self, data):
        self._make_edges(data)

    def __len__(self):
        return self.nodes_number

    def _make_edges(self, data):
        self.nodes_number = len(data)
        self.edges = [[] for _ in range(self.nodes_number + 1)]
        self.deps = [[] for _ in range(self.nodes_number + 1)]
        self.tags = [(elem[3], make_tag(elem[5])) for elem in data]
        for child, elem in enumerate(data, 1):
            root, dep = int(elem[6]), elem[7]
            self.edges[root].append(child)
            self.deps[root].append(dep)
        return

    def _order(self):
        color = [0] * (self.nodes_number + 1)
        stack = [0]
        order = []
        while len(stack) > 0:
            v = stack[-1]
            if color[v] == 0:
                color[v] = 1
                stack.extend(self.edges[v])
                continue
            elif color[v] == 1:
                color[v] = 2
                order.append(v)
            stack.pop()
        return order[::-1]

    def get_indexes(self):
        #         print(self.edges)
        verb_indexes = self._get_verb_indexes()
        subj_indexes, obj_indexes = self._get_noun_indexes(verb_indexes)
        return {"verb": verb_indexes, "subj": subj_indexes, "obj": obj_indexes}

    def _get_verb_indexes(self):
        parents_for_verbs, answer = [0], []
        while len(parents_for_verbs) > 0:
            #             for parent_index in parents_for_verbs:
            parent_index = parents_for_verbs.pop()
            curr_dep_data = self.edges[parent_index], self.deps[parent_index]
            for verb_index, dep in zip(*curr_dep_data):
                pos, tag = self.tags[verb_index - 1]
                if dep in ["root", "xcomp", "nsubj"] and is_verb(pos, tag):
                    answer.append(verb_index)
                    parents_for_verbs.append(verb_index)
        return answer

    def _get_noun_indexes(self, verb_indexes):
        subj_indexes, obj_indexes = [], []
        for verb_index in verb_indexes:
            is_passive = (self.tags[verb_index - 1][-1].get("Voice") == "Pass")
            curr_dep_data = self.edges[verb_index], self.deps[verb_index]
            for dep_index, dep in zip(*curr_dep_data):
                pos, tag = self.tags[dep_index - 1]
                if pos not in ["NOUN", "PROPN"]:
                    continue
                if dep == "nsubj":
                    dest = obj_indexes if is_passive else subj_indexes
                elif dep in ["nsubj:pass", "obj"]:
                    dest = obj_indexes
                elif self._can_be_direct_object(dep_index, dep):
                    dest = obj_indexes
                else:
                    continue
                dest.append(dep_index)
        return subj_indexes, obj_indexes

    def make_subject_types(self):
        indexes = self.get_indexes()
        answer = ["verb"] * self.nodes_number
        for key, key_indexes in indexes.items():
            if key == "verb":
                continue
            curr_indexes = key_indexes[:]
            while len(curr_indexes) > 0:
                index = curr_indexes.pop()
                answer[index-1] = key
                curr_indexes.extend(self.edges[index])
        return answer


    def _can_be_direct_object(self, index, dep):
        pos, tag = self.tags[index - 1]
        if pos not in ["NOUN", "PROPN"]:
            return False
        if dep != "obl":
            return False
        object_deps = self.deps[index]
        if "case" in object_deps:
            return False
        case = tag.get("Case")
        return case in ["Nom", "Acc", "Gen"]

def prettify_UD_output(s, attach_single_root=False):
    lines = s.split("\n")
    state = 0
    answer, curr_sent = [], []
    for line in lines:
        if state == 0:
            if line.startswith("# sent_id"):
                iter(lines).__next__()
                state = 2
                curr_sent = []
                continue
        elif state == 2:
            state = 1
        elif state == 1:
            if line == "":
                answer.append(curr_sent)
                curr_sent, state = [], 0
                continue
            line_parse = line.split("\t")
            for i in [0, 6]:
                line_parse[i] = int(line_parse[i])
            curr_sent.append(line.split("\t"))
    if curr_sent != []:
        answer.append(curr_sent)
    if attach_single_root:
        start_offsets = np.cumsum([len(x) for x in answer])
        for i, elem in enumerate(answer[1:], 1):
            for line_parse in elem:
                line_parse[0] += start_offsets[i-1]
                line_parse[6] += start_offsets[i-1]
    return answer


model = Model.load("russian-syntagrus-ud-2.0-170801.udpipe")
pipeline = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

pairs, targets = read_paraphrases("paraphraser/paraphrases_train.xml")
phrases = [phrase for pair in pairs for phrase in pair]

parses = []
with open("parses.out", "w", encoding="utf8") as fout:
    for i, phrase in enumerate(phrases[:20], 1):
        if i % 1000 == 0:
            print("{} phrases parsed".format(i))
        phrase = "\n".join(phrase.split(":"))
        if phrase[-1] not in ".?!":
            phrase += "."
        parse = prettify_UD_output(pipeline.process(phrase))[0]
        fout.write("\n".join("\t".join(elem[:8]) for elem in parse) + "\n\n")
        parses.append(parse)
parse_data = []
for parse in parses:
    parse_data.append(SyntaxTreeGraph(parse))
for j, elem in enumerate(parse_data):
    # indexes = elem.get_indexes()
    # print(j, end=" ")
    # for key in ["subj", "verb", "obj"]:
    #     for index in indexes[key]:
    #         # print(j, key, index, parses[j][index-1][1])
    #         print("{}:{}".format(key, parses[j][index-1][1]), end=" ")
    # print("")
    phrase_data = elem.make_subject_types()
    for label, elem in zip(phrase_data, parses[j]):
        print("{}:{}".format(label, elem[1]), end=" ")
    print("")