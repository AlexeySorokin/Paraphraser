from itertools import chain
from time import time

import numpy as np

import pymorphy2
import russian_tagsets.converters as tag_converters
from ufal_udpipe import Pipeline

from lemmatize_ud_with_pymorphy import are_compatible_tags


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
    
    def make_branches(self, d=3):
        answer = [[]] + [None] * self.nodes_number
        depths = [0] * self.nodes_number
        verb_indexes = self._get_verb_indexes()
        is_passive = (self.tags[self.heads - 1][-1].get("Voice") == "Pass")
        queue = [(0, True, False)]
        while len(queue) > 0:
            parent_index, is_verb_index, is_passive = parents_for_verbs.pop()
            curr_dep_data = self.edges[parent_index], self.deps[parent_index]
            for child_index, dep in zip(*curr_dep_data):
                pos, tag = self.tags[child_index - 1]
                if is_verb_index:
                    is_child_verb = (dep in ["root", "xcomp"] and is_verb(pos, tag))
                    is_child_passive = is_passive or (is_child_verb and tag.get("Voice") == "Pass")
                    if dep == "nsubj":
                        child_answer = ["nsubj"] if not is_passive else ["nsubj:pass"]
                    elif dep in ["nsubj:pass", "obj", "obl"]:
                        child_answer = [dep]
                    elif is_child_verb:
                        child_answer = [dep]
                    else:
                        child_answer = answer[parent_index] + [dep]
                else:
                    is_child_verb, is_child_passive = False, is_passive
                    child_answer = answer[parent_index] + [dep]
                queue.append((child_index, is_child_verb, is_child_passive))
                answer[child_index] = child_answer
        return answer[1:]

    
def prettify_UD_output(s, attach_single_root=False, has_header=True):
    lines = s.split("\n")
    start_state = 0 if has_header else 1
    state = start_state
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
                if len(curr_sent) > 0:
                    answer.append(curr_sent)
                curr_sent, state = [], start_state
                continue
            curr_sent.append(line.split("\t"))
    if len(curr_sent) > 0:
        answer.append(curr_sent)
    if attach_single_root:
        start_offsets = np.cumsum([len(x) for x in answer])
        for offset, elem in zip(start_offsets, answer[1:]):
            for line_parse in elem:
                line_parse[0] = str(int(line_parse[0]) + offset)
                if line_parse[6] != '0' and line_parse[6] != "_":
                    line_parse[6] = str(int(line_parse[6]) + offset)
            answer[0].extend(elem)
    return answer


def UD_list_to_str(sent):
    return "\n".join("\t".join(elem) for elem in sent)


def split_by_colons(sent):
    colon_indexes = [i for i, elem in enumerate(sent) if elem[1] == ":"]
    colon_indexes = [-1] + colon_indexes + [len(sent) - 1]
    answer = []
    for j, start in enumerate(colon_indexes[:-1]):
        end = colon_indexes[j+1]
        curr_phrase = sent[start+1:end+1]
        if j >= 1:
            for index, elem in enumerate(curr_phrase):
                elem[0] = str(index+1)
        answer.append(curr_phrase)
    return answer

def make_pos_and_feats(tag):
    splitted = tag.split(",", maxsplit=1)
    if len(splitted) == 1:
        pos, feats = splitted[0], "_"
    else:
        pos, feats = splitted
    return pos, feats


class MixedUDPreprocessor:

    UD_FIELD_INDEXES = {"word": 1, "lemma": 2, "pos": 3, "feats": 5, "head": 6, "rel": 7}

    def __init__(self, model, tagger, batch_size=32):
        self.tokenizer = Pipeline(model, "tokenize", Pipeline.NONE, Pipeline.NONE, "conllu")
        self.tagger = tagger
        self.lemmatizer = pymorphy2.MorphAnalyzer()
        self.converter = tag_converters.converter('opencorpora-int', 'ud20')
        self.parser = Pipeline(model, "conllu", Pipeline.NONE, Pipeline.DEFAULT, "conllu")
        self.batch_size = batch_size

    def process(self, sents, fields_to_return=None):
        if fields_to_return is None:
            fields_to_return = ["all"]
        t_last = time()
        UD_sents = [prettify_UD_output(self.tokenizer.process(sent), attach_single_root=True)[0]
                    for sent in sents]
        t = time()
        print("Tokenizing: {:.2f}".format(t - t_last))
        t_last = t
        word_sents = [[elem[1] for elem in sent] for sent in UD_sents]
        tag_sents = []
        for start in range(0, len(word_sents), self.batch_size):
            end = start + self.batch_size
            tag_sents.extend(self.tagger(word_sents[start:end]))
        t = time()
        print("Tagging: {:.2f}".format(t - t_last))
        t_last = t
        lemma_sents = [[self._make_lemma(word, tag) for word, tag in zip(word_sent, tag_sent)]
                       for word_sent, tag_sent in zip(word_sents, tag_sents)]
        t = time()
        print("Lemmatizing: {:.2f}".format(t - t_last))
        t_last = t
        for i, (tag_sent, lemma_sent) in enumerate(zip(tag_sents, lemma_sents)):
            for j, (tag, lemma) in enumerate(zip(tag_sent, lemma_sent)):
                pos, feats = make_pos_and_feats(tag)
                for index, value in zip([3, 5, 2], [pos, feats, lemma]):
                    UD_sents[i][j][index] = value
        if  "all" in fields_to_return or "head" in fields_to_return or 'rel' in fields_to_return:
            for i, sent in enumerate(UD_sents):
                UD_sents[i] = split_by_colons(sent)
            sents_to_parse = ["\n\n".join(UD_list_to_str(elem) for elem in sent) for sent in UD_sents]
            t = time()
            print("Preparing to syntax parse: {:.2f}".format(t - t_last))
            t_last = t
            parse_sents = [prettify_UD_output(self.parser.process(sent), has_header=False, attach_single_root=True)[0]
                           for sent in sents_to_parse]
            t = time()
            print("Parsing syntax: {:.2f}".format(t - t_last))
        else:
            parse_sents = UD_sents
        t_last = t
        answer = []
        if isinstance(fields_to_return, str):
            fields_to_return = [fields_to_return]
        for key in fields_to_return:
            if key == "all":
                answer.append(parse_sents)
                continue
            index = self.UD_FIELD_INDEXES.get(key)
            if index is None:
                answer.append(None)
            else:
                answer.append([[word_parse[index] for word_parse in elem] for elem in parse_sents])
        return answer


    def _make_lemma(self, word, tag):
        parses = self.lemmatizer.parse(word)
        for parse in parses:
            curr_tag = str(parse.tag)
            curr_lemma = parse.normal_form.replace("ё", "е")
            curr_ud_tag = self.converter(curr_tag)
            if are_compatible_tags(tag, curr_ud_tag):
                return curr_lemma
        return parses[0].normal_form