from ufal.udpipe import Model

from read import read_paraphrases
from work import *

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