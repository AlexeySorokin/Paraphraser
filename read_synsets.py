import sys
from collections import defaultdict
import xml.etree.ElementTree as ET


def make_graph(infile, use_derivates=True):
    parser = ET.XMLParser(encoding="utf-8")
    treeRelations = ET.parse(infile, parser=parser)
    rootRelations = treeRelations.getroot()
    attrRelations = [child.attrib for child in rootRelations]
    synsets = set()
    # первый проход: собираем вершины
    for elem in attrRelations:
        child, parent, name = elem['child_id'], elem['parent_id'], elem['name']
        synsets.add(child)
        synsets.add(parent)
    # перекодируем синсеты индексами, чтобы работать со списками
    # (проверять значение элемента в списке быстрее, чем принадлежность множеству)
    synsets = sorted(synsets)
    synsets_encoding = {elem: i for i, elem in enumerate(synsets)}
    # synsets_reverse_encoding = {i: elem for i, elem in enumerate(synsets)}
    # второй проход: строим граф
    graph = [[] for _ in synsets]
    derivates = dict()
    for elem in attrRelations:
        child, parent, name = elem['child_id'], elem['parent_id'], elem['name']
        if name in ['hypernym', 'instance_hypernym', 'part holonym']:
            child_code = synsets_encoding[child]
            parent_code = synsets_encoding[parent]
            graph[parent_code].append(child_code)
        if name == 'derivational' and use_derivates:
            child_code = synsets_encoding[child]
            parent_code = synsets_encoding[parent]
            derivates[parent_code]=child_code
    return synsets, synsets_encoding,  graph, derivates


def collect_synsets_for_lemmas(infile, mode="main_word"):
    senses_tree = ET.parse(infile, parser=ET.XMLParser(encoding="utf-8")).getroot()
    synsets_by_lemmas = defaultdict(set)
    lemmas_by_synsets = defaultdict(set)
    for sense_tag in senses_tree:
        # по умолчанию из синсета извлекается только main_word
        if mode == "main_word":
            lemma = (sense_tag.attrib["main_word"] if sense_tag.attrib["main_word"] != "" \
                     else sense_tag.attrib["lemma"])
            lemmas = [lemma.lower()]
        elif mode == "all":
            lemmas = [x.lower() for x in sense_tag["lemma"].split()]
        else:
            raise NotImplementedError
        synset_code = sense_tag.attrib["synset_id"]
        # for lemma in lemmas:
        #     synsets_by_lemmas[lemma].add(synset_code)
        if len(lemmas) == 1:
            synsets_by_lemmas[lemmas[0]].add(synset_code)
        lemmas_by_synsets[synset_code].update(lemmas)
    # приводим к спискам
    synsets_by_lemmas = defaultdict(
        list, {lemma: sorted(lemma_synsets)
               for lemma, lemma_synsets in synsets_by_lemmas.items()})
    lemmas_by_synsets = defaultdict(
        list, {synset: sorted(synset_lemmas)
               for synset, synset_lemmas in lemmas_by_synsets.items()})
    return synsets_by_lemmas, lemmas_by_synsets


def topological_sort(graph, reverse=False):
    """
    Выполняет топологическую сортировку графа (входящая вершина ребра всегда позже исходящей)
    """
    WHITE, GREY, BLACK = 0, 1, 2
    order = []
    # покрашена вершина или нет
    colors = [WHITE] * len(graph)
    for i, children in enumerate(graph):
        if colors[i] == WHITE:
            # вершина ещё не обрабатывалась
            stack = [i]
            while len(stack) > 0:
                v = stack[-1]
                if colors[v] == WHITE:
                    colors[v] = GREY
                    for u in graph[v]:
                        if colors[u] == WHITE:
                            stack.append(u)
                        elif colors[u] == GREY:
                            # в графе обнаружен цикл
                            raise ValueError("Graph has cycles, impossible to perform topological sort")
                elif colors[v] == GREY:
                    colors[v] = BLACK
                    order.append(v)
                    stack.pop()
                elif colors[v] == BLACK:
                    stack.pop()
    print(len(graph), len(order))
    return order if reverse else order[::-1]


def make_ancestors_lists(graph, derivates, use_derivates=False):
    order = topological_sort(graph, reverse=True)
    # ancestors[i][j] --- предки вершины i на расстоянии j
    ancestors = [[] for _ in graph]
    paths_number = [0] * len(graph)
    max_depth_in_graph = 1
    depths = dict()
    for i, u in enumerate(order, 1):
        if i % 10000 == 0:
            print("{} vertexes processed".format(i))
        depths[u] = 0
        parents = graph[u]
        # if derivates.get(u) is not None:
        #     parents = graph[u] + graph[derivates[u]]
        # else:
        #     parents = graph[u]
        if len(parents) > 0:  # у вершины есть предки
            # максимальное расстояние до предка
            depths[u] = max(len(ancestors[v]) for v in parents)+1
            if depths[u] > max_depth_in_graph:
                max_depth_in_graph=depths[u]
            ancestors[u] = [set() for _ in range(depths[u])]
            ancestors[u][0] = set(parents)
            for v in parents:
                # предки v на расстоянии d становятся предками u на расстоянии d+1
                for d, curr_ancestors in enumerate(ancestors[v]):
                    ancestors[u][d+1].update(curr_ancestors)
            paths_number[u] = sum(paths_number[x] for x in parents)
        else:
            paths_number[u] = 1
        if paths_number[u] == 0:
            print(i, u)
            # sys.exit()
    # одна и та же вершина может оказаться предком на двух разных расстояниях, оставляем минимальное
    for i, curr_ancestors in enumerate(ancestors):
        processed_ancestors = set()
        for j, curr_level_ancestors in enumerate(curr_ancestors):
            curr_ancestors[j] =\
                {x for x in curr_level_ancestors if x not in processed_ancestors}
            processed_ancestors.update(curr_ancestors[j])
    paths_number_counts = defaultdict(int)
    for elem in paths_number:
        paths_number_counts[elem] +=1
    # for n, count in sorted(paths_number_counts.items()):
    #     print(n, count)
    # sys.exit()
    return ancestors, max_depth_in_graph, depths