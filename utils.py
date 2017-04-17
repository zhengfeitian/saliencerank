#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Various util functions.
Including code from textrank, which is licensed under the MIT License.
'''
from itertools import combinations as combinations
from Queue import Queue as _Queue
import re

WINDOW_SIZE = 2

def get_first_window(split_text):
    return split_text[:WINDOW_SIZE]

#tokens is a list of words 
def set_graph_edge(graph, tokens, word_a, word_b):
    if word_a in tokens and word_b in tokens:
        edge = (word_a, word_b)
        if graph.has_node(word_a) and graph.has_node(word_b) and not graph.has_edge(*edge):
            graph.add_edge(*edge)

def process_first_window(graph, tokens, split_text):
    first_window = get_first_window(split_text)
    for word_a, word_b in combinations(first_window, 2):
        set_graph_edge(graph, tokens, word_a, word_b)

def init_queue(split_text):
    queue = _Queue()
    first_window = get_first_window(split_text)
    for word in first_window[1:]:
        queue.put(word)
    return queue

def process_word(graph, tokens, queue, word):
    for word_to_compare in queue_iterator(queue):
        set_graph_edge(graph, tokens, word, word_to_compare)


def update_queue(queue, word):
    queue.get()
    queue.put(word)
    assert queue.qsize() == (WINDOW_SIZE - 1)

def process_text(graph, tokens, split_text):
    queue = init_queue(split_text)
    for i in xrange(WINDOW_SIZE, len(split_text)):
        word = split_text[i]
        process_word(graph, tokens, queue, word) 
        update_queue(queue, word)

def queue_iterator(queue):
    iterations = queue.qsize()
    for i in xrange(iterations):
        var = queue.get()
        yield var
        queue.put(var)

def set_graph_edges(graph, tokens, split_text):
    process_first_window(graph, tokens, split_text)
    process_text(graph, tokens, split_text)

#retuns dictionary of dictionaries: {topic i : {word: count in given topic i }}
def parse_weights_from_file (filename): 
    topics_dict ={}
    
    count = 0 
    with open(filename) as f:
            for line in f: 
                single_topic ={}
                
                line_list =re.split(r',+', line)
                for el in line_list:
                    split_el = re.split(r'\t+', el)
                    if len(split_el)==2: 
                        single_topic[split_el[0]]= int(split_el[1]) 
                topics_dict[count] = single_topic
                count= count + 1
    return topics_dict

def load_docsXtopics_from_file (filename): 
    docsXtopics_list = []
    with open(filename) as f:
        for line in f: 
            line_list =re.split(r',+', line)
            line_list =[float(el) for el in line_list[:-1]]
            docsXtopics_list.append(line_list)
    return docsXtopics_list



