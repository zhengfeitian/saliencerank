#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Code containing the implementation of the three ranking methods.
'''
from __future__ import division
from utils import parse_weights_from_file,load_docsXtopics_from_file, set_graph_edges
import numpy as np 
from numpy import linalg
from tagger import tag_phrases 
import networkx, nltk
from nltk.tokenize import RegexpTokenizer

"""TextRank algorithm : no heuristic selection of candidates on top of POS tagging. 
	Ref: Mihalcea and Tarau. 2004. Textrank: Bringing order into texts.""" 
def textrank(text):
    # tokenize all words; remove stop words 
    words = []
    stop_words = nltk.corpus.stopwords.words('english') 
    stop_words = set(stop_words)
    
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    words_nostopwords = []
    for i in xrange(len(words)): 
        words[i] = words[i].lower()
        if words[i] not in stop_words: 
            words_nostopwords.append(words[i])
    words = words_nostopwords

    graph = networkx.Graph()
    graph.add_nodes_from(set(words))
    set_graph_edges(graph, words, words)
    
    # score nodes using default pagerank algorithm
    ranks = networkx.pagerank(graph)
    tagged_phrases = tag_phrases (text) # list of lists 

    tagged_phrases_scores = {}
    for p in tagged_phrases: 
        score = 0 
        for w in p: 
            if w in ranks: 
                score = score + ranks [w]
        tagged_phrases_scores [" ".join(p)]= score 
    
    if '' in tagged_phrases_scores: #remove empty character as a key 
        tagged_phrases_scores.pop('')
    sorted_phrases = sorted(tagged_phrases_scores.iteritems(), key=lambda x: x[1], reverse=True) 
    return sorted_phrases


"""Topical Pagerank (TPR) algorithm
	Ref: Liu et al. 2010. Automatic keyphrase extraction via topic decomposition."""
def tpr (topics, pt , text, file_ID):   
    # tokenize all words; remove stop words 
    words = []
    stop_words = nltk.corpus.stopwords.words('english') 
    stop_words = set(stop_words)
    
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    words_nostopwords = []
    for i in xrange(len(words)): 
        words[i] = words[i].lower()
        if words[i] not in stop_words: 
            words_nostopwords.append(words[i])
    words = words_nostopwords

    graph = networkx.Graph()
    graph.add_nodes_from(set(words))
    set_graph_edges(graph, words, words)
    
    tagged_phrases = tag_phrases (text) # list of lists
 
    #add personalization to pagerank 
    topics_nparray = np.ones((len(topics), len(topics[0])))*10e-10
    for t in xrange(len(topics)):
        count=0
        for el in sorted(topics[t]): 
            topics_nparray [t, count] = topics_nparray [t, count]+ topics[t][el]
            count = count + 1 
           
    row_sums = topics_nparray.sum(axis=1)
    phi = topics_nparray / row_sums[:, np.newaxis] #normalize row-wise: each topic(row) is a distribution
    topics_nparray = phi 
    
    col_sums = topics_nparray.sum(axis=0)
    p_tw = topics_nparray / col_sums [np.newaxis, :] #normalize column-wise: each word (col) is a distribution
    
    #run page rank for each topic 
    tagged_phrases_scores = {} # keyphrse: list of ranks for each topic
    for t in xrange(len(topics)): 
        #construct personalization vector and run PR 
        personalization = {}
        idx  = 0 
        for n,_ in graph.nodes_iter(data=True): 
            if n in sorted(topics[0]):
                if n in words:
                    personalization [n] = p_tw [t, idx]
                else: 
                    personalization [n] = 0 
            else: 
                personalization [n] = 0
            idx = idx + 1 
        ranks = networkx.pagerank(graph, 0.85, personalization)
        
        #accumulate ranks for candidate keyphrases for each topic
        for p in tagged_phrases: 
            whole_phrase = " ".join(p) 
            score = 0 
            for w in p: 
                if w in ranks: 
                    score = score + ranks [w]
            if whole_phrase in tagged_phrases_scores: 
                tagged_phrases_scores[whole_phrase].append (score) 
            else: 
                tagged_phrases_scores [whole_phrase]= [score]

    # final rank for each keyphrase: weigh candidate ranks by the document's topic distribution
    for p,v in tagged_phrases_scores.items(): 
        tagged_phrases_scores[p] = np.dot(np.array(v), pt[file_ID,:]/sum(pt[file_ID,:])) 
        
    sorted_phrases = sorted(tagged_phrases_scores.iteritems(), key=lambda x: x[1], reverse=True) 
    return sorted_phrases

"""Single Topical PageRank (SingleTPR) algorithm
	Ref: Sterckx et al. 2015. Topical word importance for fast keyphrase extraction. """
def singletpr (topics, pt , text, file_ID): 
    # tokenize all words; remove stop words 
    words = []
    stop_words = nltk.corpus.stopwords.words('english') 
    stop_words = set(stop_words)

    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    words_nostopwords = []
    for i in xrange(len(words)): 
        words[i] = words[i].lower()
        if words[i] not in stop_words: 
            words_nostopwords.append(words[i])
    words = words_nostopwords
    
    #set the graph edges
    graph = networkx.Graph()
    graph.add_nodes_from(set(words))
    set_graph_edges(graph, words, words)

    #add personalization to pagerank 
    topics_nparray = np.ones((len(topics), len(topics[0])))*10e-10
    for t in xrange(len(topics)):
        count=0
        for el in sorted(topics[t]): 
            topics_nparray [t, count] = topics_nparray [t, count]+ topics[t][el]
            count = count + 1 
    row_sums = topics_nparray.sum(axis=1)
    phi = topics_nparray / row_sums[:, np.newaxis] #normalize row-wise: each topic(row) is a distribution
    topics_nparray = phi # #topics x #words 

    pt_new_dim = pt[file_ID,:]/sum(pt[file_ID,:]) # topic distribution for one doc
    pt_new_dim = pt_new_dim[None, :] 
    weights = np.dot (phi.T, pt_new_dim.T) 
    weights = weights/linalg.norm(pt_new_dim, 'fro') # cos similarity normalization 

    personalization = {}
    count = 0 
    for n,_ in graph.nodes_iter(data=True): 
        if n in sorted(topics[0]):
            if n in words: 
                personalization[n] = weights[count]/(linalg.norm(pt_new_dim, 'fro')*linalg.norm (phi[:, count])) # cos similarity normalization 
            else: 
                personalization[n] = 0 
        else: 
            personalization[n] = 0
        count = count + 1
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    factor=1.0/sum(personalization.itervalues()) #normalize the personalization vec
    for k in personalization:
       personalization[k] = personalization[k]*factor

    ranks = networkx.pagerank(graph, 0.85, personalization)
    tagged_phrases = tag_phrases (text) # list of lists 

    tagged_phrases_scores = {}
    for p in tagged_phrases: 
        score = 0 
        for w in p: 
            if w in ranks: 
                score = score + ranks [w]
                
        tagged_phrases_scores [" ".join(p)]= score  
    if '' in tagged_phrases_scores: #remove empty character as a key 
        tagged_phrases_scores.pop('')
    sorted_phrases = sorted(tagged_phrases_scores.iteritems(), key=lambda x: x[1], reverse=True) 
    return sorted_phrases

"""Salience Rank algorithm 
	Ref: Teneva and Cheng. 2017. Salience Rank: Efficient Keyphrase Extraction with Topic Modeling."""
def saliencerank (topics, pt , text, file_ID, alpha): 
    # tokenize all words; remove stop words 
    words = []
    stop_words = nltk.corpus.stopwords.words('english') 
    stop_words = set(stop_words)

    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    words_nostopwords = []
    for i in xrange(len(words)): 
        words[i] = words[i].lower()
        if words[i] not in stop_words: 
            words_nostopwords.append(words[i])
    words = words_nostopwords
    
    #set the graph edges
    graph = networkx.Graph()
    graph.add_nodes_from(set(words))
    set_graph_edges(graph, words, words)

    #add personalization to pagerank 
    topics_nparray = np.ones((len(topics), len(topics[0])))*10e-10
    for t in xrange(len(topics)):
        count=0
        for el in sorted(topics[t]): 
            topics_nparray [t, count] = topics_nparray [t, count]+ topics[t][el]
            count = count + 1 
    row_sums = topics_nparray.sum(axis=1)
    phi = topics_nparray / row_sums[:, np.newaxis] #normalize row-wise: each topic(row) is a distribution
    topics_nparray = phi 

    col_sums = topics_nparray.sum(axis=0)
    pw = col_sums / np.sum(col_sums)
    
    p_tw = topics_nparray / col_sums [np.newaxis, :] #normalize column-wise: each word (col) is a distribution
    pt_new_dim = pt[file_ID,:]/sum(pt[file_ID,:])
    pt_new_dim = pt_new_dim[None, :]
    p_tw_by_pt = np.divide (p_tw, pt_new_dim.T) #divide each column by the vector pt elementwise 
    kernel = np.multiply(p_tw, np.log (p_tw_by_pt))
    distinct = kernel.sum(axis=0) 
    distinct = (distinct - np.min(distinct))/(np.max(distinct) - np.min(distinct)) #normalize
    
    personalization = {}
    count = 0 
    for n,_ in graph.nodes_iter(data=True): 
        if n in sorted(topics[0]):
            if n in words: 
                personalization[n] =  (1.0-alpha)*sum(phi[:, count])+ alpha*distinct[count] 
            else: 
                personalization[n] = 0 
        else: 
            personalization[n] = 0
        count = count + 1
    
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph, 0.85, personalization)
    tagged_phrases = tag_phrases (text) # list of lists 

    tagged_phrases_scores = {}
    for p in tagged_phrases: 
        score = 0 
        for w in p: 
            if w in ranks: 
                score = score + ranks [w]
                
        tagged_phrases_scores [" ".join(p)]= score  
    if '' in tagged_phrases_scores: #remove empty character as a key 
        tagged_phrases_scores.pop('')
    sorted_phrases = sorted(tagged_phrases_scores.iteritems(), key=lambda x: x[1], reverse=True) 
    return sorted_phrases

