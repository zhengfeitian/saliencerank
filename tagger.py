#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
POS tagger.
Including code from NLTK, which is licensed under the Apache License Version 2.0.
'''

import nltk

"""Finds NP (nounphrase) leaf nodes of a chunk tree."""
def leaves(tree):
    for subtree in tree.subtrees(filter = lambda t: t.node=='NP'):
        yield subtree.leaves()
        
"""Normalises words to lowercase and stems and lemmatizes it."""
def normalise(word):
    word = word.lower()
    return word

"""Checks conditions for acceptable word: length, stopword."""
def acceptable_word(word, stopwords):
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stopwords)
    return accepted

"""Get terms given a POS tree and stopwords"""
def get_terms(tree, stopwords):
    all_terms = []
    for leaf in leaves(tree):
        term = [ normalise(w) for w,t in leaf if acceptable_word(w, stopwords) ]
        if term:  
            if term not in all_terms:
                all_terms.append(term)
    return all_terms 

"""Return the words (as a list) satisfying the noun/adjective-noun regular expression"""
def tag_phrases (text):
    from nltk.corpus import stopwords
    text= text.replace ("-", " ")
    # Used when tokenizing words
    sentence_re = r'''(?x)      # set flag to allow verbose regexps
          ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
        | \w+(-\w+)*            # words with optional internal hyphens
        | \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
        | \.\.\.                # ellipsis
        | [][.,;"'?():-_`]      # these are separate tokens
    '''
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>} # Nouns and Adjectives, terminated with Nouns
            
        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    chunker = nltk.RegexpParser(grammar)
    toks = nltk.regexp_tokenize(text, sentence_re)
    postoks = nltk.tag.pos_tag(toks)
   
    tree = chunker.parse(postoks)

    stopwords = stopwords.words('english')
    return get_terms(tree, stopwords)
