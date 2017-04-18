# Source Code of Salience Rank Keyphrase Extraction Algorithm
#### by Nedelina Teneva, April 15, 2017

### Reference
  - Nedelina Teneva and Weiwei Cheng. *Salience Rank: Efficient Keyphrase Extraction with Topic Modeling.* The 55th Annual Meeting of the Association for Computational Linguistics (ACL-17). Vancouver, Canada.

### Dependencies
  - nltk 2.0 
  - matplotlib 1.3
  - networkx 1.10
  - numpy 1.8

### Files 
  - runner.py: executes the main function  
  - ranks.py: implementation of various key phrase extraction algorithms
  - tagger.py: POS tagging infrastructure 
  - utils.py: various utilities functions 
  - process.py: infrastructure for dataset processing 

### Directories
  - data: contains the two standard datasets Inspec (Hulth. 2003. Improved automatic keyword extraction given more linguistic knowledge) and 500N (Marujo et al. 2013. Supervised topical key phrase extraction of news stories using crowdsourcing, light filtering and co-reference normalization). 
  - lda: The TPR and DR algorithms rely on two LDA output files (which can be obtained with any standard LDA implementation). 
    - Each line of lda-topicsXvocab*.txt contains the topic distribution over the vocabulary for each document (documents are sorted alphabetically by filename). 
    - Each line of lda-docxXtopics*.txt contains the proportion of each topic for each document (documents are sorted alphabetically by filename).
  - results: the results for the two datasets are output here after executing runner.py

### Running the code
  - select an algorithm by setting the algorithm variable on line 2 in runner.py   
  - run the code as follows: $ python runner.py 
