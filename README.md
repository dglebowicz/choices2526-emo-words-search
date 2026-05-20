## Finding emotionally charged words

This is a repository for analysis of portfolios submitted for the course 'Choices in Data Visualisation 2025/2026' conducted by Florian Dehmelt at the University of Tuebingen. The aim is to find and quantify emotional expressions in the submissions as well as create a tool providing context for each finding.

## Getting started

The project was created with help of the uv manager. For the easiest start, install the uv manager and run uv sync in commandline in the downloaded project directory.

The script main.py includes a simple loop for finding and manually validating found emotional words in files from a given folder. It saves the final results to a csv file.

## Module structure

The classes for searching for emotional words are included in the emo_words_search module. The structure is as follows:

EmoWordsBase - provides a tool for creating a dictionary of emotional words to search for (emotion: list of words). It creates an initial pre-defined dictionary on initialization and includes methods for modifying the dictionary with user input.

EmoWordsFinder - inherits from EmoWordsBase and provides methods for searching for emotional words: 
- by stem (main method, most suitable with the pre-initialized dictionary: stems the dictionary and the sentences provided, highest likelihood of false positives)
- by lemma (lemmatizes the dictionary and sentences provided, may require modifications to the dictionary)
- by exact match (requires modifications to the dictionary to ensure that all forms of words of interests are included)

