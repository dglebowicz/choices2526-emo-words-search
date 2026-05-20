import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from google import genai
import PyPDF2
from time import sleep
from abc import ABC, abstractmethod

nltk.download('wordnet')


class EmoWordsBase(ABC):
    '''Base class for emotional words management. Provides methods to get, add, remove, stem, and lemmatize emotional words.'''
    def __init__(self):
        '''Initializes the EmoWordsBase class with predefined lists of emotional words for different emotions and a dictionary to store them.'''
        self.happy_words = ['happy','joyful','content','pleased','delighted','cheerful','ecstatic','elated']
        self.struggling_words = ['struggle','difficulty','challenge','hardship','obstacle','tribulation','ordeal','setback']
        self.surprised_words = ['surprise','astonishment','amazement','wonder','shock','disbelief','stunned','startled','flabbergasted']
        self.annoyed_words = ['annoyed','irritated','frustrated','displeased','vexed','aggravated','exasperated','discontented']
        self.unhappy_words = ['unhappy','sad','depressed','miserable','downcast','gloomy','melancholy','sorrowful','heartbroken','despondent']
        self.interested_words = ['interested','curious','fascinated','engaged','intrigued','captivated','enthusiastic']
        self.appreciative_words = ['appreciative','grateful','thankful','admired','respected','valued']
        self.nervous_words = ['nervous','anxious','uneasy','apprehensive','worried','tense','restless']
        
        self.dict_of_emotional_words = {'happiness': self.happy_words,
                                'struggle': self.struggling_words,
                                'surprise': self.surprised_words,
                                'annoyance': self.annoyed_words,
                                'unhappiness': self.unhappy_words,
                                'interest': self.interested_words,
                                'appreciation': self.appreciative_words,
                                'nervousness': self.nervous_words}
        
    
    def get_emotional_words(self, emotion):
        '''Returns the list of emotional words for a given emotion. If the emotion is not found, returns an empty list.
        Args:
            emotion (str): The emotion for which to retrieve the emotional words.
        Returns:
            list: The list of emotional words for the given emotion.'''
        return self.dict_of_emotional_words.get(emotion, [])
    
    def add_emotional_words(self, emotion, words):
        '''Adds new emotional words to the list of emotional words for a given emotion. If the emotion does not exist, it creates a new entry for it.
        Args:
            emotion (str): The emotion for which to add emotional words.
            words (list): The list of emotional words to add.
        Returns:
            None: the method modifies the existing attribute dict_of_emotional_words.'''
        if emotion in self.dict_of_emotional_words:
            self.dict_of_emotional_words[emotion].extend(words)
        else:
            self.dict_of_emotional_words[emotion] = words
    
    def remove_emotional_word(self, emotion, word):
        '''Removes an emotional word from the list of emotional words for a given emotion. If the emotion or the word is not found, it does nothing.
        Args:
            emotion (str): The emotion for which to remove the emotional word.
            word (str): The emotional word to remove.
        Returns:
            None: the method modifies the existing attribute dict_of_emotional_words.'''
        if emotion in self.dict_of_emotional_words:
            self.dict_of_emotional_words[emotion] = [w for w in self.dict_of_emotional_words[emotion] if w != word]
    
    def stem_emotional_words(self, stemmer = nltk.stem.SnowballStemmer('english')):
        '''Stems the emotional words using the provided stemmer. If no stemmer is provided, it uses the SnowballStemmer for English from NLTK.
        Args:
            stemmer (nltk.stem.SnowballStemmer): The stemmer to use for stemming the emotional words.
        Returns:
            None: the method creates the attribute dict_of_emotional_stems.'''
        self.stemmer = stemmer
        self.dict_of_emotional_stems = {}
        for key in self.dict_of_emotional_words.keys():
            self.dict_of_emotional_stems[key] = [self.stemmer.stem(word) for word in self.dict_of_emotional_words[key]]
        
    
    def lemmatize_emotional_words(self, lemmatizer = nltk.stem.WordNetLemmatizer()):
        '''Lemmatizes the emotional words using the provided lemmatizer. If no lemmatizer is provided, it uses the WordNetLemmatizer from NLTK.
        Args:
            lemmatizer (nltk.stem.WordNetLemmatizer): The lemmatizer to use for lemmatizing the emotional words.
        Returns:
            None: the method creates the attribute dict_of_emotional_lemmas.'''
        self.lemmatizer = lemmatizer
        self.dict_of_emotional_lemmas = {}
        for key in self.dict_of_emotional_words.keys():
            self.dict_of_emotional_lemmas[key] = [self.lemmatizer.lemmatize(word) for word in self.dict_of_emotional_words[key]]
        


class EmoWordsFinder(EmoWordsBase):
    '''Class for finding emotional words in sentences. Inherits from EmoWordsBase and provides methods to find emotional words by stem, exact match, and lemma.'''
    def __init__(self, generate_test, loadpath,sentences=None):
        '''Initializes the EmoWordsFinder class. If sentences are provided, it uses them. If generate_test is True, it generates a test portfolio using the generate_test_portfolio method. If loadpath is provided, it loads the sentences from the specified file (PDF or TXT).
        Args:
            generate_test (bool): Whether to generate a test portfolio.
            loadpath (str): The path to the file containing sentences (PDF or TXT).
            sentences (list): The list of sentences to analyze. Default: None.
        Returns:
            None: the method initializes the EmoWordsFinder instance.'''
        super().__init__()
        if sentences is None:
            if generate_test:
                self.sentences = self.generate_test_portfolio()
            else:
                if loadpath.endswith('.pdf'):
                    self.sentences = self.load_pdf(loadpath)
                elif loadpath.endswith('.txt'):
                    self.sentences = self.load_txt(loadpath)
        else:
            self.sentences = sentences
    
    def load_pdf(self,loadpath):
        '''Loads sentences from a PDF file. It reads the PDF file, extracts the text from each page, and splits it into sentences based on periods.
        Args:   
            loadpath (str): The path to the PDF file to load.
        Returns:
            list: The list of sentences extracted from the PDF file.'''
        full_text=''
        with open(loadpath, "rb") as pdf_file:
            read_pdf = PyPDF2.PdfReader(pdf_file)
            for page in read_pdf.pages:
                full_text += page.extract_text()
        sentences = full_text.split('.')
        return sentences

    def load_txt(self, loadpath):
        '''Loads sentences from a TXT file. It reads the TXT file, extracts the text, and splits it into sentences based on periods.
        Args:
            loadpath (str): The path to the TXT file to load.
        Returns:
            list: The list of sentences extracted from the TXT file.'''
        with open(loadpath, 'r', encoding='utf-8') as txt_file:
            full_text = txt_file.read()
        sentences = full_text.split('.')
        return sentences

    
    def find_emotional_words_by_stem(self, method = 'AI'):
        '''Finds emotional words in the sentences by stemming the words and comparing them to the stemmed emotional words in the dictionary. It can use either AI or manual validation to determine if a word is used emotionally in a sentence.
        Args:
            method (str): The method to use for validation ('AI' or 'manual'). Default: 'AI': uses the Gemini 3.1 Flash Lite model to validate if the word is used emotionally in the sentence. 'manual': prompts the user to validate if the word is used emotionally in the sentence.
        Returns:
            None: The method updates the search_results_df attribute with the findings. This DataFrame contains the columns: 'Emotion', 'Word', 'Sentence', 'Sentence_id', 'Search_method', and 'Validation_method'.'''
        if hasattr(self, 'stemmer') is False:
            self.stem_emotional_words()
        
        if hasattr(self, 'search_results_df') is False:
            self.search_results_df = pd.DataFrame(columns=['Emotion','Word','Sentence','Sentence_id','Search_method','Validation_method'])
        
        if method=='AI':
            self.client = genai.Client()
            self.sentences_found_stem_ai={}
            self.words_found_stem_ai = {}
            for key in self.dict_of_emotional_stems.keys():
                self.sentences_found_stem_ai[key] = []
                self.words_found_stem_ai[key] = []
                
            for j,sentence in enumerate(self.sentences):
                sentence=sentence.replace('\n', '')
                for word in sentence.split():
                    for key in self.dict_of_emotional_stems.keys():
                        
                        if self.stemmer.stem(word.strip()) in self.dict_of_emotional_stems[key]:
                            try:
                                response = self.client.models.generate_content(
                                    model="gemini-3.1-flash-lite", 
                                    contents=f"If the following sentence uses the word {word} to express emotions of the {key} type, then write the word emotional, else write the word nonemotional. Sentence: {sentence}"
                                )
                            except:
                                sleep(10)
                                response = self.client.models.generate_content(
                                    model="gemini-3.1-flash-lite", 
                                    contents=f"If the following sentence uses the word {word} to express emotions of the {key} type, then write the word emotional, else write the word nonemotional. Sentence: {sentence}"
                                )
                            if response.text == 'emotional':
                                self.sentences_found_stem_ai[key].append(sentence)
                                self.words_found_stem_ai[key].append(word)
                                self.search_results_df = pd.concat([
                                    self.search_results_df,
                                    pd.DataFrame({'Emotion': key, 'Word': word.strip(), 'Sentence': sentence, 'Sentence_id': j, 'Search_method': 'stem', 'Validation_method': 'AI'}, index=[0])
                                ], ignore_index=True)
        
        elif method=='manual':
            self.sentences_found_stem_manual={}
            self.words_found_stem_manual={}
            for key in self.dict_of_emotional_stems.keys():
                self.sentences_found_stem_manual[key] = []
                self.words_found_stem_manual[key] = []
            
            for j,sentence in enumerate(self.sentences):
                sentence=sentence.replace('\n', '')
                words = sentence.split()
                i=0

                while i<len(words):
                    word = words[i]
                    found_match = False
                    for key in self.dict_of_emotional_stems.keys():
                        
                        if self.stemmer.stem(word.strip()) in self.dict_of_emotional_stems[key]:
                            found_match = True
                            print(f"Emotional word found:\n emotion: {key}, word: {word}\n in: {sentence}")
                            user_input = input("Is this an emotional word? (y/n): ")
                            
                            if user_input.lower() == 'y':
                                self.sentences_found_stem_manual[key].append(sentence)
                                self.words_found_stem_manual[key].append(word)
                                i+=1
                                self.search_results_df = pd.concat([
                                    self.search_results_df,
                                    pd.DataFrame({'Emotion': key, 'Word': word.strip(), 'Sentence': sentence, 'Sentence_id': j, 'Search_method': 'stem', 'Validation_method': 'manual'}, index=[0])
                                ], ignore_index=True)
                                break
                            elif user_input.lower() == 'n':
                                i+=1
                                break
                            else:
                                print("Invalid input. Please enter 'y' or 'n'.")
                                break
                    if not found_match:
                        i+=1
        
        
    def find_emotional_words_by_exact_match(self, method = 'AI'):
        '''Finds emotional words in the sentences by exact match of the words to the emotional words in the dictionary. It can use either AI or manual validation to determine if a word is used emotionally in a sentence. When using this method, it's recommended to first update the dictionary_of_emotional_words attribute with the add_emotional_words method to include different forms of the emotional words of interest (e.g., struggle, struggles, struggling) to increase the chances of finding emotional words in the sentences.
        Args:
            method (str): The method to use for validation ('AI' or 'manual'). Default: 'AI': uses the Gemini 3.1 Flash Lite model to validate if the word is used emotionally in the sentence. 'manual': prompts the user to validate if the word is used emotionally in the sentence.
        Returns:
            None: The method updates the search_results_df attribute with the findings. This DataFrame contains the columns: 'Emotion', 'Word', 'Sentence', 'Sentence_id', 'Search_method', and 'Validation_method'.'''
        if hasattr(self, 'search_results_df') is False:
            self.search_results_df = pd.DataFrame(columns=['Emotion','Word','Sentence','Sentence_id','Search_method','Validation_method'])
        
        if method=='AI':
            self.client = genai.Client()
            self.sentences_found_exact_ai={}
            self.words_found_exact_ai = {}
            for key in self.dict_of_emotional_words.keys():
                self.sentences_found_exact_ai[key] = []
                self.words_found_exact_ai[key] = []
                
            for j,sentence in enumerate(self.sentences):
                sentence=sentence.replace('\n', '')
                for word in sentence.split():
                    for key in self.dict_of_emotional_words.keys():
                        
                        if word.strip().lower() in [w.lower() for w in self.dict_of_emotional_words[key]]:
                            try:
                                response = self.client.models.generate_content(
                                    model="gemini-3.1-flash-lite", 
                                    contents=f"If the following sentence uses the word {word} to express emotions of the {key} type, then write the word emotional, else write the word nonemotional. Sentence: {sentence}"
                                )
                            except:
                                sleep(10)
                                response = self.client.models.generate_content(
                                    model="gemini-3.1-flash-lite", 
                                    contents=f"If the following sentence uses the word {word} to express emotions of the {key} type, then write the word emotional, else write the word nonemotional. Sentence: {sentence}"
                                )
                            if response.text == 'emotional':
                                self.sentences_found_exact_ai[key].append(sentence)
                                self.words_found_exact_ai[key].append(word)
                                self.search_results_df = pd.concat([
                                    self.search_results_df,
                                    pd.DataFrame({'Emotion': key, 'Word': word.strip(), 'Sentence': sentence, 'Sentence_id': j, 'Search_method': 'exact', 'Validation_method': 'AI'}, index=[0])
                                ], ignore_index=True)
        
        elif method=='manual':
            self.sentences_found_exact_manual={}
            self.words_found_exact_manual={}
            for key in self.dict_of_emotional_words.keys():
                self.sentences_found_exact_manual[key] = []
                self.words_found_exact_manual[key] = []
            
            for j,sentence in enumerate(self.sentences):
                sentence=sentence.replace('\n', '')
                words = sentence.split()
                i=0

                while i<len(words):
                    word = words[i]
                    found_match = False
                    for key in self.dict_of_emotional_words.keys():
                        
                        if word.strip().lower() in [w.lower() for w in self.dict_of_emotional_words[key]]:
                            found_match = True
                            print(f"Emotional word found:\n emotion: {key}, word: {word}\n in: {sentence}")
                            user_input = input("Is this an emotional word? (y/n): ")
                            
                            if user_input.lower() == 'y':
                                self.sentences_found_exact_manual[key].append(sentence)
                                self.words_found_exact_manual[key].append(word)
                                i+=1
                                self.search_results_df = pd.concat([
                                    self.search_results_df,
                                    pd.DataFrame({'Emotion': key, 'Word': word.strip(), 'Sentence': sentence, 'Sentence_id': j, 'Search_method': 'exact', 'Validation_method': 'manual'}, index=[0])
                                ], ignore_index=True)
                                break
                            elif user_input.lower() == 'n':
                                i+=1
                                break
                            else:
                                print("Invalid input. Please enter 'y' or 'n'.")
                                break
                    if not found_match:
                        i+=1
        
        
            

    def find_emotional_words_by_lemma(self, method = 'AI'):
        '''Finds emotional words in the sentences by lemmatizing the words and comparing them to the lemmatized emotional words in the dictionary. It can use either AI or manual validation to determine if a word is used emotionally in a sentence. When using this method, it's recommended to first update the dictionary_of_emotional_words attribute with the add_emotional_words method, as current dictionary may not be sufficient for lemmatized search.
        Args:
            method (str): The method to use for validation ('AI' or 'manual'). Default: 'AI': uses the Gemini 3.1 Flash Lite model to validate if the word is used emotionally in the sentence. 'manual': prompts the user to validate if the word is used emotionally in the sentence.
        Returns:
            None: The method updates the search_results_df attribute with the findings. This DataFrame contains the columns: 'Emotion', 'Word', 'Sentence', 'Sentence_id', 'Search_method', and 'Validation_method'.
        '''
        if hasattr(self, 'lemmatizer') is False:
            self.lemmatize_emotional_words()
        if hasattr(self, 'search_results_df') is False:
            self.search_results_df = pd.DataFrame(columns=['Emotion','Word','Sentence','Sentence_id','Search_method','Validation_method'])
        
        if method=='AI':
            self.client = genai.Client()
            self.sentences_found_lemma_ai={}
            self.words_found_lemma_ai = {}
            for key in self.dict_of_emotional_lemmas.keys():
                self.sentences_found_lemma_ai[key] = []
                self.words_found_lemma_ai[key] = []
                
            for j,sentence in enumerate(self.sentences):
                sentence=sentence.replace('\n', '')
                for word in sentence.split():
                    for key in self.dict_of_emotional_lemmas.keys():
                        
                        if self.lemmatizer.lemmatize(word.strip()) in self.dict_of_emotional_lemmas[key]:
                            try:
                                response = self.client.models.generate_content(
                                    model="gemini-3.1-flash-lite", 
                                    contents=f"If the following sentence uses the word {word} to express emotions of the {key} type, then write the word emotional, else write the word nonemotional. Sentence: {sentence}"
                                )
                            except:
                                sleep(10)
                                response = self.client.models.generate_content(
                                    model="gemini-3.1-flash-lite", 
                                    contents=f"If the following sentence uses the word {word} to express emotions of the {key} type, then write the word emotional, else write the word nonemotional. Sentence: {sentence}"
                                )
                            if response.text == 'emotional':
                                self.sentences_found_lemma_ai[key].append(sentence)
                                self.words_found_lemma_ai[key].append(word)
                                self.search_results_df = pd.concat([
                                    self.search_results_df,
                                    pd.DataFrame({'Emotion': key, 'Word': word.strip(), 'Sentence': sentence, 'Sentence_id': j, 'Search_method': 'lemma', 'Validation_method': 'AI'}, index=[0])
                                ], ignore_index=True)
            
        
        elif method=='manual':
            self.sentences_found_lemma_manual={}
            self.words_found_lemma_manual={}
            for key in self.dict_of_emotional_lemmas.keys():
                self.sentences_found_lemma_manual[key] = []
                self.words_found_lemma_manual[key] = []
            for j,sentence in enumerate(self.sentences):
                sentence=sentence.replace('\n', '')
                words = sentence.split()
                i=0

                while i<len(words):
                    word = words[i]
                    found_match = False
                    for key in self.dict_of_emotional_lemmas.keys():
                        
                        if self.lemmatizer.lemmatize(word.strip()) in self.dict_of_emotional_lemmas[key]:
                            found_match = True
                            print(f"Emotional word found:\n emotion: {key}, word: {word}\n in: {sentence}")
                            user_input = input("Is this an emotional word? (y/n): ")
                            
                            if user_input.lower() == 'y':
                                self.sentences_found_lemma_manual[key].append(sentence)
                                self.words_found_lemma_manual[key].append(word)
                                self.search_results_df = pd.concat([self.search_results_df, pd.DataFrame({'Emotion': key, 'Word': word.strip(), 'Sentence': sentence, 'Sentence_id': j, 'Search_method': 'lemma', 'Validation_method': 'manual'}, index=[0])], ignore_index=True)
                                i+=1
                                break
                            elif user_input.lower() == 'n':
                                i+=1
                                break
                            else:
                                print("Invalid input. Please enter 'y' or 'n'.")
                                break
                    if not found_match:
                        i+=1
            
                                
    
        

                           
    def generate_test_portfolio(self, n_sentences=1):
        '''Generates a test portfolio with sentences that express different emotions using the emotional words from the dictionary. For each emotion, it generates n_sentences sentences that express the emotion and n_sentences sentences that use the emotional words but do not express the emotion. It uses the Gemini 3.1 Flash Lite model to generate the sentences.
        Args:         
            n_sentences (int): The number of sentences to generate for each emotion. Default: 1.
        Returns:
            list: The list of generated sentences for the test portfolio.'''
        self.client = genai.Client()
        self.test_portfolio = []
        for emotion in self.dict_of_emotional_words.keys():
            for _ in range(n_sentences):
                response = self.client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=f"Generate one sentence that expresses the emotion of {emotion} using one of the following words: {self.dict_of_emotional_words[emotion]} in the context of a portfolio for a course on data visualisations. The word can be used in any form as long as it expresses the emotion of the student in the context of the course. For example for the emotion of excitement: I was excited about being able to explore visualisations for this dataset in a more relaxed environment. Respond with only the sentence, without any additional text."
                )
                
                
                self.test_portfolio.append(response.text)
                sleep(2)
            
            response = self.client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=f"Generate one sentence that uses one of the following words: {self.dict_of_emotional_words[emotion]} in the context of a portfolio for a course on data visualisations but does not use the word to express an emotion. For example: The values on the X-axis represent the hormone levels. Respond with only the sentence, without any additional text. If you cannot generate such a sentence, respond with the word 'impossible'."
                )
            sleep(2)
            if response.text!='impossible':
                self.test_portfolio.append(response.text)
        return self.test_portfolio

                


        
        
