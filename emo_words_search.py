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
    def __init__(self):
        self.happy_words = ['happy','joyful','content','pleased','delighted','cheerful','ecstatic','elated']
        self.struggling_words = ['struggle','difficulty','challenge','hardship','obstacle','tribulation','ordeal','setback']
        self.surprised_words = ['surprise','astonishment','amazement','wonder','shock','disbelief','stunned','startled','flabbergasted']
        self.annoyed_words = ['annoyed','irritated','frustrated','displeased','vexed','aggravated','exasperated','discontented']
        self.unhappy_words = ['unhappy','sad','depressed','miserable','downcast','gloomy','melancholy','sorrowful','heartbroken','despondent']
        self.interested_words = ['interested','curious','fascinated','engaged','intrigued','captivated','enthusiastic', 'eager']
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
        return self.dict_of_emotional_words.get(emotion, [])
    
    def add_emotional_words(self, emotion, words):
        if emotion in self.dict_of_emotional_words:
            self.dict_of_emotional_words[emotion].extend(words)
        else:
            self.dict_of_emotional_words[emotion] = words
    
    def remove_emotional_word(self, emotion, word):
        if emotion in self.dict_of_emotional_words:
            self.dict_of_emotional_words[emotion] = [w for w in self.dict_of_emotional_words[emotion] if w != word]
    
    def stem_emotional_words(self, stemmer = nltk.stem.SnowballStemmer('english')):
        self.stemmer = stemmer
        self.dict_of_emotional_stems = {}
        for key in self.dict_of_emotional_words.keys():
            self.dict_of_emotional_stems[key] = [self.stemmer.stem(word) for word in self.dict_of_emotional_words[key]]
        
    
    def lemmatize_emotional_words(self, lemmatizer = nltk.stem.WordNetLemmatizer()):
        self.lemmatizer = lemmatizer
        self.dict_of_emotional_lemmas = {}
        for key in self.dict_of_emotional_words.keys():
            self.dict_of_emotional_lemmas[key] = [self.lemmatizer.lemmatize(word) for word in self.dict_of_emotional_words[key]]
        


class EmoWordsFinder(EmoWordsBase):
    def __init__(self,sentences=None, generate_test=True, loadpath=None):
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
        full_text=''
        with open(loadpath, "rb") as pdf_file:
            read_pdf = PyPDF2.PdfReader(pdf_file)
            for page in read_pdf.pages:
                full_text += page.extract_text()
        sentences = full_text.split('.')
        return sentences

    def load_txt(self, loadpath):
        with open(loadpath, 'r', encoding='utf-8') as txt_file:
            full_text = txt_file.read()
        sentences = full_text.split('.')
        return sentences

    
    def find_emotional_words_by_stem(self, method = 'AI'):
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

                


        
        
