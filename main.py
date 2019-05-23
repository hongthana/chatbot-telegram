#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:39:49 2019

@author: thanawatraibroycharoen
"""
# Import Libraries
import gensim
import json
import numpy as np
import pickle
import re
import requests
import time
from nltk.corpus import stopwords
from requests.compat import urljoin
from sklearn.metrics.pairwise import pairwise_distances_argmin

class BotHandler(object):
    """
    BotHandler is a class which implements all backend of the bot.
    It has three main functions:
        'get_updates' - checks for new messages
        'send_message' - posts new message to user
        'get_answer' - computes the most relevant on a user's question
    """
    
    def __init__(self, token, dialogue_manager):
        self.token = token # Put the Telegram Access Token here
        self.api_url = "https://api.telegram.org/bot{}/".format(token)
        self.dialogue_manager = dialogue_manager
        
    def get_updates(self, offset = None, timeout = 30):
        params = {"timeout": timeout, "offset": offset}
        # Get latest user's message in form of JSON
        raw_resp = requests.get(urljoin(self.api_url, "getUpdates"), params)
        try:
            resp = raw_resp.json() # Convert JSON into dict
        except json.decoder.JSONDecodeError as e:
            print('Failed to parse response {}: {}.'.format(raw_resp.content, e))
            return []
        # if key "result" doesn't exist create one
        if "result" not in resp:
            return []
        return resp["result"] # "result" is a list variable
    
    def send_message(self, chat_id, text):
        params = {"chat_id": chat_id, "text": text}
        # Reply message to user
        return requests.post(urljoin(self.api_url, "sendMessage"), params)
    
    def get_answer(self, question):
        if question == '/start':
            return "Hi, I am your project bot. How can I help you today?"
        # Check whether "question" is "dialogue" or "post" & get the answer
        return self.dialogue_manager.generate_answer(question)
    
def is_unicode(text):
    return len(text) == len(text.encode())

# Prepare text at the prediction time
def text_prepare(text):
    """
    Message preprocessing and tokenization
    """
    replace_by_space = re.compile('[/()P{}\[\]\|@,;]') # Compile a regular expression patern
    bad_symbol = re.compile('[^0-9a-z #+_]')           # into a regular expression object.
    stopwords_set = set(stopwords.words('english')) # set english stopwords
    text = text.lower() # Convert to lower case
    text = replace_by_space.sub(' ', text) # replace symbols with space
    text = bad_symbol.sub(' ', text) # replace a single character & symbols with space
    # Delete stopwords
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])
    
    return text.strip() # return "text" with both leading & trailing char stripped

# Convert questions asked by user to vectors
def question_to_vec(question, embeddings, dim = 300):
    """
    Convert a string "question" into a vector:
        question: a string
        embeddings: dict where the key is a word and a value is its embedding
        dim: size of the representation
    
    result: vector representation for the question
    """
    word_tokens = question.split(" ") # split word with space
    question_len = len(word_tokens) # Count number of splitted words
    # Create numpy array (float 32) dimension of (question_len x dim)
    question_mat = np.zeros((question_len, dim), dtype = np.float32)
    
    # use enumerate to loop over "word_tokens" and return index (idx) and word
    for idx, word in enumerate(word_tokens):
        if word in embeddings:
            # Assign embeddings to question_mat with corresponding index
            question_mat[idx, :] = embeddings[word]
            
    # Remove zero-rows which stand for Out-Of_Vocabulary (OOV) words
    question_mat = question_mat[~np.all(question_mat == 0, axis = 1)]
    
    # Compute the mean of each word along the sentence
    if question_mat.shape[0] > 0:
        vec = np.array(np.mean(question_mat, axis = 0), dtype = np.float32).reshape((1, dim))
    else:
        vec = np.zeros((1, dim), dtype = np.float32)
    return vec

class SimpleDialogueManager(object):
    """
    A simple dialogue manager for testing the telegram bot.
    """
    def __init__(self):
        # Import Libraries
        from chatterbot import ChatBot
        from chatterbot.trainers import ChatterBotCorpusTrainer
        
        print('Loading resoures...')
        chatbot = ChatBot('AjarnBot') # Bot name 'Linkedin99bot'
        # Allows the chat bot to be trained using data from ChatterBot dialog corpus
        trainer = ChatterBotCorpusTrainer(chatbot)
        trainer.train('chatterbot.corpus.english') # Train chat bot with english corpus
        self.chitchat_bot = chatbot
        print('Loading Word2Vec model...')
        # Load Google's pre-trained Word2Vec model
        self.model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)
        print('Loading Classifier objects...')
        # Load pre-trained Intent Classifier model
        self.intent_recognizer = pickle.load(open('resources/intent_clf.pkl', 'rb'))
        # Load pre-trained Tag Classifier model
        self.tag_classifier = pickle.load(open('resources/tag_clf.pkl', 'rb'))
        # Load pre-trained TF-IDF model
        self.tfidf_vectorizer = pickle.load(open('resources/tfidf.pkl', 'rb'))
        print('Finished Loading Resources')
        
    def get_similar_question(self, question, tag):
        """
        Find similar question using tag:
            question: a string of question
            tag: a tag
            
            return: post_ids with min distance
        """
        # Specify pre-classified post tag
        embeddings_path = 'resources/embeddings_folder/' + tag + ".pkl"
        # Load pre-classified post tag assign post-id and embedding
        post_ids, post_embeddings = pickle.load(open(embeddings_path, 'rb'))
        # Convert question to vector
        question_vec = question_to_vec(question, self.model, 300)
        # Compute min distances between question_vec and post_embeddings
        best_post_index = pairwise_distances_argmin(question_vec, post_embeddings)
        
        return post_ids[best_post_index]
    
    def generate_answer(self, question):
        """
        Send answer to Telegram:
            question: a string of question
            
            return: response
        """
        # word preprocessing and tokenizer
        prepared_question = text_prepare(question)
        # Transform using pre-train TF-IDF model
        features = self.tfidf_vectorizer.transform([prepared_question])
        # Predict intent using pre-train Intent model: "Dialogue" or "Post"
        intent = self.intent_recognizer.predict(features)[0]
        # Dialogue
        if intent == 'dialogue':
            response = self.chitchat_bot.get_response(question)
        # Post
        else:
            # Predict tag of the question using pre-train Tag model
            tag = self.tag_classifier.predict(features)[0]
            # Find most similar question post_id
            post_id = self.get_similar_question(question, tag)[0]
            # Send response to Telegram
            response = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s' % (tag, post_id)
        return response
        
def main():
    token = '835876331:AAGACjUbJ2_dZvK-bpNOdRK5BxDCK7vT020' # AjarnBot
    token = 'YOUR:TOKEN'
    # Call SimpleDialogueManager
    simple_manager = SimpleDialogueManager()
    # Call BotHandler
    bot = BotHandler(token, simple_manager)
    
    print('++++ LET TALK! ++++')
    offset = 0
    while True:
        updates = bot.get_updates(offset = offset)
        for update in updates:
            chat_id = update["message"]["chat"]["id"]
            if "text" in update["message"]:
                text = update["message"]["text"]
                if is_unicode(text):
                    print('Update content: {}'.format(update))
                    bot.send_message(chat_id, bot.get_answer(update["message"]["text"]))
                else:
                    bot.send_message(chat_id, "Hmm, you are sending some weird characters to me...")
            offset = max(offset, update['update_id'] + 1)
        time.sleep(1)
        
if __name__ == "__main__":
    main()
