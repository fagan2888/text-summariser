from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

STOP_WORDS = {'re', 'further', 'but', 'don', 'as', 'if', 'be', 'being', 'than', 'only', 'wasn', 'of', 'all', 'such', 'that', 'her', 'no', 'both', 'any', 'themselves', 'been', "mustn't", 'were', 'those', "she's", 'during', "didn't", 'ourselves', "it's", 'same', 'y', "haven't", 'most', 'there', 'in', 'down', 'to', "weren't", 'shan', 'because', 'couldn', 'by', 'up', 'each', 'did', "mightn't", 'these', 'too', 'hasn', 'weren', 'below', 'hers', "aren't", 'nor', 'and', 'its', 'few', "couldn't", 'it', 'after', 'needn', 'which', 'the', 'whom', 'you', 'over', 'me', 'they', 'has', "you're", 'their', 'into', 'from', "hasn't", 'd', 'we', 'aren', 'mightn', 'ours', 'who', 'on', 'do', 'then', 'our', 'doesn', 'had', 'can', 'haven', 'once', 'about', 'ma', 'doing', 'a', "should've", 'am', 'now', 'own', 't', 'this', 'some', "isn't", 'myself', "that'll", 'mustn', 'through', 'or', 'him', 'off', 'how', 'he', 'so', "shouldn't", 'was', 'your', 'very', 'having', 'theirs', "you'll", 'does', 'other', 'shouldn', 'before', "needn't", 'his', 'won', 'just', 'when', "wouldn't", 'she', 'until', 'above', 'o', 'are', 'what', 'is', 'herself', 'yourself', 'my', "wasn't", "you've", 'again', 'hadn', "won't", "doesn't", 'between', 'itself', 'against', 'wouldn', 'while', 'i', 'ain', 'not', 'more', 's', 'should', 'll', "don't", 'under', 'with', 'himself', 'yourselves', 'at', 'out', 'an', 'm', 'will', 'where', 'yours', 'isn', "you'd", 'here', "shan't", 'have', 've', "hadn't", 'why', 'didn', 'them', 'for'}
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
    "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
    "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
    "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
    "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
    "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
    "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
   "you're": "you are", "you've": "you have"}

#with open('contractions.json') as json_file:
 #   contraction_mapping = json.load(json_file)

lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = re.sub(r"\n","",text).strip() 
    print(text)
    text = text.lower()
    words = text.strip().split(' ')
    words = filter(lambda t: t not in STOP_WORDS, words)
    cleaned_text = (' ').join([contraction_mapping[w] if w in contraction_mapping
                                                        else lemmatizer.lemmatize(w) for w in words])
    cleaned_text = re.sub(r"[^a-zA-Z]"," ",cleaned_text).strip()                                            
    return cleaned_text                                                


def generate_summary(full_text, n_components=2, info_ratio = 0.2, min_length=2, max_length=4, order_factor=1.05, min_sentence_len = 4):
   # full_text = full_text.replace('\n',' ')
  #  sentences = full_text.split('. ')
    sentences = re.split(r"\. |\.\n|\?|\?\n|\!|\!\n|\.\r|\r", full_text)
   # print('input text:')
   # for i in sentences:
   #     print(i)
   #     print('break')
   # print('***')
    for i in range(0,len(sentences)):
       # sentences[i] = re.sub(r"\n","",sentences[i]).strip() 
        sentences[i] = sentences[i].strip()
    data = pd.DataFrame(sentences, columns = ['sentence'])
    cleaned_sentences = []
    for i in range(0,data.shape[0]):
        cleaned_sentences.append(clean_text(data.loc[i,'sentence'].strip()))
    data['cleaned_sentence'] = cleaned_sentences
    vectorizer = TfidfVectorizer()
    sla_matrix = vectorizer.fit_transform(data['cleaned_sentence'])
#    print(data['cleaned_sentence'].head())

    svd = TruncatedSVD(n_components = n_components)
    VT = np.abs(svd.fit_transform(sla_matrix))
    dictionary = vectorizer.get_feature_names()
    sigma = svd.explained_variance_ratio_
    B = (VT*(sigma**2))
    B = (B.sum(axis=1))*0.5
    encoding_matrix=pd.DataFrame(svd.components_,columns=dictionary).T
    for i in range (0, B.shape[0]):
        B[i] = B[i]*(order_factor**i)
    B = B/sum(B)

    #print(VT)
    #print(encoding_matrix)
    #print(sigma)

    scores = enumerate(B)
    sorted_scores = sorted(scores, key = lambda x: x[1])
    summary_indices = []
    total_information = 0
    sum_count = 0
    info_ratio = 0.2
    for score in sorted_scores:
        if (sum_count < min_length) or ((sum_count < max_length) and (total_information < info_ratio)):
            words = data.loc[ score[0], 'sentence'].split(' ')
            if len(words) >= min_sentence_len:
                summary_indices.append(score[0])
                total_information += score[1]
                sum_count += 1
                print('total info ', total_information)

    summary_text = ''
    MAX_SENTENCES = 4
    count = 0
    for i in range(data.shape[0]):
        print(count)
        if count > MAX_SENTENCES: break
        if i in summary_indices:
            summary_text = summary_text + data.loc[i,'sentence'] + '.\n'
            count += 1
    
  #  print('SUMMMARY:')
    return summary_text[:-1]

def app_post(full_text):
    sentence_count = len(full_text.split('.'))
    word_count = len(full_text.split(' '))
    if sentence_count < 4 or word_count < 25:
        summarised_text = "You do not need a summary for that tiny text, try something longer ..."
    else:
        summarised_text = generate_summary(full_text, 
                                n_components=3, 
                                min_sentence_len = 3)    
    return summarised_text

def text_post(event, context):
    obj = event
    #full_text = obj['text']
    print('START OF PRINT ...')
    full_text = obj['queryStringParameters']['text']
    full_text = obj['body']
   # full_text = json.load(full_text)
    full_text = json.loads(full_text)
    full_text = full_text['text']
    sentence_count = len(full_text.split('.'))
    word_count = len(full_text.split(' '))
    if sentence_count < 4 or word_count < 25:
        summarised_text = "You do not need a summary for that..."
    else:
        summarised_text = generate_summary(full_text, 
                                n_components=3, 
                                min_sentence_len = 3)

    returned_obj = json.dumps({"summary": summarised_text})
    response = {
        "statusCode": 200,
        "headers": {},
        "body": returned_obj
    }
    return response

#file = open('input.txt')
#full_text = file.read()
#generate_summary(full_text, n_components=3, min_sentence_len = 0)
