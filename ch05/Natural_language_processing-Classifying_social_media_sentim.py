import pandas as pd
import numpy as np
import random

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.pipeline import Pipeline
import time

tweet_df = pd.read_csv(r"D:\python\feature engineering bookcamp\ch05\cleaned_airline_tweets.csv")

tweet_df.head()

from pandas_profiling import ProfileReport

profile = ProfileReport(tweet_df, title="Tweets Report", explorative=True)

profile

tweet_df['sentiment'].value_counts(normalize=True)

from sklearn.model_selection import train_test_split

train, test = train_test_split(tweet_df, test_size=0.2, random_state=0, stratify=tweet_df['sentiment'])

print(f'Count of tweets in training set: {train.shape[0]:,}')
print(f'Count of tweets in testing set: {test.shape[0]:,}')

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()  # A
single_word = cv.fit_transform(train['text'])  #  B

print(single_word.shape)

for i, (token, token_index) in enumerate(cv.vocabulary_.items()):
    print(f'Token: {token} Index {token_index}')
    if i == 10:
        break

pd.DataFrame(single_word.todense(), columns=cv.get_feature_names_out())

single_word

cv = CountVectorizer(max_features=20)  # A 

limited_vocab = cv.fit_transform(train['text'])

pd.DataFrame(limited_vocab.toarray(), index = train['text'], columns = cv.get_feature_names_out())

cv = CountVectorizer(ngram_range=(1, 3))  # A

more_ngrams = cv.fit_transform(train['text'])

print(more_ngrams.shape)  # 70,613 features!

pd.DataFrame(more_ngrams.toarray(), index = train['text'], columns = cv.get_feature_names_out()).head()

single_word.sum(axis=0).argsort()[::-1]

cv = CountVectorizer()

cv.inverse_transform(cv.fit_transform(train['text'].head()))

cv = CountVectorizer(max_features=10)
cv.fit(train['text'])

cv.get_feature_names_out()

cv = CountVectorizer(stop_words='english', max_features=10)  # A
cv.fit(train['text'])

cv.get_feature_names_out()

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=10000)  # A

ml_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),  # B
    ('classifier', clf)
])

params = {
    'vectorizer__lowercase': [True, False],  # C
    'vectorizer__stop_words': [None, 'english'],
    'vectorizer__max_features': [100, 1000, 5000],
    'vectorizer__ngram_range': [(1, 1), (1, 3)],
    
    'classifier__C': [1e-1, 1e0, 1e1]  
    
}

def advanced_grid_search(x_train, y_train, x_test, y_test, ml_pipeline, params, cv=3, include_probas=False, is_regression=False):
    ''' 
    This helper function will grid search a machine learning pipeline with feature engineering included
    and print out a classification report for the best param set. 
    Best here is defined as having the best cross-validated accuracy on the training set
    '''
    
    model_grid_search = GridSearchCV(ml_pipeline, param_grid=params, cv=cv, error_score=-1)
    start_time = time.time()  # capture the start time

    model_grid_search.fit(x_train, y_train)

    best_model = model_grid_search.best_estimator_
    
    y_preds = best_model.predict(x_test)
    
    if is_regression:
        rmse = np.sqrt(mean_squared_error(y_pred=y_preds, y_true=test_set['pct_change_eod']))
        print(f'RMSE: {rmse:.5f}')
    else:
        print(classification_report(y_true=y_test, y_pred=y_preds))
    print(f'Best params: {model_grid_search.best_params_}')
    end_time = time.time()
    print(f"Overall took {(end_time - start_time):.2f} seconds")
    
    if include_probas:
        y_probas = best_model.predict_proba(x_test).max(axis=1)
        return best_model, y_preds, y_probas
    
    return best_model, y_preds

print("Count Vectorizer + Log Reg\n=====================")
advanced_grid_search(  # D
    train['text'], train['sentiment'], test['text'], test['sentiment'], 
    ml_pipeline, params
)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10)

tfdf_text = tfidf_vectorizer.fit_transform(train['text'])
pd.DataFrame(tfdf_text.toarray(), index = train['text'], columns = tfidf_vectorizer.get_feature_names_out())

tfidf_vectorizer = TfidfVectorizer()  # A

tfidf_vectorizer.fit(train['text'])

idf = pd.DataFrame({'feature_name':tfidf_vectorizer.get_feature_names_out(), 'idf_weights':tfidf_vectorizer.idf_})
idf.sort_values('idf_weights', ascending=True)

np.log((1 + train.shape[0]) / (1 + 1)) + 1

ml_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),  # A
    ('classifier', clf)
])

print("TF-IDF Vectorizer + Log Reg\n=====================")
advanced_grid_search(
    train['text'], train['sentiment'], test['text'], test['sentiment'], 
    ml_pipeline, params  # B
)

import preprocessor as tweet_preprocessor

# remove urls and mentions
tweet_preprocessor.set_options(
    tweet_preprocessor.OPT.URL, tweet_preprocessor.OPT.NUMBER
)

tweet_preprocessor.clean(
    '@United is #awesome 👍 https://a.link/s/redirect 100%'
)

tweet_preprocessor.set_options(
    tweet_preprocessor.OPT.URL, tweet_preprocessor.OPT.NUMBER
)

ml_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),  # TfidfVectorizer gave us better results
    ('classifier', clf)
])

params = {
    'vectorizer__lowercase': [True, False],
    'vectorizer__stop_words': [None, 'english'],
    'vectorizer__max_features': [100, 1000, 5000],
    'vectorizer__ngram_range': [(1, 1), (1, 3)],
    
    'classifier__C': [1e-1, 1e0, 1e1]  

}

print("Tweet Cleaning + Log Reg\n=====================")
advanced_grid_search(
    # apply cleaning here because it does not change given the training data
    train['text'].apply(tweet_preprocessor.clean), train['sentiment'], 
    test['text'].apply(tweet_preprocessor.clean), test['sentiment'], 
    ml_pipeline, params
)

from nltk.stem import SnowballStemmer  # A

snowball_stemmer = SnowballStemmer(language='english')  # B

snowball_stemmer.stem('waiting')

import nltk  # A

nltk.download('stopwords')
from nltk.corpus import stopwords

stemmed_stopwords = list(map(snowball_stemmer.stem, stopwords.words('english')))  # B

import re

def stem_tokenizer(_input):  #  C
    tokenized_words = re.sub(r"[^A-Za-z0-9\-]", " ", _input).lower().split()
    return [snowball_stemmer.stem(word) for word in tokenized_words if snowball_stemmer.stem(word) not in stemmed_stopwords]

stem_tokenizer('waiting for the plane')

ml_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=stem_tokenizer)),  # A
    ('classifier', clf)
])

params = {
#     'vectorizer__lowercase': [True, False],
#     'vectorizer__stop_words': [],  # B
    
    'vectorizer__max_features': [100, 1000, 5000],
    'vectorizer__ngram_range': [(1, 1), (1, 3)],
    
    'classifier__C': [1e-1, 1e0, 1e1]  

}

print("Stemming + Log Reg\n=====================")
advanced_grid_search(
    # remove cleaning
    train['text'], train['sentiment'], 
    test['text'], test['sentiment'], 
    ml_pipeline, params
)

from sklearn.decomposition import TruncatedSVD  # A

ml_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),  # B
    ('reducer', TruncatedSVD()),
    ('classifier', clf)
])

params = {
    'vectorizer__lowercase': [True, False],
    'vectorizer__stop_words': [None, 'english'],
    'vectorizer__max_features': [5000],
    'vectorizer__ngram_range': [(1, 3)],
    
    'reducer__n_components': [500, 1000, 1500, 2000],  # number of components to reduce to
    
    'classifier__C': [1e-1, 1e0, 1e1]

}

print("SVD + Log Reg\n=====================")
advanced_grid_search(
    train['text'], train['sentiment'], 
    test['text'], test['sentiment'], 
    ml_pipeline, params
)

t = TfidfVectorizer(max_features=1000)
X = t.fit_transform(train['text']).toarray()

svd = TruncatedSVD(n_components=10)

svd.fit(X)

svd.components_.shape

(np.matmul(X, svd.components_.T) == svd.transform(X)).mean()  # A

vectorizer = TfidfVectorizer(**{
    'lowercase': True, 'max_features': 5000, 'ngram_range': (1, 3), 'stop_words': None
})

vectorized_X_train = vectorizer.fit_transform(train['text']).toarray()  # A
vectorized_X_test = vectorizer.transform(test['text']).toarray()  # A

vectorized_X_train.shape, vectorized_X_test.shape

from keras.layers import Input, Dense      # A
from keras.models import Model, Sequential # A
import tensorflow as tf                    # A

n_inputs = vectorized_X_train.shape[1]
n_bottleneck = 2000  # B

# encoder
visible = Input(shape=(n_inputs,), name='input')
e = Dense(n_inputs//2, activation='relu', name='encoder')(visible)
# code/bottleneck
bottleneck = Dense(n_bottleneck, name='bottleneck')(e)

# decoder
d = Dense(n_inputs//2, activation='relu', name='decoder')(bottleneck)
# output layer
output = Dense(n_inputs, activation='relu', name='output')(d)

# define autoencoder model
autoencoder = Model(inputs=visible, outputs=output)


autoencoder.compile(optimizer='adam', loss='mse')  # C

from keras.utils import plot_model

plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True, show_layer_names=True)

import matplotlib.pyplot as plt

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)  # A

# B
autoencoder_history = autoencoder.fit(vectorized_X_train, vectorized_X_train, 
                batch_size = 512, epochs = 100,  callbacks=[early_stopping_callback],
                shuffle = True, validation_split = 0.10)


plt.plot(autoencoder_history.history['loss'], label='Loss')
plt.plot(autoencoder_history.history['val_loss'], label='Val Loss')

plt.title('Autoencoder Loss')
plt.legend()

latent_representation = Model(inputs=visible, outputs=bottleneck)  # A

encoded_X_train = latent_representation.predict(vectorized_X_train)  # B
encoded_X_test = latent_representation.predict(vectorized_X_test)  # B


ml_pipeline = Pipeline([
    ('classifier', clf)
])

params = {
    'classifier__C': [1e-1, 1e0, 1e1]  
}

print("Autoencoder + Log Reg\n=====================")
advanced_grid_search(
    encoded_X_train, train['sentiment'], encoded_X_test, test['sentiment'], 
    ml_pipeline, params
)

from transformers import BertTokenizer, BertModel  # A
import torch

bert_model = BertModel.from_pretrained('bert-base-uncased')  # B

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # C

tweet = 'I hate this airline'

token_ids = torch.tensor(bert_tokenizer.encode(tweet)).unsqueeze(0)  # D

bert_model(token_ids)[1].shape

from tqdm import tqdm
import numpy as np

def batch_embed_text(bert_model, tokenizer, text_iterable, batch_size=256):
    ''' This helper method will batch embed an iterable of text using a given tokenizer and bert model '''
    encoding = tokenizer(
    text_iterable, 
    padding=True, 
    truncation=True, 
    return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    def batch_array_idx(np_array, batch_size):
        for i in tqdm(range(0, np_array.shape[0], batch_size)):
            yield i, i + batch_size
            
    embedded = None

    for start_idx, end_idx in batch_array_idx(input_ids, batch_size=batch_size):
        batch_bert = bert_model(
            torch.tensor(input_ids[start_idx:end_idx]), 
            attention_mask=torch.tensor(attention_mask[start_idx:end_idx])
        )[1].detach().numpy()
        if embedded is None:
            embedded = batch_bert
        else:
            embedded = np.vstack([embedded, batch_bert])

    return embedded

bert_X_train = batch_embed_text(bert_model, bert_tokenizer, train['text'].tolist())

bert_X_test = batch_embed_text(bert_model, bert_tokenizer, test['text'].tolist())

ml_pipeline = Pipeline([
    ('classifier', clf)
])

params = {
    'classifier__C': [1e-1, 1e0, 1e1]  
}

print("BERT + Log Reg\n=====================")
advanced_grid_search(
    bert_X_train, train['sentiment'], bert_X_test, test['sentiment'], 
    ml_pipeline, params
)

from datasets import load_dataset, Dataset

sample_tweets, _ = train_test_split(train, test_size=0.9, random_state=0, stratify=train['sentiment'])

# Let's map our sentiment to a numerical classes. negative -> 0, neutral -> 1, and positive -> 2
sample_tweets['sentiment'] = sample_tweets['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})

# the trainer is expecting a 'label' (see the forward method in the docs)
sample_tweets['label'] = sample_tweets['sentiment']

print(sample_tweets['label'].value_counts())

sample_dataset = Dataset.from_pandas(sample_tweets)

# Dataset has a built in train test split method
sample_dataset = sample_dataset.train_test_split(test_size=0.3)

train_set = sample_dataset['train']
test_set = sample_dataset['test']

def preprocess(data):
    return bert_tokenizer(data['text'], padding=True, truncation=True)

train_set = train_set.map(preprocess, batched=True, batch_size=len(train_set))
test_set = test_set.map(preprocess, batched=True, batch_size=len(test_set))

train_set.set_format('torch', 
                      columns=['input_ids', 'attention_mask', 'label'])
test_set.set_format('torch', 
                     columns=['input_ids', 'attention_mask', 'label'])

from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

sequence_classification_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=3
)

batch_size = 128
epochs = 10

warmup_steps = 500
weight_decay = 0.01

# Define the trainer: 

trainer = Trainer(
    model=sequence_classification_model,
    train_dataset=train_set,
    eval_dataset=test_set
)

# Get initial metrics
trainer.evaluate()

trainer.train()
fine_tuned_bert_model = sequence_classification_model.bert
fine_tuned_bert_X_train = batch_embed_text(fine_tuned_bert_model, bert_tokenizer, train['text'].tolist())

fine_tuned_bert_X_test = batch_embed_text(fine_tuned_bert_model, bert_tokenizer, test['text'].tolist())
ml_pipeline = Pipeline([
    ('classifier', clf)
])

params = {
    'classifier__C': [1e-1, 1e0, 1e1]  
}

print("Fine-tuned BERT + Log Reg\n=====================")
advanced_grid_search(
    fine_tuned_bert_X_train, train['sentiment'],
    fine_tuned_bert_X_test, test['sentiment'],
    ml_pipeline, params
)

