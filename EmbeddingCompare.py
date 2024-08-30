import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
from keras.layers import Conv1D, Add, Activation, Concatenate
from keras.layers import Attention

from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
from keras import backend as K
import os, random
from transformers import BertTokenizer, TFBertModel

def load_data(file_name):
    """Load and preprocess data."""
    df = pd.read_csv("./" + file_name, encoding="ISO-8859-1")
    df = df.dropna(subset=["ProcessedMessage"], axis=0)
    return df.reset_index(drop=True)

def balance_data(X_train, y_train):
    """Oversampling the dataset using SMOTE"""
    counter = Counter(y_train)
    print("Before SMOTE: ", counter)

    smote_model = SMOTE()
    X_train, y_train = smote_model.fit_resample(X_train, y_train)

    counter = Counter(y_train)
    print("After SMOTE: ", counter)
    return X_train, y_train

def create_embedding_matrix(tokenizer, embedding_dim):
    file_name = f"glove.6B.{embedding_dim}d.txt"
    embeddings_index = {}

    with open("./" + file_name, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def encode_text(docs_X_train, docs_X_test, max_sequence_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(docs_X_train)

    encoded_docs_train = tokenizer.texts_to_sequences(docs_X_train)
    encoded_docs_test = tokenizer.texts_to_sequences(docs_X_test)

    padded_docs_train = pad_sequences(
        encoded_docs_train,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post",
    )
    padded_docs_test = pad_sequences(
        encoded_docs_test, maxlen=max_sequence_length, padding="post", truncating="post"
    )

    return padded_docs_train, padded_docs_test, tokenizer

def tfidf_embedding(docs_X_train, docs_X_test):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(docs_X_train).toarray()
    X_test_tfidf = vectorizer.transform(docs_X_test).toarray()
    return X_train_tfidf, X_test_tfidf, vectorizer

def bert_embedding(docs_X_train, docs_X_test, max_len=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')

    def preprocess(docs):
        input_ids, attention_masks = [], []
        for doc in docs:
            encoded_dict = tokenizer.encode_plus(
                doc,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='tf',
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = tf.concat(input_ids, axis=0)
        attention_masks = tf.concat(attention_masks, axis=0)
        return input_ids, attention_masks

    train_input_ids, train_attention_masks = preprocess(docs_X_train)
    test_input_ids, test_attention_masks = preprocess(docs_X_test)

    return model, train_input_ids, train_attention_masks, test_input_ids, test_attention_masks

def build_resnet_block(input_layer, filters, kernel_size):
    x1 = Conv1D(filters, kernel_size, padding='same', activation='relu')(input_layer)
    x1 = Conv1D(filters, kernel_size, padding='same')(x1)
    shortcut1 = Conv1D(filters, kernel_size, padding='same')(input_layer)
    x1 = Add()([x1, shortcut1])
    x1 = Activation('relu')(x1)
    x2 = Conv1D(filters, kernel_size, padding='same', activation='relu')(input_layer)
    x2 = Conv1D(filters, kernel_size, padding='same')(x2)
    shortcut2 = Conv1D(filters, kernel_size, padding='same')(input_layer)
    x2 = Add()([x2, shortcut2])
    x2 = Activation('relu')(x2)
    x1_attention = Attention()([x1, x1])
    x2_attention = Attention()([x2, x2])
    x = Concatenate()([x1_attention, x2_attention])
    return x

def DCAR(X_train, X_test, y_train, y_test, params, class_weights, vocab_size=None, max_sequence_length=None, embedding_matrix=None, tfidf_vectorizer=None, use_tfidf=False):
    filters = params["filters"]
    kernel_size = params["kernel_size"]
    epochs = params["epochs"]
    num_resnet_blocks = params.get("num_resnet_blocks", 2)
    dropout_rate = params.get("dropout_rate", 0.5)

    if use_tfidf:
        input_layer = tf.keras.layers.Input(shape=(X_train.shape[1], 1))
        x = input_layer
    else:
        input_layer = tf.keras.layers.Input(shape=(max_sequence_length,))
        x = tf.keras.layers.Embedding(vocab_size, 100, input_length=max_sequence_length, weights=[embedding_matrix], trainable=True, mask_zero=True)(input_layer)
        
    for _ in range(num_resnet_blocks):
        x = build_resnet_block(x, filters, kernel_size)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled_output = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    x = tf.keras.layers.Dropout(dropout_rate)(pooled_output)
    output_layer = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    model.fit(X_train, y_train, class_weight=class_weights, epochs=epochs, verbose=0)
    feature_maps_model = tf.keras.models.Model(model.input, model.get_layer(index=-2).output)
    X_train_features = feature_maps_model.predict(X_train)
    X_test_features = feature_maps_model.predict(X_test)
    X_train_balanced, y_train_balanced = balance_data(X_train_features, y_train)
    probs_train, probs_test, y_pred, y_proba = DCAR_classify(X_train_balanced, X_test_features, y_train_balanced, y_test, class_weights)
    evaluate_model(y_test, y_pred)
    fpr, tpr, auc = roc(y_test, y_proba)
    return fpr, tpr, auc

def DCAR_classify(X_train, X_test, y_train, y_test, class_weights):
    classifier1 = RandomForestClassifier(n_estimators=100, criterion="gini", class_weight=class_weights, random_state=0)
    classifier2 = RandomForestClassifier(n_estimators=100, criterion="entropy", class_weight=class_weights, random_state=0)
    classifier3 = ExtraTreesClassifier(n_estimators=100, criterion="gini", class_weight=class_weights, random_state=0)
    classifier4 = ExtraTreesClassifier(n_estimators=100, criterion="entropy", class_weight=class_weights, random_state=0)

    # Fit classifiers
    classifier1.fit(X_train, y_train)
    classifier2.fit(X_train, y_train)
    classifier3.fit(X_train, y_train)
    classifier4.fit(X_train, y_train)

    # Get probabilities
    probs_train1 = classifier1.predict_proba(X_train)
    probs_test1 = classifier1.predict_proba(X_test)
    probs_train2 = classifier2.predict_proba(X_train)
    probs_test2 = classifier2.predict_proba(X_test)
    probs_train3 = classifier3.predict_proba(X_train)
    probs_test3 = classifier3.predict_proba(X_test)
    probs_train4 = classifier4.predict_proba(X_train)
    probs_test4 = classifier4.predict_proba(X_test)

    # Concatenate probabilities
    probs_train = np.concatenate((probs_train1, probs_train2, probs_train3, probs_train4), axis=1)
    probs_test = np.concatenate((probs_test1, probs_test2, probs_test3, probs_test4), axis=1)

    y_pred = []
    y_proba = []
    for i in range(len(X_test)):
        avg_ham = (probs_test1[i][0] + probs_test2[i][0] + probs_test3[i][0] + probs_test4[i][0]) / 4
        avg_spam = (probs_test1[i][1] + probs_test2[i][1] + probs_test3[i][1] + probs_test4[i][1]) / 4
        y_pred.append(0 if avg_ham > avg_spam else 1)
        max_spam = max(probs_test1[i][1], probs_test2[i][1], probs_test3[i][1], probs_test4[i][1])
        y_proba.append(max_spam)

    return probs_train, probs_test, y_pred, y_proba

def evaluate_model(y_test, y_pred):
    target_names = ["Ham", "Spam"]
    print("Unique classes in y_test:", np.unique(y_test))
    print("Unique classes in y_pred:", np.unique(y_pred))
    labels = [0, 1]
    plot_confusion_matrix(y_test, y_pred, target_names)
    print(metrics.classification_report(y_test, y_pred, labels=labels, target_names=target_names, digits=4))

def plot_confusion_matrix(y_test, y_pred, classes):
    cm = metrics.confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def roc(y_test, y_proba):
    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    auc = metrics.roc_auc_score(y_test, y_proba)
    return fpr, tpr, auc

def plot_ROC_all(plot_name, result_table):
    fig = plt.figure(figsize=(8, 6))
    for i in result_table.index:
        plt.plot(result_table.loc[i]["fpr"], result_table.loc[i]["tpr"], label="{}, AUC={:.3f}".format(i, result_table.loc[i]["auc"]))
    plt.plot([0, 1], [0, 1], color="orange", linestyle="--")
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.legend(prop={"size": 13}, loc="lower right")
    plt.show()
    fig.savefig("./Figures/" + plot_name + ".pdf")

def get_max_input_length(docs):
    return max(len(doc.split()) for doc in docs)

df = load_data("processed_spam.csv")

data = df.drop(["Label"], axis=1)
targets = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.20, random_state=42)

docs_X_train = X_train["ProcessedMessage"]
docs_X_test = X_test["ProcessedMessage"]
max_sequence_length = get_max_input_length(docs_X_train)

# GloVe Embedding
# print("\n*******************************Run DCAR with GloVe Embedding*******************************\n")
# padded_docs_train, padded_docs_test, tokenizer = encode_text(docs_X_train, docs_X_test, max_sequence_length)
# vocab_size = len(tokenizer.word_index) + 1
# embedding_dim = 100
# embedding_matrix = create_embedding_matrix(tokenizer, embedding_dim)

params = {"filters": 64, "kernel_size": 2, "units": 64, "epochs": 10}
class_weights = None

# DCAR_fpr, DCAR_tpr, DCAR_auc = DCAR(
# padded_docs_train,
# padded_docs_test,
# y_train,
# y_test,
# params=params,
# class_weights=class_weights,
# vocab_size=vocab_size,
# max_sequence_length=max_sequence_length,
# embedding_matrix=embedding_matrix
# )

# print(f"GloVe AUC: {DCAR_auc}")

# TF-IDF Embedding
print("\n*******************************Run DCAR with TF-IDF Embedding*******************************\n")

X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_embedding(docs_X_train, docs_X_test)
X_train_tfidf = np.expand_dims(X_train_tfidf, axis=-1)
X_test_tfidf = np.expand_dims(X_test_tfidf, axis=-1)

DCAR_fpr, DCAR_tpr, DCAR_auc = DCAR(
X_train_tfidf,
X_test_tfidf,
y_train,
y_test,
params=params,
class_weights=class_weights,
vocab_size=None,
max_sequence_length=None,
embedding_matrix=None,
tfidf_vectorizer=tfidf_vectorizer,
use_tfidf=True
)

print(f"TF-IDF AUC: {DCAR_auc}")

# BERT Embedding
print("\n*******************************Run DCAR with BERT Embedding*******************************\n")
model, train_input_ids, train_attention_masks, test_input_ids, test_attention_masks = bert_embedding(docs_X_train, docs_X_test)

bert_params = {"filters": 64, "kernel_size": 2, "units": 64, "epochs": 5}
input_ids, attention_masks = train_input_ids, train_attention_masks
bert_model = build_bert_model(model, 128)

start_time = time.time()
bert_model.fit([input_ids, attention_masks], y_train, epochs=bert_params["epochs"], batch_size=16, verbose=1)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time for BERT-EL is {}".format(execution_time))

y_pred_proba = bert_model.predict([test_input_ids, test_attention_masks])
y_pred_labels = np.argmax(y_pred_proba, axis=1)
print("Classification report for BERT-based model:")
print(metrics.classification_report(y_test, y_pred_labels, digits=4))
evaluate_model(y_test, y_pred_labels)
y_proba = y_pred_proba[:, 1]  # 选择第二列作为预测的spam概率
fpr, tpr, auc = roc(y_test, y_proba)
print(f"BERT AUC: {auc}, Execution time: {execution_time}")
