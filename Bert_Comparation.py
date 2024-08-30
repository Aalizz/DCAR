import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
from keras.layers import Conv1D, Add, Activation, Concatenate
from keras.layers import Attention
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
import os,random
import time
from transformers import BertTokenizer, TFBertModel
def load_data(file_name):
    """Load and preprocess data."""
    df = pd.read_csv("./" + file_name, encoding="ISO-8859-1")
    df = df.dropna(subset=["ProcessedMessage"], axis=0)
    # df.info()
    return df.reset_index(drop=True)

def get_max_input_length(docs):
    """Get the maximum input length from the documents."""
    return max(len(doc.split()) for doc in docs)

def balance_data(X_train, y_train, input_type="numerical"):
    """Oversampling the dataset using SMOTE"""
    counter = Counter(y_train)
    print("Before SMOTE: ", counter)

    if input_type == "text":
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train)

    smote_model = SMOTE()
    X_train, y_train = smote_model.fit_resample(X_train, y_train)

    counter = Counter(y_train)
    print("After SMOTE: ", counter)

    if input_type == "text":
        X_train = vectorizer.inverse_transform(X_train)
        for i in range(len(X_train)):
            X_train[i] = " ".join(reversed(X_train[i]))

    return X_train, y_train

def create_embedding_matrix(tokenizer, embedding_dim):
    """Create an embedding matrix using GloVe embeddings."""
    file_name = ""
    embeddings_index = {}
    if embedding_dim == 50:
        file_name = "glove.6B.50d.txt"
    elif embedding_dim == 100:
        file_name = "glove.6B.100d.txt"

    with open("./" + file_name, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    # print("Found %s word vectors." % len(embeddings_index))
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def encode_text(docs_X_train, docs_X_test):
    """Encode text data into padded sequences."""
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


def build_bert_model(transformer_model, max_len):
    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    attention_masks = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='attention_masks')

    bert_output = transformer_model(input_ids, attention_mask=attention_masks)[0]
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(bert_output)

    dropout = tf.keras.layers.Dropout(0.3)(pooled_output)
    output = tf.keras.layers.Dense(2, activation='softmax')(dropout)

    model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def evaluate_model(y_test, y_pred):
    """Plot confusion matrix and print classification report"""
    target_names = ["Ham", "Spam"]

    # 检查 y_test 和 y_pred 中的类别
    print("Unique classes in y_test:", np.unique(y_test))
    print("Unique classes in y_pred:", np.unique(y_pred))

    # 显式指定 labels 参数
    labels = [0, 1]
    plot_confusion_matrix(y_test, y_pred, target_names)
    print(metrics.classification_report(y_test, y_pred, labels=labels, target_names=target_names, digits=4))

def plot_confusion_matrix(y_test, y_pred, classes):
    cm = metrics.confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.title(title)
    # plt.savefig(f"./Figures/confusion_matrix.pdf")
    plt.show()

def roc(y_test, y_proba):
    """Calculate ROC curve"""
    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    auc = metrics.roc_auc_score(y_test, y_proba)
    return fpr, tpr, auc

def plot_ROC_all(plot_name, result_table):
    """Plot multiple ROC-Curve"""
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

def load_transformer_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model
def preprocess_data_with_bert(tokenizer, docs, max_len):
    input_ids = []
    attention_masks = []

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

def transformer_based_model(X_train, X_test, y_train, y_test, params, max_len):
    tokenizer, transformer_model = load_transformer_model()
    train_input_ids, train_attention_masks = preprocess_data_with_bert(tokenizer, X_train, max_len)
    test_input_ids, test_attention_masks = preprocess_data_with_bert(tokenizer, X_test, max_len)

    model = build_bert_model(transformer_model, max_len)

    start_time = time.time()
    model.fit(
        [train_input_ids, train_attention_masks],
        y_train,
        epochs=params["epochs"],
        batch_size=16,
        validation_split=0.1,
        verbose=1
    )
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time for BERT-EL is {}".format(execution_time))

    y_pred_proba = model.predict([test_input_ids, test_attention_masks])
    y_pred_labels = np.argmax(y_pred_proba, axis=1)
    print("Classification report for BERT-based model:")
    print(metrics.classification_report(y_test, y_pred_labels, digits=4))
    evaluate_model(y_test, y_pred_labels)
    y_proba = y_pred_proba[:, 1]  # 选择第二列作为预测的spam概率
    fpr, tpr, auc = roc(y_test, y_proba)
    print(fpr, tpr, auc, execution_time)
    return fpr, tpr, auc, execution_time

def transformer_based_models(X_train, X_test, y_train, y_test, params):
    result_table = pd.DataFrame(columns=["classifiers", "fpr", "tpr", "auc", "time"])
    print("\n*******************************BERT-EL*******************************\n")
    fpr, tpr, auc, execution_time = transformer_based_model(
        X_train["ProcessedMessage"].tolist(),
        X_test["ProcessedMessage"].tolist(),
        y_train,
        y_test,
        params=params,
        max_len=128  # Example max length, adjust as needed
    )
    result_table = result_table.append(
        {"classifiers": "BERT-EL", "fpr": fpr, "tpr": tpr, "auc": auc, "time": execution_time}, ignore_index=True
    )

    # Set name of the classifiers as index labels
    result_table.set_index("classifiers", inplace=True)

    # Plot multiple ROC-Curve
    plot_ROC_all("ROC for TS models", result_table)


# Main execution
df = load_data("processed_spam.csv")

data = df.drop(["Label"], axis=1)
targets = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.20, random_state=42)

docs_X_train = X_train["ProcessedMessage"]
docs_X_test = X_test["ProcessedMessage"]
max_sequence_length = get_max_input_length(docs_X_train)

params = {"epochs": 10}
class_weights = None

print("\n**************************Run BERT-based Transformer model*************************\n")
docs_X_train, docs_y_train = balance_data(docs_X_train, y_train, input_type="text")

transformer_based_models(
    X_train,
    X_test,
    y_train,
    y_test,
    params=params
)