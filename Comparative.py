import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
from keras.layers import Conv1D, Add, Activation, Concatenate
from keras.layers import Attention
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
import os, random
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


set_seed(42)


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


def build_resnet_block(input_layer, filters, kernel_size):
    """Build a ResNet block with dual channels and spatial attention."""
    # First branch
    x1 = Conv1D(filters, kernel_size, padding='same', activation='relu')(input_layer)
    x1 = Conv1D(filters, kernel_size, padding='same')(x1)
    shortcut1 = Conv1D(filters, kernel_size, padding='same')(input_layer)
    x1 = Add()([x1, shortcut1])
    x1 = Activation('relu')(x1)

    # Second branch
    x2 = Conv1D(filters, kernel_size, padding='same', activation='relu')(input_layer)
    x2 = Conv1D(filters, kernel_size, padding='same')(x2)
    shortcut2 = Conv1D(filters, kernel_size, padding='same')(input_layer)
    x2 = Add()([x2, shortcut2])
    x2 = Activation('relu')(x2)

    # Apply attention mechanism to each branch output
    x1_attention = Attention()([x1, x1])
    x2_attention = Attention()([x2, x2])

    x = Concatenate()([x1_attention, x2_attention])

    return x


def DCAR(X_train, X_test, y_train, y_test, params, class_weights, vocab_size, max_sequence_length, embedding_matrix,
         embedding_dim):
    filters = params["filters"]
    kernel_size = params["kernel_size"]
    epochs = params["epochs"]
    num_resnet_blocks = params.get("num_resnet_blocks", 2)  # Number of ResNet blocks
    # attention_units = params.get("attention_units", 64)    # Number of attention units
    dropout_rate = params.get("dropout_rate", 0.5)  # Dropout rate

    input_layer = tf.keras.layers.Input(shape=(max_sequence_length,))
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_sequence_length,
                                  weights=[embedding_matrix], trainable=True, mask_zero=True)(input_layer)

    # Build ResNet blocks
    for _ in range(num_resnet_blocks):
        x = build_resnet_block(x, filters, kernel_size)

    # Dual-channel pooling
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled_output = tf.keras.layers.Concatenate()([max_pool, avg_pool])

    x = tf.keras.layers.Dropout(dropout_rate)(pooled_output)  # Dropout layer
    output_layer = tf.keras.layers.Dense(2, activation='softmax')(x)

    # Compile and train the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    start_time = time.time()
    model.fit(X_train, y_train, class_weight=class_weights, epochs=epochs, verbose=0)
    end_time = time.time()
    execution_time = end_time - start_time
    print("time of DCAR is {}".format(execution_time))

    # Extract feature maps for further analysis or classification
    feature_maps_model = tf.keras.models.Model(model.input, model.get_layer(index=-2).output)
    X_train_features = feature_maps_model.predict(X_train)
    X_test_features = feature_maps_model.predict(X_test)

    # Balance data if necessary
    X_train_balanced, y_train_balanced = balance_data(X_train_features, y_train)

    # Classify using DCAR_classify function (assuming it's defined elsewhere)
    probs_train, probs_test, y_pred, y_proba = DCAR_classify(X_train_balanced, X_test_features, y_train_balanced,
                                                             y_test, class_weights)

    # Evaluate model performance
    evaluate_model(y_test, y_pred)
    fpr, tpr, auc = roc(y_test, y_proba)

    return fpr, tpr, auc


def DCAR_classify(X_train, X_test, y_train, y_test, class_weights):
    """Build DCAR's base classifiers (RandomForestClassifier and ExtraTreesClassifier)"""

    classifier1 = RandomForestClassifier(
        n_estimators=100, criterion="gini", class_weight=class_weights, random_state=0
    )
    classifier2 = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        class_weight=class_weights,
        random_state=0,
    )
    classifier3 = ExtraTreesClassifier(
        n_estimators=100, criterion="gini", class_weight=class_weights, random_state=0
    )
    classifier4 = ExtraTreesClassifier(
        n_estimators=100,
        criterion="entropy",
        class_weight=class_weights,
        random_state=0,
    )

    # Classifier 1
    classifier1.fit(X_train, y_train)
    probs_train1 = classifier1.predict_proba(X_train)
    probs_test1 = classifier1.predict_proba(X_test)

    # Classifier 2
    classifier2.fit(X_train, y_train)
    probs_train2 = classifier2.predict_proba(X_train)
    probs_test2 = classifier2.predict_proba(X_test)

    # Classifier 3
    classifier3.fit(X_train, y_train)
    probs_train3 = classifier3.predict_proba(X_train)
    probs_test3 = classifier3.predict_proba(X_test)

    # Classifier 4
    classifier4.fit(X_train, y_train)
    probs_train4 = classifier4.predict_proba(X_train)
    probs_test4 = classifier4.predict_proba(X_test)

    probs_train = np.concatenate(
        (probs_train1, probs_train2, probs_train3, probs_train4), axis=1
    )
    probs_test = np.concatenate(
        (probs_test1, probs_test2, probs_test3, probs_test4), axis=1
    )

    y_pred = []
    y_proba = []
    for i in range(0, len(X_test)):
        avg_ham = (
                          probs_test1[i][0]
                          + probs_test2[i][0]
                          + probs_test3[i][0]
                          + probs_test4[i][0]
                  ) / 4
        avg_spam = (
                           probs_test1[i][1]
                           + probs_test2[i][1]
                           + probs_test3[i][1]
                           + probs_test4[i][1]
                   ) / 4
        if avg_ham > avg_spam:
            # ham
            y_pred.append(0)
        else:
            # spam
            y_pred.append(1)

        max_spam = max(
            [probs_test1[i][1], probs_test2[i][1], probs_test3[i][1], probs_test4[i][1]]
        )
        y_proba.append(max_spam)

    return probs_train, probs_test, y_pred, y_proba


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
        plt.plot(result_table.loc[i]["fpr"], result_table.loc[i]["tpr"],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]["auc"]))
    plt.plot([0, 1], [0, 1], color="orange", linestyle="--")
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.legend(prop={"size": 13}, loc="lower right")
    plt.show()
    fig.savefig("./Figures/" + plot_name + "2" + ".pdf")

def ml_models_comparison(
        DCAR_fpr,
        DCAR_tpr,
        DCAR_auc,
        X_train,
        X_test,
        y_train,
        y_test,
        params
):
    """Compare DCAR with ML algorithms"""
    result_table = pd.DataFrame(columns=["classifiers", "fpr", "tpr", "auc"])

    # Random Forest
    print("\n*******************************Random Forest*******************************\n")
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=params["n_estimators"], random_state=42)
    rf_model.fit(X_train, y_train)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    y_pred_rf = rf_model.predict(X_test)
    evaluate_model(y_test, y_pred_rf)
    end_time = time.time()
    execution_time = end_time - start_time
    rf_fpr, rf_tpr, rf_auc = roc(y_test, y_proba_rf)
    print(f"time of RF is {execution_time}")
    print({"classifiers": "Random Forest", "fpr": rf_fpr, "tpr": rf_tpr, "auc": rf_auc})
    result_table = result_table.append(
        {"classifiers": "Random Forest", "fpr": rf_fpr, "tpr": rf_tpr, "auc": rf_auc}, ignore_index=True
    )

    # start_time = time.time()
    # svm_model = SVC(probability=True, random_state=42)
    # svm_model.fit(X_train, y_train)
    # y_pred_svm = svm_model.predict(X_test)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # svm_fpr, svm_tpr, svm_auc = roc(y_test, y_pred_svm)
    # evaluate_model(y_test, y_pred_svm)
    # print(f"time of SVM is {execution_time}")
    # result_table = result_table.append(
    #     {"classifiers": "Support Vector Machine", "fpr": svm_fpr, "tpr": svm_tpr, "auc": svm_auc}, ignore_index=True
    # )

    print("\n*******************************Support Vector Machine*******************************\n")
    start_time = time.time()
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    y_proba_svm = svm_model.predict_proba(X_test)[:, 1]
    y_pred_svm = svm_model.predict(X_test)
    end_time = time.time()
    execution_time = end_time - start_time
    svm_fpr, svm_tpr, svm_auc = roc(y_test, y_proba_svm)
    evaluate_model(y_test, y_pred_svm)
    print(f"time of SVM is {execution_time}")
    print({"classifiers": "Support Vector Machine", "fpr": svm_fpr, "tpr": svm_tpr, "auc": svm_auc})
    result_table = result_table.append(
        {"classifiers": "Support Vector Machine", "fpr": svm_fpr, "tpr": svm_tpr, "auc": svm_auc}, ignore_index=True
    ).dropna()

    # Decision Tree
    print("\n*******************************Decision Tree*******************************\n")
    start_time = time.time()
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_proba_dt = dt_model.predict_proba(X_test)[:, 1]
    y_pred_dt = dt_model.predict(X_test)
    end_time = time.time()
    execution_time = end_time - start_time
    dt_fpr, dt_tpr, dt_auc = roc(y_test, y_proba_dt)
    evaluate_model(y_test, y_pred_dt)
    print(f"time of DT is {execution_time}")
    print( {"classifiers": "Decision Tree", "fpr": dt_fpr, "tpr": dt_tpr, "auc": dt_auc})
    result_table = result_table.append(
        {"classifiers": "Decision Tree", "fpr": dt_fpr, "tpr": dt_tpr, "auc": dt_auc}, ignore_index=True
    ).dropna()

    # K-Nearest Neighbors
    print("\n*******************************K-Nearest Neighbors*******************************\n")
    start_time = time.time()
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_proba_knn = knn_model.predict_proba(X_test)[:, 1]
    y_pred_knn = knn_model.predict(X_test)
    end_time = time.time()
    execution_time = end_time - start_time
    knn_fpr, knn_tpr, knn_auc = roc(y_test, y_proba_knn)
    evaluate_model(y_test, y_pred_knn)
    print(f"time of KNN is {execution_time}")
    print({"classifiers": "K-Nearest Neighbors", "fpr": knn_fpr, "tpr": knn_tpr, "auc": knn_auc})
    result_table = result_table.append(
        {"classifiers": "K-Nearest Neighbors", "fpr": knn_fpr, "tpr": knn_tpr, "auc": knn_auc}, ignore_index=True
    ).dropna()

    # Gradient Boosting
    print("\n*******************************Gradient Boosting*******************************\n")
    start_time = time.time()
    gb_model = GradientBoostingClassifier(n_estimators=params["n_estimators"], random_state=42)
    gb_model.fit(X_train, y_train)
    y_proba_gb = gb_model.predict_proba(X_test)[:, 1]
    y_pred_gb = gb_model.predict(X_test)
    end_time = time.time()
    execution_time = end_time - start_time
    gb_fpr, gb_tpr, gb_auc = roc(y_test, y_proba_gb)
    evaluate_model(y_test, y_pred_gb)
    print(f"time of GB is {execution_time}")
    print({"classifiers": "Gradient Boosting", "fpr": gb_fpr, "tpr": gb_tpr, "auc": gb_auc})
    result_table = result_table.append(
        {"classifiers": "Gradient Boosting", "fpr": gb_fpr, "tpr": gb_tpr, "auc": gb_auc}, ignore_index=True
    ).dropna()

    # Set name of the classifiers as index labels
    result_table.set_index("classifiers", inplace=True)

    # Plot multiple ROC-Curve
    plot_ROC_all("ROC for ML models", result_table)
    return result_table


from keras import Sequential


def cnn_model(
        X_train,
        X_test,
        y_train,
        y_test,
        params,
        class_weights,
        vocab_size,
        max_sequence_length,
        embedding_matrix,
        embedding_dim,
):
    filters = params["filters"]
    kernel_size = params["kernel_size"]
    epochs = params["epochs"]

    model = Sequential()
    model.add(
        tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=max_sequence_length,
            weights=[embedding_matrix],
            trainable=True,
            mask_zero=True,
        )
    )
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation="relu"))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(3, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    start_time = time.time()
    model.fit(X_train, y_train, class_weight=class_weights, epochs=epochs, verbose=0)

    model.evaluate(X_test, y_test)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_proba = model.predict(X_test)[::, 1]

    end_time = time.time()
    execution_time = end_time - start_time
    print("time of CNN is {}".format(execution_time))
    print(metrics.classification_report(y_test, y_pred, digits=4))

    fpr, tpr, auc = roc(y_test, y_proba)

    return fpr, tpr, auc


# LSTM
def lstm_model(
        X_train,
        X_test,
        y_train,
        y_test,
        params,
        class_weights,
        vocab_size,
        max_sequence_length,
        embedding_matrix,
        embedding_dim,
):
    units = params["units"]
    epochs = params["epochs"]

    model = Sequential()
    model.add(
        tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=max_sequence_length,
            weights=[embedding_matrix],
            trainable=True,
            mask_zero=True,
        )
    )
    model.add(tf.keras.layers.LSTM(units=units, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    start_time = time.time()
    model.fit(X_train, y_train, class_weight=class_weights, epochs=epochs, verbose=0)

    model.evaluate(X_test, y_test)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_proba = model.predict(X_test)[::, 1]

    print("Classification metrics of LSTM")
    print(metrics.classification_report(y_test, y_pred, digits=4))
    end_time = time.time()
    execution_time = end_time - start_time
    print("time of LSTM is {}".format(execution_time))
    fpr, tpr, auc = roc(y_test, y_proba)

    return fpr, tpr, auc
# RNN
def rnn_model(
        X_train,
        X_test,
        y_train,
        y_test,
        params,
        class_weights,
        vocab_size,
        max_sequence_length,
        embedding_matrix,
        embedding_dim,
):
    units = params["units"]
    epochs = params["epochs"]

    model = Sequential()
    model.add(
        tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=max_sequence_length,
            weights=[embedding_matrix],
            trainable=True,
            mask_zero=True,
        )
    )
    model.add(tf.keras.layers.SimpleRNN(units=units, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    start_time = time.time()
    model.fit(X_train, y_train, class_weight=class_weights, epochs=epochs, verbose=0)

    model.evaluate(X_test, y_test)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_proba = model.predict(X_test)[::, 1]

    print("Classification metrics of RNN")
    print(metrics.classification_report(y_test, y_pred, digits=4))
    end_time = time.time()
    execution_time = end_time - start_time
    print("time of RNN is {}".format(execution_time))

    fpr, tpr, auc = roc(y_test, y_proba)

    return fpr, tpr, auc


def DL_models(
        X_train,
        X_test,
        y_train,
        y_test,
        params,
        class_weights,
        vocab_size,
        max_sequence_length,
        embedding_matrix,
        embedding_dim,
):
    """Compare DCAR with deep learning techniques"""
    # result_table = pd.DataFrame(columns=["classifiers", "fpr", "tpr", "auc"])
    result_table = pd.DataFrame(columns=["classifiers", "fpr", "tpr", "auc"])

    # Random Forest
    print("\n*******************************Random Forest*******************************\n")
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    y_pred_rf = rf_model.predict(X_test)
    evaluate_model(y_test, y_pred_rf)
    end_time = time.time()
    execution_time = end_time - start_time
    rf_fpr, rf_tpr, rf_auc = roc(y_test, y_proba_rf)
    print(f"time of RF is {execution_time}")
    # print({"classifiers": "Random Forest", "fpr": rf_fpr, "tpr": rf_tpr, "auc": rf_auc})
    result_table = result_table.append(
        {"classifiers": "Random Forest", "fpr": rf_fpr, "tpr": rf_tpr, "auc": rf_auc}, ignore_index=True
    )

    print("\n*******************************Support Vector Machine*******************************\n")
    start_time = time.time()
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    y_proba_svm = svm_model.predict_proba(X_test)[:, 1]
    y_pred_svm = svm_model.predict(X_test)
    end_time = time.time()
    execution_time = end_time - start_time
    svm_fpr, svm_tpr, svm_auc = roc(y_test, y_proba_svm)
    evaluate_model(y_test, y_pred_svm)
    print(f"time of SVM is {execution_time}")
    # print({"classifiers": "Support Vector Machine", "fpr": svm_fpr, "tpr": svm_tpr, "auc": svm_auc})
    result_table = result_table.append(
        {"classifiers": "Support Vector Machine", "fpr": svm_fpr, "tpr": svm_tpr, "auc": svm_auc}, ignore_index=True
    )

    # Decision Tree
    print("\n*******************************Decision Tree*******************************\n")
    start_time = time.time()
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_proba_dt = dt_model.predict_proba(X_test)[:, 1]
    y_pred_dt = dt_model.predict(X_test)
    end_time = time.time()
    execution_time = end_time - start_time
    dt_fpr, dt_tpr, dt_auc = roc(y_test, y_proba_dt)
    evaluate_model(y_test, y_pred_dt)
    print(f"time of DT is {execution_time}")
    # print({"classifiers": "Decision Tree", "fpr": dt_fpr, "tpr": dt_tpr, "auc": dt_auc})
    result_table = result_table.append(
        {"classifiers": "Decision Tree", "fpr": dt_fpr, "tpr": dt_tpr, "auc": dt_auc}, ignore_index=True
    )

    # K-Nearest Neighbors
    print("\n*******************************K-Nearest Neighbors*******************************\n")
    start_time = time.time()
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_proba_knn = knn_model.predict_proba(X_test)[:, 1]
    y_pred_knn = knn_model.predict(X_test)
    end_time = time.time()
    execution_time = end_time - start_time
    knn_fpr, knn_tpr, knn_auc = roc(y_test, y_proba_knn)
    evaluate_model(y_test, y_pred_knn)
    print(f"time of KNN is {execution_time}")
    # print({"classifiers": "K-Nearest Neighbors", "fpr": knn_fpr, "tpr": knn_tpr, "auc": knn_auc})
    result_table = result_table.append(
        {"classifiers": "K-Nearest Neighbors", "fpr": knn_fpr, "tpr": knn_tpr, "auc": knn_auc}, ignore_index=True
    )

    # Gradient Boosting
    print("\n*******************************Gradient Boosting*******************************\n")
    start_time = time.time()
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    y_proba_gb = gb_model.predict_proba(X_test)[:, 1]
    y_pred_gb = gb_model.predict(X_test)
    end_time = time.time()
    execution_time = end_time - start_time
    gb_fpr, gb_tpr, gb_auc = roc(y_test, y_proba_gb)
    evaluate_model(y_test, y_pred_gb)
    print(f"time of GB is {execution_time}")
    # print({"classifiers": "Gradient Boosting", "fpr": gb_fpr, "tpr": gb_tpr, "auc": gb_auc})
    result_table = result_table.append(
        {"classifiers": "Gradient Boosting", "fpr": gb_fpr, "tpr": gb_tpr, "auc": gb_auc}, ignore_index=True
    )

    print("\n*******************************CNN*******************************\n")
    fpr, tpr, auc = cnn_model(
        X_train,
        X_test,
        y_train,
        y_test,
        params=params,
        class_weights=class_weights,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        embedding_matrix=embedding_matrix,
        embedding_dim=embedding_dim,
    )
    result_table = result_table.append(
        {"classifiers": "CNN", "fpr": fpr, "tpr": tpr, "auc": auc}, ignore_index=True
    )

    print("\n*******************************LSTM*******************************\n")
    fpr, tpr, auc = lstm_model(
        X_train,
        X_test,
        y_train,
        y_test,
        params=params,
        class_weights=class_weights,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        embedding_matrix=embedding_matrix,
        embedding_dim=embedding_dim,
    )
    result_table = result_table.append(
        {"classifiers": "LSTM", "fpr": fpr, "tpr": tpr, "auc": auc}, ignore_index=True
    )

    print("\n*******************************RNN*******************************\n")
    fpr, tpr, auc = rnn_model(
        X_train,
        X_test,
        y_train,
        y_test,
        params=params,
        class_weights=class_weights,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        embedding_matrix=embedding_matrix,
        embedding_dim=embedding_dim,
    )
    result_table = result_table.append(
        {"classifiers": "RNN", "fpr": fpr, "tpr": tpr, "auc": auc}, ignore_index=True
    )

    fpr = np.array([0.00000000, 0.00896057, 0.00896057, 0.01075269, 0.01433692, 0.01433692,
0.01433692, 0.01433692, 0.01433692, 0.01433692, 0.01433692, 0.01433692,
0.01433692, 0.01612903, 0.01612903, 0.01612903, 0.01612903, 0.01612903,
0.01792115, 0.01792115, 0.01971326, 0.01971326, 0.01971326, 0.01971326,
0.01971326, 0.01971326, 0.02329749, 0.02329749, 0.02508961, 0.02508961,
0.02688172, 0.02867384, 0.03405018, 0.03405018, 0.03405018, 0.03405018,
0.03405018, 0.03942652, 0.03942652, 0.04121864, 0.04121864, 0.04301075,
0.04301075, 0.04480287, 0.04480287, 0.04480287, 0.04480287, 0.04659498,
0.04659498, 0.04659498, 0.04659498, 0.04659498, 0.04838710, 0.04838710,
0.05017921, 0.05017921, 0.05197133, 0.05197133, 0.05376344, 0.05376344,
0.05555556, 0.05555556, 0.05734767, 0.05734767, 0.05913978, 0.05913978,
0.06093190, 0.06093190, 0.06272401, 0.06451613, 0.06451613, 0.06989247,
0.06989247, 0.07168459, 0.07168459, 0.07526882, 0.07885305, 0.07885305,
0.08064516, 0.08781362, 0.08960573, 0.08960573, 0.09498208, 0.09498208,
0.10035842, 0.10035842, 0.11648746, 0.11648746, 0.12544803, 0.12724014,
0.13082437, 0.13082437, 0.13261649, 0.13261649, 0.13440860, 0.13799283,
0.14157706, 0.14157706, 0.15053763, 0.16308244, 0.16308244, 0.16487455,
0.16845878, 0.17025090, 0.17025090, 0.17741935, 0.18100358, 0.19354839,
0.19354839, 0.19534050, 0.19534050, 0.19534050, 0.19892473, 0.20071685,
0.20071685, 0.20430108, 0.20788530, 0.21146953, 0.21146953, 0.21505376,
0.22222222, 0.22759857, 0.22759857, 0.22939068, 0.22939068, 0.23297491,
0.23297491, 0.23297491, 0.24014337, 0.24014337, 0.24193548, 0.24193548,
0.25627240, 0.25627240, 0.26164875, 0.26523297, 0.27777778, 0.27777778,
0.28494624, 0.29032258, 0.29569892, 0.29928315, 0.31003584, 0.31362007,
0.31899642, 0.32258065, 0.33691756, 0.34050179, 0.34229391, 0.34587814,
0.35663082, 0.36200717, 0.36917563, 0.37634409, 0.38172043, 0.38172043,
0.38709677, 0.39964158, 0.41218638, 0.41577061, 0.41935484, 0.43189964,
0.43906810, 0.44265233, 0.44802867, 0.45878136, 0.46236559, 0.46774194,
0.49103943, 0.49641577, 0.50000000, 0.50537634, 0.51254480, 0.51612903,
0.53942652, 0.54301075, 0.54659498, 0.55017921, 0.55376344, 0.55555556,
0.56093190, 0.56630824, 0.60035842, 0.60752688, 0.61469534, 0.61827957,
0.62365591, 0.63261649, 0.71684588, 0.71863799, 0.72222222, 0.72939068,
0.73118280, 0.73476703, 0.73655914, 0.74193548, 0.74372760, 0.74731183,
0.74910394, 1.00000000
])

    tpr = np.array([0.00000000, 0.57534247, 0.58219178, 0.60730594, 0.60730594, 0.61415525,
0.62100457, 0.63013699, 0.64383562, 0.65296804, 0.65753425, 0.66438356,
0.66894977, 0.66894977, 0.67351598, 0.70319635, 0.70776256, 0.72602740,
0.72602740, 0.73744292, 0.73744292, 0.73972603, 0.74429224, 0.74657534,
0.75114155, 0.75799087, 0.76255708, 0.76712329, 0.76712329, 0.76940639,
0.76940639, 0.77397260, 0.77397260, 0.77625571, 0.78082192, 0.79223744,
0.79680365, 0.79680365, 0.79908676, 0.79908676, 0.80136986, 0.80136986,
0.81050228, 0.81050228, 0.81506849, 0.81963470, 0.82191781, 0.82191781,
0.82420091, 0.82876712, 0.83561644, 0.84018265, 0.84018265, 0.84474886,
0.84703196, 0.84931507, 0.84931507, 0.85159817, 0.85159817, 0.85616438,
0.85616438, 0.85844749, 0.85844749, 0.86301370, 0.86301370, 0.86757991,
0.86757991, 0.86986301, 0.87214612, 0.87214612, 0.87671233, 0.87671233,
0.88127854, 0.88127854, 0.88584475, 0.88584475, 0.88584475, 0.89041096,
0.89041096, 0.90410959, 0.90410959, 0.90867580, 0.90867580, 0.91095890,
0.91095890, 0.91780822, 0.91780822, 0.92237443, 0.92237443, 0.92465753,
0.92465753, 0.92694064, 0.92694064, 0.93150685, 0.93150685, 0.93378995,
0.93378995, 0.93607306, 0.94292237, 0.94292237, 0.94520548, 0.94520548,
0.94520548, 0.94520548, 0.94977169, 0.94977169, 0.94977169, 0.94977169,
0.95205479, 0.95205479, 0.95662100, 0.95890411, 0.95890411, 0.95890411,
0.96118721, 0.96575342, 0.96575342, 0.96575342, 0.96803653, 0.96803653,
0.96803653, 0.96803653, 0.97031963, 0.97031963, 0.97260274, 0.97260274,
0.97716895, 0.97945205, 0.97945205, 0.98173516, 0.98173516, 0.98401826,
0.98401826, 0.98630137, 0.98630137, 0.98630137, 0.98630137, 0.98858447,
0.98858447, 0.98858447, 0.98858447, 0.98858447, 0.98858447, 0.98858447,
0.98858447, 0.98858447, 0.98858447, 0.98858447, 0.98858447, 0.99086758,
0.99086758, 0.99086758, 0.99086758, 0.99086758, 0.99086758, 0.99315068,
0.99315068, 0.99543379, 0.99543379, 0.99543379, 0.99543379, 0.99543379,
0.99543379, 0.99543379, 0.99543379, 0.99543379, 0.99543379, 0.99543379,
0.99543379, 0.99543379, 0.99543379, 0.99543379, 0.99543379, 0.99543379,
1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000,
1.00000000, 1.00000000
])

    auc = 0.9679956138197411

    result_table = result_table.append(
        {"classifiers": "DCF", "fpr": fpr, "tpr": tpr, "auc": auc}, ignore_index=True
    )

#     BERT_fpr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00105263, 0.00105263, 0.01578947, 0.01578947, 0.02631579, 0.02631579, 0.05052632, 0.05052632, 0.13157895, 0.13368421, 0.15157895, 0.15157895, 0.15789474, 0.15789474, 0.22, 0.22, 0.22315789, 0.22526316, 0.24526316, 0.24736842, 0.37368421, 0.37578947, 0.52, 0.52210526, 0.65473684, 0.65684211, 0.67157895, 0.67368421, 0.68315789, 0.68526316, 0.72105263, 0.72315789, 0.72842105, 0.73052632, 0.74421053, 0.75157895, 0.79263158, 0.79684211, 0.79789474, 0.8, 0.81368421, 0.81789474, 0.85789474, 0.86, 0.89894737, 0.90105263, 0.95473684, 0.95684211, 0.96421053, 0.96947368, 1.0
# ])
#     BERT_tpr = np.array([0.0, 0.00613497, 0.01226994, 0.02453988, 0.03067485, 0.04294479, 0.04907975, 0.06134969, 0.07361963, 0.09202454, 0.11656442, 0.12269939, 0.13496933, 0.14110429, 0.16564417, 0.18404908, 0.20245399, 0.20858896, 0.2208589, 0.2392638, 0.25153374, 0.26380368, 0.28220859, 0.28834356, 0.31288344, 0.32515337, 0.34355828, 0.36809816, 0.38650307, 0.39877301, 0.40490798, 0.44171779, 0.4601227, 0.47239264, 0.49079755, 0.50306748, 0.51533742, 0.5398773, 0.61349693, 0.63803681, 0.68711656, 0.6993865, 0.91411043, 0.91411043, 0.96319018, 0.96319018, 0.96932515, 0.96932515, 0.97546012, 0.97546012, 0.98159509, 0.98159509, 0.98159509, 0.98159509, 0.98773006, 0.98773006, 0.99386503, 0.99386503, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
# ])
#     BERT_auc = 0.9961317403939296

    BERT_fpr = np.array([0.00000000, 0.01052632, 0.01052632, 0.01052632, 0.01578947, 0.01578947,
0.01754386, 0.01929825, 0.01929825, 0.02456140, 0.02456140, 0.02631579,
0.02631579, 0.02631579, 0.02631579, 0.02807018, 0.02982456, 0.02982456,
0.02982456, 0.03157895, 0.03333333, 0.03508772, 0.03508772, 0.03508772,
0.03508772, 0.03684211, 0.03859649, 0.03859649, 0.04035088, 0.04210526,
0.04210526, 0.04736842, 0.05087719, 0.05438596, 0.05438596, 0.05614035,
0.05789474, 0.05789474, 0.05964912, 0.06140351, 0.06140351, 0.06315789,
0.06491228, 0.06842105, 0.07192982, 0.07719298, 0.08070175, 0.08070175,
0.08245614, 0.08245614, 0.08596491, 0.08771930, 0.08947368, 0.09122807,
0.09298246, 0.09298246, 0.09473684, 0.09824561, 0.10175439, 0.10175439,
0.10350877, 0.10701754, 0.10701754, 0.11052632, 0.11578947, 0.11578947,
0.11578947, 0.11929825, 0.11929825, 0.12280702, 0.12280702, 0.12631579,
0.12807018, 0.12982456, 0.12982456, 0.13157895, 0.13508772, 0.14210526,
0.14912281, 0.15263158, 0.15614035, 0.16140351, 0.17192982, 0.17894737,
0.18245614, 0.18596491, 0.18947368, 0.19473684, 0.19824561, 0.19824561,
0.20000000, 0.20000000, 0.20175439, 0.20350877, 0.21228070, 0.21228070,
0.21403509, 0.22456140, 0.22456140, 0.22807018, 0.23157895, 0.24035088,
0.24385965, 0.24912281, 0.25438596, 0.25964912, 0.26140351, 0.26491228,
0.27192982, 0.28421053, 0.28596491, 0.29824561, 0.31403509, 0.32456140,
0.33859649, 0.35087719, 0.38771930, 0.44035088, 0.50701754, 0.64561404,
1.00000000])

    BERT_tpr = np.array([0.00000000, 0.53755869, 0.53990610, 0.59389671, 0.62206573, 0.62676056,
0.65727700, 0.67605634, 0.67840376, 0.69014085, 0.70657277, 0.72535211,
0.73943662, 0.74882629, 0.75117371, 0.76056338, 0.77230047, 0.77934272,
0.78169014, 0.78403756, 0.78403756, 0.78873239, 0.79342723, 0.80046948,
0.80516432, 0.80516432, 0.80985915, 0.81220657, 0.81220657, 0.81455399,
0.81690141, 0.81690141, 0.82394366, 0.82394366, 0.82629108, 0.83098592,
0.83098592, 0.83333333, 0.83333333, 0.83802817, 0.84272300, 0.84741784,
0.84741784, 0.85211268, 0.85211268, 0.85446009, 0.85446009, 0.85915493,
0.86619718, 0.87089202, 0.87089202, 0.87323944, 0.87323944, 0.87558685,
0.87558685, 0.88262911, 0.88497653, 0.88497653, 0.88497653, 0.89201878,
0.89201878, 0.89671362, 0.89906103, 0.90140845, 0.90140845, 0.90375587,
0.91079812, 0.91079812, 0.91549296, 0.92253521, 0.92723005, 0.92723005,
0.92957746, 0.92957746, 0.93192488, 0.93192488, 0.93192488, 0.93192488,
0.93661972, 0.94131455, 0.94131455, 0.94131455, 0.94131455, 0.94131455,
0.94131455, 0.94131455, 0.94131455, 0.94366197, 0.94366197, 0.94835681,
0.94835681, 0.95305164, 0.95774648, 0.96009390, 0.96009390, 0.96244131,
0.96244131, 0.96244131, 0.96478873, 0.96948357, 0.97183099, 0.97183099,
0.97183099, 0.97183099, 0.97417840, 0.97417840, 0.97652582, 0.98122066,
0.98122066, 0.98122066, 0.98122066, 0.98122066, 0.98122066, 0.98356808,
0.98356808, 0.98356808, 0.98356808, 0.98591549, 0.98826291, 0.99061033,
1.00000000])

    BERT_auc = 0.9589078329626883

    result_table = result_table.append(
        {"classifiers": "ACR", "fpr": BERT_fpr, "tpr": BERT_tpr, "auc": BERT_auc}, ignore_index=True
    )

 #    DCAR_fpr = np.array([0.0,         0.0,         0.0,         0.001002,   0.001002,   0.00501002,
 # 0.00501002, 0.00601202, 0.00601202, 0.00801603, 0.00801603, 0.01002004,
 # 0.01302605, 0.01603206, 0.01703407, 0.01903808, 0.02204409, 0.02705411,
 # 0.02905812, 0.03006012, 0.03306613, 0.03807615, 0.05811623, 0.09619238,
 # 1.0])
 #
 #    DCAR_tpr = np.array([0.0,0.92173913,0.93913043, 0.93913043, 0.94782609, 0.94782609,
 # 0.95652174, 0.95652174, 0.96521739, 0.96521739, 0.97391304, 0.97391304,
 # 0.97391304, 0.9826087,  0.9826087,  0.9826087,  0.9826087,  0.9826087,
 # 0.99130435, 0.99130435, 0.99130435, 1.0,1.0,1.0,
 # 1.0])
 #
 #    DCAR_auc = 0.9991461183236038
    DCAR_fpr = np.array([0.0, 0.00724638, 0.00724638, 0.00724638, 0.01086957, 0.01086957,
0.01449275, 0.01449275, 0.01811594, 0.01811594, 0.02173913, 0.02173913,
0.02898551, 0.02898551, 0.0326087, 0.0326087, 0.03623188, 0.03623188,
0.04347826, 0.04347826, 0.05434783, 0.05434783, 0.07246377, 0.07246377,
0.07608696, 0.07608696, 0.09782609, 0.09782609, 0.10507246, 0.11231884,
0.12318841, 0.12318841, 0.13768116, 0.13768116, 0.15217391, 0.15217391,
0.15942029, 0.15942029, 0.18478261, 0.18478261, 0.19927536, 0.2173913,
0.22463768, 0.23913043, 0.25, 0.25724638, 0.25724638, 0.31521739,
0.32246377, 0.32246377, 0.32608696, 1.0])

    DCAR_tpr = np.array([0.0, 0.77027027, 0.7972973, 0.80630631, 0.80630631, 0.81081081,
0.81081081, 0.81531532, 0.81531532, 0.81981982, 0.81981982, 0.85585586,
0.85585586, 0.88738739, 0.8963964, 0.91891892, 0.91891892, 0.92342342,
0.92342342, 0.93243243, 0.93243243, 0.93693694, 0.93693694, 0.94144144,
0.94144144, 0.94594595, 0.94594595, 0.95045045, 0.95045045, 0.95495495,
0.95495495, 0.95945946, 0.95945946, 0.96396396, 0.96396396, 0.96846847,
0.96846847, 0.97297297, 0.97297297, 0.97747748, 0.97747748, 0.97747748,
0.97747748, 0.97747748, 0.97747748, 0.97747748, 0.98198198, 0.98648649,
0.98648649, 0.99099099, 0.99099099, 1.0])

    DCAR_auc = 0.9783587935761848
    # Adding DCAR metrics to result_table
    result_table = result_table.append(
        {"classifiers": "DCAR", "fpr": DCAR_fpr, "tpr": DCAR_tpr, "auc": DCAR_auc},
        ignore_index=True,
    )

    # Set name of the classifiers as index labels
    result_table.set_index("classifiers", inplace=True)

    # Plot multiple ROC-Curve
    plot_ROC_all("ROC for ALL models", result_table)

df = load_data("ExAIS_SMS_SPAM_DATA2.csv")

data = df.drop(["Label"], axis=1)
targets = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.20, random_state=42)

docs_X_train = X_train["ProcessedMessage"]
docs_X_test = X_test["ProcessedMessage"]
max_sequence_length = get_max_input_length(docs_X_train)
padded_docs_train, padded_docs_test, tokenizer = encode_text(docs_X_train, docs_X_test)
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
embedding_matrix = create_embedding_matrix(tokenizer, embedding_dim)

params = {"filters": 64, "kernel_size": 3, "units": 64, "epochs": 10}
class_weights = None

docs_X_train = X_train["ProcessedMessage"]
docs_X_test = X_test["ProcessedMessage"]
docs_X_train, docs_y_train = balance_data(docs_X_train, y_train, input_type="text")

max_sequence_length = get_max_input_length(docs_X_train)
padded_docs_train, padded_docs_test, tokenizer = encode_text(docs_X_train, docs_X_test)
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
# Retrieve GloVe word embeddings
embedding_matrix = create_embedding_matrix(tokenizer, embedding_dim)

print("\n**************************Run DL classifiers*************************\n")
DL_models(
    padded_docs_train,
    padded_docs_test,
    docs_y_train,
    y_test,
    params=params,
    class_weights=class_weights,
    vocab_size=vocab_size,
    max_sequence_length=max_sequence_length,
    embedding_matrix=embedding_matrix,
    embedding_dim=embedding_dim
)
