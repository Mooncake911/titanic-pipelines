from IPython.display import display
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.inspection import permutation_importance


def draw_confusion_matrix(y_test, y_pred, name):
    """ Compute confusion matrix """
    conf_matrix = confusion_matrix(y_test, y_pred)
    cnf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    true_class_names = ['True', 'False']
    predicted_class_names = ['Predicted True', 'Predicted False']
    df_cnf_matrix = pd.DataFrame(conf_matrix, index=true_class_names, columns=predicted_class_names)
    df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, index=true_class_names, columns=predicted_class_names)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(name)
    sns.heatmap(df_cnf_matrix, annot=True, ax=axes[0], fmt='d')
    axes[0].title.set_text('Perceptron: values')
    sns.heatmap(df_cnf_matrix_percent, annot=True, ax=axes[1])
    axes[1].title.set_text('Perceptron: %')


def check_models(df, pred_column, ratio=0.7):
    # Split dataframe
    length = int(len(df) * ratio)
    train = df[:length]
    test = df[length:]

    x_train = train.drop(pred_column, axis=1)
    y_train = train[pred_column]
    x_test = test.drop(pred_column, axis=1)
    y_test = test[pred_column]

    print(f'\n{x_train.shape} - len of train data\n'
          f'{x_test.shape} - len of test data')

    y_pred = []
    models = {}
    acc = []
    coefficients = []

    # Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    models['LogisticRegression'] = model
    coff = model.coef_[0]
    y_pred.append(model.predict(x_test))
    acc.append(round(model.score(x_train, y_train) * 100, 2))
    coefficients.append(coff)

    # Support Vector Machine
    model = SVC(kernel='linear')
    model.fit(x_train, y_train)
    models['SVC'] = model
    coff = model.coef_[0]
    y_pred.append(model.predict(x_test))
    acc.append(round(model.score(x_train, y_train) * 100, 2))
    coefficients.append(coff)

    # Lining SVM
    model = LinearSVC(dual=True)
    model.fit(x_train, y_train)
    models['LiningSVC'] = model
    coff = model.coef_[0]
    y_pred.append(model.predict(x_test))
    acc.append(round(model.score(x_train, y_train) * 100, 2))
    coefficients.append(coff)

    # Perceptron
    model = Perceptron(max_iter=5, tol=None)
    model.fit(x_train, y_train)
    models['Perceptron'] = model
    coff = model.coef_[0]
    y_pred.append(model.predict(x_test))
    acc.append(round(model.score(x_train, y_train) * 100, 2))
    coefficients.append(coff)

    # SGD
    model = SGDClassifier(max_iter=5, tol=None)
    model.fit(x_train, y_train)
    models['SGDClassifier'] = model
    coff = model.coef_[0]
    y_pred.append(model.predict(x_test))
    acc.append(round(model.score(x_train, y_train) * 100, 2))
    coefficients.append(coff)

    # K-NeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)
    models['KNeighborsClassifier'] = model
    coff = permutation_importance(model, x_train, y_train, n_repeats=10, random_state=42)
    y_pred.append(model.predict(x_test))
    acc.append(round(model.score(x_train, y_train) * 100, 2))
    coefficients.append(coff)

    # Decision TreeClassifier
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    models['DecisionTreeClassifier'] = model
    coff = permutation_importance(model, x_train, y_train, n_repeats=10, random_state=42)
    y_pred.append(model.predict(x_test))
    acc.append(round(model.score(x_train, y_train) * 100, 2))
    coefficients.append(coff)

    # Random Forest
    model = RandomForestClassifier(n_estimators=100)
    model.fit(x_train, y_train)
    models['RandomForestClassifier'] = model
    coff = permutation_importance(model, x_train, y_train, n_repeats=10, random_state=42)
    y_pred.append(model.predict(x_test))
    acc.append(round(model.score(x_train, y_train) * 100, 2))
    coefficients.append(coff)

    # Gaussian Naive Bayes
    model = GaussianNB()
    model.fit(x_train, y_train)
    models['GaussianNB'] = model
    y_pred.append(model.predict(x_test))
    acc.append(round(model.score(x_train, y_train) * 100, 2))
    coefficients.append([])

    metrics_data = []
    feature_data = []
    for i, methods_names in enumerate(models.keys()):
        draw_confusion_matrix(y_test=y_test, y_pred=y_pred[i], name=methods_names)

        # Compute score
        accuracy = accuracy_score(y_test, y_pred[i])
        precision = precision_score(y_test, y_pred[i], zero_division=1)
        recall = recall_score(y_test, y_pred[i])
        f1 = f1_score(y_test, y_pred[i])
        mse = mean_squared_error(y_test, y_pred[i])
        roc_auc = roc_auc_score(y_test, y_pred[i])

        fpr, tpr, thresholds = roc_curve(y_test, y_pred[i])
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred[i]).ravel()
        specificity = tn / (tn + fp)

        metrics_data.append([methods_names, accuracy, precision, recall, f1, mse, specificity, roc_auc])
        feature_data.append([methods_names, *coefficients[i]])

    metrics_df = pd.DataFrame(metrics_data, columns=['Method_Name', 'Accuracy', 'Precision', 'Recall',
                                                     'F1', 'MSE', 'Specificity', 'Roc_Auc'])
    feature_df = pd.DataFrame(feature_data, columns=['Method_Name', *df.columns[:-1]])

    display(metrics_df)
    display(feature_df)

    return models, metrics_df, feature_df
