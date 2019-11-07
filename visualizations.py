#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:30:34 2019

@author: flatironschool
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as py
from plotly.graph_objs import Scatter, Layout
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix


def plot_personality_distribution(df, personality_cols):
    
    '''
    This function plots distribution of personalities.
    Parameter
    ----------
    df: Dataframe with personalities
    Returns
    --------
    Dataframe with categories
    '''
    
    fig, ax = plt.subplots(3, 3, figsize=(15, 17))
    for i, col in enumerate(personality_cols):
        sns.distplot(df[col], ax=ax[i//3, i%3])
        ax[i//3, i % 3].set_title(f'Distribution of {col}')
    fig.delaxes(ax[2, 1])
    fig.delaxes(ax[2, 2]);


def plot_countplot(df, drug_cols, item, xlabels):
    ''' 
    Encoding the categorical features as Age, Education and Gender
    Parameters
    -----------
    df: Dataframe with age, gender and education as a categorical variable
    Return
    -----------
    Dataframe with age and education as an ordinal variable, gender as dummy variable
    '''
    
    fig, ax = plt.subplots(5, 4, figsize=(18, 20))
    for i, drugs in enumerate(drug_cols):
        if item == 'class':
            sns.countplot(df[drugs], ax=ax[i//4, i%4])
            ax[i//4, i%4].set_title(f'Count of Each Class for {drugs}')
        else:
            sns.countplot(df[f'{drugs}_User'], ax=ax[i//4, i%4])
            ax[i//4, i%4].set_title(f'Count of Users vs. Non-users for {drugs}')
        ax[i//4, i%4].set_xticklabels(xlabels, rotation=20, ha='right')
    fig.delaxes(ax[4, 3])
    plt.tight_layout();


def plot_map(df, country):
    data = [dict(
            type='choropleth',
            locations=country,
            locationmode='country names',
            z=(df['Country'].value_counts().values),
            text=country,
            colorscale='portland',
            reversescale=True,
            )]
    layout = dict(
        title='A Map About Population of Drug Addicted in Each Country',
        geo=dict(showframe=False, showcoastlines=True,
                 projection=dict(type='Mercator'))
    )
    fig = dict(data=data, layout=layout)
    py.iplot(fig, validate=False, filename='world-map')

def plot_correlation(df, corrmat):
    plt.figure(figsize=(20, 20))
    sns.set(font_scale=1)
    hm = sns.heatmap(corrmat, cmap='RdYlGn', annot=True,
                     yticklabels=df.columns, xticklabels=df.columns)
    plt.xticks(fontsize=13, rotation=50)
    plt.yticks(fontsize=13)
    plt.title("Correlation B/W Different Features", fontsize=18)


def plot_user_count(drugs, drug_names):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    plt.subplots_adjust(wspace = 0.4)
    for i, drug in enumerate(drugs):
        sns.countplot(drug['User'], ax=ax[i])
        ax[i].set_title(f'Number of Users vs Non-Users of {drug_names[i]}\n',
                        fontsize=14)
    for x in ax:
        x.set_xticklabels(labels=['Non-user', 'User'], fontsize=14)
        x.set_ylabel('count', fontsize=14)
        x.set_xlabel('')
    sns.despine(left=False, bottom=False)
    plt.savefig('img/users_vs_nonusers.png', bbox_inches='tight');


def plot_personality(df, personality_cols, drug_name):
    fig = plt.figure(figsize=(16, 20))
    sns.set_palette(sns.light_palette((360, 90, 50), n_colors=7, input='husl'))
    plt.suptitle(f'{drug_name} consumption and personality scores', size=16)
    for row, personality in enumerate(personality_cols):
        plt.subplot(4, 2, row+1)
        sns.boxplot(x=df[drug_name], y=df[personality],
                    order=['CL0', 'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6'])
        plt.xlabel(' ')

    vc = df[drug_name].value_counts()/len(df)
    plt.subplot(4, 2, 8)
    sns.barplot(x=vc.index, y=vc)
    plt.ylabel('Percent of the total');


def plot_feature_dist(df, drug_name):
    fig, ax = plt.subplots(2, 2, figsize=(12, 13))
    plt.subplots_adjust(hspace=1.2)
    sns.set_palette(sns.light_palette((220, 90, 60), n_colors=2, input='husl'))
    for i, dem in enumerate(['Age', 'Education', 'Gender', 'Ethnicity']):
        sns.countplot(x = df[dem], hue=df[f'{drug_name}_User'], ax=ax[i//2, i%2])
        ax[i//2, i%2].set_title(f"{dem} Distribution for {drug_name}")
        ax[i//2, i%2].set_xticklabels(ax[i//2, i%2].get_xticklabels(),
                                     rotation=45, horizontalalignment='right')
    plt.tight_layout;


def plot_feat_imp(model, df, drug):
    plt.figure(figsize=(8, 8))
    if model == 'SVM':
        plt.barh(df['Feature'],
                 df['Absolute Coefficient'],
                 align='center')
        sns.despine(left=False, bottom=False)
        coefs = df['Coefficient'].apply(lambda x: round(x, 2))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=18)
        plt.xlabel('Feature importance', fontsize=18)
        plt.ylabel('Feature', fontsize=18)
        plt.title(f'Predicting {drug} usage\n', fontsize=18)
        for i, v in enumerate(coefs):
            plt.text(np.abs(v) + 0.007, i - 0.1, str(np.abs(v)), color='black', fontsize=18)
            if v > 0:
                plt.text(v - 0.01, i - 0.1, '+', color='black', fontsize=12, fontweight='bold')
            else:
                plt.text(np.abs(v) - 0.03, i - 0.1, '-', color='black', fontsize=18, fontweight='bold')
    else:
        n_features = X_train.shape[1]
        plt.barh(range(n_features),
                 model.best_estimator_.feature_importances_,
                 align='center')
        plt.yticks(np.arange(n_features), X_train.columns.values)
        plt.xlabel('Feature importance')
        plt.ylabel('Feature')
        plt.title(f'Feature importance in predicting {drug} usage');
    
    
def plot_roc_curve(models, model_names, X_test, X_test_scale, y_test, drug):
    plt.figure(figsize=(10, 8))
    for idx, model in enumerate(models):
        if (model_names[idx] == 'Logistic Regression') | (model_names[idx] == 'KNN') | (model_names[idx] == 'SVM'):
            auc_score = roc_auc_score(
                y_test, model.best_estimator_.predict_proba(X_test_scale)[:, 1])
            fpr, tpr, thresholds = roc_curve(
                y_test, model.best_estimator_.predict_proba(X_test_scale)[:, 1])
        else:
            auc_score = roc_auc_score(
                y_test, model.best_estimator_.predict_proba(X_test)[:, 1])
            fpr, tpr, thresholds = roc_curve(
                y_test, model.best_estimator_.predict_proba(X_test)[:, 1])
        # fpr = false positive, #tpr = true positive
        plt.plot(fpr, tpr,
                 label=f'{model_names[idx]} (auc = %0.2f)' % auc_score,
                 lw=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0, 0, 1], [0, 1, 1], 'k--', color='red')
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(f'ROC Curves for {drug} Consumption', fontsize=20)
    plt.xlim(-0.005, 1.005)
    plt.ylim(-0.005, 1.005)
    plt.legend(loc='best', fontsize=12, frameon=False);

    
def plot_confusion_matrix(y_test, X_test, model, drug):
    matrix = confusion_matrix(y_test, model.best_estimator_.predict(X_test))
    matrix = matrix / matrix.astype(np.float).sum()
    df_matrix = pd.DataFrame(matrix,
                             columns=['predicted non-user', 'predicted user'],
                             index=['actual non-user', 'actual user'])
    plt.figure(figsize=(8, 5))
    col = sns.color_palette("PuBu", 10)
    sns.heatmap(df_matrix, annot=True, annot_kws={"size": 40},
                fmt='.0%', cmap=col, cbar=False)
    plt.title(f'Confusion Matrix for {drug}\n', fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    plt.xticks(fontsize=20);