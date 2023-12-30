#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn import metrics
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from treeinterpreter import treeinterpreter as ti
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

DRUG_NAME = "Vancomycin"
DRUG_DATA = pd.read_csv("/home/racheal/Downloads/drug_response.csv")
GENE_DATA= pd.read_csv('/home/racheal/Downloads/gene_present.csv')

def main():
    drug_data()
    gene_data()
    ML_data()
    logistic_Regression()

def drug_data():
    #drug_response = pd.read_csv("/home/racheal/Downloads/drug_response.csv")
    drug_response = DRUG_DATA
    drug_response.head()
    drug_response.drop(drug_response.columns[0], axis=1)
    for x in drug_response.iloc[:, 2:]:
        print(drug_response[x].value_counts())
    
    index_loc = drug_response.columns.get_loc(DRUG_NAME) # Vancomycin = -2, 
    drug_response = drug_response.iloc[:,[1, index_loc]]
    drug_response


    # create a Boolean mask for the rows to remove
    remove_label = ['N', 'I']
    for i in remove_label:
        mask = drug_response[DRUG_NAME] == i
    #mask = drug_response[DRUG_NAME] == 'N'
    #mask = streptomycin_response['Streptomycin'] == 'I'
    # select all rows except the ones that contain 'Coca Cola'
        drug_response = drug_response[~mask]

    # print the resulting DataFrame
    print(drug_response)


    ## Encoding the drugs bases on drug response label as O or 1
    drug_response[DRUG_NAME].value_counts()
    cleanup_nums = {DRUG_NAME: {"S": 0, "R": 1}}
    drug_response = drug_response.replace(cleanup_nums)
    return drug_response


def gene_data():
    drug_response = drug_data()
    GENE_DATA.drop(GENE_DATA.columns[0], axis=1)
    gene_encoded = pd.merge(drug_response, GENE_DATA, on = "Gene")
    gene_encoded = gene_encoded.drop(gene_encoded.columns[2], axis=1)
    #check for missing values
    gene_encoded[gene_encoded.isna().any(axis=1)]
    print (gene_encoded)
    return gene_encoded


def ML_data():
    gene_encoded = gene_data()
    train_X = gene_encoded.drop(gene_encoded.columns[0:2], axis =1)
    train_X
    train_Y = gene_encoded[DRUG_NAME]
    train_Y
    return train_X, train_Y

def logistic_Regression():
    train_X, train_Y = ML_data()
    from sklearn.linear_model import LogisticRegression
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    coefficients = model.coef_[0]
    coefficients

    coef = model.coef_
    coef
    # feature_importance = pd.DataFrame({'Feature': train_X.columns, 'Importance': np.abs(coef)})
    # feature_importance = feature_importance.sort_values('Importance', ascending=True)

    feature_importance = pd.DataFrame({'Feature': train_X.columns, 'Importance': np.abs(coefficients)})
    feature_importance = feature_importance.sort_values('Importance', ascending=True)

    feature_importance.nlargest(10, "Importance").plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6), title = 'Important Gene in {} Resistance'.format(DRUG_NAME))
    plt.savefig("/home/racheal/Downloads/plot/{}_plot.png".format(DRUG_NAME))

    pd.set_option('display.max_columns', None)

    feature_importance

    model.classes_

    model.score(X_test, y_test)

    confusion_matrix(y_test, model.predict(X_test))


    print(classification_report(y_test, model.predict(X_test)))



    pred_prob1 = model.predict_proba(X_test)

    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
    # roc curve for tpr = fpr 
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


    auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
    print(auc_score1)

    plt.style.available

    # matplotlib
    plt.style.use('seaborn-v0_8')

    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
    #plt.plot(fpr2, tpr2, linestyle='--',color='green', label='KNN')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('{}     ROC curve'.format(DRUG_NAME))
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig("/home/racheal/Downloads/plot/{}_ROC.png".format(DRUG_NAME), dpi=300)
    #plt.savefig('ROC',dpi=300)
    plt.show()

    cm = confusion_matrix(y_test, model.predict(X_test))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Susceptible", "Resistant"])
    cm_display.plot()
    plt.tick_params(axis=u'both', which=u'both',length=0)
    plt.grid(False)
    #plt.show()
    plt.savefig("/home/racheal/Downloads/plot/{}_matrix.png".format(DRUG_NAME))

main()

# # cm = confusion_matrix(y_test, model.predict(X_test))
# # fig, ax = plt.subplots(figsize=(8, 8))
# # ax.imshow(cm)
# # ax.grid(False)
# # ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
# # ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
# # ax.set_ylim(1.5, -0.5)
# # for i in range(2):
# #     for j in range(2):
# #         ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
# # plt.show()
# # plt.savefig("/home/racheal/Downloads/plot/vancomycin_matrix.png")



# # #PCA
# # pca_BSI = PCA(n_components=2)
# # principalComponents_BSI = pca_BSI.fit_transform(train_X)
# # principal_BSI_Df = pd.DataFrame(data = principalComponents_BSI
# #              , columns = ['principal component 1', 'principal component 2'])
# # principal_BSI_Df.tail()
# # print('Explained variation per principal component: {}'.format(pca_BSI.explained_variance_ratio_))

# # plt.figure(figsize=(10,10))
# # plt.xticks(fontsize=12)
# # plt.yticks(fontsize=14)
# # plt.xlabel('Principal Component - 1',fontsize=20)
# # plt.ylabel('Principal Component - 2',fontsize=20)
# # plt.title("Principal Component Analysis of BSI Dataset",fontsize=20)
# # targets = ["S", "R"]
# # colors = ['r', 'g']
# # for target, color in zip(targets,colors):
# #     indicesToKeep = gene_vancomycin_encoded['Vancomycin'] == target
# #     plt.scatter(principal_BSI_Df.loc[indicesToKeep, 'principal component 1']
# #                , principal_BSI_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

# # plt.legend(targets,prop={'size': 15})



# # randomly split the data
# # train_x, test_x, train_y, test_y = train_test_split(train_X, train_Y,test_size=0.25,random_state=0)



# # # shape of train and test splits
# # train_x.shape, test_x.shape, train_y.shape, test_y.shape




# # #create an object of the RandomForestRegressor
# # model_RFR = RandomForestRegressor(max_depth=10)

# # # fit the model with the training data
# # model_RFR.fit(train_x, train_y)

# # # predict the target on train and test data
# # predict_train = model_RFR.predict(train_x)
# # predict_test = model_RFR.predict(test_x)

# # # Root Mean Squared Error on train and test data
# # print('RMSE on train data: ', mean_squared_error(train_y, predict_train)**(0.5))
# # print('RMSE on test data: ',  mean_squared_error(test_y, predict_test)**(0.5))




# # # plot the 7 most important features 
# # plt.figure(figsize=(10,7))
# # feat_importances = pd.Series(model_RFR.feature_importances_, index = train_x.columns)
# # feat_importances.nlargest(30).plot(kind='barh');



# # prediction, bias, contributions = ti.predict(model_RFR, train_X)


# # for i in range(len(train_X)):
# #     print ("train_X", i)
# #     print ("Feature contributions:")
# #     for c, feature in sorted(zip(contributions[i], 
# #                                  train_X.columns), 
# #                              key=lambda x: -abs(x[0])):
# #         print (feature, round(c, 2))
# #     print ("-"*20)



# # pd.set_option('display.max_rows', None)
# # import scipy.stats as stats
# # from scipy.stats import pearsonr
# # train_X.corrwith(gene_vancomycin_encoded["Vancomycin"])



# # target_col_name = 'Vancomycin'
# # ID = "Gene"
# # feature_target_corr = {}
# # for col in gene_vancomycin_encoded:
# #     if target_col_name != col and ID != col:
# #         feature_target_corr[col + '_' + target_col_name] = \
# #             pearsonr(gene_vancomycin_encoded[col], gene_vancomycin_encoded[target_col_name])[0]
# # print("Feature-Target Correlations") 
# # print(feature_target_corr)



# # import sys
# # np.set_printoptions(threshold=sys.maxsize)
# # coorr = train_X.apply(lambda x: x.corr(train_Y))
# # print(coorr)

# # gene_vancomycin_encoded.Vancomycin.value_counts()



# # ### view bar chart
# # import plotly.express as px
# # vancomycin_counts = gene_vancomycin_encoded.Vancomycin.value_counts()
# # fig = px.bar(x=vancomycin_counts.index, y=vancomycin_counts.values)
# # fig.show()



# # ax = sns.countplot(x='Vancomycin', data=gene_vancomycin_encoded)
# # plt.show()


# drug_response=drug_response.drop(drug_response.columns[0], axis=1)



# drug_response



# for x in drug_response.iloc[:, 1:]:
#     print(drug_response[x].value_counts())


# drug_response=drug_response.drop(drug_response[['5-Fluorocytosine','Amikacin', 'Amoxicillin','Amoxicillin-clavulanate', 'Amphotericin B' ]], axis=1)


# for x in drug_response.iloc[:, 2:]:
#     value1 = ['S']
#     value2= ['R']
#     if value1 and value2 in drug_response[x].values: #and drug_response[x].value_counts()['N'] > 600:
#         print(drug_response[x].value_counts())
#         #drug_response=drug_response.drop(drug_response[x], axis=1)


# # print (drug_response["Vancomycin"].value_counts()['R'])
# # drug_response.iloc[:, 2:]
# drug_response


# drug_response.columns.get_loc('Gentamicin')


# #drug_response.columns.get_loc("Tetracycline")
# #drug_response.columns[56]
# gentamincin_response = drug_response.iloc[:,[0, 34]]
# gentamincin_response
# gentamincin_response.Gentamicin.value_counts()


# # create a Boolean mask for the rows to remove
# mask = gentamincin_response['Gentamicin'] == 'N'
# mask = gentamincin_response['Gentamicin'] == 'I' 

# #if mask == 'N' or mask == 'I':
#     # select all rows except the ones that contain 'N'
# gentamincin_response = gentamincin_response[~mask]
# # print the resulting DataFrame
# print(gentamincin_response)


# print(gentamincin_response.Gentamicin.value_counts())



# ax = sns.countplot(x='Gentamicin', data=gentamincin_response)
# plt.show()



# cleanup_nums = {"Gentamicin":     {"S": 0, "R": 1}}


# gentamincin_response = gentamincin_response.replace(cleanup_nums)
# gentamincin_response.head()


# ax = sns.countplot(x='Gentamicin', data=gentamincin_response)
# plt.show()


# gene_present= pd.read_csv('/home/racheal/Downloads/gene_present.csv')


# gene_present.drop(gene_present.columns[0], axis=1)


# # merge the drug and gene data
# gene_gentamicin_encoded = pd.merge(gentamincin_response, gene_present, on = "Gene")
# gene_gentamicin_encoded = gene_gentamicin_encoded.drop(gene_gentamicin_encoded.columns[2], axis=1)



# gene_gentamicin_encoded



# #check for missing values
# gene_gentamicin_encoded[gene_gentamicin_encoded.isna().any(axis=1)]


# train_X = gene_gentamicin_encoded.drop(gene_gentamicin_encoded.columns[0:2], axis =1)



# train_X



# train_Y = gene_gentamicin_encoded['Gentamicin']



# train_Y




# from sklearn.linear_model import LogisticRegression
# X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.3, random_state=42)
# model = LogisticRegression()
# model.fit(X_train, y_train)



# coefficients = model.coef_[0]
# coefficients



# feature_importance = pd.DataFrame({'Feature': train_X.columns, 'Importance': np.abs(coefficients)})
# feature_importance = feature_importance.sort_values('Importance', ascending=True)




# feature_importance.nlargest(15, "Importance").plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6), title='Important Genes for Gentamicin Resistance')
# plt.savefig("/home/racheal/Downloads/plot/gentamicin_plot")


# from sklearn.metrics import confusion_matrix, classification_report
# confusion_matrix(y_test, model.predict(X_test))
# print(classification_report(y_test, model.predict(X_test)))



# from sklearn.metrics import roc_curve, roc_auc_score
# pred_prob1 = model.predict_proba(X_test)
# # roc curve for models
# fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
# # roc curve for tpr = fpr 
# random_probs = [0 for i in range(len(y_test))]
# p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# # matplotlib
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-v0_8')

# # plot roc curves
# plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
# #plt.plot(fpr2, tpr2, linestyle='--',color='green', label='KNN')
# plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# # title
# plt.title('Gentamicin     ROC curve')
# # x label
# plt.xlabel('False Positive Rate')
# # y label
# plt.ylabel('True Positive rate')
# plt.legend(loc='best')
# plt.savefig("/home/racheal/Downloads/plot/gentamicin_ROC.png", dpi=300)
# #plt.savefig('ROC',dpi=300)
# plt.show()



# cm = confusion_matrix(y_test, model.predict(X_test))
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Susceptible", "Resistant"])
# cm_display.plot()
# plt.tick_params(axis=u'both', which=u'both',length=0)
# plt.grid(False)
# #plt.show()
# plt.savefig("/home/racheal/Downloads/plot/gentamicin_matrix.png")




