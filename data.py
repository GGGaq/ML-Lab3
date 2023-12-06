import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# from kneed import KneeLocator

# https://www.kaggle.com/code/vamsimohanchalla/unsupervised-learning-k-means-and-medoids

class data_loader():
    def __init__(self) -> None:
        self.data = pd.read_csv('Country-data.csv')
        self.df_0 = None        
    
    def analysis(self):
        print(pd.read_csv('data_dictionary.csv'))
        '''There are 10 columns country, child_mortality, exports, health spending, imports, income per person, inflation rate, life expetancy of a new born child, total fertility rate and gdp per capita.
        Exports, Imports, inflation, gdpp give an estimate of economic information of the country.
        Child mortality, health, life expectancy of newborns and total fertility are related to social information of the country.
        Income is a parameter which is both social and economic information in it.
        We are required to find the socio-economic status of the countries based on these features.
        Can we find a pattern and cluster similar countries and categorize the countries which require aid from HELP International.'''

        print(self.data.info())
        print(self.data.describe().T)


    def draw_pic(self):
        self.data.hist(bins=30,figsize=(10,10));
        for i in range(1, 10) :
            fig, ax = plt.subplots(1, 2, figsize=(15, 2))
            plt.suptitle(self.data.columns[i], fontsize=20, fontweight='bold', color='navy')
            # Left Plot
            sns.boxplot(x=self.data.columns[i], data=self.data, ax=ax[0])
            # Right Plot
            sns.distplot(self.data[self.colmns[i]], ax=ax[1])
        sns.heatmap(data=self.data.iloc[:, 1:].corr(), annot=True, fmt=".2f", linewidth=0.75, cmap="Blues")
        plt.show()
    
    def normalization(self):
        self.data.drop(columns='country', inplace=True)
        scaler = MinMaxScaler().fit_transform(self.data)
        self.df_0 = pd.DataFrame(scaler, columns=self.data.columns)
    
    def myPCA(self):
        pca = PCA(n_components=9).fit(self.df_0)
        exp = pca.explained_variance_ratio_

        plt.plot(np.cumsum(exp), linewidth=2, marker = 'o', linestyle = '--')
        plt.title("PCA", fontsize=20)
        plt.xlabel('n_component')
        plt.ylabel('Cumulative explained Variance Ratio')
        plt.yticks(np.arange(0.55, 1.05, 0.05))
        plt.show()
        
        final_pca = IncrementalPCA(n_components=5).fit_transform(self.df_0)
        pc = np.transpose(final_pca)
        corrmat = np.corrcoef(pc)
        sns.heatmap(data=corrmat, annot=True, fmt=".2f", linewidth=0.75, cmap="Blues")
        plt.show()

        self.data = pd.DataFrame({
            'PC1':pc[0],
            'PC2':pc[1],
            'PC3':pc[2],
            'PC4':pc[3],
            'PC5':pc[4],
        })
        fig, ax = plt.subplots(figsize=(15,6))
        sns.boxplot(data=self.data)
        plt.show()



class evaluation():
    def __init__(self,X,labels) -> None:
        self.X = X
        self.labels = labels
        self.cal()

    def cal(self):
        # 计算轮廓系数
        silhouette_avg = silhouette_score(self.X, self.labels)
        print(f"Silhouette Coefficient: {silhouette_avg}")

        # 计算Calinski-Harabasz指数
        calinski_harabasz_score_val = calinski_harabasz_score(self.X, self.labels)
        print(f"Calinski-Harabasz Index: {calinski_harabasz_score_val}")

        # 计算Davies-Bouldin指数
        davies_bouldin_score_val = davies_bouldin_score(self.X, self.labels)
        print(f"Davies-Bouldin Index: {davies_bouldin_score_val}")