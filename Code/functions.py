import pandas as pd
import gzip
import pickle as pkl
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
from sklearn import decomposition, preprocessing
import seaborn as sns
from sklearn.manifold import TSNE

def calcul_acp_tsne(matrice, nb_axes=50, seed=123):
    pca = decomposition.PCA(n_components = nb_axes, random_state=seed)
    pca.fit(matrice)
    print('Cumulative variance explained by', nb_axes,' principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))
    X_projected = pca.transform(matrice)
    tsne = TSNE(random_state=seed).fit_transform(X_projected)


    #critere du coude
    plt.figure(figsize=(15, 5))
    #plt.subplot(2,4,1)
    plt.bar(np.arange(len(pca.explained_variance_ratio_))+1, pca.explained_variance_ratio_)
    plt.show()
    # projection des individus/CV
    X_projected = pca.transform(matrice)
    return X_projected, tsne


def representation_acp_tsne(tsne, acp, size_plot_tsne = (15,10), clusters = None, colors=None, plot_acp=False, seed=123):# Calcul des composantes principales
    X_projected = acp
    if plot_acp==True:
        plt.figure(figsize=(15, 10))
        plt.subplot(2,3,1)
        sns.scatterplot(x=X_projected[:, 0], y=X_projected[:, 1], hue=clusters, palette=colors)
        plt.xlabel("axe 0")
        plt.ylabel("axe 1")
        plt.subplot(2,3,2)
        sns.scatterplot(x=X_projected[:, 0], y=X_projected[:, 2], hue=clusters, palette=colors)
        plt.xlabel("axe 0")
        plt.ylabel("axe 2")
        plt.subplot(2,3,3)
        sns.scatterplot(x=X_projected[:, 1], y=X_projected[:, 2], hue=clusters, palette=colors)
        plt.xlabel("axe 1")
        plt.ylabel("axe 2")
        plt.subplot(2,3,4)
        sns.scatterplot(x=X_projected[:, 0], y=X_projected[:, 3], hue=clusters, palette=colors)
        plt.xlabel("axe 0")
        plt.ylabel("axe 3")
        plt.subplot(2,3,5)
        sns.scatterplot(x=X_projected[:, 1], y=X_projected[:, 3], hue=clusters, palette=colors)
        plt.xlabel("axe 1")
        plt.ylabel("axe 3")
        plt.subplot(2,3,6)
        sns.scatterplot(x=X_projected[:, 2], y=X_projected[:, 3], hue=clusters, palette=colors)
        plt.xlabel("axe 2")
        plt.ylabel("axe 3")
        plt.show()

    plt.figure(figsize=size_plot_tsne)
    sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=clusters, palette=colors, markers=False, edgecolors=None)
    plt.title("ACP + TSNE")
    if clusters!=None:
        uniq_classes = np.unique(clusters)
        for c in uniq_classes:
            cvs_id = [i for i in range(len(clusters)) if clusters[i]==c]
            xtext, ytext = np.median(tsne[cvs_id, :], axis=0)
            plt.text(xtext, ytext, str(c), fontsize=24)
    plt.show()



