import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mplt



def main():
    df = pd.read_csv("data.csv")
    scaler = StandardScaler().fit(df)
    df = scaler.transform(df)
    pca = PCA(n_components=2).fit(df)
    df = pca.transform(df)

    db = DBSCAN(eps=0.55, min_samples=40).fit(df)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    
    test = pd.read_csv("test.csv")
    test = scaler.transform(test)
    test = pca.transform(test)
    test_labels = db.fit_predict(test)
    print(f"Test prediction label = {test_labels}")
    print(df[0])
    print(test[0])

#def show_plot(labels, df, core_samples_mask, n_clusters_):
#    # Plot results
#    unique_labels = set(labels)
#    colors = [mplt.cm.Spectral(each)
#              for each in np.linspace(0, 1, len(unique_labels))]
#    for k, col in zip(unique_labels, colors):
#        if k == -1:
#            # Black used for noise.
#            col = [0, 0, 0, 1]
#
#        class_member_mask = (labels == k)
#
#        xy = df[class_member_mask & core_samples_mask]
#        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                 markeredgecolor='k', markersize=14)
#
#        xy = df[class_member_mask & ~core_samples_mask]
#        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                 markeredgecolor='k', markersize=6)
#
#    plt.title('Estimated number of clusters: %d' % n_clusters_)
#    plt.show()

if __name__ == "__main__":
    main()