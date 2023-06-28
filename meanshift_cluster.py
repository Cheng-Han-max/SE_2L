import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import cycle
from datetime import datetime



def data_cluster(input_data, out_path, genome, quantile, n_samples):

    bandwidth = estimate_bandwidth(input_data, quantile=quantile, n_samples=n_samples)
    # print(bandwidth)

    ms = MeanShift(bandwidth=bandwidth, n_jobs=1)
    ms.fit(input_data)
    y_pre = ms.labels_
    labels = pd.DataFrame(y_pre)
    labels.value_counts()
    cluster_centers = ms.cluster_centers_


    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)

    quantity = pd.Series(ms.labels_).value_counts()


    for i in range(n_clusters):
        points_under_each_cluster = np.where(ms.labels_ == i)[0]

        np.savetxt('%s/%s/ID/cluster' + str(i) + '.csv'% out_path %genome, points_under_each_cluster, fmt='%d', delimiter=',')

    data = pd.DataFrame(input_data)
    data_label = pd.concat([data, labels], axis=1)
    data_label.columns = [f'vec{i}' for i in range(1, datas.shape[1] + 1)] + ['label']
    data_label.head()

    t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print('Start time：' + t1)

    tsne = TSNE()
    tsne.fit_transform(data_label)
    tsne = pd.DataFrame(tsne.embedding_, index=data_label.index)

    colors = ['#DDA0DD', '#696969', '#FF6347', '#008080', '#FF8C00', '#6A5ACD', '#D2B48C']

    colors_indexs = cycle('0123456')
    for k, color_index in zip(range(n_clusters), colors_indexs):
        color = colors[int(color_index)]
        d = tsne[data_label['label'] == k]
        plt.plot(d[0], d[1], marker='o', color=color, linestyle='')

    plt.title('Estimated number of clusters: %d' % n_clusters)
    plt.savefig('%s/%s/ID/%s_meanshift_cluster_result.jpg' % out_path %genome %genome)
    plt.show()

    t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print('end time：' + t2)

    return t1, t2


def extract_data(all_data, index):

    data_after_extract = []
    index = np.array(index)

    for i in range(len(index)):
        x = index[i]
        selected_idx = x[0]
        instances = all_data[selected_idx]
        data_after_extract.append(instances)

    return data_after_extract
