"""
Learn the network's normal behavior by clustering.
Compare new frames to created clusters to find anomalies.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d, Delaunay
from os import path
import subprocess
from tcp_latency import measure_latency
import platform
import random
import csv

ip = '172.16.3.203'


def main():
    scaler, pca, cluster = learnNormalState()
    test_frame = createTestFrame(ip) #pd.read_csv("resources/test.csv")
    frameOK = testNewFrame(test_frame, scaler, pca, cluster)
    

def testNewFrame(df, scaler, pca, cluster):
    df_scaled = scaler.transform(df)
    test_point_pca = pca.transform(df_scaled)[0]

    for c in cluster:
        cluster_array = np.array(c)
        hull = Delaunay(cluster_array)
        if hull.find_simplex(test_point_pca) >= 0:
            print(f"OK: {df}")
            return True
    print(f"ANOMALY:\n {df}")        
    return False
      

def learnNormalState():
    #df = pd.read_csv("resources/data.csv")
     
    if not path.isfile("resources/real_data.csv") or path.getsize("resources/real_data.csv") == 0:
        print("Creating trainigs data...")
        with open("resources/real_data.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["priority", "latency", "jitter", "framelength"])
            for i in range(0,1000):
                latency, jitter, framelength = get_tcp_latency(ip)
                writer.writerow([1, latency, jitter, framelength])
                print(f"#{i}")
    
    # Create cluster on collected data
    df = pd.read_csv("resources/real_data.csv")
    scaler = StandardScaler().fit(df)
    df_scaled = scaler.transform(df)
    pca = PCA(n_components=2).fit(df_scaled)
    train_data = pca.transform(df_scaled)

    db = DBSCAN(eps=0.6, min_samples=10).fit(train_data)
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    cluster = [[] for i in range(0, n_clusters_)]
    for i in range(0, len(db.labels_)):
        label = db.labels_[i]
        if label != -1:
            cluster[label].append(train_data[i])

    # Plot all Points
    point_array = np.array(train_data)
    plt.plot(point_array[:,0], point_array[:,1], 'o')

    # Plot convex hull
    for c in cluster:
        cluster_array = np.array(c)
        hull = ConvexHull(cluster_array)
        #plt.plot(cluster_array[:,0], cluster_array[:,1], 'o')
        for simplex in hull.simplices:
            plt.plot(cluster_array[simplex, 0], cluster_array[simplex, 1], 'k-', color='r')
    plt.show()
    return scaler, pca, cluster

def createTestFrame(ip):
    latency, jitter, framelength = sendPing(ip)
    return pd.DataFrame([[1, latency, jitter, 64]], columns=['priority', 'latency', 'jitter', 'framelength'], dtype = float)

def sendPing(ip):
    # TODO: Use fping for faster ping
    packet_size = random.randint(56, 248) # + 8 Byte icmp header
    latency = []
    jitter = []
    try:
        output = subprocess.check_output("ping -{} 2 {}".format('n' if platform.system().lower(
        ) == "windows" else 'c', ip ), shell=True, universal_newlines=True)
        
        for response in output.splitlines():
            pos_of_time = response.find("time=")
            if pos_of_time != -1:
                latency.append(float(response[pos_of_time + 5 : pos_of_time + 10]))
        for i in range(0, len(latency)-1):
            jitter.append(abs(latency[i] - latency[i+1]))
        
        latency.pop(0)
        return latency[0], jitter[0], packet_size + 8
    except:
        print(f"Unable to reach {ip}")
        quit()

def get_tcp_latency(ip):
    print(f"ping {ip}")
    latency = measure_latency(host=ip, runs=2, timeout=0.005, wait=0.05)
    print(latency)
    jitter = abs(latency[0] - latency[1])
    return latency [1], jitter, 64

if __name__ == "__main__":
    main()