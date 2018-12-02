import numpy as np

class KMeans:

    def recomputeCentroids(self, cluster):
        centroids = []
        for key in cluster:
            center = np.mean(cluster[key])
            centroids += [center]
        return centroids

    def assignPoints(self):
        clus = {i : [] for i in range(self.k)}
        self.clusArray = []
        for i in self.X:
            mindist = float("inf")
            mindistk = 0 

            for j in range(self.k):
                dist = np.linalg.norm(i-self.centroids[j])
                if(mindist > dist):
                    mindist = dist
                    mindistk = j
            clus[mindistk] = clus[mindistk] + [i]
            self.clusArray.append(mindistk)
        return clus

    def __init__ (self, X, k):
        np.random.seed(1234)
        # 1. Select k points at random from X
        # 2. While centroid not stable, reassign and recompute
        self.clusArray = []
        self.m, self.d = X.shape
        self.X = X
        self.k = k

        self.centroids = np.empty([self.k, self.d])
        for t,i in enumerate(np.random.choice(self.m, k, replace=False)):
            self.centroids[t] = self.X[i]
        while True:
            self.clustering = KMeans.assignPoints(self)
            self.oldcentroids = self.centroids
            self.centroids = KMeans.recomputeCentroids(self, self.clustering)
            # print(str(self.oldcentroids))
            if(np.array_equal(np.array(self.oldcentroids), np.array(self.centroids))):
                print(self.centroids)
                break
    def giveCluster(self):
        return self.clustering
    
    def giveClusterArray(self):
        return self.clusArray