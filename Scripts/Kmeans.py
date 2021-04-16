# -*- coding: utf-8 -*-
"""
# PYTHON: 3.7.8
# DATE: 03-12-2020
# CONTACT: sagar.datascientist@gmail.com
"""
# Basic imports
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3425)

#########################################################################################################################################
# This class kmeans run all the different stages of k-means clustering algoritms.
#########################################################################################################################################

class Kmeans():
    def __init__(self,norm,k_cluster_no,no_of_dimensions,no_of_iteration):
        """
        This method set norm type, cluster no,no of dimensions and no of iterations.

        :param norm:
        :param k_cluster_no:
        :param no_of_dimensions:
        :param no_of_iteration:
        """
        self.norm = norm
        self.k_cluster_no = k_cluster_no
        self.no_of_dimensions = no_of_dimensions
        self.no_of_iteration=no_of_iteration


    def objective_function(self,data,centroid):
        """
        This method calculate norm based on user specified norm type.
        :param data:
        :param centroid:
        :return:
        """
        return np.linalg.norm(data - centroid, ord=self.norm)

    def recalculate_clusters(self,X, centroids, k):
        """  This method recalculates the clusters
        :param X:
        :param centroids:
        :param k:
        :return:
        """
        # Initiate empty clusters
        clusters = {}
        # Set the range for value of k (number of centroids)
        for i in range(k):
            clusters[i] = []
        # Setting the plot points using dataframe (X) and the vector norm (magnitude/length)
        cluster_Id_List = []
        for data in X:
            # Set up list of euclidian distance and iterate through
            euc_dist = []
            for j in range(k):
                cent=centroids[j]
                distance=self.objective_function(data,cent)
                euc_dist.append(distance)
            # Append the cluster of data to the dictionary
            cluster_id = euc_dist.index(min(euc_dist))
            cluster_Id_List.append(cluster_id)
            clusters[cluster_id].append(data)
        return clusters, cluster_Id_List

    def convert_N_Clusters_3(self,centroids,y_predict,each_class_mean):
        """
        This method convert N cluster to 3 clusters by calculating euclidian distance to each class label mean of training data and assing closet class label to it.

        :param centroids:
        :param y_predict:
        :param each_class_mean:
        :return:
        """
        map_N_classes_actual={}
        for i in range(self.k_cluster_no):
            map_N_classes_actual[i]=""

        for pre_id,pre_centroid in centroids.items():
            euc_dist=[]
            for classid,actual_class_mean in each_class_mean.items():
                 distance=self.objective_function(pre_centroid,actual_class_mean)
                 euc_dist.append(distance)
            # Assgin cluster id based on min euc distance
            cluster_id = euc_dist.index(min(euc_dist))
            #add inside mapping dict
            map_N_classes_actual[pre_id] =cluster_id

        mapped_y_predict=[]
        for each_pred in y_predict:
            id=int(each_pred)
            if(map_N_classes_actual.__contains__(id)):
                acutal_class_id=map_N_classes_actual[id]
                mapped_y_predict.append(acutal_class_id)
            else:
                print("Error can't map cluster id ")

        return  mapped_y_predict

    def recalculate_centroids(self,centroids, clusters, k):
        """ Recalculates the centroid position based on the plot
        :param centroids:
        :param clusters:
        :param k:
        :return:
        """
        for i in range(k):
            # Finds the average of the cluster at given index
            centroids[i] = np.average(clusters[i], axis=0)
        return centroids

    def centroid_Init_PlusMethod(self,df):
        """
        This method used for centroid initialization and  create cluster centroids like the k-means++ algorithm.

        :param ds:
        :return:
        """
        centroids = {}
        centroids[0]=df[0]

        for k in range(1, self.k_cluster_no):
            dist_sq = np.array([min([np.inner(c - x, c - x) for key,c in centroids.items()]) for x in df])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break
            centroids[k]=df[i]
        return centroids

    def centroid_Init_Random(self,df):
        """
        Create random cluster centroids.
        :param df:
        :return:
        """
        centroids = {}
        m = np.shape(df)[0]

        for index in range(self.k_cluster_no):
            r = np.random.randint(0, m - 1)
            centroids[index]=(df[r])

        return centroids

    def fit(self,X):
        """ Calculates full k_means_clustering algorithm
        :param X:
        :return:
        """
        #Option 1 Define centroids like k ++ method- efficent way to define it and algorithm converging and getting lower error.
        centroids=self.centroid_Init_PlusMethod(X)

        # Option 2 Define centroids based on data points or define randomly or may be use naive_sharding
       # centroids=self.centroid_Init_Random(X)

        #   centroids = {}
        # for i in range(self.k_cluster_no):
        #     # Sets up the centroids based on the data
        #     centroids[i] = X[i]

        #Outputs the recalculated clusters and centroids
        # clusters=[]
        cluster_id_list=[]
        for i in range(self.no_of_iteration):
            clusters, cluster_id_list = self.recalculate_clusters(X, centroids,self.k_cluster_no)
            centroids = self.recalculate_centroids(centroids, clusters, self.k_cluster_no)

        return centroids, cluster_id_list

    def predict(self,X, centroids, k):
        """
        This method calculate new clusters based on freezed centroids calculated through training process.
        :param X:
        :param centroids:
        :param k:
        :return:
        """
        clusters, cluster_id_list=self.recalculate_clusters(X,centroids,k)
        centroids = self.recalculate_centroids(centroids, clusters, k)

        return centroids, cluster_id_list

    def define_OptimalClusters(self,X,K_max):
        """
        Define optimal no of clusters using elbow method.
        Implement the Silhouette Method for it.

        :param X:
        :param K_max:
        """
        sse=[]
        K_max=K_max+1
        for K in range(1, K_max):
            self.k_cluster_no=K
            Centroids,Output=self.fit(X)
            curr_sse=0
            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(X)):
                curr_center = Centroids[Output[i]]
                curr_sse += (X[i, 0] - curr_center[0]) ** 2 + (X[i, 1] - curr_center[1]) ** 2

            sse.append(curr_sse)

        K_array = np.arange(1, K_max, 1)
        plt.plot(K_array, sse,'bx-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Within-cluster sums of squares (WCSS)')
        plt.title('Elbow method to determine optimum number of clusters')
        plt.show()






