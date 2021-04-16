# -*- coding: utf-8 -*-
"""
# PYTHON: 3.7.8
# DATE: 03-12-2020
# CONTACT: sagar.datascientist@gmail.com
"""

from Kmeans import Kmeans

# Basic imports
import sys, getopt,traceback,pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

np.random.seed(3425)

#########################################################################################################################################
# This Main class will connect all the different modules that required to run K-Means clustering algorithm.
# This class connect different training, testing modules and generate accuracy and plot graph.
# Arguments to run project: python Train_Test.py -i <Trainfile> -t <Testfile> -n <norm> -k <clusterno>
#########################################################################################################################################

class Main:

    def __init__(self):
        """
        This init method initialize all the static/dynamic variables used by main class

        """
        self.train_path = ""
        self.test_path = ""
        self.scaler = False # Minmax scaler [scale data and pass to algorithm]
        self.norm = -1
        self.k_cluster_no = -1
        self.no_of_dimensions = 100
        self.no_of_iteration = 100
        self.max_clusters=10
        self.each_class_mean = {}
        self.optimized_centroids = {}
        self.class_labels = {}
        self.figsize = (15, 11)
        self.global_mean = ''
        self.global_std = ''
        self.global_Min=''
        self.global_Max=''
        self.found = False
        self.df_actual_colname = 'Actual_ClassLabel'
        self.df_predicted_colname = 'Predicted_ClassLabel'
        self.col_names_irisdata=['Sepal length (cm)','Sepal width (cm)','Petal length (cm)','Petal width (cm)']


    def save_model(self):
        """
        This method store all the final variables as model file for prediction.
        """
        try:
            model = {}
            model['Dimensions'] = self.no_of_dimensions
            model['KNoofClusrers'] = self.k_cluster_no
            model['MapClassLabels'] = (self.class_labels)
            model['Centroids'] = (self.optimized_centroids)
            model['FeatureMean'] = (self.each_class_mean)
            if (self.scaler):
                model['Scaler'] = self.scaler
                model['GlobalMean'] = self.global_mean
                model['GlobalStd'] = self.global_std

            with open(self.model_op_path, 'wb') as f:
                pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        except:
            print("Found Error in the save_model method")
            traceback.print_exc()

    def load_model(self):
        """
        This method load stored model file.

        """
        with open(self.model_op_path, 'rb') as f:
            model_loaded = pickle.load(f)

    def convert_labels(self, Y):
        """
        This method map class labels to unique Ids and return list.

        :param Y:
        :return:
        """
        idList = []
        self.class_labels_list = list(self.class_labels)
        for eachType in Y:
            try:
                if eachType in self.class_labels_list:
                    id = self.class_labels_list.index(eachType)
                    idList.append(id)
            except:
                print("Error Diff Class label passed than existing")
                sys.exit(2)
        return idList

    def generate_Columns(self):
        """
        This method generate columns based on dimensions. For Iris dataset assume col in ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
        :return:
        """
        colNames = []
        if (self.no_of_dimensions>4):
            for i in range(self.no_of_dimensions):
                colNames.append("Col_" + str(int(i)))
            colNames.append(self.df_actual_colname)
        else:
            for col in self.col_names_irisdata:
                colNames.append(col)
            colNames.append(self.df_actual_colname)

        return colNames

    def checkDimensions(self, X, status):
        """
        This method will check dimensions of input data and if it is more than 100 than exit code

        :param X:
        :param status:
        """
        dim = X.shape[1] - 1
        if (dim > self.no_of_dimensions):
            print("No of dimensions are greather than 100. Please pass dimesions below 100 to run this code.")
            sys.exit(2)
        else:
            if (status == 'Train'):
                self.no_of_dimensions = dim
            elif (status == 'Test'):
                if (dim != self.no_of_dimensions):
                    print("Please pass same dimensions in the Train and Test files")
                    sys.exit(2)

    def preprocessing_Data(self, df, scaler, status):
        """
        This method convert data into normal distribution which has mean 0 and std 1

        :param df:
        :param scaler:
        :param status:
        :return:
        """
        if (scaler == True):
            normalized_df = ''
            if (status == "Train"):
                #Normalization data into [0,1]
                self.globalMin = df.min().values
                self.globalMax = df.max().values
                normalized_df = (df - self.globalMin) / (self.globalMax-self.globalMin)
                # Standardization [mean 0 and std 1]
                # self.global_mean = df.mean().values
                # self.global_std = df.std().values
                # normalized_df = (df - self.global_mean) / self.global_std
            elif (status == "Test"):
                normalized_df = (df - self.globalMin) / (self.globalMax-self.globalMin)
                # Standard scaler
                # normalized_df = (df - self.global_mean) / self.global_std

            return normalized_df
        else:
            return df

    def mean_ClassLabels(self, iris_df):
        """
        This method calculates mean per each unique class label. It will use to convert N cluster to 3 for classification

        :param iris_df:
        """
        for each_class in list(self.class_labels):
            df = iris_df.loc[iris_df[self.df_actual_colname] == each_class]
            mean_per_class = df.mean().values
            self.each_class_mean[each_class] = mean_per_class

    def plot_accuracy(self, cf, categories, stats_text, title, save_path):
        """
        This method plot accuracy of given data frame and save confusion matrix with accuracy details for train and test data sets.

        :param cf:
        :param categories:
        :param stats_text:
        :param title:
        :param save_path:
        """
        cmap = 'Blues'
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        box_labels = [f"{v1}".strip() for v1 in (group_counts)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

        plt.figure(figsize=self.figsize)
        sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=True, xticklabels=categories, yticklabels=categories)

        plt.ylabel('True label', color='r')
        plt.xlabel('Predicted label', color='r')
        plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=12)
        title = title + "\n\n\n" + "Confusion matrix"

        plt.title(title, color='r', fontsize=15)
        plt.savefig(save_path)
        plt.show()

    def calculate_Accuracy(self, y_true, y_pred, graph_title, save_path):
        """
        This method calculate confusino matrix for multi class with precesion,recall and f1 score for all classes.
        :param y_true:
        :param y_pred:
        :param graph_title:
        :param save_path:
        """
        classes = set(y_true)
        number_of_classes = len(classes)
        confMat = pd.DataFrame(
            np.zeros((number_of_classes, number_of_classes), dtype=int),
            index=classes,
            columns=classes)

        for i, j in zip(y_true, y_pred):
            confMat.loc[i, j] += 1
        confMat = confMat.values
        confMat = np.where(np.isnan(confMat), 0, confMat)

        FP = confMat.sum(axis=0) - np.diag(confMat)
        FN = confMat.sum(axis=1) - np.diag(confMat)
        TP = np.diag(confMat)
        TN = confMat.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        # Sensitivity, hit rate, recall, or true positive rate
        recall = TP / (TP + FN)
        # print("Recall", TPR)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        precision = TP / (TP + FP)
        # print("Precesion", PPV)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)
        # Calculate error
        Error = (FP + FN) / (TP + TN + FP + FN)
        # Remove NAN values
        error_avg = np.where(np.isnan(Error), 0, Error)
        precision = np.where(np.isnan(precision), 0, precision)
        recall = np.where(np.isnan(recall), 0, recall)
        # Overall accuracy for each class plot accuracy per each class
        acc_per_class = ((TP + TN) / (TP + FP + FN + TN))
        f1_score_per_class = 2 * precision * recall / (precision + recall)
        f1_score_per_class = np.where(np.isnan(f1_score_per_class), 0, f1_score_per_class)

        # generate stats for each class label
        stats_text_per_class = "F1 Score(Accuracy) for each class :"
        for each_class, f1_score in zip(self.class_labels, f1_score_per_class):
            # print("F1 Score(Accuracy) for each class : " + each_class + " : " + str(float(f1_score * 100)))
            if (f1_score == 0):
                self.found = True
            stats_text_per_class += each_class + " : {:0.2f}%".format(float(f1_score * 100)) + " , "

        # generate avg accuracy
        precision_avg = np.mean(precision, axis=0) * 100
        recall_avg = np.mean(recall, axis=0) * 100
        f1_score_avg = np.mean(f1_score_per_class, axis=0) * 100
        acc_avg = np.mean(acc_per_class, axis=0) * 100
        error_avg = np.mean(error_avg, axis=0) * 100
        class_labels_ids = list(self.class_labels)

        # generate stats for each class label
        stats_text = "\n\nAvg. Accuracy={:0.2f}%, Avg. Error ={:0.2f}%, Avg. Precision={:0.2f}%, Avg. Recall={:0.2f}%, Avg. F1 Score={:0.2f}%".format(
            acc_avg, error_avg, precision_avg, recall_avg, f1_score_avg) + "\n\n" + stats_text_per_class[:-2]

        print(stats_text)

        # plot confusion matrix with accuracy
        self.plot_accuracy(confMat, class_labels_ids, stats_text, graph_title, save_path)

    def find_variable_correlations(self, df,y_true):
        """
        This method calculates correlations between columns

        :param df:
        """
        df['species_id']=y_true
        print("\n Pivot Table: ")
        print(df.pivot_table(index=self.df_actual_colname,
                         values=['Sepal length (cm)', 'Sepal width (cm)', 'Petal length (cm)', 'Petal width (cm)',
                                 'species_id'], aggfunc=np.mean).to_string()+"\n\n")

        print("\nCorrelation matrix:")
        d_corr = df.corr()
        print(d_corr.to_string())
        sns.set_style('whitegrid')
        plt.figure(figsize=(15, 6))
        mask = np.zeros_like(d_corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(d_corr, mask=mask, cmap=cmap, vmax=.3,
                    square=True,
                    linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()
        del df['species_id']

    def generate_Multivariate_Plot(self, df, title, save_path):
        """
        This method store multivariate classification grph and display it.

        :param df:
        :param title:
        :param save_path:
        """
        sns.set_style("whitegrid")

        if (self.found):
            g = sns.pairplot(df, hue=self.df_predicted_colname, diag_kind="hist")
            self.found = False
        else:
            val=np.unique(df[self.df_predicted_colname].values)
            if(len(val)==3):
                g = sns.pairplot(df, hue=self.df_predicted_colname, diag_kind="hist", markers=["o", "s", 'd'])
            else:
                g = sns.pairplot(df, hue=self.df_predicted_colname, diag_kind="hist")
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(title, color='r')
        if (save_path != ""):
            g.savefig(save_path)
        plt.show()

    def decide_Optimal_NoClusters(self,x_norm):
        """
        Decide no of clusters using

        :param x_norm:
        """
        # Intialize kmeans algorithm and set parameters
        self.obj_kmeans = Kmeans(self.norm, self.k_cluster_no, self.no_of_dimensions, self.no_of_iteration)
        # Define optimal no of cluster
        self.obj_kmeans.define_OptimalClusters(x_norm, self.max_clusters)
    def kMeans_Training(self):
        """
        This method read train dataset and run kmeans clustering algorithm and
        plot multivariate graph, accuracy report and save outputfile for norm type.

        """
        try:
            # Read Train CSV without header
            iris_df_train = pd.read_csv(self.train_path,header=None)

            # check dimemsions and if it more than 100 then stop code
            self.checkDimensions(iris_df_train, 'Train')
            headers = self.generate_Columns()
            iris_df_train.columns = headers

            # Select all col expect last
            x_raw = iris_df_train.iloc[:, :-1]
            # Assume last col is class label
            y_labels = iris_df_train.iloc[:, -1].values
            # map class labels to unique IDs
            y_true, self.class_labels = pd.factorize(y_labels)


            # this method find variable correlations and we can remove highly correlated variables to get good accuracy [Not doing in this code]
            self.find_variable_correlations(iris_df_train,y_true)


            # calculate mean per each class label for all the dimensions- to convert N clusters to 3 for calculating accuracy
            self.mean_ClassLabels(iris_df_train)

            # Apply Data Preprocessing
            x_norm = self.preprocessing_Data(x_raw, self.scaler, 'Train')
            x_norm = x_norm.values

            ########################### Turn on below code to decide optimal no of clusters size #########################
            #self.decide_Optimal_NoClusters(x_norm)
            ##############################################################################################################

            # Intialize kmeans algorithm and set parameters
            self.obj_kmeans = Kmeans(self.norm, self.k_cluster_no, self.no_of_dimensions, self.no_of_iteration)

            # Run kmeans clustering for no of iterations,norm type and for k no of clusters
            centroids, y_predict = self.obj_kmeans.fit(x_norm)
            # Find final optimized centroids
            self.optimized_centroids = centroids

            if (self.k_cluster_no > 3):
                # Store k_cluster and plot multivariate graph
                iris_df_train[self.df_predicted_colname] = y_predict
                plot_title = "Multivariate Plot for Training Dataset using Norm-" + "L" + str(
                    int(self.norm)) + " for K cluster " + str(int(self.k_cluster_no))
                self.found = True
                self.generate_Multivariate_Plot(iris_df_train, plot_title, "")
                del iris_df_train[self.df_predicted_colname]

                # Below method convert N clusters to  3 clusters based on each class mean and  calculate accuracy
                y_predict = self.obj_kmeans.convert_N_Clusters_3(centroids, y_predict, self.each_class_mean)

            y_predict_class = self.class_labels[y_predict]
            # append final predicted class label to frame
            iris_df_train[self.df_predicted_colname] = y_predict_class

            # Save Train file with appended predicted column in csv file
            iris_df_train.to_csv(self.train_op_path, index=False)
            print("\n\n\n############################################################### Training Accuracy ###############################################################")

            # Generate multivariate plot and save it for norm type [if k_cluster>3 then merge clusters to 3 based on actual class mean by calculating min eucliden]
            plot_title = "Multivariate classfication Plot for Training Dataset using Norm-" + "L" + str(
                int(self.norm)) + " for K cluster " + str(int(self.k_cluster_no))
            self.generate_Multivariate_Plot(iris_df_train, plot_title, self.pair_plot_train)

            graph_title = "Accuracy Report for Training Dataset using Norm-" + "L" + str(
                int(self.norm)) + " for K cluster " + str(int(self.k_cluster_no))
            self.calculate_Accuracy(y_true, y_predict, graph_title, self.plot_accuracy_train)
            print("################################################################################################################################################\n\n\n\n")

        except:
            print("Found Error in the kMeans_Training method")
            traceback.print_exc()

    def kMeans_Testing(self):
        """
        This method use freezed centers generated by k-means algorithm on train set and generate accuracy for norm.

        """
        try:
            # read test dataset
            iris_df_test = pd.read_csv(self.test_path,header=None)

            x_test = iris_df_test.iloc[:, :-1]
            #Assume last colum is class label
            y_test_labels = iris_df_test.iloc[:, -1].values

            #Check data dimensions [make sure train and test datasets have same dimesions]
            self.checkDimensions(iris_df_test, 'Test')
            headers = self.generate_Columns()
            iris_df_test.columns = headers

            #Apply data preprocessing
            x_test_norm = self.preprocessing_Data(x_test, self.scaler, 'Test')
            x_test_norm = x_test_norm.values
            y_true = self.convert_labels(y_test_labels)

            #pass freezed centroids, calculate distances to each testing data and assign cluster id
            centroids, y_pred = self.obj_kmeans.predict(x_test_norm, self.optimized_centroids, self.k_cluster_no)
            if (self.k_cluster_no > 3):
                iris_df_test[self.df_predicted_colname] = y_pred
                plot_title = "Multivariate Plot for Testing Dataset using Norm-" + "L" + str(
                    int(self.norm)) + " for K cluster " + str(int(self.k_cluster_no))
                self.found = True
                self.generate_Multivariate_Plot(iris_df_test, plot_title, "")
                del iris_df_test[self.df_predicted_colname]
                # Below method convert N clusters to  3 clusters based on each class mean and  calculate accuracy
                y_pred = self.obj_kmeans.convert_N_Clusters_3(centroids, y_pred, self.each_class_mean)

            print("############################################################### Testing Accuracy ###############################################################")

            y_predict_class = self.class_labels[y_pred]
            iris_df_test[self.df_predicted_colname] = y_predict_class
            iris_df_test.to_csv(self.test_op_path, index=False)
            #Calculate accuracy for testing data
            graph_title = "Accuracy Report for Testing Dataset using Norm-" + "L" + str(
                int(self.norm)) + " for K cluster " + str(int(self.k_cluster_no))
            self.calculate_Accuracy(y_true, y_pred, graph_title, self.plot_accuracy_test)

            # Generate multivariate plot for testing data
            plot_title = "Multivariate Plot for Testing Dataset using Norm-" + "L" + str(
                int(self.norm)) + " for K cluster " + str(int(self.k_cluster_no))
            self.generate_Multivariate_Plot(iris_df_test, plot_title, self.pair_plot_test)
            print("################################################################################################################################################")

        except:
            print("Found Error in the kMeans_Testing method")
            traceback.print_exc()

    def generate_OutputPath(self):
        """
        This method generate output path location to store model prediction, accuracy, graph and model file.
        """
        try:
            # Store train and test file with appened classification column
            self.train_op_path = self.train_path[0:-4] + "_Norm_L" + str(int(self.norm))+"_Cluster_"+str(int(self.k_cluster_no)) + "_op.csv"
            self.test_op_path = self.test_path[0:-4] + "_Norm_L" + str(int(self.norm))+"_Cluster_"+str(int(self.k_cluster_no)) + "_op.csv"

            # save model file
            self.model_op_path = self.train_path[0:-4] + "_Norm_L" + str(int(self.norm))+"_Cluster_"+str(int(self.k_cluster_no)) + '_op.pkl'

            # Save plots for train data
            self.pair_plot_train = self.train_path[0:-4] + "_Norm_L" + str(int(self.norm))+"_Cluster_"+str(int(self.k_cluster_no)) + '_TrainData_Plot_Multivariate.png'
            self.plot_accuracy_train = self.train_path[0:-4] + "_Norm_L" + str(int(self.norm))+"_Cluster_"+str(int(self.k_cluster_no)) + '_TrainData_Accuracy.png'

            # Save plots for test data
            self.pair_plot_test = self.test_path[0:-4] + "_Norm_L" + str(int(self.norm))+"_Cluster_"+str(int(self.k_cluster_no)) + '_TestData_Plot_Multivariate.png'
            self.plot_accuracy_test = self.test_path[0:-4] + "_Norm_L" + str(int(self.norm))+"_Cluster_"+str(int(self.k_cluster_no)) + '_TestData_Accuracy.png'

        except:
            print("Found Error in the generate_OutputPath method")
            traceback.print_exc()

    def pipeline_ML(self, argv):
        """
        This pipeline method combine all the different modules

        :param argv:
        """
        self.parserArgument(argv)
        self.generate_OutputPath()
        self.kMeans_Training()
        self.kMeans_Testing()
        self.save_model()

    def parserArgument(self, argv):
        """

        :param argv:
        """
        try:
            opts, args = getopt.getopt(argv, "hi:t:n:k:", ["ifile=", "tfile=", "nfile=", "kcluster="])
            if (len(opts) == 0):
                print('Please pass argument: Train_Test.py -i <Trainfile> -t <Testfile> -n <norm> -k <clusterno>')
                sys.exit(2)
        except getopt.GetoptError:
            print('Train_Test.py -i <Trainfile> -t <Testfile> -n <norm> -k <clusterno>')
            sys.exit(2)
        try:
            for opt, arg in opts:
                if opt == '-h':
                    print('Train_Test.py -i <inputfile> -t <outputfile> -n <norm> -k <clusterno>')
                    sys.exit()
                elif opt in ("-i", "--ifile"):
                    self.train_path = arg
                elif opt in ("-t", "--tfile"):
                    self.test_path = arg
                elif opt in ("-n", "--Norm"):
                    norm_type = arg
                    self.norm = (int)(norm_type)
                elif opt in ("-k", "--cluster"):
                    no_of_clusters = arg
                    self.k_cluster_no = (int)(no_of_clusters)
                    if(self.k_cluster_no<3):
                        print("Please pass cluster no greater than equal to 3")
                        sys.exit()
            if self.train_path == "" or self.test_path == "" or self.norm == "" or self.k_cluster_no=="":
                print('Please pass argument: Train_Test.py -i <Trainfile> -t <Testfile> -n <norm> -k <clusterno>')
                sys.exit()

            print("Running K-Means algorithm for Norm(L) " + str(int(self.norm)) + " & cluster size " + str(
                int(self.k_cluster_no)) +"For Training file: "+str(self.train_path)+" For Testing file: "+ str(self.train_path))
        except:
            print("Found Error in the parserArgument method")
            traceback.print_exc()

# This is main function takes arguments and run pipeline_ML for kmeans clustering algorithm.
# Arguments to run project: python Train_Test.py -i <Trainfile> -t <Testfile> -n <norm> -k <clusterno>

if __name__ == '__main__':
    obj = Main()
    obj.pipeline_ML(sys.argv[1:])
