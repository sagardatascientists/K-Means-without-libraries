# K-Means-without-libraries

![image](https://user-images.githubusercontent.com/17669852/114973089-d1cb9400-9ed3-11eb-81af-73bbce7e1fc4.png)


# Dataset Description
Provided dataset consists total of 150 samples divided into two files irirs_train.csv and irirs_test.csv having 130
and 20 samples, respectively. As data set is iris flower, I assumed the column names as:

Column 1 - Sepal Length in cm
Column 2 - Sepal Width in cm
Column 3 - Petal Length in cm
Column 4 - Petal Width in cm
Column 5 – Species: Iris-Setosa, Iris-Versicolor and Iris-Virginica

Findings
1. Data Cleaning and normalization:
    No Missing value or Null value found in input dataset.
    Calculated min-max normalization scaler to normalize data before passing to algorithm.
2. Correlation Analysis:
  Outcomes of Correlation analysis:
  • Setosa petal lengths and widths are much smaller than Versicolor and Virginica.
  • Strong linear relationship between all the variables except sepal width, which is much weaker and
  negative.
The below table identifies trends between variables. Depending on strength of the relationship, it
assigns a number between -1 and 1.•

Looking at the below correlation table, we can see that there are 3 main variables (sepal length, petal
length and petal width) that have a strong linear relationship with species_id. These variables are
likely to be strong variables in predicting the species of a given data.

![image](https://user-images.githubusercontent.com/17669852/114973279-38e94880-9ed4-11eb-90a9-a663f7c9ea72.png)


![Correlation_Matrix] (https://user-images.githubusercontent.com/17669852/114973306-4a325500-9ed4-11eb-82df-f40ff9489c20.png)
