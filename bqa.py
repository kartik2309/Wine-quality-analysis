from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from pandas import read_csv
from pandas import DataFrame
from pandas import concat


import matplotlib.pyplot as plt

# Converting the dataset to binary labelled one.
def data_cleaning(bev_set, c):
    if c == 'red':
        X = bev_set.iloc[:, 0:11]
        y = bev_set.iloc[:, 11]
        normalizer = Normalizer().fit(X)
        X_norm = normalizer.transform(X)
        X_norm_pd = DataFrame(X_norm)

        for i in range(0, len(y)):
            if y.iloc[i] == 0:
                y.iloc[i] = 'Average'
            elif y.iloc[i] == 1:
                y.iloc[i] = 'Average'
            elif y.iloc[i] == 2:
                y.iloc[i] = 'Average'

            elif y.iloc[i] == 3:
                y.iloc[i] = 'Average'
            elif y.iloc[i] == 4:
                y.iloc[i] = 'Average'
            elif y.iloc[i] == 5:
                y.iloc[i] = 'Average'

            elif y.iloc[i] == 6:
                y.iloc[i] = 'Good'
            elif y.iloc[i] == 7:
                y.iloc[i] = 'Good'
            elif y.iloc[i] == 8:
                y.iloc[i] = 'Good'

            elif y.iloc[i] == 8:
                y.iloc[i] = 'Good'
            elif y.iloc[i] == 9:
                y.iloc[i] = 'Good'
            elif y.iloc[i] == 10:
                y.iloc[i] = 'Good'

        wine_data_norm = concat([X_norm_pd, y], axis=1)
        wine_data_norm.to_csv('winequalityclean-red.csv', sep=',', encoding='utf-8')
    else:
        X = bev_set.iloc[:, 0:11]
        y = bev_set.iloc[:, 11]
        normalizer = Normalizer().fit(X)
        X_norm = normalizer.transform(X)
        X_norm_pd = DataFrame(X_norm)

        for i in range(0, len(y)):
            if y.iloc[i] == 0:
                y.iloc[i] = 'Average'
            elif y.iloc[i] == 1:
                y.iloc[i] = 'Average'
            elif y.iloc[i] == 2:
                y.iloc[i] = 'Average'

            elif y.iloc[i] == 3:
                y.iloc[i] = 'Average'
            elif y.iloc[i] == 4:
                y.iloc[i] = 'Average'
            elif y.iloc[i] == 5:
                y.iloc[i] = 'Average'

            elif y.iloc[i] == 6:
                y.iloc[i] = 'Good'
            elif y.iloc[i] == 7:
                y.iloc[i] = 'Good'
            elif y.iloc[i] == 8:
                y.iloc[i] = 'Good'

            elif y.iloc[i] == 8:
                y.iloc[i] = 'Good'
            elif y.iloc[i] == 9:
                y.iloc[i] = 'Good'
            elif y.iloc[i] == 10:
                y.iloc[i] = 'Good'

        wine_data_norm = concat([X_norm_pd, y], axis=1)
        wine_data_norm.to_csv('winequalityclean-white.csv', sep=',', encoding='utf-8')
    return wine_data_norm


def decision_tree(wine_set_copy, c):
    if c == 'red':
        bev_set = wine_set_copy.copy()

        # Splitting the attributes and label from the data
        X = bev_set.iloc[:, 0:11]
        y = bev_set.iloc[:, 11]

        # Splitting data for testing and training, with 30% for testing and 70% for training ---
        # for cross validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Training the Decision Tree Classifier.
        dt_clf = DecisionTreeClassifier()
        dt_clf.fit(X_train, y_train)

        # Predict the test attributes
        y_predicted = dt_clf.predict(X_test)

        # Generate Accuracy Score by comparing it with the y_test.
        accr_score = accuracy_score(y_test, y_predicted)

        # Generating Confusion Matrix by comparing the y_test and predicted values.
        confn_matr = confusion_matrix(y_test, y_predicted)

        # Printing the values
        print("-----Decision Tree Classifier----")
        print("By using whole Dataset")
        print("Accuracy:", accr_score)
        print("Confusion Matrix:\n", confn_matr)

        # FOR pH AND ETHANOL ONLY
        X = bev_set.iloc[:, [8, 9]]
        y = bev_set.iloc[:, 11]

        # Splitting data for testing and training, with 30% for testing and 70% for training ---
        # for cross validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Training the Decision Tree Classifier.
        dt_clf = DecisionTreeClassifier()
        dt_clf.fit(X_train, y_train)

        # Predict the test attributes
        y_predicted = dt_clf.predict(X_test)

        # Generate Accuracy Score by comparing it with the y_test.
        accr_score = accuracy_score(y_test, y_predicted)

        # Generating Confusion Matrix by comparing the y_test and predicted values.
        confn_matr = confusion_matrix(y_test, y_predicted)

        # accuracy and confusion matrix
        print("By using pH and Sulphates only")
        print("Accuracy:", accr_score)
        print("Confusion Matrix:\n", confn_matr)
    else:
        bev_set = wine_set_copy.copy()

        # Splitting the attributes and label from the data
        X = bev_set.iloc[:, 0:11]
        y = bev_set.iloc[:, 11]

        # Splitting data for testing and training, with 30% for testing and 70% for training ---
        # for cross validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Training the Decision Tree Classifier.
        dt_clf = DecisionTreeClassifier()
        dt_clf.fit(X_train, y_train)

        # Predict the test attributes
        y_predicted = dt_clf.predict(X_test)

        # Generate Accuracy Score by comparing it with the y_test.
        accr_score = accuracy_score(y_test, y_predicted)

        # Generating Confusion Matrix by comparing the y_test and predicted values.
        confn_matr = confusion_matrix(y_test, y_predicted)

        # Printing the values
        print("-----Decision Tree Classifier----")
        print("By using whole Dataset")
        print("Accuracy:", accr_score)
        print("Confusion Matrix:\n", confn_matr)

        # FOR pH AND ETHANOL ONLY
        X = bev_set.iloc[:, [8, 10]]
        y = bev_set.iloc[:, 11]

        # Splitting data for testing and training, with 30% for testing and 70% for training ---
        # for cross validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Training the Decision Tree Classifier.
        dt_clf = DecisionTreeClassifier()
        dt_clf.fit(X_train, y_train)

        # Predict the test attributes
        y_predicted = dt_clf.predict(X_test)

        # Generate Accuracy Score by comparing it with the y_test.
        accr_score = accuracy_score(y_test, y_predicted)

        # Generating Confusion Matrix by comparing the y_test and predicted values.
        confn_matr = confusion_matrix(y_test, y_predicted)

        # accuracy and confusion matrix
        print("By using pH and Ethanol only")
        print("Accuracy:", accr_score)
        print("Confusion Matrix:\n", confn_matr)


def random_forest(wine_set_copy, c):
    if c == 'red':
        bev_set = wine_set_copy.copy()

        # Splitting the attributes and label from the data
        X = bev_set.iloc[:, 0:11]
        y = bev_set.iloc[:, 11]

        # Splitting data for testing and training, with 30% for testing and 70% for training ---
        # for cross validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Get accuracy for individual trees
        accuracy = []
        for i in range(0, 50):
            rf_clf_2 = RandomForestClassifier(n_estimators=i + 1)
            rf_clf_2.fit(X_train, y_train)
            pred = rf_clf_2.predict(X_test)
            accr = accuracy_score(y_test, pred)
            accuracy.append(accr)

        plt.plot(range(0, 50), accuracy)
        plt.show()

        # Training the Decision Tree Classifier.
        rf_clf = RandomForestClassifier(n_estimators=25)
        rf_clf.fit(X_train, y_train)

        # Predict the test attributes
        y_predicted = rf_clf.predict(X_test)

        # Generate Accuracy Score by comparing it with the y_test.
        accr_score = accuracy_score(y_test, y_predicted)

        # Generating Confusion Matrix by comparing the y_test and predicted values.
        confn_matr = confusion_matrix(y_test, y_predicted)

        # Printing the values
        print("\n-----Random Forest Classifier-----")
        print("By using whole Dataset")
        print("Accuracy:", accr_score)
        print("Confusion Matrix:\n", confn_matr)

        # FOR pH AND ETHANOL ONLY
        X = bev_set.iloc[:, [8, 9]]
        y = bev_set.iloc[:, 11]

        # Splitting data for testing and training, with 30% for testing and 70% for training ---
        # for cross validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Get accuracy for individual trees
        accuracy = []
        for i in range(0, 50):
            rf_clf_2 = RandomForestClassifier(n_estimators=i + 1)
            rf_clf_2.fit(X_train, y_train)
            pred = rf_clf_2.predict(X_test)
            accr = accuracy_score(y_test, pred)
            accuracy.append(accr)

        plt.plot(range(0, 50), accuracy)
        plt.show()

        # Training the Decision Tree Classifier.
        rf_clf = RandomForestClassifier(n_estimators=25)
        rf_clf.fit(X_train, y_train)

        # Predict the test attributes
        y_predicted = rf_clf.predict(X_test)

        # Generate Accuracy Score by comparing it with the y_test.
        accr_score = accuracy_score(y_test, y_predicted)

        # Generating Confusion Matrix by comparing the y_test and predicted values.
        confn_matr = confusion_matrix(y_test, y_predicted)

        # Printing the values
        print("\nBy using pH and Sulphates only")
        print("Accuracy:", accr_score)
        print("Confusion Matrix:\n", confn_matr)
    else:
        bev_set = wine_set_copy.copy()

        # Splitting the attributes and label from the data
        X = bev_set.iloc[:, 0:11]
        y = bev_set.iloc[:, 11]

        # Splitting data for testing and training, with 30% for testing and 70% for training ---
        # for cross validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Get accuracy for individual trees
        accuracy = []
        for i in range(0, 50):
            rf_clf_2 = RandomForestClassifier(n_estimators=i + 1)
            rf_clf_2.fit(X_train, y_train)
            pred = rf_clf_2.predict(X_test)
            accr = accuracy_score(y_test, pred)
            accuracy.append(accr)

        plt.plot(range(0, 50), accuracy)
        plt.show()

        # Training the Decision Tree Classifier.
        rf_clf = RandomForestClassifier(n_estimators=25)
        rf_clf.fit(X_train, y_train)

        # Predict the test attributes
        y_predicted = rf_clf.predict(X_test)

        # Generate Accuracy Score by comparing it with the y_test.
        accr_score = accuracy_score(y_test, y_predicted)

        # Generating Confusion Matrix by comparing the y_test and predicted values.
        confn_matr = confusion_matrix(y_test, y_predicted)

        # Printing the values
        print("\n------Random Forest Classifier------")
        print("By using whole Dataset")
        print("Accuracy:", accr_score)
        print("Confusion Matrix:\n", confn_matr)

        # FOR pH AND ETHANOL ONLY
        X = bev_set.iloc[:, [8, 10]]
        y = bev_set.iloc[:, 11]

        # Splitting data for testing and training, with 30% for testing and 70% for training ---
        # for cross validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Get accuracy for individual trees
        accuracy = []
        for i in range(0, 50):
            rf_clf_2 = RandomForestClassifier(n_estimators=i + 1)
            rf_clf_2.fit(X_train, y_train)
            pred = rf_clf_2.predict(X_test)
            accr = accuracy_score(y_test, pred)
            accuracy.append(accr)

        plt.plot(range(0, 50), accuracy)
        plt.show()

        # Training the Decision Tree Classifier.
        rf_clf = RandomForestClassifier(n_estimators=25)
        rf_clf.fit(X_train, y_train)

        # Predict the test attributes
        y_predicted = rf_clf.predict(X_test)

        # Generate Accuracy Score by comparing it with the y_test.
        accr_score = accuracy_score(y_test, y_predicted)

        # Generating Confusion Matrix by comparing the y_test and predicted values.
        confn_matr = confusion_matrix(y_test, y_predicted)

        # Printing the values
        print("\nBy using pH and Ethanol only")
        print("Accuracy:", accr_score)
        print("Confusion Matrix:\n", confn_matr)


def gaussian_nb(wine_set_copy, c):
    if c == 'red':
        bev_set = wine_set_copy.copy()

        # Splitting the attributes and label from the data
        X = bev_set.iloc[:, 0:11]
        y = bev_set.iloc[:, 11]

        # Splitting data for testing and training, with 30% for testing and 70% for training ---
        # for cross validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Training the Decision Tree Classifier.
        dt_clf = GaussianNB()
        dt_clf.fit(X_train, y_train)

        # Predict the test attributes
        y_predicted = dt_clf.predict(X_test)

        # Generate Accuracy Score by comparing it with the y_test.
        accr_score = accuracy_score(y_test, y_predicted)

        # Generating Confusion Matrix by comparing the y_test and predicted values.
        confn_matr = confusion_matrix(y_test, y_predicted)

        # Printing the values
        print("\n-----Gaussian Naive Bayes Classifier----")
        print("By using whole Dataset")
        print("Accuracy:", accr_score)
        print("Confusion Matrix:\n", confn_matr)

        # FOR pH AND ETHANOL ONLY
        X = bev_set.iloc[:, [8, 9]]
        y = bev_set.iloc[:, 11]

        # Splitting data for testing and training, with 30% for testing and 70% for training ---
        # for cross validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Training the Decision Tree Classifier.
        dt_clf = GaussianNB()
        dt_clf.fit(X_train, y_train)

        # Predict the test attributes
        y_predicted = dt_clf.predict(X_test)

        # Generate Accuracy Score by comparing it with the y_test.
        accr_score = accuracy_score(y_test, y_predicted)

        # Generating Confusion Matrix by comparing the y_test and predicted values.
        confn_matr = confusion_matrix(y_test, y_predicted)

        # accuracy and confusion matrix
        print("By using pH and Sulphates only")
        print("Accuracy:", accr_score)
        print("Confusion Matrix:\n", confn_matr)
    else:
        bev_set = wine_set_copy.copy()

        # Splitting the attributes and label from the data
        X = bev_set.iloc[:, 0:11]
        y = bev_set.iloc[:, 11]

        # Splitting data for testing and training, with 30% for testing and 70% for training ---
        # for cross validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Training the Decision Tree Classifier.
        dt_clf = GaussianNB()
        dt_clf.fit(X_train, y_train)

        # Predict the test attributes
        y_predicted = dt_clf.predict(X_test)

        # Generate Accuracy Score by comparing it with the y_test.
        accr_score = accuracy_score(y_test, y_predicted)

        # Generating Confusion Matrix by comparing the y_test and predicted values.
        confn_matr = confusion_matrix(y_test, y_predicted)

        # Printing the values
        print("\n-----Gaussian Naive Bayes Classifier----")
        print("By using whole Dataset")
        print("Accuracy:", accr_score)
        print("Confusion Matrix:\n", confn_matr)

        # FOR pH AND ETHANOL ONLY
        X = bev_set.iloc[:, [8, 10]]
        y = bev_set.iloc[:, 11]

        # Splitting data for testing and training, with 30% for testing and 70% for training ---
        # for cross validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Training the Decision Tree Classifier.
        dt_clf = GaussianNB()
        dt_clf.fit(X_train, y_train)

        # Predict the test attributes
        y_predicted = dt_clf.predict(X_test)

        # Generate Accuracy Score by comparing it with the y_test.
        accr_score = accuracy_score(y_test, y_predicted)

        # Generating Confusion Matrix by comparing the y_test and predicted values.
        confn_matr = confusion_matrix(y_test, y_predicted)

        # accuracy and confusion matrix
        print("By using pH and Ethanol only")
        print("Accuracy:", accr_score)
        print("Confusion Matrix:\n", confn_matr)

def k_means_clustering(wine_data_set, c):
    if c == 'red':
        bev_set = wine_data_set.copy()

        x = bev_set.iloc[:, 8]
        y = bev_set.iloc[:, 9]

        # Splitting the attributes and label from the data
        df_eth = DataFrame(x)
        df_sul = DataFrame(y)
        df = concat([df_eth, df_sul], axis=1)

        # plotting Ethanol vs. Quality relation
        plt.xlabel('pH')
        plt.ylabel('Sulphates')
        plt.scatter(x, y)
        plt.show()

        # Applying K-Means algorithm
        kmeans = KMeans(init='k-means++', n_clusters=3)
        clusters = kmeans.fit_predict(df)

        # Plotting the clustered data
        centres = kmeans.cluster_centers_
        colors = [[1, 0, 0]]
        plt.xlabel('pH')
        plt.ylabel('Sulphates')
        plt.scatter(x, y, c=clusters)
        for i, j in centres:
            plt.scatter(i, j, c=colors, marker='x', s=100)
        plt.show()
    else:
        bev_set = wine_data_set.copy()

        x = bev_set.iloc[:, 8]
        y = bev_set.iloc[:, 10]

        # Splitting the attributes and label from the data
        df_eth = DataFrame(x)
        df_sul = DataFrame(y)
        df = concat([df_eth, df_sul], axis=1)

        # plotting Ethanol vs. Quality relation
        plt.xlabel('pH')
        plt.ylabel('Ethanol')
        plt.scatter(x, y)
        plt.show()

        # Applying K-Means algorithm
        kmeans = KMeans(init='k-means++', n_clusters=3)
        clusters = kmeans.fit_predict(df)

        # Plotting the clustered data
        centres = kmeans.cluster_centers_
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        plt.xlabel('pH')
        plt.ylabel('Ethanol')
        plt.scatter(x, y, c=clusters)
        for i, j in centres:
            plt.scatter(i, j, c=colors, marker='x', s=100)
        plt.show()


def logistic_regression(wine_set_copy):
    bev_set = wine_set_copy.copy()

    # Splitting the attributes and label from the data
    X = bev_set.iloc[:, 0:11]
    y = bev_set.iloc[:, 11]

    clf = LogisticRegression(random_state=0, C=100, multi_class='ovr')
    clf.fit(X, y)
    coefs = clf.coef_
    print("\n---Logistic Regression Coefficients---")
    print(coefs)
    plt.plot(coefs[0])
    plt.show()


#READING BOTH DATASETS(ALREADY CLEANED ONE)
bev_data_red = read_csv("Beverage quality clean-red.csv")
bev_data_white = read_csv("Beverage quality clean-white.csv")
#newdata = data_cleaning(beverage_data)

#ANALYSING RED BEVERAGE DATA
print("\n\n-------------For Beverage White-----------------")
logistic_regression(bev_data_red)

decision_tree(bev_data_red, c='red')
random_forest(bev_data_red, c='red')
gaussian_nb(bev_data_red, c='red')

k_means_clustering(bev_data_red, c='red')

#ANALYSING WHITE BEVERAGE DATA
print("\n\n---------------For Beverage Red------------------")
logistic_regression(bev_data_red)

decision_tree(bev_data_red, c='white')
random_forest(bev_data_red, c='white')
gaussian_nb(bev_data_red, c='white')

k_means_clustering(bev_data_red, c='white')
