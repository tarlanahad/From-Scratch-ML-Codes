import numpy as np


class NaiveBayes:

    def __init__(self, X, y):
        self.priors = []  # List of priors - P(y) for each class
        self.stds = []  # List of STDs of feature columns
        self.means = []  # List of Mean of feature columns
        self.classes = np.unique(y)  # unique class labels

        self.X = X
        self.y = y

    def fit(self):
        for c in self.classes:
            x_c = self.X[c == self.y]  # Get a feature vector that belongs to the class
            self.means.append(x_c.mean(axis=0))  # axis = 0 => column-wise Mean
            self.stds.append(x_c.std(axis=0))  # axis = 0 => column-wise STD
            self.priors.append(len(x_c) / len(self.X))  # Calculate frequency of each class

    def predict(self, X):
        y_pred = [self.__predict__(x) for x in X]
        return y_pred

    def __predict__(self, x):  # Helper Function for the function 'predict'
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = self.priors[idx]  # P(yi)

            P_X_yi = self.__gauss_pdf__(idx, x)  # P(X | yi) - Likelihood

            # Equation 4.6
            posterior = 0
            for P_x_yi in P_X_yi:
                posterior += np.log(P_x_yi)  # P(x1 | yi) + ... + P(xn | yi)
            posterior = posterior + np.log(prior)

            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]  # Select class with highest Posterior

    def __gauss_pdf__(self, idx, x):

        mu = self.means[idx]
        std = self.stds[idx]

        return np.exp(-(x - mu) ** 2 / (2 * std ** 2)) / np.sqrt(2 * np.pi * std ** 2)  # Equation 4.7


from sklearn.model_selection import train_test_split
from sklearn import datasets

data_x, data_y = datasets.make_classification(n_samples=10000, n_features=5, n_classes=2, random_state=123)
x_tr, x_ts, y_tr, y_ts = train_test_split(data_x, data_y, test_size=0.2, random_state=123)


nb = NaiveBayes(x_tr, y_tr)
nb.fit()
preds = nb.predict(x_ts)

print('Accuracy:', np.sum(y_ts == preds) / len(x_ts))

# '> Accuracy: 0.9145

#######
