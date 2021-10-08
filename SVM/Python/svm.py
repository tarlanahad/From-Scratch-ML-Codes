import numpy as np


class SVM:
    def __init__(self, X, y, num_of_epochs, lr, C):

        self.X = X
        self.y = y
        self.num_of_epochs = num_of_epochs
        self.lr = lr
        self.C = C

        # Add column vector of ones for computational convenience
        self.X = np.column_stack((np.ones(len(X)), X))

        # Initialize normal vector
        self.w = np.ones(len(self.X[0]))

    def distances(self, w, with_lagrange=True):
        distances = self.y * (np.dot(self.X, w)) - 1

        # get distance from the current decision boundary
        # by considering 1 width of margin

        if with_lagrange:  # if lagrange multiplier considered
            # if distance is more than 0
            # sample is not on the support vector
            # Lagrange multiplier will be 0
            distances[distances > 0] = 0

        return distances

    def get_cost_grads(self, X, w, y):

        distances = self.distances(w)

        # Get current cost
        L = 1 / 2 * np.dot(w, w) - self.C * np.sum(distances)

        dw = np.zeros(len(w))

        for ind, d in enumerate(distances):
            if d == 0:  # if sample is not on the support vector
                di = w  # (alpha * y[ind] * X[ind]) = 0
            else:
                # (alpha * y[ind] * X[ind]) = y[ind] * X[ind]
                di = w - (self.C * y[ind] * X[ind])
            dw += di
        return L, dw / len(X)

    def fit(self):
        for i in range(self.num_of_epochs):
            L, dw = self.get_cost_grads(self.X, self.w, self.y)
            self.w = self.w - self.lr * dw
            if i % 1000 == 0:
                print(i, ' | ', L)

    def predict(self, X):
        X = np.column_stack((np.ones(len(X)), X))
        return np.sign(X @ self.w)


data = np.genfromtxt('files/data.csv', dtype=float, delimiter=',')
np.random.shuffle(data)

train_y = data[:, 0]
train_x = data[:, 1:]

clf = SVM(train_x, train_y, num_of_epochs=10000, lr=1e-3, C=30)
clf.fit()

p = clf.predict(train_x)
p = p - train_y.flatten()

# Prediction accuracy should be 1.0 for the training set
print("Accuracy |", len(np.where(p == 0)[0]) / len(p))
