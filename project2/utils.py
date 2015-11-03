from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

# compute correlation between feature
def compute_correlation(Xtrain):
    for i in range(0, Xtrain.shape[1]):
        for j in range(i+1, Xtrain.shape[1]):
            correlation = pearsonr(Xtrain[:, i], Xtrain[:, j])[0]
            if correlation > 0.3 or correlation < -0.3:
                print ('correlation between', i, "and", j, " feature is", correlation)

# output moments of each feature
def momentsFeatures(X):

    for i in range(0, X.shape[1]):
        std = np.std(np.abs(X[:, i]))
        mean = np.mean(np.abs(X[:, i]))
        print "mean of feature {0} is {1}".format(i, mean)
        print "std dev of feature {0} is {1}".format(i, std)
        print "std/mean {0}".format(std/mean)
        print ""

# plot features with y, x sorted
def plotFeatures (X, Y):

    MAX_FEATURES = 15

    Y = np.exp(Y)
    permY = np.argsort(Y, axis=0)
    plt.title("Y")
    plt.plot(Y[permY])
    plt.show()

    for i in range(0, np.min(MAX_FEATURES, X.shape[1])):
        column = X[:, i]
        perm = np.argsort(column, axis=0)
        plt.title("feature " + str(i))
        plt.plot(column[perm], Y[perm], 'bo')
        plt.show()

