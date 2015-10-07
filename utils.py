from scipy.stats.stats import pearsonr


# compute correlation between features
def compute_correlation(Xtrain):
    for i in range(0, Xtrain.shape[1]):
        for j in range(i+1, Xtrain.shape[1]):
            correlation = pearsonr(Xtrain[:, i], Xtrain[:, j])[0]
            if correlation > 0.3 or correlation < -0.3:
                print ('correlation between', i, "and", j, " feature is", correlation)

