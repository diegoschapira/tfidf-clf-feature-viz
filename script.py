

import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

tfidf = TfidfVectorizer()
clf = LinearSVC()

# add script to fit models above

def plot_coefficients(classifier, feature_names, top_features=40):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(20, 5))
    colors = ['blue' if c < 0 else 'red' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()

plot_coefficients(clf, tfidf.get_feature_names())
