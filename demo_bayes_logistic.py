from sklearn.metrics import roc_auc_score, r2_score
import bayes_logistic as bl
import numpy as np

N1 = 101000
p = 11

# make a 'time' vector for the sine waves
t = np.linspace(0, 100, N1)

# ------------------------------------------------------------------------------
# Make a Correlated Feature matrix from sine waves of different frequencies
X1 = np.zeros([N1, p])
for _ in np.arange(10):
    X1[:, _] = np.sin(2 * np.pi * 0.5 * _ * t)
X1[:, 0] = np.ones([N1, ])  # bias term
for _ in np.arange(5):
    X1[:, _ + 6] = (X1[:, _ + 1] * X1[:, _ + 6])  # this is where impose the correlation

# ------------------------------------------------------------------------------
# make a parameter vector
w_true = np.random.uniform(-0.5, 0.5, p)
w_true[0] = -1  # bias parameter

# ------------------------------------------------------------------------------
# make some binary responses
mu = bl.logistic_prob(X1, w_true)
y1 = np.empty([N1])
for _ in np.arange(N1):
    y1[_] = np.random.binomial(1, mu[_])

# to get going, set a prior parameter of zeros, and a diagonal hessian
w_prior = np.zeros(p)
H_prior = np.diag(np.ones(p)) * 0.001

#----------------------------------------------------------------------------------------
# Do a bayesian fit with this random sample
# The default uses a full Hessian matrix and a Newton's conjugate gradient solver
w_posterior, H_posterior = bl.fit_bayes_logistic(y1, X1, w_prior, H_prior)
logistic_prob = bl.logistic_prob(X1, w_posterior)

print(r2_score(mu, logistic_prob))
print(roc_auc_score(y1, logistic_prob))
