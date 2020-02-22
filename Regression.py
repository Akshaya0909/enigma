#1 Linear Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x = 10 * np.random.rand(50)
y = 2 * x - 5 + np.random.randn(50)
x = x.reshape(-1,1)
y = y.reshape(-1,1)
linear_regressor = LinearRegression()
linear_regressor.fit(x,y)
y_pred = linear_regressor.predict(x)
intercept = linear_regressor.intercept_
coefficient = linear_regressor.coef_
print("Intercept of the line fit is ",intercept)
print("Slope of the line fit is ",coefficient[0])
plt.scatter(x, y)
plt.plot(x,y_pred,color='red')
plt.show()

#2 Polynomial Regression (degree = 6)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
x = 10 * np.random.rand(50)
y = np.sin(x) + 0.1 * np.random.rand(50)
x = x[:, np.newaxis]
y = y[:, np.newaxis]
polynomial_features = PolynomialFeatures(degree=6)
x_polynomial = polynomial_features.fit_transform(x)
model = LinearRegression()
model.fit(x_polynomial,y)
y_polynomial_pred = model.predict(x_polynomial)
RMSE = np.sqrt(mean_squared_error(y,y_polynomial_pred))
print("Root mean squared error of polynomial regression of degree 6 is ",RMSE)
plt.scatter(x, y)
plt.plot(x, y_polynomial_pred, color ='m')
plt.show()

#2 Polynomial Regression using pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
x = 10 * np.random.rand(50)
y = np.sin(x) + 0.1 * np.random.rand(50)
poly_model = make_pipeline(PolynomialFeatures(7),LinearRegression())
poly_model.fit(x[:,np.newaxis],y)
yfit = poly_model.predict(x[:,np.newaxis])
plt.scatter(x,y)
plt.plot(x,yfit)
plt.show()

#3 Linear Regression using Gaussian Basis (order = 6)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
class GaussianFeatures(BaseEstimator, TransformerMixin):
  def __init__(self, N, width_factor=2.0):
    self.N = N
    self.width_factor = width_factor

  @staticmethod
  def _gauss_basis_(x, y, width, axis=None):
    arg = (x-y)/width
    return np.exp(-0.5 * np.sum(arg **2, axis=1))

  def fit(self, X, y=None):
    self.centers_ = np.linspace(X.min(),X.max(),self.N)
    self.width_ = self.width_factor * (self.centers_[1]-self.centers_[0])
    return self
  
  def transform(self, X):
    return self._gauss_basis_(X[:,:,np.newaxis], self.centers_, self.width_,axis=1)

  
x = np.array([4.17022005e+00, 7.20324493e+00, 1.14374817e-03, 3.02332573e+00,
       1.46755891e+00, 9.23385948e-01, 1.86260211e+00, 3.45560727e+00,
       3.96767474e+00, 5.38816734e+00, 4.19194514e+00, 6.85219500e+00,
       2.04452250e+00, 8.78117436e+00, 2.73875932e-01, 6.70467510e+00,
       4.17304802e+00, 5.58689828e+00, 1.40386939e+00, 1.98101489e+00,
       8.00744569e+00, 9.68261576e+00, 3.13424178e+00, 6.92322616e+00,
       8.76389152e+00, 8.94606664e+00, 8.50442114e-01, 3.90547832e-01,
       1.69830420e+00, 8.78142503e+00, 9.83468338e-01, 4.21107625e+00,
       9.57889530e+00, 5.33165285e+00, 6.91877114e+00, 3.15515631e+00,
       6.86500928e+00, 8.34625672e+00, 1.82882773e-01, 7.50144315e+00,
       9.88861089e+00, 7.48165654e+00, 2.80443992e+00, 7.89279328e+00,
       1.03226007e+00, 4.47893526e+00, 9.08595503e+00, 2.93614148e+00,
       2.87775339e+00, 1.30028572e+00])
y = np.array([-0.92530881,  0.71111718, -0.06598087,  0.11672496,  0.88294471,
        0.8210899 ,  1.12370616, -0.23467501, -0.75446517, -0.86898322,
       -0.94231439,  0.70804351,  0.89495535,  0.53638242,  0.28955648,
        0.61914583, -0.84603144, -0.5796531 ,  1.01611705,  0.88180869,
        0.87399567, -0.28992469, -0.01353862,  0.65589053,  0.69771523,
        0.55374595,  0.78013085,  0.46920917,  0.91644209,  0.72516826,
        0.8837173 , -0.90676173, -0.10465615, -0.82186313,  0.70681199,
        0.13841844,  0.76810625,  0.74161023,  0.03745364,  0.88805266,
       -0.43137564,  1.01910093,  0.36236496,  0.7970268 ,  0.82783992,
       -0.89007576,  0.35538665,  0.28020998,  0.23855606,  0.94355877])

  
gauss_model = make_pipeline(GaussianFeatures(6),LinearRegression())
gauss_model.fit(x[:,np.newaxis],y)
y_pred = gauss_model.predict(x[:,np.newaxis])
  
plt.scatter(x,y)
plt.plot(x,y_pred, color='m')
plt.show()

#4 L1 Regularization 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from statistics import mean

x = np.array([4.17022005e+00, 7.20324493e+00, 1.14374817e-03, 3.02332573e+00,
       1.46755891e+00, 9.23385948e-01, 1.86260211e+00, 3.45560727e+00,
       3.96767474e+00, 5.38816734e+00, 4.19194514e+00, 6.85219500e+00,
       2.04452250e+00, 8.78117436e+00, 2.73875932e-01, 6.70467510e+00,
       4.17304802e+00, 5.58689828e+00, 1.40386939e+00, 1.98101489e+00,
       8.00744569e+00, 9.68261576e+00, 3.13424178e+00, 6.92322616e+00,
       8.76389152e+00, 8.94606664e+00, 8.50442114e-01, 3.90547832e-01,
       1.69830420e+00, 8.78142503e+00, 9.83468338e-01, 4.21107625e+00,
       9.57889530e+00, 5.33165285e+00, 6.91877114e+00, 3.15515631e+00,
       6.86500928e+00, 8.34625672e+00, 1.82882773e-01, 7.50144315e+00,
       9.88861089e+00, 7.48165654e+00, 2.80443992e+00, 7.89279328e+00,
       1.03226007e+00, 4.47893526e+00, 9.08595503e+00, 2.93614148e+00,
       2.87775339e+00, 1.30028572e+00])
y = np.array([-0.92530881,  0.71111718, -0.06598087,  0.11672496,  0.88294471,
        0.8210899 ,  1.12370616, -0.23467501, -0.75446517, -0.86898322,
       -0.94231439,  0.70804351,  0.89495535,  0.53638242,  0.28955648,
        0.61914583, -0.84603144, -0.5796531 ,  1.01611705,  0.88180869,
        0.87399567, -0.28992469, -0.01353862,  0.65589053,  0.69771523,
        0.55374595,  0.78013085,  0.46920917,  0.91644209,  0.72516826,
        0.8837173 , -0.90676173, -0.10465615, -0.82186313,  0.70681199,
        0.13841844,  0.76810625,  0.74161023,  0.03745364,  0.88805266,
       -0.43137564,  1.01910093,  0.36236496,  0.7970268 ,  0.82783992,
       -0.89007576,  0.35538665,  0.28020998,  0.23855606,  0.94355877])

x = x.reshape(-1,1)
y = y.reshape(-1,1)

# Lasso or L1 Regularization
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
cross_scores_lasso = []
lambda_val = []
for i in range(1,10):
  lasso_model = Lasso(alpha = 0.2*i,tol = 0.075)
  lasso_model.fit(x_train,y_train)
  scores = cross_val_score(lasso_model,x,y,cv=10)
  avg_score = mean(scores)*100
  cross_scores_lasso.append(avg_score)
  lambda_val.append(i*0.2)
  
for i  in range(1,10):
  print((0.2*i)," : ",cross_scores_lasso[i-1])

parameters = {'alpha':[0.0000000001, 0.00000001, 0.000001, 0.00001, 0.0001, 0.001, 0.2, 0.4, 0.6, 0.8, 1, 5, 10, 20]}
lasso =  Lasso()
lasso_regressor = GridSearchCV(lasso,parameters, scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(x,y)
print("Best value of parameters : ",lasso_regressor.best_params_)
print("Best value of negative mean squared error : ",lasso_regressor.best_score_)

# We conclude alpha =2 gives the least error in both manual and automatic check for parameters
lassomodelchosen = Lasso(alpha = 0.2, tol = 0.075)
lassomodelchosen.fit(x_train,y_train)
y_pred = lassomodelchosen.predict(x)
plt.scatter(x,y)
plt.plot(x,y_pred)
plt.show()

#4 L2 Regularization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from statistics import mean

x = np.array([4.17022005e+00, 7.20324493e+00, 1.14374817e-03, 3.02332573e+00,
       1.46755891e+00, 9.23385948e-01, 1.86260211e+00, 3.45560727e+00,
       3.96767474e+00, 5.38816734e+00, 4.19194514e+00, 6.85219500e+00,
       2.04452250e+00, 8.78117436e+00, 2.73875932e-01, 6.70467510e+00,
       4.17304802e+00, 5.58689828e+00, 1.40386939e+00, 1.98101489e+00,
       8.00744569e+00, 9.68261576e+00, 3.13424178e+00, 6.92322616e+00,
       8.76389152e+00, 8.94606664e+00, 8.50442114e-01, 3.90547832e-01,
       1.69830420e+00, 8.78142503e+00, 9.83468338e-01, 4.21107625e+00,
       9.57889530e+00, 5.33165285e+00, 6.91877114e+00, 3.15515631e+00,
       6.86500928e+00, 8.34625672e+00, 1.82882773e-01, 7.50144315e+00,
       9.88861089e+00, 7.48165654e+00, 2.80443992e+00, 7.89279328e+00,
       1.03226007e+00, 4.47893526e+00, 9.08595503e+00, 2.93614148e+00,
       2.87775339e+00, 1.30028572e+00])
y = np.array([-0.92530881,  0.71111718, -0.06598087,  0.11672496,  0.88294471,
        0.8210899 ,  1.12370616, -0.23467501, -0.75446517, -0.86898322,
       -0.94231439,  0.70804351,  0.89495535,  0.53638242,  0.28955648,
        0.61914583, -0.84603144, -0.5796531 ,  1.01611705,  0.88180869,
        0.87399567, -0.28992469, -0.01353862,  0.65589053,  0.69771523,
        0.55374595,  0.78013085,  0.46920917,  0.91644209,  0.72516826,
        0.8837173 , -0.90676173, -0.10465615, -0.82186313,  0.70681199,
        0.13841844,  0.76810625,  0.74161023,  0.03745364,  0.88805266,
       -0.43137564,  1.01910093,  0.36236496,  0.7970268 ,  0.82783992,
       -0.89007576,  0.35538665,  0.28020998,  0.23855606,  0.94355877])

x = x.reshape(-1,1)
y = y.reshape(-1,1)

# L2 Regularization
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
cross_scores_ridge = []
lambda_val = []
for i in range(1,10):
  ridge_model = Ridge(alpha = 0.2*i,tol = 0.075)
  ridge_model.fit(x_train,y_train)
  scores = cross_val_score(ridge_model,x,y,cv=10)
  avg_score = mean(scores)*100
  cross_scores_ridge.append(avg_score)
  lambda_val.append(i*0.2)
  
for i  in range(1,10):
  print((0.2*i)," : ",cross_scores_ridge[i-1])

parameters = {'alpha':[0.0000000001, 0.00000001, 0.000001, 0.00001, 0.0001, 0.001, 0.2, 0.4, 0.6, 0.8, 1, 5, 10, 20]}
ridge =  Ridge()
ridge_regressor = GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(x,y)
print("Best value of parameters : ",ridge_regressor.best_params_)
print("Best value of negative mean squared error : ",ridge_regressor.best_score_)

# We conclude alpha =20 gives the least error in automatic check for parameters. No change in cross validation score while checking manually for values between 0.2 to 2
ridgemodelchosen = Ridge(alpha = 0.2, tol = 0.075)
ridgemodelchosen.fit(x_train,y_train)
y_pred = ridgemodelchosen.predict(x)
plt.scatter(x,y)
plt.plot(x,y_pred)
plt.show()

#5 a) L2 Regularization using polynomial basis function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from statistics import mean
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
# Number of training samples 
N = 75
# Generate equispaced floats in the interval [0, 2π] 
x = np.linspace(0, 2*np.pi, N) 
# Generate noise 
mean = 0 
std = 0.05
# Generate some numbers from the sine function 
y = np.sin(x) 
# Add noise 
y += np.random.normal(mean, std, N)
x = x[:, np.newaxis]
y = y[:, np.newaxis]
polynomial_features = PolynomialFeatures(degree=3) # degree 3 seems to be a better choice as matrix becomes ill-conditioned with higher orders
x_polynomial = polynomial_features.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_polynomial,y,test_size = 0.25)

parameters = {'alpha':[0.0000000001, 0.00000001, 0.000001, 0.00001, 0.0001, 0.001, 0.2, 0.4, 0.6, 0.8, 1, 5, 10, 20]}
ridge =  Ridge()
ridge_regressor = GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(x_polynomial,y)
print("Best value of parameters : ",ridge_regressor.best_params_)
print("Best value of negative mean squared error : ",ridge_regressor.best_score_)

# we get the best value of alpha to be 1e-10
ridgemodelchosen = Ridge(alpha = 1e-10, tol = 0.075)
ridgemodelchosen.fit(x_train,y_train)
y_pred = ridgemodelchosen.predict(x_polynomial)
plt.scatter(x,y)
plt.plot(x,y_pred)
plt.show()

#5 b) Maximum Likelihood estimation with gaussian distribution
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy.optimize as opt
def norm_pdf(xvals,mu,sigma,cutoff):
  if cutoff == 'None':
    prob_notcut = 1.0
  else:
    prob_notcut = sts.norm.cdf(cutoff,loc=mu,scale=sigma)
  pdf_vals = ((1/(sigma*np.sqrt(2*np.pi))*np.exp(-(xvals-mu)**2/(2*sigma**2)))/prob_notcut)
  return pdf_vals

def log_lik_norm(xvals,mu,sigma,cutoff):
  pdf_vals = norm_pdf(xvals,mu,sigma,cutoff)
  ln_pdf_vals = np.log(pdf_vals)
  log_lik_vals = ln_pdf_vals.sum()
  return log_lik_vals

def criterion_func(params,*args):
  mu,sigma = params
  xvals,cutoff = args
  log_lik_val = log_lik_norm(xvals,mu,sigma,cutoff)
  neg_log_lik_val = -log_lik_val
  return neg_log_lik_val

N = 1000
# Generate equispaced floats in the interval [0, 2π] 
x = np.linspace(0, 2*np.pi, N) 
# Generate noise 
mean = 0 
std = 0.05
# Generate some numbers from the sine function 
y = np.sin(x) 
# Add noise 
y += np.random.normal(mean, std, N)

mu_init = 0
sigma_init = 0.5
params_init = np.array([mu_init,sigma_init])
mle_args = (y,'None')
results = opt.minimize(criterion_func,params_init,args = (mle_args))
mu_mle,sigma_mle = results.x
print("mu_MLE : ",mu_mle," sigma_MLE : ",sigma_mle)

#5 c) Maximum Aposteriori Probability(MAP) estimation with Gaussian priors
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy.optimize as opt
mu_0 = 0
sigma_0 = 1
mu_1 = 0.5
sigma_1 = 0.75
def norm_pdf(xvals,mu,sigma,cutoff):
  if cutoff == 'None':
    prob_notcut = 1.0
  else:
    prob_notcut = sts.norm.cdf(cutoff,loc=mu,scale=sigma)
  pdf_vals = ((1/(sigma*np.sqrt(2*np.pi))*np.exp(-(xvals-mu)**2/(2*sigma**2)))/prob_notcut)
  return pdf_vals

def log_lik_norm(xvals,mu,sigma,cutoff):
  pdf_vals = norm_pdf(xvals,mu,sigma,cutoff)
  ln_pdf_vals = np.log(pdf_vals)
  log_lik_vals = ln_pdf_vals.sum()
  return log_lik_vals

def criterion_func(params,*args):
  mu,sigma = params
  xvals,cutoff = args
  log_lik_val = log_lik_norm(xvals,mu,sigma,cutoff)
  log_lik_mu = log_lik_norm(mu,mu_0,sigma_0,cutoff)
  log_lik_sigma = log_lik_norm(sigma,mu_1,sigma_1,cutoff)
  neg_log_lik_val = -(log_lik_val+log_lik_mu+log_lik_sigma)
  return neg_log_lik_val

N = 1000
# Generate equispaced floats in the interval [0, 2π] 
x = np.linspace(0, 2*np.pi, N) 
# Generate noise 
mean = 0 
std = 0.05
# Generate some numbers from the sine function 
y = np.sin(x) 
# Add noise 
y += np.random.normal(mean, std, N)

mu_init = 0
sigma_init = 0.5


params_init = np.array([mu_init,sigma_init])
mle_args = (y,'None')
results = opt.minimize(criterion_func,params_init,args = (mle_args))
mu_map,sigma_map = results.x
print("mu_MAP : ",mu_map," sigma_MAP : ",sigma_map)

#6 MLE Estimation
# d) Gaussian Distribution

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy.optimize as opt

def norm_pdf(xvals,mu,sigma,cutoff):
  if cutoff == 'None':
    prob_notcut = 1.0
  else:
    prob_notcut = sts.norm.cdf(cutoff,loc=mu,scale=sigma)
  pdf_vals = ((1/(sigma*np.sqrt(2*np.pi))*np.exp(-(xvals-mu)**2/(2*sigma**2)))/prob_notcut)
  return pdf_vals

def log_lik_norm(xvals,mu,sigma,cutoff):
  pdf_vals = norm_pdf(xvals,mu,sigma,cutoff)
  ln_pdf_vals = np.log(pdf_vals)
  log_lik_vals = ln_pdf_vals.sum()
  return log_lik_vals

def criterion_func(params,*args):
  mu,sigma = params
  xvals,cutoff = args
  log_lik_val = log_lik_norm(xvals,mu,sigma,cutoff)
  neg_log_lik_val = -log_lik_val
  return neg_log_lik_val

mean = 300
std_dev = 30
dist_pts = np.random.normal(mean,std_dev,500)
y = dist_pts


mu_init = 1000
sigma_init = 50
params_init = np.array([mu_init,sigma_init])
mle_args = (y,'None')
results = opt.minimize(criterion_func,params_init,args = (mle_args))
mu_mle,sigma_mle = results.x
print("mu_MLE : ",mu_mle," sigma_MLE : ",sigma_mle)

#Estimated mean and variance by Maximum likelihood estimation turns out to be mean = 299.735, variance = 29.94

plt.scatter(y,norm_pdf(y,mean,std_dev,'None'),color='red',label='1: Original Dataset')

z = np.random.normal(mu_mle,sigma_mle,500)
plt.scatter(z,norm_pdf(z,mu_mle,sigma_mle,'None'),label='2: Dataset using estimated parameters')
plt.legend(loc='upper left')

#6 c) MLE Estimation for "exponential Distribution"

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy.optimize as opt

def exp_pdf(xvals,lambda_p,cutoff):
  if cutoff == 'None':
    prob_notcut = 1.0
  else:
    prob_notcut = sts.expon.cdf(cutoff,loc=0,scale=1/lambda_p)
  pdf_vals = sts.expon.pdf(xvals,loc=0,scale=1/lambda_p)/prob_notcut
  return pdf_vals

def log_lik_exp(xvals,lambda_p,cutoff):
  pdf_vals = exp_pdf(xvals,lambda_p,cutoff)
  ln_pdf_vals = np.log(pdf_vals)
  log_lik_vals = ln_pdf_vals.sum()
  return log_lik_vals

def criterion_func(params,*args):
  lambda_p = params
  xvals,cutoff = args
  log_lik_val = log_lik_exp(xvals,lambda_p,cutoff)
  neg_log_lik_val = -log_lik_val
  return neg_log_lik_val

lambda_0 = 5
dist_pts = np.random.exponential(1/lambda_0,500)
y = dist_pts


lambda_init = 20
params_init = np.array([lambda_init])
mle_args = (y,'None')
results = opt.minimize(criterion_func,params_init,args = (mle_args))
lambda_mle = results.x
print("lambda_MLE : ",lambda_mle)

#Estimated mean and variance by Maximum likelihood estimation turns out to be lambda = 4.739

plt.scatter(y,exp_pdf(y,lambda_0,'None'),color='red',label='1: Original Dataset')

z = np.random.exponential(lambda_mle,500)
plt.scatter(z,exp_pdf(z,lambda_mle,'None'),label='2: Dataset using estimated parameters')
plt.legend(loc='upper left')

# 6 e) MLE estimation with "Laplacian Distribution"
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy.optimize as opt

def laplace_pdf(xvals,mu,sigma,cutoff):
  if cutoff == 'None':
    prob_notcut = 1.0
  else:
    prob_notcut = sts.laplace.cdf(cutoff,loc=mu,scale=sigma)
  pdf_vals = (sts.laplace.pdf(xvals,mu,sigma)/prob_notcut)
  return pdf_vals

def log_lik_laplace(xvals,mu,sigma,cutoff):
  pdf_vals = laplace_pdf(xvals,mu,sigma,cutoff)
  ln_pdf_vals = np.log(pdf_vals)
  log_lik_vals = ln_pdf_vals.sum()
  return log_lik_vals

def criterion_func(params,*args):
  mu,sigma = params
  xvals,cutoff = args
  log_lik_val = log_lik_laplace(xvals,mu,sigma,cutoff)
  neg_log_lik_val = -log_lik_val
  return neg_log_lik_val

mean = 300
std_dev = 3
dist_pts = np.random.laplace(mean,std_dev,500)
y = dist_pts


mu_init = 100
sigma_init = 5
params_init = np.array([mu_init,sigma_init])
mle_args = (y,'None')
results = opt.minimize(criterion_func,params_init,args = (mle_args))
mu_mle,sigma_mle = results.x
print("loc_MLE : ",mu_mle," scale_MLE : ",sigma_mle)

#Estimated mean and variance by Maximum likelihood estimation turns out to be loc = 300.0252, variance = 2.94

plt.scatter(y,laplace_pdf(y,mean,std_dev,'None'),color='red',label='1: Original Dataset')

z = np.random.laplace(mu_mle,sigma_mle,500)
plt.scatter(z,laplace_pdf(z,mu_mle,sigma_mle,'None'),label='2: Dataset using estimated parameters')
plt.legend(loc='upper left')

