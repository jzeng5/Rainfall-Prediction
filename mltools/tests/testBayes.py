import numpy as np
import sys
sys.path.append('../../')

import mltools as ml
import mltools.bayes
import matplotlib.pyplot as plt

XY = np.loadtxt('data/iris.txt')
X, Y = XY[:,:-1], XY[:,-1]

model = ml.bayes.gaussClassify( X, Y );

Yhat = model.predict(X)

print model.err(X,Y)
# 0.0202702702703
print model.nll(X,Y)
# 0.0360394614996

model = ml.bayes.gaussClassify( X, Y, equal=True );
print model.err(X,Y)
# 0.0135135135135
print model.nll(X,Y)
# 0.0880380736893

model = ml.bayes.gaussClassify( X, Y, diagonal=True );
print model.err(X,Y)
# 0.0405405405405
print model.nll(X,Y)
# 0.112463365158


model = ml.bayes.gaussClassify( X[:,:2], Y );
ml.plotClassify2D( model, X[:,:2], Y)
plt.show()

model = ml.bayes.gaussClassify( X[:,:2], Y, equal=True );
ml.plotClassify2D( model, X[:,:2], Y)
plt.show()

model = ml.bayes.gaussClassify( X[:,:2], Y, diagonal=True );
ml.plotClassify2D( model, X[:,:2], Y)
plt.show()


