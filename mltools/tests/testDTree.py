import numpy as np
import sys
sys.path.append('../../')

import mltools as ml
import mltools.dtree
import matplotlib.pyplot as plt

np.random.seed(0)

XY = np.loadtxt('data/iris.txt')
X, Y = XY[:,:-1], XY[:,-1]
X,Y = ml.shuffleData(X,Y)
Xtr,Xva,Ytr,Yva = ml.splitData(X,Y,0.80)

model = ml.dtree.treeClassify( Xtr, Ytr );
print model.err(Xva,Yva)
# 0.0666666666667
print model.nll(Xva,Yva)
# 0.584066443011

model = ml.dtree.treeClassify( Xtr, Ytr, maxDepth=2 );
print model.err(Xva,Yva)
# 0.0666666666667
print model.nll(Xva,Yva)
# 0.283356420648

model = ml.dtree.treeClassify( Xtr, Ytr, minParent=20 );
print model.err(X,Y)
# 0.0135135135135
print model.nll(X,Y)
# 0.293316254332


model = ml.dtree.treeClassify( X[:,:2], Y );
ml.plotClassify2D( model, X[:,:2], Y)
plt.show()

model = ml.dtree.treeClassify( X[:,:2], Y, maxDepth=2);
ml.plotClassify2D( model, X[:,:2], Y)
plt.show()

model = ml.dtree.treeClassify( X[:,:2], Y, minParent=20);
ml.plotClassify2D( model, X[:,:2], Y)
plt.show()


