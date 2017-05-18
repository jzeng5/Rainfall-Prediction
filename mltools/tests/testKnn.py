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

Xtr, scale = ml.transforms.rescale(Xtr)
Xva, _ = ml.transforms.rescale(Xva, scale)


model = ml.knn.knnClassify( Xtr, Ytr );
print model.err(Xtr,Ytr), model.err(Xva,Yva)
# 0.0 0.0666666666667
print model.nll(Xtr,Ytr), model.nll(Xva,Yva)
# -0.0 inf

model = ml.knn.knnClassify( Xtr, Ytr, K=5 );
print model.err(Xtr,Ytr), model.err(Xva,Yva)
# 0.0254237288136 0.0333333333333
print model.nll(Xtr,Ytr), model.nll(Xva,Yva)
# 0.0672872960653 inf

model = ml.knn.knnClassify( Xtr, Ytr, K=100, alpha=0.1 );
print model.err(Xtr,Ytr), model.err(Xva,Yva)
# 0.0932203389831 0.133333333333
print model.nll(Xtr,Ytr), model.nll(Xva,Yva)
# 0.641303145273 0.673828596126


model = ml.knn.knnClassify( Xtr[:,:2], Ytr );
ml.plotClassify2D( model, Xtr[:,:2], Ytr)
plt.show()

model = ml.knn.knnClassify( Xtr[:,:2], Ytr, K=5);
ml.plotClassify2D( model, Xtr[:,:2], Ytr)
plt.show()

model = ml.knn.knnClassify( Xtr[:,:2], Ytr, K=100, alpha=.1);
ml.plotClassify2D( model, Xtr[:,:2], Ytr)
plt.show()


