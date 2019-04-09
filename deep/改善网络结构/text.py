import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec


Da = np.random.rand(4, 3)
Db = np.random.randn(4, 3)
print(Da)
print(Db)
Da = Da < 0.5
print(Da)
Dc = np.multiply(Da, Db)
print(Dc)
Dc = Dc / 0.5
print(Dc)

a = np.zeros(Dc.shape)
print(a)
