# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h2o
h2o.init()

# Import data
url = "http://coursera.h2o.ai/cacao.882.csv"
data = h2o.import_file(url)
train, test, valid = data.split_frame([0.8, 0.1])

# Set the dependent and explanaroty variables
y = 'Maker Location'
ignoreFields = ['Maker Location']
x = [i for i in data.names if i not in ignoreFields]

# Baseline model (takes 30 seconds to run)
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
baseline = H2ODeepLearningEstimator(seed = 10, epochs = 30)
%time baseline.train(x, y, train, validation_frame = valid)
baseline.model_performance(test) # MSE: 0.107

# Tuned model (takes 3'40'' to run - less than 10 times baseline model)
from h2o.grid.grid_search import H2OGridSearch
search_criteria = {'strategy': 'RandomDiscrete',
                   'seed' : 10,
                   'max_models' : 8}
hyper_params = {'activation': ["RectifierWithDropout"],
                "input_dropout_ratio" : [0, 0.05],         
                'l1': [0, 0.000001],
                'l2': [0, 0.000001],
                'hidden_dropout_ratios' : [[0, 0], [0.2,0.2]]}
tuned = H2OGridSearch(model = H2ODeepLearningEstimator(epochs = 30, stopping_rounds = 2),
                      grid_id = 'grid_test',
                      hyper_params = hyper_params,
                      search_criteria = search_criteria)
%time tuned.train(x, y, train, validation_frame = valid)

# Best model
grid = tuned.get_grid(sort_by = 'mse', decreasing = False)
tuned_model = grid[0]
tuned_model.model_performance(test) # MSE: 0.078 < baseline MSE (=0.107)

# Save the model
h2o.save_model(model = baseline, force=True)
h2o.save_model(model = tuned_model, force=True)
