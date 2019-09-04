#!/usr/bin/env python
# coding: utf-8

import h2o

from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator


URL = "http://coursera.h2o.ai/house_data.3487.csv"

h2o.init()

dataset = h2o.import_file(URL)

datetime_column = dataset['date'].as_date('%Y%M%dT000000')

dataset['month'] = datetime_column.month() 

dataset['year'] = datetime_column.year()

dataset = dataset.drop(['date', 'id'])

y = "price"

x = dataset.columns

x.remove(y)

train, test = dataset.split_frame([0.9], seed=123)

train, valid = train.split_frame([0.9], seed=123)

# Check we have the correct number of rows as specified in the project
# description.
assert train.nrows + valid.nrows == 19462
assert test.nrows == 2151

print("train rows:", train.nrows)
print("valid rows:", valid.nrows)
print("test rows:", test.nrows)

nfolds = 5

m1 = H2ORandomForestEstimator(
    model_id = "RF",
    nfolds = nfolds,
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = True,
)

m2 = H2OGradientBoostingEstimator(
    model_id = "GBM",
    ntrees = 500,       # default: 50
    max_depth = 4,      # default: 5
    nfolds = nfolds,
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = True
)

m3 = H2OXGBoostEstimator(
    model_id = "XGB",
    learn_rate = 0.1,   # default: 0.3
    ntrees = 500,       # default: 50
    max_depth = 4,      # default: 6
    nfolds = nfolds,
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = True
)

m4 = H2ODeepLearningEstimator(
    model_id = "DL",
    l1=1e-5,
    nfolds = nfolds,
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = True
)

m1.train(x, y, train)
m2.train(x, y, train)
m3.train(x, y, train)
m4.train(x, y, train)

base_models = [m1, m2, m3, m4]

m5 = H2OStackedEnsembleEstimator(
    model_id = "SE",
    base_models = base_models,
    metalearner_algorithm = "deeplearning",
    metalearner_params = {
        'epochs': 20,
        'hidden': [200, 200, 200],
        'l1': 1e-5,
    },
    metalearner_nfolds = nfolds,
    validation_frame = valid
)

m5.train(x, y, train)

for m in [m1, m2, m3, m4, m5]:
    print("%3s RMSE: %.2f" % (m.model_id, m.model_performance(test).rmse()))

best_model = sorted(set(base_models + [m5]), key=lambda x: x.model_performance(test).rmse())[0]

print(best_model.model_performance(test))

print("Model saved to: %s" % best_model.save_mojo('/tmp', force=True))

# ModelMetricsRegression: gbm
# ** Reported on test data. **
# 
# MSE: 13039579755.115294
# RMSE: 114190.97930710329
# MAE: 65266.318884520806
# RMSLE: 0.16689610604112504
# Mean Residual Deviance: 13039579755.115294
