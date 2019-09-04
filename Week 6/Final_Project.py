# Overall runtime is ~8 minutes
# My H20 cluster has: 1 node, 4 cores, 3Gb free memory

# Set up the libraries
import h2o
import pandas as pd
h2o.init()

# Load the data
data = h2o.import_file("http://coursera.h2o.ai/house_data.3487.csv")

# Convert date into year and month
data = data.as_data_frame() # didn't manage to work with H2O frames, sorry for that
data['year'] = [data['date'][i][:4] for i in range(len(data))]
data['month'] = [data['date'][i][4:6] for i in range(len(data))]
data = h2o.H2OFrame(data)

# Split the data
train, test = data.split_frame([0.9], seed = 123)
print("Train length: %d / Test length: %d" %(train.nrows, test.nrows))

# Split again to get a validation set
train, valid = train.split_frame([0.85], seed = 123)
print("Train length: %d / Test length: %d / Valid length: %d" %(train.nrows, test.nrows, valid.nrows))

# Specify the x and y variables
y = 'price'
ignoreFields = ['id', 'date', 'price', 'year', 'month']
x = [i for i in data.names if i not in ignoreFields]

## First model : Gradient Boosting Machine - 2'31'' to run
from h2o.estimators.gbm import H2OGradientBoostingEstimator
my_GBM = H2OGradientBoostingEstimator(
    model_id = 'my_GBM',
    ntrees = 1200,
    max_depth = 4,
    learn_rate = 0.05,
    distribution = 'gaussian',
    seed = 123, 
    nfolds = 5,
    fold_assignment = "Modulo", 
    keep_cross_validation_predictions = True
    )

# Training and evaluation
my_GBM.train(x, y, train, validation_frame = valid)
my_GBM.model_performance(valid)

## Second model: Neural Network - 6'26''
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
my_NN = H2ODeepLearningEstimator(
        model_id = 'my_NN',
        epochs = 20,
        hidden = [400, 200],
        stopping_rounds = 2,
        seed = 123, 
        nfolds = 5,
        fold_assignment = "Modulo", 
        keep_cross_validation_predictions = True)

# Training & evaluation
my_NN.train(x, y, train, validation_frame = valid)
my_NN.model_performance(valid)

## Third model: Random Forest - 1'23'' to run
from h2o.estimators.random_forest import H2ORandomForestEstimator
my_RF = H2ORandomForestEstimator(
        model_id = 'my_RF',
        ntrees = 100,
        max_depth = 50,
        seed = 123, 
        nfolds = 5,
        fold_assignment = "Modulo", 
        keep_cross_validation_predictions = True)
my_RF.train(x, y, train, validation_frame = valid)
my_RF.model_performance(valid)

## Fourth model: GLM - 4'' to run
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
my_GLM = H2OGeneralizedLinearEstimator(
         model_id = 'my_GLM',
         alpha = 0.9,
         lambda_ = 0.001,
         family = 'poisson',
         standardize = True,
         seed = 123, 
         nfolds = 5,
         fold_assignment = "Modulo", 
         keep_cross_validation_predictions = True)
my_GLM.train(x, y, train, validation_frame = valid)
my_GLM.model_performance(valid)

## Save models
h2o.save_model(model = my_GBM, force = True)
h2o.save_model(model = my_NN, force = True)
h2o.save_model(model = my_RF, force = True)
h2o.save_model(model = my_GLM, force = True)

## Stacked ensemble - 21'' to run
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
models = [my_GBM.model_id, my_NN.model_id, my_RF.model_id, my_GLM.model_id]
my_SE = H2OStackedEnsembleEstimator(model_id = "SE_glm_gbm_rf_nn",
                                   base_models = models,
                                   metalearner_algorithm = 'deeplearning',
                                   seed = 12)
my_SE.train(x, y, train, validation_frame = valid)
h2o.save_model(model = my_SE, force = True)

# Compare models performance
all_models = [my_GBM, my_NN, my_RF, my_GLM, my_SE]
names = ["GBM", "NN", "RF", "GLM", "SE"]
test_perf = list(map(lambda x: x.model_performance(valid), all_models))
pd.Series(map(lambda p: p.rmse(), test_perf), names)
# Better performance on the Stacked Ensemble!

## Performance on the test data
my_SE.model_performance(test)

# Performance reached on the test data : RMSE = 120575 < 123000
# For some reasons, every SE training, gives different evaluation metrics
# You might not find 120575 but I ran several and every SE RMSE was < 123000.