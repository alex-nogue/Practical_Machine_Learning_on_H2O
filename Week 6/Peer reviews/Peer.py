import h2o
import pandas as pd
import matplotlib as plot
%matplotlib inline
h2o.init()
url="http://coursera.h2o.ai/house_data.3487.csv"
data = h2o.import_file(url) 
data= data.as_data_frame()
data['date']=pd.to_datetime(data['date'])
data['date_year']=data['date'].map(lambda x: x.year)
data['date_month']=data['date'].map(lambda x: x.month)
data=h2o.H2OFrame(data)
ignore_columns = ['id', 'price', 'date']
training_columns=[n for n in data.names if n not in ignore_columns]
predict_column='price'
nfolds = 5
train, test = data.split_frame(ratios=[0.9], seed=123)
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
model_gbm = H2OGradientBoostingEstimator(
                                            model_id= "model_gbm",
                                            nfolds= nfolds,
                                            fold_assignment="Modulo",
                                            keep_cross_validation_predictions=True,
                                            ntrees = 200,
                                            sample_rate=0.7,
                                            col_sample_rate=0.4) 
model_gbm.train(x=training_columns, y=predict_column, training_frame = train )
model_rf= H2ORandomForestEstimator(
                                            model_id= "model_rf",
                                            nfolds= nfolds,
                                            fold_assignment="Modulo",
                                            keep_cross_validation_predictions=True,
                                            ntrees = 200)
model_rf.train(x=training_columns, y=predict_column, training_frame = train )
model_glm = H2OGeneralizedLinearEstimator(  
                                            
                                            model_id= "model_glm",
                                            nfolds= nfolds,
                                            fold_assignment="Modulo",
                                            keep_cross_validation_predictions=True)										
model_glm.train(x=training_columns, y=predict_column, training_frame = train)
models = [model_gbm, model_rf, model_glm]
final_se = H2OStackedEnsembleEstimator(
                                    model_id = "se_model",
                                    base_models=models)
final_se.train(x=training_columns, y=predict_column, training_frame = train)
model_gbm.model_performance(test)