library(h2o)

h2o.init()

url <- "http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv"
iris <- h2o.importFile(url)

parts <- h2o.splitFrame(iris, 0.8) # we can specify a seed as argument: seed = 123
train <- parts[[1]]
test <- parts[[2]]

summary(train)
nrow(train)
nrow(test)

# Automatic Deep Learning model (with default parameters)
mDL <- h2o.deeplearning(1:4, 5, train)
mDL

# Predictions
p <- h2o.predict(mDL, test)
p

h2o.performance(mDL, test)

# We can also run a AutoML algorithm (an Algorithm that checks several algorithms?)

mA <- h2o.automl(1:4, 5, train, max_runtime_secs = 30)
mA

mA@leaderboard # compares the classification error in the TRAINING dataset

p <- h2o.predict(mA@leader, test)
p

h2o.performance(mA@leader, test)


# Random Forest Algorithm

mRF <- h2o.randomForest(1:4, 5, train)
mRF
p <- h2o.predict(mRF, test)
h2o.performance(mRF, test)
?h2o.randomForest # to see the features available with the function h2o.randomForest


# Gradient Boosting Machines estimator

mGBM <- h2o.gbm(1:4, 5, train)
mGBM
p <- h2o.predict(mGBM, test)
h2o.performance(mGBM, test)
