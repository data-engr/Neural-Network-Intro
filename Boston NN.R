# https://rpubs.com/julianhatwell/annr

# Using my standard boiler plate for machine learning
# Run through some standard checks for data quality
dt <- setData(Boston, "medv")

# problem type: either classification or regression
dt$ptype

# check for NA vals
na.vals.check(dt)

# and near zero variance
nzv.check(dt) 

# and highly correlated variables
cor.vars.check(dt, 0.8)

# and linear combinations
lin.comb.check(dt)

# Have a closer look at the tax variable
corrplot(cor(dt$dt.frm))

#########################################
# ANN Model fits with neuralnet Library #
#########################################
# The predictor vars must be scaled data for the ANN fitting
Boston.scaled <- as.data.frame(scale(Boston))
min.medv <- min(Boston$medv)
max.medv <- max(Boston$medv)
# response var must be scaled to [0 < resp < 1]
Boston.scaled$medv <- scale(Boston$medv
                            , center = min.medv
                            , scale = max.medv - min.medv)

# Train-test split
Boston.train.scaled <- Boston.scaled[Boston.split, ]
Boston.test.scaled <- Boston.scaled[!Boston.split, ]

# neuralnet doesn't accept resp~. (dot) notation
# so a utility function to create a verbose formula is used
Boston.nn.fmla <- generate.full.fmla("medv", Boston)

# 2 models, one with 2 layers of 5 and 3
# the second with one layer of 8
# linear output is used for a regression problem
Boston.nn.5.3 <- neuralnet(Boston.nn.fmla
                           , data=Boston.train.scaled
                           , hidden=c(5,3)
                           , linear.output=TRUE)

Boston.nn.8 <- neuralnet(Boston.nn.fmla
                         , data=Boston.train.scaled
                         , hidden=8
                         , linear.output=TRUE)

##########################
# Visualizing Neural Net #
##########################

#neuralnet Library come with built in plot function

###################################
# Determining Variable Importance #
###################################

# garson function on two hidden layer model returns an error
tryCatch.W.E(garson(Boston.nn.5.3))


# works fine on single hidden layer model
garson(Boston.nn.8)

###################
# Model Profiling #
###################

# not working
# expected to fail
tryCatch.W.E(lekprofile(Boston.nn.5.3)) # ex

# something wrong as there is one hidden layer
tryCatch.W.E(lekprofile(Boston.nn.8))

#######################################
# Cross Validation to find best model #
#######################################

# Linear model cross validation
# same seed value each time
seed.val <- 2016
Boston.glm.full <- glm(medv~.,data=Boston) # full data set

set.seed(seed.val)
Boston.cv.RMSE <- cv.glm(Boston, Boston.glm.full
                         , cost = RMSE
                         , K=10)$delta[1]

set.seed(seed.val)
Boston.cv.MAD <- cv.glm(Boston, Boston.glm.full
                        , cost = MAD
                        , K=10)$delta[1]
# delta[1] is unadjusted for fair comparison
Boston.cv.RMSE

Boston.cv.MAD

set.seed(seed.val)
k <- 10
cv.error <- matrix(nrow = k, ncol = 4)

folds <- sample(1:k, nrow(Boston)
                , replace = TRUE)

for(i in 1:k){
  Boston.train.cv <- Boston.scaled[folds != i,]
  Boston.test.cv <- Boston.scaled[folds == i,]
  
  nn.5.3 <- neuralnet(Boston.nn.fmla
                      , data=Boston.train.cv
                      , hidden=c(5,3)
                      , linear.output=TRUE)
  
  nn.8 <- neuralnet(Boston.nn.fmla
                    , data=Boston.train.cv
                    , hidden=8
                    , linear.output=TRUE)
  
  Boston.5.3.preds.scaled <- neuralnet::compute(nn.5.3, Boston.test.cv[, 1:13])
  Boston.8.preds.scaled <- neuralnet::compute(nn.8, Boston.test.cv[, 1:13])
  
  Boston.5.3.preds <- Boston.5.3.preds.scaled$net.result * (max(Boston$medv) - min(Boston$medv)) + min(Boston$medv)
  Boston.8.preds <- Boston.8.preds.scaled$net.result * (max(Boston$medv) - min(Boston$medv)) + min(Boston$medv)
  
  medv.unscaled <- (Boston.test.cv$medv) * (max(Boston$medv) - min(Boston$medv)) + min(Boston$medv)
  
  cv.error[i, ] <- c(
    RMSE(medv.unscaled, Boston.5.3.preds)
    , MAD(medv.unscaled, Boston.5.3.preds)
    , RMSE(medv.unscaled, Boston.8.preds)
    , MAD(medv.unscaled, Boston.8.preds)
  )
}

t.test(cv.error[,1], cv.error[,3]) # RMSE

t.test(cv.error[,2], cv.error[,4]) # MAD

# MAD shows signif difference
# shows majority of errors are within a smaller bound
# a few poor error predictions

