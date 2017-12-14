## Customer Churn Prediction
## ANN with 'keras' package
## Created : 12/2/2017
## Modified : 12/2/2017
##============================

## References : 
## http://www.business-science.io/business/2017/11/28/customer_churn_analysis_keras.html

## Analysis of the customer churn prediction problem with ANN with the keras package in R. Customer acquisition is a costly affair for businesses. It is critical to both be able to predict churn and identify the features that relate to customer churn.

## ANN is a subgroup of deep learning, and they can be superior to traditional regression and classification methods because of the ability to model interactions between features that would otherwise go undetected. 

rm(list = ls(all = TRUE))

## Load the necessary packages
library(keras)
library(lime)
library(tidyverse)
library(rsample)
library(recipes)
library(yardstick)
library(corrr)
library(summarytools)
library(tidyquant)

## Install Keras for the first time
# install_keras()

## Read in the data file and save it 
fileLoc <- "https://community.watsonanalytics.com/wp-content/uploads/2015/03/WA_Fn-UseC_-Telco-Customer-Churn.csv?cm_mc_uid=60538928144815121676500&cm_mc_sid_50200000=1512167650&cm_mc_sid_52640000=1512167650"

churn <- read.csv(fileLoc, stringsAsFactors = FALSE)
glimpse(churn)

##==========================================##
##         Preprocessing the data
##==========================================##

## Repeat the preprocessing steps as earlier
# - remove customerID
# - drop observaions with NAs in total charges

churn.clean <- churn %>% 
        dplyr::select(- customerID) %>% 
        drop_na() %>% 
        dplyr::select(Churn, everything())

glimpse(churn.clean)

## Split the data into train and test - use package rsample
set.seed(100)
train.test.split <- initial_split(churn.clean, prop = 0.8)
train.test.split

## Retrieve test and train datasets
train <- training(train.test.split)
test <- testing(train.test.split)

## Data transformations
## ANN runs best when the data is one-hot encoded, scaled and centered.

## 1. Discretize the tenure feature
ggplot(churn.clean, aes(x = tenure)) + 
        geom_bar(color = "white") +
        xlab("tenure (months)") +
        ggtitle("Tenure Counts without Binning")

binsize <- diff(range(churn$tenure))/5
ggplot(churn.clean, aes(x = tenure)) + 
        geom_histogram(color = "white", binwidth = binsize) +
        xlab("tenure (months)") +
        ggtitle("Tenure Counts with 6 Bins")


## 2. Transform the Totalcharges feature
ggplot(churn.clean, aes(x = TotalCharges)) + 
        geom_histogram(color = "white", binwidth = 100) +
        xlab("tenure (months)") +
        ggtitle("TotalCharges Histogram, 100 Bins")
## Total charges has a skewed distribution. Log transformation to get a normal-ish distribution.

## Convert it to log(TotalCharges)
ggplot(churn.clean, aes(x = log(TotalCharges))) + 
        geom_histogram(color = "white") +
        xlab("tenure (months)") +
        ggtitle("Log(TotalCharges) Histogram, 100 Bins")

## Test to check the magnitude of the correlation b/w Totalcharges and Churn
train %>% 
        select(Churn, TotalCharges) %>% 
        mutate(Churn = Churn %>% as.factor() %>%  as.numeric(),
               LogTotalCharges = log(TotalCharges)) %>% 
        corrr::correlate() %>% 
        focus(Churn) %>%     ## similar to select()
        fashion()            ## tabular layout for readability

## One-hot encoding the data
dfSummary(churn, style = "grid", plain.ascii = TRUE)
## Features that have multiple categories are : Contract, MultipleLines, InternetService, PaymentMethod, OnlineBackup, DeviceProtection etc.

## Correlation has improved. 
## Next, do one-hot encoding or creating dummy variables by converting categorical values to 0 or 1.
## ANN's perform faster if features are scaled and/or normalized. ANN's use gradient descent, weights tend to update faster here. Feature scaling is important in the following cases also:
## knn with Euclidean distance (if all features should contribute equally), k-means, logistic regression, SVMs, NN etc. that use gradient descent.
## LDA, PCA etc. to find variables contributing to highest overall variance

## Pre-processing steps 
## 1. Discretize the tenure feature
## 2. Log transformation of TotalCharges
## 3. One-hot encode categorical data
## 4. Feature Scaling

## All the above steps can be accomplished with recepies package
# Create recipe object
rec_obj <- recipe(Churn ~ ., data = train) %>%
        step_discretize(tenure, options = list(cuts = 6)) %>%
        step_log(TotalCharges) %>%
        step_dummy(all_nominal(), -all_outcomes()) %>%
        step_center(all_predictors(), -all_outcomes()) %>%    ## centering
        step_scale(all_predictors(), -all_outcomes()) %>%     ## scaling
        prep(data = train)
rec_obj

## Bake with this recepie
## Predictors
x_train <- bake(rec_obj, newdata = train) %>% select(-Churn)
x_test <- bake(rec_obj, newdata = test) %>% select(-Churn)

glimpse(x_train)

## Response variable for train and test sets
y_train <- ifelse(pull(train, Churn) == "Yes", 1, 0)
y_test <- ifelse(pull(test, Churn) == "Yes", 1, 0)

## Deep Learning with Keras
## https://www.xenonstack.com/blog/overview-of-artificial-neural-networks-and-its-applications

## Building a multi-layer perceptron
## good for regression, binary and multi classification

## 1. Initialize keras model
## 2. Apply layers - number of hidden layers, nodes, activation function etc.
##                  dropout layers - eliminate weights below a threshold (prevent low weights from overfitting)
##                  output layer - sigmoid activation
## 3. compile the layer

# Building our Artificial Neural Network
model_keras <- keras_model_sequential()

model_keras %>% 
        # First hidden layer
        layer_dense(
                units              = 16,                      ## number of nodes
                kernel_initializer = "uniform",               
                activation         = "relu",                  ## activation fn.
                input_shape        = ncol(x_train)) %>%         ## number of inputs
        # Dropout to prevent overfitting
        layer_dropout(rate = 0.1) %>%                         ## dropout threshold (remove weights <10%)
        # Second hidden layer
        layer_dense(
                units              = 16,                      ## similar to first hidden layer
                kernel_initializer = "uniform", 
                activation         = "relu") %>% 
        # Dropout to prevent overfitting
        layer_dropout(rate = 0.1) %>%
        # Output layer
        layer_dense(
                units              = 1,                       ## single node for binary values
                kernel_initializer = "uniform", 
                activation         = "sigmoid") %>%           ## activation function
        # Compile ANN
        compile(          
                optimizer = 'adam',                           ## optimization algorithm
                loss      = 'binary_crossentropy',            ## loss fn. for binary
                metrics   = c('accuracy')                     ## performance metric
        ) 
model_keras

## About Adam https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

# Fit the keras model to the training data
fit_keras <- fit(
        object           = model_keras,        ## the model defined above
        x                = as.matrix(x_train), ## input
        y                = y_train,            ## target
        batch_size       = 50,                 ## num. of samples per gradient update
        epochs           = 35,                 ## num. of training cycles
        validation_split = 0.30                ## set 30% data for validation (reduce overfitting)
)

##print the final model
fit_keras

# Plot the training/validation history of our Keras model
plot(fit_keras) +
        theme_tq() +
        scale_color_tq() +
        scale_fill_tq() +
        labs(title = "Deep Learning Training Results")

## Prediction on test data
# Predicted Class - class values as 1 and 0
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(x_test)) %>%
        as.vector()

# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(x_test)) %>%
        as.vector()

## Performance metrics with yardstick
# Format test data and predictions for yardstick metrics
estimates_keras_tbl <- tibble(
        truth      = as.factor(y_test) %>% fct_recode(yes = "1", no = "0"),
        estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
        class_prob = yhat_keras_prob_vec
)

estimates_keras_tbl

## Set classify 1 as the positive class instead of 0 (default)
options(yardstick.event_first = FALSE)

## Confusion Matrix
# Confusion Table
estimates_keras_tbl %>% conf_mat(truth, estimate)

# Accuracy
estimates_keras_tbl %>% metrics(truth, estimate)

# AUC
estimates_keras_tbl %>% roc_auc(truth, class_prob)
## AUC is often a good metric used to compare different classifiers and to compare to randomly guessing (AUC_random = 0.50). Our model has AUC = 0.85, which is much better than randomly guessing. 

# Precision
tibble(
        precision = estimates_keras_tbl %>% precision(truth, estimate),
        recall    = estimates_keras_tbl %>% recall(truth, estimate)
)
## Precision and Recall: It is important to identify the customers who are likely to remain or leave as far as the business case is concerned. 

# F1-Statistic
estimates_keras_tbl %>% f_meas(truth, estimate, beta = 1)


## Explain the model with LIME package

## setup the model
class(model_keras)

# Setup lime::model_type() function for keras
model_type.keras.models.Sequential <- function(x, ...) {
        return("classification")
}       ## tell lime what type the model is - classification, regression etc. 

# Setup lime::predict_model() function for keras
predict_model.keras.models.Sequential <- function(x, newdata, type, ...) {
        pred <- predict_proba(object = x, x = as.matrix(newdata))
        return(data.frame(Yes = pred, No = 1 - pred))
}               ## perform predictions that the lime algorithm can interpret

# Test our predict_model() function
predict_model(x = model_keras, newdata = x_test, type = 'raw') %>%
        tibble::as_tibble()     ## pass arguments to the function defined. 

# Run lime() on training set
explainer <- lime::lime(
        x              = x_train, 
        model          = model_keras, 
        bin_continuous = FALSE)     

# Run explain() on explainer for test data
explanation <- lime::explain(
        x_test[1:10,],               ## limit to a subset
        explainer    = explainer, 
        n_labels     = 1, 
        n_features   = 4,
        kernel_width = 0.5)   ## return the top 4 features that explain Churn ("1")

## Feature Importance Visualization
## For the first 10 cases from the test data, the top 4 features are visualized. the green bars imply the feature supports the model conclusion, redd does not

plot_features(explanation) +
        labs(title = "LIME Feature Importance Visualization",
             subtitle = "Hold Out (Test) Set, First 10 Cases Shown")

plot_explanations(explanation) +
        labs(title = "LIME Feature Importance Heatmap",
             subtitle = "Hold Out (Test) Set, First 10 Cases Shown")

## Analysis with correlation for the entire train data
# Feature correlations to Churn
corrr_analysis <- x_train %>%
        mutate(Churn = y_train) %>%
        correlate() %>%
        focus(Churn) %>%
        rename(feature = rowname) %>%
        arrange(abs(Churn)) %>%
        mutate(feature = as_factor(feature)) 
corrr_analysis

# Correlation visualization
corrr_analysis %>%
        ggplot(aes(x = Churn, y = fct_reorder(feature, desc(Churn)))) +
        geom_point() +
        # Positive Correlations - Contribute to churn
        geom_segment(aes(xend = 0, yend = feature), 
                     color = palette_light()[[2]], 
                     data = corrr_analysis %>% filter(Churn > 0)) +
        geom_point(color = palette_light()[[2]], 
                   data = corrr_analysis %>% filter(Churn > 0)) +
        # Negative Correlations - Prevent churn
        geom_segment(aes(xend = 0, yend = feature), 
                     color = palette_light()[[1]], 
                     data = corrr_analysis %>% filter(Churn < 0)) +
        geom_point(color = palette_light()[[1]], 
                   data = corrr_analysis %>% filter(Churn < 0)) +
        # Vertical lines
        geom_vline(xintercept = 0, color = palette_light()[[5]], size = 1, linetype = 2) +
        geom_vline(xintercept = -0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
        geom_vline(xintercept = 0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
        # Aesthetics
        theme_tq() +
        labs(title = "Churn Correlation Analysis",
             subtitle = "Positive Correlations (contribute to churn), Negative Correlations (prevent churn)",
             y = "Feature Importance")

## We can identify features that have positive and negative correlation with churn
