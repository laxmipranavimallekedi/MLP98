library(mlbench)
library(purrr)

data("PimaIndiansDiabetes2")
ds <- as.data.frame(na.omit(PimaIndiansDiabetes2))
## fit a logistic regression model to obtain a parametric equation
logmodel <- glm(diabetes ~ .,
                data = ds,
                family = "binomial")
summary(logmodel)

cfs <- coefficients(logmodel) ## extract the coefficients
prednames <- variable.names(ds)[-9] ## fetch the names of predictors in a vector
prednames

sz <- 100000000 ## to be used in sampling
##sample(ds$pregnant, size = sz, replace = T)

dfdata <- map_dfc(prednames,
                  function(nm){ ## function to create a sample-with-replacement for each pred.
                    eval(parse(text = paste0("sample(ds$",nm,
                                             ", size = sz, replace = T)")))
                  }) ## map the sample-generator on to the vector of predictors
## and combine them into a dataframe

names(dfdata) <- prednames
dfdata

class(cfs[2:length(cfs)])

length(cfs)
length(prednames)
## Next, compute the logit values
pvec <- map((1:8),
            function(pnum){
              cfs[pnum+1] * eval(parse(text = paste0("dfdata$",
                                                     prednames[pnum])))
            }) %>% ## create beta[i] * x[i]
  reduce(`+`) + ## sum(beta[i] * x[i])
  cfs[1] ## add the intercept

## exponentiate the logit to obtain probability values of thee outcome variable
dfdata$outcome <- ifelse(1/(1 + exp(-(pvec))) > 0.5,
                         1, 0)


# Load required packages
library(xgboost)
library(caret)
library(dplyr)
library(Metrics)
library(pROC)

# Your dfdata is already created with the outcome variable

# Function to evaluate XGBoost at different sample sizes
evaluate_xgboost <- function(data, sizes = c(100, 1000, 10000, 100000, 1000000)) {
  results <- data.frame(
    Size = sizes,
    Method = "XGBoost in R - direct use of xgboost() with simple cross-validation",
    Accuracy = NA,
    Precision = NA,
    Recall = NA,
    F1 = NA,
    AUC = NA,
    Time = NA
  )
  
  for (i in seq_along(sizes)) {
    size <- sizes[i]
    
    # Sample the data
    set.seed(123)
    indices <- sample(1:nrow(data), size = min(size, nrow(data)))
    sample_data <- data[indices, ]
    
    # Prepare data for XGBoost
    features <- as.matrix(sample_data[, 1:8])
    labels <- as.numeric(sample_data$outcome)
    
    # Split into training and testing sets (80% training, 20% testing)
    set.seed(123)
    train_indices <- createDataPartition(labels, p = 0.8, list = FALSE)
    train_features <- features[train_indices, ]
    train_labels <- labels[train_indices]
    test_features <- features[-train_indices, ]
    test_labels <- labels[-train_indices]
    
    # Create DMatrix objects
    dtrain <- xgb.DMatrix(data = train_features, label = train_labels)
    dtest <- xgb.DMatrix(data = test_features, label = test_labels)
    
    # Set XGBoost parameters
    params <- list(
      objective = "binary:logistic",
      eval_metric = "error",
      eta = 0.1,
      max_depth = 6,
      subsample = 0.8,
      colsample_bytree = 0.8
    )
    
    # Start timing
    start_time <- Sys.time()
    
    # Train XGBoost model with cross-validation
    cv_model <- xgb.cv(
      params = params,
      data = dtrain,
      nrounds = 100,
      nfold = 5,
      early_stopping_rounds = 10,
      verbose = 0
    )
    
    # Get the best number of rounds
    best_nrounds <- cv_model$best_iteration
    
    # Train final model with best number of rounds
    model <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = best_nrounds,
      watchlist = list(train = dtrain, test = dtest),
      verbose = 0
    )
    
    # End timing
    end_time <- Sys.time()
    time_taken <- as.numeric(difftime(end_time, start_time, units = "secs"))
    
    # Make predictions
    predictions_prob <- predict(model, dtest)
    predictions <- ifelse(predictions_prob > 0.5, 1, 0)
    
    # Calculate metrics
    conf_matrix <- confusionMatrix(factor(predictions), factor(test_labels))
    roc_result <- roc(test_labels, predictions_prob)
    
    # Store results
    results$Accuracy[i] <- conf_matrix$overall["Accuracy"]
    results$Precision[i] <- conf_matrix$byClass["Pos Pred Value"]
    results$Recall[i] <- conf_matrix$byClass["Sensitivity"]
    results$F1[i] <- 2 * (results$Precision[i] * results$Recall[i]) / 
      (results$Precision[i] + results$Recall[i])
    results$AUC[i] <- auc(roc_result)
    results$Time[i] <- time_taken
  }
  
  return(results)
}

# Run evaluation
set.seed(123)
results <- evaluate_xgboost(dfdata)

library(knitr)
# Display results in the format requested
kable(results[, c("Size", "Method", "Accuracy", "Time")], 
      caption = "XGBoost Performance Across Different Dataset Sizes")



# Function to evaluate XGBoost using caret with 5-fold CV
evaluate_xgboost_caret <- function(data, sizes = c(100, 1000, 10000, 1000000, 10000000)) {
  results <- data.frame(
    Size = sizes,
    Method = rep("XGBoost in R – via caret, with 5-fold CV simple cross-validation", length(sizes)),
    Accuracy = NA,
    Time = NA
  )
  
  for (i in seq_along(sizes)) {
    size <- sizes[i]
    cat("Processing size:", size, "\n")
    
    # Sample the data (ensure we don't try to sample more than available)
    set.seed(123)
    sample_size <- min(size, nrow(data))
    indices <- sample(1:nrow(data), size = sample_size)
    sample_data <- data[indices, ]
    
    # Check if there are any issues with the outcome values
    unique_outcomes <- unique(sample_data$outcome)
    if(length(unique_outcomes) < 2) {
      cat("Warning: Dataset at size", size, "doesn't have both outcome classes\n")
      results$Accuracy[i] <- NA
      results$Time[i] <- NA
      next
    }
    
    # Convert the outcome to a proper factor with labels that definitely exist in the data
    sample_data$outcome <- factor(sample_data$outcome)
    
    # Split data (80% training, 20% testing)
    set.seed(123)
    train_index <- createDataPartition(sample_data$outcome, p = 0.8, list = FALSE)
    train_data <- sample_data[train_index, ]
    test_data <- sample_data[-train_index, ]
    
    # Verify both classes exist in training data
    if(length(unique(train_data$outcome)) < 2) {
      cat("Warning: Training set doesn't have both outcome classes\n")
      results$Accuracy[i] <- NA
      results$Time[i] <- NA
      next
    }
    
    # Define training control
    ctrl <- trainControl(
      method = "cv",
      number = 5,
      verboseIter = FALSE
    )
    
    # Start timing
    start_time <- Sys.time()
    
    # Train the model
    tryCatch({
      model <- train(
        x = train_data[, 1:8],
        y = train_data$outcome,
        method = "xgbTree",
        trControl = ctrl,
        tuneLength = 3,
        verbose = FALSE
      )
      
      # End timing
      end_time <- Sys.time()
      time_taken <- difftime(end_time, start_time, units = "secs")
      
      # Make predictions
      preds <- predict(model, test_data[, 1:8])
      
      # Calculate accuracy
      accuracy <- sum(preds == test_data$outcome) / length(preds)
      
      # Store results
      results$Accuracy[i] <- accuracy
      results$Time[i] <- as.numeric(time_taken)
    }, error = function(e) {
      cat("Error in training model for size", size, ":", e$message, "\n")
      results$Accuracy[i] <- NA
      results$Time[i] <- NA
    })
  }
  
  return(results)
}

print(table(dfdata$outcome))
dfdata$outcome <- as.numeric(dfdata$outcome)
print(table(dfdata$outcome))
results_caret <- evaluate_xgboost_caret(dfdata)
print(results_caret)