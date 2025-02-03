library(MASS)
library(boot)
library(parallel)
library(ggplot2)
# Set seed for reproducibility
set.seed(123)
n_samples <- 100

# Generate bootstrap samples
bootstrap_indices <- lapply(1:n_samples, function(i) sample(1:nrow(Boston), replace = TRUE))

# Function to fit GLM and extract
fit_glm_and_extract_aic <- function(indices) {
  sample_data <- Boston[indices, ]
  model <- glm(medv ~ ., data = sample_data)
  aic_value <- AIC(model)
  return(aic_value)
}

# Serial Execution
time_serial <- system.time({
  aic_serial <- sapply(bootstrap_indices, fit_glm_and_extract_aic)
})

# Parallel Execution
num_cores <- detectCores() - 1

# Create a cluster
cl <- makeCluster(num_cores)

# Export necessary objects to the cluster
clusterExport(cl, varlist = c("Boston", "fit_glm_and_extract_aic", "bootstrap_indices"))

# Measure execution time for parallel processing
time_parallel <- system.time({
  aic_parallel <- parLapply(cl, bootstrap_indices, fit_glm_and_extract_aic)
  aic_parallel <- unlist(aic_parallel)
})

# Stop the cluster after processing
stopCluster(cl)

# Aggregating Results
mean_serial <- mean(aic_serial, na.rm = TRUE)
iqr_serial <- IQR(aic_serial, na.rm = TRUE)

mean_parallel <- mean(aic_parallel, na.rm = TRUE)
iqr_parallel <- IQR(aic_parallel, na.rm = TRUE)

# Plotting Results
data_plot <- data.frame(
  AIC = c(aic_serial, aic_parallel),
  Method = rep(c("Serial", "Parallel"), each = n_samples)
)

ggplot(data_plot, aes(x = Method, y = AIC, fill = Method)) +
  geom_boxplot() +
  labs(title = "Comparison of AIC from Serial and Parallel Execution", y = "AIC") +
  theme_minimal()

# Display execution times and statistics
print(time_serial)
print(time_parallel)
print(paste("Mean (Serial):", mean_serial, "IQR (Serial):", iqr_serial))
print(paste("Mean (Parallel):", mean_parallel, "IQR (Parallel):", iqr_parallel))
