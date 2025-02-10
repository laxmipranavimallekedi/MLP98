# Load necessary libraries
library(readxl)

# Read the data
df <- read_excel("clinics.xls")

# Convert locLat and locLong to numeric
df$locLat <- as.numeric(df$locLat)
df$locLong <- as.numeric(df$locLong)

# Define the Haversine function
haversine <- function(lat1, lon1, lat2, lon2) {
  R <- 3959 # Radius of Earth in miles
  lat1 <- lat1 * pi / 180
  lon1 <- lon1 * pi / 180
  lat2 <- lat2 * pi / 180
  lon2 <- lon2 * pi / 180
  
  dlat <- lat2 - lat1
  dlon <- lon2 - lon1
  
  a <- sin(dlat/2)^2 + cos(lat1) * cos(lat2) * sin(dlon/2)^2
  c <- 2 * asin(sqrt(a))
  
  return(R * c)
}

# For-loop approach
start_time <- Sys.time()
distances_loop <- numeric(nrow(df))
for (i in 1:nrow(df)) {
  distances_loop[i] <- haversine(40.671, -73.985, df$locLat[i], df$locLong[i])
}
end_time <- Sys.time()
cat("For-loop time (seconds):", as.numeric(difftime(end_time, start_time, units="secs")), "\n")

# Apply function approach
start_time <- Sys.time()
distances_apply <- apply(df[, c("locLat", "locLong")], 1, function(row) {
  haversine(40.671, -73.985, row[1], row[2])
})
end_time <- Sys.time()
cat("Apply function time (seconds):", as.numeric(difftime(end_time, start_time, units="secs")), "\n")

# Vectorized approach
start_time <- Sys.time()
distances_vectorized <- haversine(40.671, -73.985, df$locLat, df$locLong)
end_time <- Sys.time()
cat("Vectorized time (seconds):", as.numeric(difftime(end_time, start_time, units="secs")), "\n")
