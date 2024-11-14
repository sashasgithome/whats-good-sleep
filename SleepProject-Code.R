# ========================================================================================================
# Purpose:      CBA Paper (NTU Assignment)
# Author:       Sasha A.
# DOC:          22/07/2024
# Topics:       Linear Regression, CART, VIFs.
# Data Source:  sleep.csv
#=========================================================================================================

# Load necessary libraries
library(data.table)
library(ggplot2)
library(Metrics)
library(car)
library(rpart)
library(rpart.plot)

# Load the dataset
data <- fread("C:/Users/sasha/Documents/RDirectory/sleep.csv")

# Convert categorical variables to factors
data[, Gender := factor(Gender)]
data[, `Smoking status` := factor(`Smoking status`)]
data[, `Caffeine consumption` := as.numeric(`Caffeine consumption`)]
data[, `Alcohol consumption` := as.numeric(`Alcohol consumption`)]

# Check the structure of the dataset to ensure conversions
str(data)

# ========================================= EXPLORATORY DATA ANALYSIS (EDA)

# Plot distribution of Sleep Efficiency
ggplot(data, aes(x = `Sleep efficiency`)) +
  geom_histogram(binwidth = 0.05, fill = "purple", color = "black") +
  labs(title = "Distribution of Sleep Efficiency", x = "Sleep Efficiency", y = "Frequency")


# Scatter plot of Sleep Efficiency vs. Sleep Duration
ggplot(data, aes(x = `Sleep duration`, y = `Sleep efficiency`)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Sleep Efficiency vs. Sleep Duration", x = "Sleep Duration (Hours)", y = "Sleep Efficiency")

# Scatter plot of Sleep Efficiency vs. Awakenings
ggplot(data, aes(x = Awakenings, y = `Sleep efficiency`)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Sleep Efficiency vs. Awakenings", x = "Number of Awakenings", y = "Sleep Efficiency")

# Scatter plot of Sleep Efficiency vs. Caffeine Consumption
ggplot(data, aes(x = `Caffeine consumption`, y = `Sleep efficiency`)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Sleep Efficiency vs. Caffeine Consumption", x = "Caffeine Consumption (mg)", y = "Sleep Efficiency")

# Scatter plot of Sleep Efficiency vs. Alcohol Consumption
ggplot(data, aes(x = `Alcohol consumption`, y = `Sleep efficiency`)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Sleep Efficiency vs. Alcohol Consumption", x = "Alcohol Consumption Level (0 Lowest)", y = "Sleep Efficiency")

# Scatter plot of Sleep Efficiency vs. Exercise Frequency
ggplot(data, aes(x = `Exercise frequency`, y = `Sleep efficiency`)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Sleep Efficiency vs. Exercise Frequency", x = "Exercise Frequency (times per week)", y = "Sleep Efficiency")


# Boxplot of Sleep Efficiency by Smoking Status
ggplot(data, aes(x = `Smoking status`, y = `Sleep efficiency`)) +
  geom_boxplot(fill = c("red", "green")) +
  labs(title = "Sleep Efficiency by Smoking Status", x = "Smoking Status", y = "Sleep Efficiency")

# ========================================= TRAIN/TEST DATA SPLITTING


# Split the data into training and testing sets
set.seed(777)
trainIndex <- sample(1:nrow(data), 0.8 * nrow(data))
dataTrain <- data[trainIndex, ]
dataTest  <- data[-trainIndex, ]

# ========================================= LINEAR REGRESSION MODEL

model_v1 <- lm(`Sleep efficiency` ~ Awakenings + `Caffeine consumption` + `Alcohol consumption` + `Exercise frequency` + `Daily Steps` + `REM sleep percentage` + `Deep sleep percentage` + `Light sleep percentage` + `Gender` + `Smoking status` + `Age` + `Sleep duration`, data = dataTrain)
summary(model_v1)
vif(model_v1)

# Drop variables due to multicollinearity and re-fit models
model_v2 <- lm(`Sleep efficiency` ~ Awakenings + `Alcohol consumption` + `Exercise frequency` + `REM sleep percentage` + `Deep sleep percentage` + `Smoking status` + `Age`, data = dataTrain)
summary(model_v2)
vif(model_v2)

# Final LR model after insignificant variables are dropped
model_v3 <- lm(`Sleep efficiency` ~ Awakenings + `Alcohol consumption` + `REM sleep percentage` + `Deep sleep percentage` + `Smoking status` + `Age`, data = dataTrain)
summary(model_v3)
vif(model_v3)

# Predict and calculate RMSE for linear regression
linreg_model_results <- predict(model_v3, dataTest)
linreg_model_results 
linreg_rmse <- rmse(dataTest$`Sleep efficiency`, linreg_model_results)

# Create a data frame for comparison
linreg_comparison <- data.frame(
  Actual = dataTest$`Sleep efficiency`,
  Predicted = linreg_model_results
)

# Show the top 10 results for linear regression
head(linreg_comparison, 10)

# ========================================= CART MODEL (containing both Deep Sleep Percentage & Light Sleep Percentage which are multicollinear)

model_cart <- rpart(`Sleep efficiency` ~ Awakenings + `Caffeine consumption` + `Alcohol consumption` + `Exercise frequency` + `Daily Steps` + `REM sleep percentage` + `Deep sleep percentage` + `Light sleep percentage` + `Gender` + `Smoking status` + `Age` + `Sleep duration`, data = dataTrain, method = 'anova', control = rpart.control(minsplit = 30, cp = 0))
printcp(model_cart)
plotcp(model_cart)
print(model_cart)

# Prune the tree using the optimal CP value
optimal_cp <- model_cart$cptable[which.min(model_cart$cptable[,"xerror"]), "CP"]
model_cart_pruned <- prune(model_cart, cp = optimal_cp)
rpart.plot(model_cart_pruned)
summary(model_cart_pruned)

# Predict and calculate RMSE for CART
cart_model_results <- predict(model_cart_pruned, dataTest)
cart_model_results 
cart_rmse <- rmse(dataTest$`Sleep efficiency`, cart_model_results)

# Create a data frame for comparison
cart_comparison <- data.frame(
  Actual = dataTest$`Sleep efficiency`,
  Predicted = cart_model_results
)

# Show the top 10 results for CART
head(cart_comparison, 10)

# ========================================= CART MODEL (containing ONLY Deep Sleep Percentage, Light Sleep Percentage excluded to prevent multicollinearity)

model_cart2 <- rpart(`Sleep efficiency` ~ Awakenings + `Caffeine consumption` + `Alcohol consumption` + `Exercise frequency` + `Daily Steps` + `REM sleep percentage` + `Deep sleep percentage` + `Gender` + `Smoking status` + `Age` + `Sleep duration`, data = dataTrain, method = 'anova', control = rpart.control(minsplit = 30, cp = 0))
printcp(model_cart2)
plotcp(model_cart2)
print(model_cart2)

# Prune the tree using the optimal CP value
optimal_cp2 <- model_cart$cptable[which.min(model_cart2$cptable[,"xerror"]), "CP"]
model_cart_pruned2 <- prune(model_cart2, cp = optimal_cp2)
rpart.plot(model_cart_pruned2)
summary(model_cart_pruned2)

# Predict and calculate RMSE for CART
cart_model_results2 <- predict(model_cart_pruned2, dataTest)
cart_model_results2 
cart_rmse2 <- rmse(dataTest$`Sleep efficiency`, cart_model_results2)

# Create a data frame for comparison
cart_comparison2 <- data.frame(
  Actual = dataTest$`Sleep efficiency`,
  Predicted = cart_model_results2
)

# Show the top 10 results for CART
head(cart_comparison2, 10)

# ========================================= EVALUATION PURPOSES

# Display RMSEs
linreg_rmse
cart_rmse
cart_rmse2
