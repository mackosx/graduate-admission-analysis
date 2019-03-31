library(MASS)
library(MLmetrics)

# Setup data
graduateAdmissions.csv <- read.csv("Admission_Predict_Ver1.1.csv")
graduateAdmissions.data <- data.frame(graduateAdmissions.csv)
graduateAdmissions.catData <- graduateAdmissions.data[, -c(1)]
graduateAdmissions.catData$Chance.of.Admit <- as.numeric(graduateAdmissions.data$Chance.of.Admit > 0.5)


# Predict Chance of Admit
graduateAdmission.lda <- lda(Chance.of.Admit ~ ., data = graduateAdmissions.catData, cv = TRUE)
# Perform LDA
plda <- predict(graduateAdmission.lda, graduateAdmissions.catData)

# Classification Table
table(graduateAdmissions.catData$Chance.of.Admit,plda$class)
# Classification Metrics
LogLoss(plda$posterior[,2], graduateAdmissions.catData$Chance.of.Admit)
Accuracy(graduateAdmissions.catData$Chance.of.Admit,plda$class)
Sensitivity(graduateAdmissions.catData$Chance.of.Admit,plda$class)

# Perform QDA on Chance of admission
graduateAdmission.qda <- qda(Chance.of.Admit ~ ., data = graduateAdmissions.catData, cv = TRUE)

# Run predictions
pqda <- predict(graduateAdmission.qda, graduateAdmissions.catData)
# Classification table for chance of admission
table(graduateAdmissions.catData$Chance.of.Admit, pqda$class )

# Metrics
LogLoss(pqda$posterior[,2], graduateAdmissions.catData$Chance.of.Admit)
Accuracy(graduateAdmissions.catData$Chance.of.Admit, pqda$class)
Sensitivity(graduateAdmissions.catData$Chance.of.Admit, pqda$class)
F1_Score(graduateAdmissions.catData$Chance.of.Admit, pqda$class)



# Predict Research

# Perform LDA on Research
graduateAdmission.lda <- lda(Research ~ ., data = graduateAdmissions.catData, cv = TRUE)
plda <- predict(graduateAdmission.lda, graduateAdmissions.catData)

# Classificationtablee
table(graduateAdmissions.catData$Research,plda$class)
# Metrics
LogLoss(plda$posterior[,2], graduateAdmissions.catData$Research)
Accuracy(graduateAdmissions.catData$Research,plda$class)
Sensitivity(graduateAdmissions.catData$Research,plda$class)

# Perform QDA on research
graduateAdmission.qda <- qda(Research ~ ., data = graduateAdmissions.catData, cv = TRUE)

# Predicr research with QDA
pqda <- predict(graduateAdmission.qda, graduateAdmissions.catData)

# Class table
table(graduateAdmissions.catData$Research, pqda$class)
# Metrics
LogLoss(pqda$posterior[,2], graduateAdmissions.catData$Research)
Accuracy(graduateAdmissions.catData$Research, pqda$class)
Sensitivity(graduateAdmissions.catData$Research, pqda$class)
F1_Score(graduateAdmissions.catData$Research, pqda$class)


# Neural Net
library(neuralnet)
graduateAdmissions.numeric <- graduateAdmissions.data[, -c(1)]
graduateAdmissions.normalizedNumeric <- apply(graduateAdmissions.numeric, 2, function(v) (v-min(v)) / (max(v) - min(v)))
graduateAdmissions.normalizedNumeric <- data.frame(graduateAdmissions.normalizedNumeric)
graduateAdmissions.normalizedNumeric$Research <- as.factor(graduateAdmissions.normalizedNumeric$Research)
graduateAdmissions.nnAllData <- neuralnet(Research ~ ., data = graduateAdmissions.normalizedNumeric, hidden = 5, linear.output = FALSE, stepmax = 1e+06)


# Training/Testing Set
ind <- sample(1:nrow(graduateAdmissions.normalizedNumeric), 250)
train <- graduateAdmissions.normalizedNumeric[ind,]
test <- graduateAdmissions.normalizedNumeric[-ind,]

graduateAdmissions.nnWithTrain <- neuralnet(Research ~ ., data = train, hidden = 5, linear.output = FALSE, stepmax = 1e+06)
predicted = data.frame(round(predict(graduateAdmissions.nnWithTrain, graduateAdmissions.normalizedNumeric, type = "class"), 0))$X2
table(predicted, actual = graduateAdmissions.normalizedNumeric$Research)

Sensitivity(graduateAdmissions.normalizedNumeric$Research, predicted)
Accuracy(graduateAdmissions.normalizedNumeric$Research, predicted)
F1_Score(graduateAdmissions.normalizedNumeric$Research, predicted)
