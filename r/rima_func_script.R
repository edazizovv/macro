
library("forecast")
library("data.table")
library("tidyverse")
library("jsonlite")

setwd('C:/TET/macro/')
# getwd()

parameters <- read_json("./rima_par.json")

input_lag <- { str_c("./data/", parameters$f, "_lag.csv") }
data_lag <- fread(input_lag)
input_app <- { str_c("./data/", parameters$f, "_app.csv") }
data_app <- fread(input_app)

# data_lag <- data_lag[, list(get(parameters$f))]
# data_app <- data_app[, list(get(parameters$f))]

data_joint <- rbind(data_lag, data_app)

y_global_t <- ts(data_lag[, list(get(parameters$f))], frequency=parameters$m)
fit_base <- auto.arima(y_global_t, trace=FALSE)

fc_actuals <- c()
fc_means <- c()
fc_lows <- c()
fc_ups <- c()
tts <- c()

fold_n <- dim(data_app)[1]
for (i in 1:fold_n) {
  y_sub <- data_joint[(i):(dim(data_lag)[1]+i-parameters$h)]
  y_sub_t <- ts(y_sub[, list(get(parameters$f))], frequency=parameters$m)
  
  fit <- Arima(y_sub_t, model=fit_base)
  
  fc <- forecast(fit,h=parameters$h)
  
  fc_actual <- data_joint[dim(data_lag)[1]+i, get(parameters$f)]
  
  fc_mean <- fc$mean[length(fc$mean)]
  fc_low <- fc$lower[dim(fc$lower)[1], 2]
  fc_up <- fc$upper[dim(fc$upper)[1], 2]
  
  fc_actuals <- c(fc_actuals, fc_actual)
  
  fc_means <- c(fc_means, fc_mean)
  fc_lows <- c(fc_lows, fc_low)
  fc_ups <- c(fc_ups, fc_up)
  tts <- c(tts, data_joint[dim(data_lag)[1]+i, DATE])
}

tts <- unlist(tts)

resulted <- data.table(tts, fc_actuals, fc_means, fc_lows, fc_ups)
colnames(resulted) <- c("DATE", parameters$f, str_c(parameters$f, "_arima_hat"), str_c(parameters$f, "_arima_hat_lower"), str_c(parameters$f, "_arima_hat_upper"))

resulted <- resulted[, DATE:=as.Date(DATE)]

s_link <- str_c("./data/", parameters$f, "_arima_pred.csv")
write.csv(resulted, s_link)
