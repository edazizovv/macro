

rima_predictor <- function(f, m, h) {
  
  setwd('C:/TET/macro/')
  # getwd()
  
  input_lag <- { str_c("./data/", f, "_lag.csv") }
  data_lag <- fread(input_lag)
  input_app <- { str_c("./data/", f, "_app.csv") }
  data_app <- fread(input_app)
  
  data_joint <- rbind(data_lag, data_app)
  
  y_global_t <- ts(data_lag, frequency=m)
  fit_base <- auto.arima(y_global_t, trace=FALSE)
  
  fc_rel_means <- c()
  fc_rel_lows <- c()
  fc_rel_ups <- c()
  tts <- c()
  
  fold_n <- dim(data_app)[1]
  for (i in 1:fold_n) {
    y_sub <- data_joint[(i):(dim(data_lag)[1]+i-h)]
    y_sub_t <- ts(y_sub, frequency=m)
    
    fit <- Arima(y_sub_t, model=fit_base)
    
    fc <- forecast(fit,h=h)
    
    fc_actual <- data_joint[dim(data_lag)[1]+i, get(f)]
    
    fc_rel_mean <- (fc$mean[length(fc$mean)] / fc_actual) - 1
    fc_rel_low <- (fc$lower[dim(fc$lower)[1], 2] / fc_actual) - 1
    fc_rel_up <- (fc$upper[dim(fc$upper)[1], 2] / fc_actual) - 1

    fc_rel_means <- c(fc_means, fc_mean)
    fc_rel_lows <- c(fc_lows, fc_low)
    fc_rel_ups <- c(fc_ups, fc_up)
    tts <- c(tts, data_joint[dim(data_lag)[1]+i, DATE])
  }
  
  resulted <- data.table(
    DATE = unlist(tts),
    get(f) = fc_actuals,
    get(str_c(f, "_arima_hat")) = fc_means,
    get(str_c(f, "_arima_hat_low")) = fc_lows,
    get(str_c(f, "_arima_hat_up")) = fc_ups,
  )
  
  resulted <- resulted[, ds:=as.Date(ds)]
  
  s_link <- str_c("./data/", f, "_arima_pred.csv")
  write.csv(resulted, s_link)
  res <- s_link
}
