
library("forecast")
library("data.table")
library("tidyverse")

start_time <- Sys.time()

setwd('C:/TET/macro/')
getwd()

f <- 'JTU5300QUL'

input <- { str_c("./data/", f, ".csv") }
data <- fread(input)

# xi = ts(data[,2], frequency = 12, start = c(Year, Period))
xi = data[,2]

th = 0.5
ix = ceiling(dim(xi)[1] * th)
y_dev = xi[1:ix]
y_test = xi[ix:dim(xi)[1]]

fo = 4

ixx = ceiling(ix / (fo + 1))

fc_actuals <- list()
fc_means <- list()
fc_lows <- list()
fc_ups <- list()
tts <- list()
for (j in 1:fo) {
  for (i in 1:ixx) {
    y_sub = y_dev[(ixx * (j - 1))+i:(ixx * j)+i-1]
    fit <- auto.arima(y_sub, trace=FALSE)
    fc <- forecast(fit,h=1)
    fc_actual <- data[,2][1:ix][(ixx * j)+i]
    fc_mean <- fc$mean[1]
    fc_low <- fc$lower[2]
    fc_up <- fc$upper[2]
    fc_actuals <- append(fc_actuals,fc_actual)
    fc_means <- append(fc_means,fc_mean)
    fc_lows <- append(fc_lows,fc_low)
    fc_ups <- append(fc_ups,fc_up)
    tts <- append(tts, data[,1][1:ix][(ixx * j)+i])
    print(j)
    print(i)
    print(data[,1][1:ix][(ixx * j)+i+1])
  }
}

resulted <- data.table(
  ds = tts,
  y = fc_actuals,
  yhat = fc_means,
  ylow = fc_lows,
  yup = fc_ups
)

# fit <- auto.arima(xi, trace=TRUE)
# plot(forecast(fit,h=20))

# resulted <- transform(resulted, ds = as.Date(as.character(ds), "%Y%m%d"))

library("ggplot2")
p <- ggplot(data=resulted)
