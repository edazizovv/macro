
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

fit <- auto.arima(y_dev, m=12, trace=FALSE)
refit <- Arima(y_test, model=fit)

library("ggplot2")
p <- ggplot(data=resulted)
