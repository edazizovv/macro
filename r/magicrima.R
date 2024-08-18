
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

th = 0.75
ix = ceiling(dim(xi)[1] * th)
y_dev = xi[1:ix]
y_test = xi[ix:dim(xi)[1]]

fo = 3

ixx = ceiling(dim(y_dev)[1] / (fo + 1))

lens <- dim(y_dev)[1] - ixx
fc_actuals <- c()
fc_means <- c()
fc_lows <- c()
fc_ups <- c()
tts <- c()
gts <- c()
pp <- c()
qq <- c()
PP <- c()
QQ <- c()
mm <- c()
dd <- c()
DD <- c()
# DD <- vector("list", lens)
# class(tts) <- "Date"
z <- 0
for (j in 1:fo) {
  fold_n <- min(ixx * (j + 1), dim(y_dev)[1]) - ixx * j
  for (i in 1:fold_n) {
    z <- z + 1
    y_sub = y_dev[(ixx * (j - 1))+i:(ixx * j)+i-1]
    y_sub_t <- ts(y_sub, frequency=12)
    fit <- auto.arima(y_sub_t, trace=FALSE)
    fc <- forecast(fit,h=1)
    fc_actual <- y_dev[(ixx * j)+i, JTU5300QUL]
    fc_mean <- fc$mean[1]
    fc_low <- fc$lower[2]
    fc_up <- fc$upper[2]
    fc_actuals <- c(fc_actuals, fc_actual)
    fc_means <- c(fc_means, fc_mean)
    fc_lows <- c(fc_lows, fc_low)
    fc_ups <- c(fc_ups, fc_up)
    tts <- c(tts, data[(ixx * j)+i,DATE])
    orders <- fit$arma
    pp <- c(pp, orders[1])
    qq <- c(qq, orders[2])
    PP <- c(PP, orders[3])
    QQ <- c(QQ, orders[4])
    mm <- c(mm, orders[5])
    dd <- c(dd, orders[6])
    DD <- c(DD, orders[7])
    
    # DD[[z]] <- orders[7]
    print(j)
    print(i)
    print(data[(ixx * j)+i,DATE])
  }
}
# fit$arma
# p q P Q m d D

resulted <- data.table(
  ds = unlist(tts),
  gs = gts,
  y = fc_actuals,
  yhat = fc_means,
  ylow = fc_lows,
  yup = fc_ups,
  p = pp,
  q = qq,
  P = PP,
  Q = QQ,
  m = mm,
  d = dd,
  D = DD
)

resulted <- resulted[, ds:=as.Date(ds)]

resulted <- resulted[, y:=as.numeric(y)]
resulted <- resulted[, yhat:=as.numeric(yhat)]
resulted <- resulted[, ylow:=as.numeric(ylow)]
resulted <- resulted[, yup:=as.numeric(yup)]

# fit <- auto.arima(xi, trace=TRUE)
# plot(forecast(fit,h=20))

# resulted <- transform(resulted, ds = as.Date(as.character(ds), "%Y%m%d"))

library("ggplot2")
ggplot(data=resulted, aes(x = ds)) + 
  geom_line(aes(y = fc_actuals, color = "actual")) + 
  geom_line(aes(y = fc_means, color = "estimated")) + 
  geom_ribbon(aes(ymin=fc_lows,ymax=fc_ups), fill="blue", alpha=0.5)+
  scale_color_manual(name = "Y series", values = c("actual" = "navy", "estimated" = "orange"))

ggplot(data=resulted, aes(x = ds)) + 
  geom_line(aes(y = p, color = "p")) + 
  geom_line(aes(y = q, color = "q")) + 
  geom_line(aes(y = P, color = "P")) + 
  geom_line(aes(y = Q, color = "Q")) + 
  geom_line(aes(y = d, color = "d")) + 
  geom_line(aes(y = D, color = "D"))

