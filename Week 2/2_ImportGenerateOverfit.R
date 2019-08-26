library(h2o)
h2o.init()

# We can also use our datasets or create our own data
x <- seq(0, 10, 0.01)
y <- jitter(sin(x), 1000)
plot(x, y, type = "l")

sineWave <- data.frame(a=x, b=y)

# Put the data frame into a h2o format
sineWave.h2o <- as.h2o(sineWave)

sineWave.h2o
tail(sineWave.h2o)

# To pass h2o dataframes into R dataframes
d <- as.data.frame(sineWave.h2o)
head(d)
