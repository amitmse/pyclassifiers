# Compare those results to what you'd get with an LDA, just for fun
# Kyle Gorman <gormanky@ohsu.edu>

require(MASS)

## scoring metrics

precision <- function(tp, tn, fp, fn) {
    (tp / (tp + fp))
}

recall <- function(tp, tn, fp, fn) {
    (tp / (tp + fn))
}

accuracy <- function(tp, tn, fp, fn) {
    num <- tp + tn
    (num / (num + fp + fn))
}

F1 <- function(tp, tn, fp, fn) {
    p <- precision(tp, tn, fp, fn)
    r <- recall(tp, tn, fp, fn)
    ((2. * p * r) / (p + r))
}

## helpers

Fdrop <- function(data) {
    data[] <- lapply(data, function(x) x[, drop=1])
    return(data)
}

## read in
d <- Fdrop(subset(iris, Species != 'setosa'))

## score
predicted <- lda(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data=d, kernel='linear', CV=TRUE)$class
hit <- 'versicolor'
tp <- sum(predicted == hit & d$Species == hit)
tn <- sum(predicted != hit & d$Species != hit)
fp <- sum(predicted == hit & d$Species != hit)
fn <- sum(predicted != hit & d$Species == hit)
cat('Accuracy:', round(accuracy(tp, tn, fp, fn), 3), '\n')
cat('F1:', round(F1(tp, fn, fp, fn), 3), '\n')
