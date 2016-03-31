library(tgp)
library(ggplot2)
# Load the cmb data
cmb <- read.table("cmb.dat",header=FALSE);
names(cmb)<-c('X','Y','a','b','c')
# X is the covariate, Y is the response
cmbSizes <- dim(cmb);
n <- cmbSizes[1];
nTrain <- 500;
nTest <- n-nTrain;

# Parition into training and testing data
indList <- sample.int(n,n,replace=FALSE,prob=NULL);
xx <- cmb$X[indList];
yy <- cmb$Y[indList];
xTrain <- xx[1:nTrain];
xTest <- xx[(nTrain+1):n];
yTrain <- yy[1:nTrain];
yTest <- yy[(nTrain+1):n];

# Run a plain old GP
cmb.bgp <- bgp(X = xTrain, Z = yTrain, XX = xTest, verb = 0)
# And a Bayesian treed linear model
cmb.btlm <- btlm(X = xTrain, Z = yTrain, XX = xTest)
# And a Bayesian treed GP
cmb.btgp <- btgp(X = xTrain, Z = yTrain, XX = xTest, verb = 0)

# And plot the outputs
pdf(file="cmb_bgp500.pdf")
qplot(cmb.bgp,main="GP",layout= "surf")
dev.off()

pdf(file="cmb_btlm500.pdf")
qplot(cmb.btlm,main="Bayesian Treed Linear Model (CART)",layout= "surf")
dev.off()

pdf(file="cmb_btgp500.pdf")
qplot(cmb.btgp,main="Treed GP",layout= "surf")
dev.off()