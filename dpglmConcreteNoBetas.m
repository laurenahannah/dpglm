% This function takes inputs of the number of iterations and the iteration
% at which to start burnin and outputs a regression of the diabetes data
% presented in "Least Angle Regression" by Efron, Hastie, Johnstone and
% Tibshirani, The Annals of Statistics 32(2),407--451.
%
% Contiuous variables are modeled by mixtures of normals with
% normal-inverse gamma priors. Categorical data are modeled by multinomial
% (Bernoulli) distributions with Dirichlet (beta) priors. Priors on the
% slope parameters are normal-inverse gamma. To sample from the posterior,
% we use a partially collapsed MCMC sampler, a combination of the samplers
% given in Neal (2000), algorithms 3 and 8.

function dpglmConcreteNoBetas(iter,burnin)

global N d;

% The number of training samples
dataSize = 500;

load concrete.dat;
load permList.dat;

[N, d] = size(concrete)
d = d-2; % The number of dimensions

% Center the non-categorical data
data = concrete;
data = data - repmat(mean(data),N,1);
data = data./repmat(std(data),N,1);

x = data(:,1:d);
y = data(:,d+1);

% Get a random sample
indList = permList(30,:);
trainInd = indList(1:dataSize);
testInd = indList(dataSize+1:end);

xTrain = x(trainInd,:);
xTest = x(testInd,:);
yTrain = y(trainInd);
yTest = y(testInd);
nTest = length(yTest);

[newC, mBar] = dpMix(xTrain,xTest,yTrain,yTest,iter,burnin);

% newC is a list of which cluster each training point is in
% mBar is an estimate for each testing data point in each iteration

mHat = mean(mBar,1);

%figure
%plot(yTest,mHat,'.')
%xlabel('True Value')
%ylabel('Estimated Value')

err = mHat'-yTest;
%figure
%hold on
%plot(yTest,err,'.')
%xlabel('True Value')
%ylabel('Error')

disp('L1 Error')
L1err = sum(abs(err))/nTest

disp('L2 Error')
L2err = err'*err/nTest

% Make a matrix of [index #, true (normalized) value, estimated (normalized) value]
%values = [indList' mHat];

%save ConcreteClusterList.dat newC -ascii
%save ConcreteOutputValues.dat values -ascii

function [newC, mBar] = dpMix(xTrain,xTest,yTrain,yTest,iter,burnin)

% This function does the DP mixing

% In collapsed sampler, newC is our state variable

% Globalize necessary variables (hyperparameters and the like)
global N d;
global nTest nTrain;
global alpha n0 s02 lambda nu ap bp ny0 sy02 m0 sb02 tau;

% Let's set up priors and such
nTest = length(yTest); nTrain = length(yTrain);

% Model:
% X_1 ~ N(m1,s1), (mi,si) ~ N-inv-gamma()
% X_2 ~ Ber(p), p ~ beta(ap, ab)
% X_3:8 ~ N(m3:8,s3:8)
% Y ~ N(X_1b1 + X_3b3+...+X_8b_8,sy), bi ~ N(), sy ~ Inv-gamma()

% NOTE: extra (unused) param for Gaussian locations to make looping easier

% Continuous location variance prior:
% s2 ~ Inv-chi2(n0, s02)
n0 = ones(d,1); s02 = ones(d,1);

% Continuous location mean prior:
% m | s2, lambda, nu ~ N(lambda, s2/nu)
lambda = zeros(d,1); nu = ones(d,1);

% Categorical p prior:
% p ~ beta(ap,bp)
ap = .5; bp = .5; 
% We want peaks near 0 & 1 to make almost hard categories

% Response variance prior:
% sy2 ~ Inv-gamma(ny0,sy02)
ny0 = 1; sy02 = .5;

% Beta priors:
% beta ~ N(m0, sb02)
m0 = zeros(d+1,1); sb02 = .5*ones(d+1,1);
sb02(1) = .01;
tau = 1;

% No hyperdistribution on alpha: add one if you want
alpha = 1;

% Get a good starting partition for the data
partInd = 3; % Covariate over which we partition
yyy = yTrain + randn(nTrain,1)*.01; % Add a bit of noise to partition
partit = .01:.01:.99; % Number of partions we want
np = length(partit);
quants = quantile(yyy,partit);
newNj = zeros(1,np+1); % Number in each cluster
for i = 1:np+1
    ind(i).vec = [];
    ind(i).n = i;
end

for i = 1:nTrain % Initial clustering based on spatial stuff
    for j = 1:np+1
        if j==1
            if (yyy(i) <= quants(1))
                vecs = [];
                vecs = ind(j).vec;
                vecs = [vecs, i];
                ind(j).vec = vecs;
                newC(i) = j;
                newNj(j) = newNj(j)+1;
            end
        elseif j <= np
            if(yyy(i) <= quants(j))&&(yyy(i) > quants(j-1))
                vecs = [];
                vecs = ind(j).vec;
                vecs = [vecs, i];
                ind(j).vec = vecs;
                newC(i) = j;
                newNj(j) = newNj(j)+1;
            end
        else
            if (yyy(i) > quants(j-1))
                vecs = [];
                vecs = ind(j).vec;
                vecs = [vecs, i];
                ind(j).vec = vecs;
                newC(i) = j;
                newNj(j) = newNj(j)+1;
            end
        end
    end
    % Debugging: if any have 0, we have problems.
    % Try partitioning on a different index, or change for categorical
    if newC(i) == 0
        xTrain(i)
    end
end
newNj

newC(1:5)

startTime = cputime; % For timekeeping

% For mbar:
% - for each testing point, assign probabilities for each cluster,
% including "new"
% - then, give weighted average of expected outputs (expectation of m0*[1,
% x] in "new")
% - record values in mbar
mBar = [];

for i = 1:iter
    i
    [newC, newNj] = MCMC(xTrain,yTrain,newC,newNj);
    disp('Number of data points associated with each component:')
    disp(newNj)
    if (mod(i,5)==0)&&(i >= burnin)
        % Save data for the clusters
        k = length(newNj);
        yOut = zeros(1,nTest);
        for el = 1:nTest
            pVec = zeros(k+1,1);
            yVecs = zeros(k+1,1);
            for j = 1:k
                xStar = [];
                yStar = [];
                xStar = xTrain(newC==j,:);
                yStar = yTrain(newC==j);
                pVec(j) = getLogLikeCtsCov(xTest(el,:),xStar);
                pVec(j) = pVec(j) + log(newNj(j))-log(alpha+N);
                yVecs(j) = getBetaX(xTest(el,:),xStar,yStar);
            end
            pVec(k+1) = getLogLikeCtsG0(xTest(el,:));
            pVec(k+1) = pVec(k+1) + log(alpha)-log(alpha+N);
            yVecs(k+1) = [1 xTest(el,:)]*m0;
            pVec = pVec - max(pVec);
            pVec = exp(pVec);
            pVec = pVec/sum(pVec);
            yOut(el) = pVec'*yVecs;
        end
        mBar = [mBar; yOut];
    end
end
stoptime = cputime-startTime


function [C, Nj] = MCMC(x,y,C,Nj)
global N d;
global nTest nTrain;
global alpha n0 s02 lambda nu ap bp ny0 sy02 m0 sb02;

for i = 1:nTrain
    if mod(i,100)==0
        i
    end
    curInd = C(i);
    Nj(curInd) = Nj(curInd)-1;
    if Nj(curInd) == 0 % It was in a cluster of 1 and we removed it
        Nj(curInd) = [];
        C(C>curInd) = C(C>curInd) - 1;
    end
    kHat = length(Nj);
    
    q = zeros(kHat+1,1); % For log probs for original 
    for k = 1:(kHat+1)
        if k <= kHat
            xStar = [];
            yStar = [];
            cHat = [];
            cHat = C;
            cHat(i)=0;
            xStar = x(cHat==k,:);
            yStar = y(cHat==k,:);
            q(k) = getLogLike(x(i,:),y(i,:),xStar,yStar);
            q(k) = q(k) + log(Nj(k)) - log(N-1+alpha);
        else
            q(k) = getLogLikeG0(x(i,:),y(i,:));
            q(k) = q(k) + log(alpha) - log(N-1+alpha);
        end
    end
    % Let's trim probs to avoid NaN and get probs
    qMax = max(q); 
    qRel = q - qMax;
    q = exp(qRel);
    q = q./sum(q);
    qCumSum = repmat(0,length(q),1);
    qCumSum = cumsum(q);
    
    % Choose a component
    u = rand;
    k0 = find(qCumSum >=u);
    picked = k0(1);
    
    % Now update
    if picked <= kHat
        C(i) = picked;
        Nj(picked) = Nj(picked)+1;
    else
        C(i) = kHat+1;
        Nj = [Nj, 1];
    end
end
% No updating values

function logLike = getLogLike(x,y,xStar,yStar)
global N d;
global nTest nTrain;
global alpha n0 s02 lambda nu ap bp ny0 sy02 m0 sb02;

logLike = getLogLikeCtsCov(x,xStar);
logLike = logLike + getLogLikeY(x,y,xStar,yStar);


function logLike = getLogLikeCtsCov(x,xStar)
global N d;
global nTest nTrain;
global alpha n0 s02 lambda nu ap bp ny0 sy02 m0 sb02;
% The continuous covariates
logLike = 0;
nj = length(xStar(:,1));
xBar = mean(xStar,1);
if nj>1
    s2 = var(xStar);
else
    s2 = zeros(1,d);
end
for i = 1:d
    mn = nu(i)/(nu(i)+nj)*lambda(i)+nj/(nu(i)+nj)*xBar(i);
    kn = nu(i)+nj;
    nun = n0(i)+nj;
    sn2 = (n0(i)*s02(i)+(nj-1)*s2(i)+nu(i)*nj/(nu(i)+nj)*(xBar(i)-lambda(i))^2)/nun;
    logLike = logLike - (nun+1)/2*log(1+(x(i)-mn)^2/(nun*sn2*(1+1/kn)));
end

function logLike = getLogLikeY(x,y,xStar,yStar)
global N d;
global nTest nTrain;
global alpha n0 s02 lambda nu ap bp ny0 sy02 m0 sb02;
% p(y|x,xStar,yStar)
logLike = 0;
nj = length(yStar);
yBar = mean(yStar);
if nj>1
    s2 = var(yStar);
else
    s2 = 0;
end
% Go from IG dist to Inv-chi2
nnu = 2*ny0;
sy2 = 2*sy02/ny0;
mn = sb02(1)/(sb02(1)+nj)*m0(1)+nj/(sb02(1)+nj)*yBar;
kn = sb02(1)+nj;
nun = nnu+nj;
sn2 = (nnu*sy2+(nj-1)*s2+sb02(1)*nj/(sb02(1)+nj)*(yBar-m0(1))^2)/nun;
logLike = logLike - (nun+1)/2*log(1+(y-mn)^2/(nun*sn2*(1+1/kn)));


function logLike = getLogLikeG0(x,y)
logLike = 0;
logLike = logLike + getLogLikeCtsG0(x);
logLike = logLike + getLogLikeYG0(x,y);

function logLike = getLogLikeCtsG0(x)
global N d;
global nTest nTrain;
global alpha n0 s02 lambda nu ap bp ny0 sy02 m0 sb02;
% The continuous covariates are 1,3:d
logLike = 0;
nj = 0;
xBar = zeros(1,d);
s2 = zeros(1,d);
for i = 1:d
    mn = nu(i)/(nu(i)+nj)*lambda(i)+nj/(nu(i)+nj)*xBar(i);
    kn = nu(i)+nj;
    nun = n0(i)+nj;
    sn2 = (n0(i)*s02(i)+(nj-1)*s2(i)+nu(i)*nj/(nu(i)+nj)*(xBar(i)-lambda(i))^2)/nun;
    logLike = logLike - (nun+1)/2*log(1+(x(i)-mn)^2/(nun*sn2*(1+1/kn)));
end

function logLike = getLogLikeYG0(x,y)
global N d;
global nTest nTrain;
global alpha n0 s02 lambda nu ap bp ny0 sy02 m0 sb02;
% p(y | beta, s2)
% p(beta, s2)

nnu = 2*ny0;
sy2 = 2*sy02/ny0;
mn = sb02(1)/(sb02(1))*m0(1);
kn = sb02(1);
nun = nnu;
sn2 = (nnu*sy2)/nun;
logLike = - (nun+1)/2*log(1+(y-mn)^2/(nun*sn2*(1+1/kn)));


function mn = getBetaX(x,xStar,yStar)

% This gives a posterior draw from a regression model, beta~N(0,tau*s2)
% with an inverse-chi^2 prior on sigma^2. This allows it
% to be used in cases where n is less than the number of dimensions.
global N d;
global nTest nTrain;
global alpha n0 s02 lambda nu ap bp ny0 sy02 m0 sb02;
nj = length(yStar);
yBar = mean(yStar);
if nj>1
    s2 = var(yStar);
else
    s2 = 0;
end
% Go from IG dist to Inv-chi2
nnu = 2*ny0;
sy2 = 2*sy02/ny0;
mn = sb02(1)/(sb02(1)+nj)*m0(1)+nj/(sb02(1)+nj)*yBar;


