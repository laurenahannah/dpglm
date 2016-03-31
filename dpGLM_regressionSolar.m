% This is for producing outputs for my paper.

function vec = dpGLM_regressionSolar(numberOfIterations, burnIn,listNum);

dataSize = 1300;

load solarX.dat;
load solarY.dat;
rodeoX = solarX;
rodeoY = solarY;
%indList = randperm(1030);
load solarList.dat; % A list of permutations for comparison
indList = solarList(listNum,:);
xTrain = rodeoX(indList(1:dataSize),:);
yTrain = rodeoY(indList(1:dataSize));
xTest = rodeoX(indList(dataSize+1:end),:);
yTest = rodeoY(indList(dataSize+1:end));

% Only center continuous variables: so no centering



[newJ, vec] = dpMnl(xTrain, yTrain, xTest, yTest, numberOfIterations, burnIn);


% This part calculates the accuracy rate and the F1 measure on the test
% set.
%result = getResults(yTest, p) 

% This part saves the results in a file.
%dlmwrite('dpGLM_results.dat', result, '-append');


function [newJ, valVec] = dpMnl(xTrain, yTrain, xTest, yTest, numberOfIterations, burnIn)

global nTrain nTest n d nLeaf nCov;
global a0 b0;
global abNuA abNuB sigBeta;
global p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11;

global leapFrog eps regInd;
global kk;

leapFrog = 2; % Number of steps for the Hamiltonian dynamics
eps = 0.01; % This is the constant multiplier for the step size 

[nTrain, nCov] = size(xTrain);
nTest = size(xTest, 1);
regInd = [11 12];
[n, nCov] = size(xTrain);
nLeaf = 1;

kk = .5*ones(n,1);

% These are the parameters of the gamma prior for scale parameter, alpha.
a0 = -1; b0 = 2; 



abNuA  = [-1, 2]; % Paramters of the prior for the intercept
abNuB  = [0, .01]; % Parameters of the prior for the coefficients

% Initial values
nuA = 0.01;
nuB = 0.01;
sigBeta = ones(nCov+1, 1);

sigComp = sigBeta;
sigComp(1, :) = nuA*sigBeta(1, :);
sigComp(2:end, :) = nuB*sigBeta(2:end, :);

% Give a Dirichlet prior for p1, p2
p1 = 2*ones(1,2);
p2 = 2*ones(1,6);
p3 = 2*ones(1,6);
p4 = 2*ones(1,4);
p5 = 2*ones(1,2);
p6 = 2*ones(1,3);
p7 = 2*ones(1,2);
p8 = 2*ones(1,2);
p9 = 2*ones(1,2);
p10 = 2*ones(1,2);
p11 = 2*ones(1,2);

% An initial value for the Scale parameter of the Drichlet process
alpha(1) = .0001;
mBar = [];
partInd = 12;
scrsz = get(0,'ScreenSize');

partit = .5:.5:.5;
np = length(partit)
%yyy = xTrain(:,partInd) + randn(nTrain,1)*.0001;
yyy = xTrain(:,4)+xTrain(:,5)+xTrain(:,11)+xTrain(:,15)+xTrain(:,22) + randn(nTrain,1)*.0001;
quants = quantile(yyy,partit)
%quants = partit
newNj = zeros(1,np+1);
%newNj = zeros(1,np);
for i = 1:np+1
    ind(i).vec = [];
    ind(i).n = i;
end
min(xTrain(:,partInd));
for i = 1:n % Initial clustering based on spatial stuff
    %for j = 1:np
    for j = 1:np+1
        if j==1
            if (yyy(i) <= quants(1))
            %if (xTrain(i,partInd) == quants(1))
                %xTrain(i)
                vecs = [];
                vecs = ind(j).vec;
                vecs = [vecs, i];
                ind(j).vec = vecs;
                newJ(i) = j;
                newNj(j) = newNj(j)+1;
            end
        elseif j <= np
            if(yyy(i) <= quants(j))&&(yyy(i) > quants(j-1))
            %if(xTrain(i,partInd) == quants(j))
                vecs = [];
                vecs = ind(j).vec;
                vecs = [vecs, i];
                ind(j).vec = vecs;
                newJ(i) = j;
                newNj(j) = newNj(j)+1;
            end
        else
            if (yyy(i) > quants(j-1))
                vecs = [];
                vecs = ind(j).vec;
                vecs = [vecs, i];
                ind(j).vec = vecs;
                newJ(i) = j;
                newNj(j) = newNj(j)+1;
            end
        end
    end
    if newJ(i) == 0
        xTrain(i)
    end
end
newNj

%newJ(1:50)

for i = 1:np+1
vecs = ind(i).vec;
beta = zeros(nCov+1,1);
%beta = glmfit(xTrain(vecs,:), yTrain(vecs),'poisson')
beta = updateBeta([ones(newNj(i),1) xTrain(vecs,:)], yTrain(vecs), vecs)
% This is the error term for the GLM
thetaStar(i).beta = beta;
thetaStar(i).nuA = nuA;
thetaStar(i).nuB = nuB;
thetaStar(i).sigComp = sigComp;
end

% thetaStar holds the unique parameters sampled from the Dirichlet process
% thetaStar.mu = zeros(1, nCov);
% thetaStar.sd = 2*ones(1, nCov);
% thetaStar.nuA = nuA;
% thetaStar.nuB = nuB;
% thetaStar.sigComp = sigComp;
% thetaStar.beta = ones(1+nCov, nLeaf);
% thetaStar.eps = 2; % This is the error term for the GLM

% % An initial value for the cluster identifiers, J
% newJ = repmat(1, n, 1); 
% 
% % An initial value for the frequencey of each cluster
% newNj = n;

startTime = cputime;

newProb = zeros(nTest, nLeaf);
countP = 0; 
counter = 0;
sumPX = 0;
sumProb = 0;
meanAccept = 0;
for iter = 1:numberOfIterations
if mod(iter,10)==0
    iter
end

% This part calls the MCMC algorithm
[thetaStar, newJ, newNj] = main_MCMC(xTrain, yTrain, thetaStar, newJ, newNj, alpha(iter));
% This prints the new frequncy of each cluster on the screen 
disp('Number of samples in each component:')
disp(newNj)
% This part resamples the parameters of the Dirichlet process given 
% the current assignment of data points to cluster.
[thetaStar, acceptP] = remix(xTrain, yTrain, newJ, thetaStar); 
%disp('Acceptance prob when resampling coefficients: ')
%disp(acceptP)

%disp('The cluster [mean; sd; alpha; beta] are:')
%disp([thetaStar.mu; thetaStar.sd; thetaStar.beta])

% This part samples new alpha from its posterior distribution 
alpha(iter+1) = pickAlpha(alpha(iter), length(newNj), a0, b0); 

nComp = length(newNj); % Number of compontnst (i.e., clusters) in the mixture


% This parts sets the constant multiplier for the step size of the
% Hamiltonian dynamics. If the acceptance rates are very low or very high,
% you can change these values to obtain a more appropriate acceptance rate.
if rem(iter, 5) ==0 
    eps = 0.04;
else
    eps = 0.02;
end

% This part obtains the predictive probability for the test set when the
% number of iterations is larger than "burnIn" value.
xClusterVec = [];
thetaVec = [];
iterCount = 0;
maxD = 0;

if iter > burnIn % I want to write the thetas, the J vectors
    countP = countP+1;
    if (mod(iterCount,5)==0)
        xClusterVec = [xClusterVec; newJ];
        maxD = max(maxD,length(thetaStar));
        thetaTemp = [];
        if (maxD > length(thetaStar))
            thetaTemp = [thetaStar, repmat(thetaStar(1),1,maxD - length(thetaStar))];
        elseif (maxD > size(thetaVec,2))
            thetaVec = [thetaVec, repmat(thetaStar(1),size(thetaVec,1),maxD-size(thetaVec,2))];
            thetaTemp = thetaStar;
        else
            thetaTemp = thetaStar;
        end
        thetaVec = [thetaVec; thetaStar];
    end
    iterCount = iterCount + 1;
    %[pY, pX, q] = getPredProb(xTest, thetaStar, newNj, alpha(iter+1), mu0, Sigma0, mu00, Sigma00, sigBeta);
    %pX = repmat(pX, 1, nLeaf);
    %sumPX = pX+sumPX;
    %p = pY.*pX;
    %sumProb = (p+sumProb);    
    %predProb = (sumProb./sumPX);
    %dlmwrite('dpMnlProb.dat', predProb);
end


if (iter >= burnIn)&&(mod(iter,1)==0)
yOut = zeros(nTest,1);
err = zeros(nTest,1);
dims = max(newJ);
nn = sum(newNj);

aaa = randperm(length(yTest));
for i = 1:length(xTest)
    % Get probabilities for each cluster
    pVec = zeros(dims,1);
    yVecs = zeros(dims,1);
    for j = 1:dims
        xOld = [];
        indx = [];
        indx = find(newJ==j);
        xOld = xTrain(indx,:);
        pp = getLogLikeX(xTest(i,:),xOld) + log(newNj(j))-log(nn);
        pVec(j) = pp;
        beta = thetaStar(j).beta;
        yVecs(j) = glmval(beta,xTest(i,:),'log');
    end
    pVec = pVec-max(pVec);
    pVec = exp(pVec);
    pVec = pVec/sum(pVec);
    if i == aaa(1)
        i
        pVec
        yVecs
        yTest(i)
    end
    yOut(i) = pVec'*yVecs;
    err(i) = yTest(i) - yOut(i);
end
for j = 1:dims
    beta = thetaStar(j).beta
end
mBar = [mBar; yOut'];
end
end
mHat = mean(mBar,1);
metaErr = yTest-mHat';

timePassed = cputime - startTime
figure
hold on
plot(yTest(1:50),mHat(1:50),'.')
xlabel('Y Actual')
ylabel('Y Predicted')
title('DP-GLM Regression')

% Make a regression tree for comparison
solarTree = [];
solarTree = classregtree(xTrain,yTrain);
yTree = [];
yTree = eval(solarTree,xTest);
treeErr = [];
treeErr = yTest-yTree;

view(solarTree)

%betaReg = glmfit(xTrain, yTrain, 'poisson')
%if max(abs(betaReg > 10))
    betaReg = updateBeta([ones(nTrain,1) xTrain],yTrain,[1:nTrain])
    betaReg = getBeta([ones(nTrain,1) xTrain],yTrain,betaReg)
%end
yReg = glmval(betaReg,xTest,'log');
plot(yTest(1:50),yReg(1:50),'g.')
plot(yTest(1:50),yTree(1:50),'c.')
plot([-2,3],[-2,3])

legend('DP-GLM','Poisson Regression','Regression Tree')
regErr = yTest - yReg;

figure
hold on
plot(yTest(1:50),metaErr(1:50),'r.')
plot(yTest(1:50),regErr(1:50),'m.')
plot(yTest(1:50),treeErr(1:50),'c.')
xlabel('Y Actual')
ylabel('Error')
title('DP-GLM Regression')
legend('DP Error','Poisson Error','Tree Error')

disp('DP-GLM L1')
L1dp = sum(abs(metaErr))/nTest
disp('DP-GLM L2')
L2dp = metaErr'*metaErr/nTest

disp('Poisson L1')
L1lin = sum(abs(regErr))/nTest
disp('Poisson L2')
L2lin = regErr'*regErr/nTest

disp('Tree L1')
L1tree = sum(abs(treeErr))/nTest
disp('Tree L2')
L2tree = treeErr'*treeErr/nTest

valVec = [L1dp; L2dp; L1lin; L2lin; L1tree; L2tree];
% testProbs = [testProbs; ones(1,maxClusters)/maxClusters];
% subplot(4,1,4)
% hold on
% %axis([-2,2,0,1])
% hh = bar(xTestNew',testProbs,1.25,'stack');
% set(hh,'LineStyle','none');
% colormap(jet)
%h = gcf;
%frame = getframe(h);
%dpClusters = addframe(dpClusters,frame);

%close(fig1);
%close(fig2);
%close(fig3);

save yOut4.dat yOut -ascii






function [thetaStar, updatedJ, updatedNj] = main_MCMC(x, y, thetaStar, J, nj, alpha)

global abNuA abNuB sigBeta;
global n nTrain nTest nLeaf;
M=5; % Number of auxillary components

for i = 1:n
    curInd = J(i);        
    nj(curInd) = nj(curInd) - 1;
    if nj(curInd) == 0
        phi = thetaStar(curInd);
        nj(curInd) =[];
        thetaStar(curInd)= [];
        J(J>curInd) = J(J>curInd) - 1;
        kBar = length(nj);
        thetaStar(kBar+1) = phi;
        for m=1:(M-1)
            nuA = sqrt(exp(normrnd(abNuA(1), abNuA(2))));
            nuB = sqrt(exp(normrnd(abNuB(1), abNuB(2))));
            sigComp = sigBeta;
            sigComp(1, :) = nuA*sigBeta(1, :);
            sigComp(2:end, :) = nuB*sigBeta(2:end, :);
            beta = normrnd(0, sigComp);
            thetaStar(kBar+1+m).nuA = nuA;
            thetaStar(kBar+1+m).nuB = nuB;
            thetaStar(kBar+1+m).sigComp = sigComp;
            thetaStar(kBar+1+m).beta = beta;
        end
    else
        kBar = length(nj);
        for m=1:M
            nuA = sqrt(exp(normrnd(abNuA(1), abNuA(2))));
            nuB = sqrt(exp(normrnd(abNuB(1), abNuB(2))));
            sigComp = sigBeta;
            sigComp(1, :) = nuA*sigBeta(1, :);
            sigComp(2:end, :) = nuB*sigBeta(2:end, :);
            beta = normrnd(0, sigComp);
            thetaStar(kBar+m).nuA = nuA;
            thetaStar(kBar+m).nuB = nuB;
            thetaStar(kBar+m).sigComp = sigComp;
            thetaStar(kBar+m).beta = beta;
         end
    end

    q1 = zeros(1,kBar);
        
    for k = 1:kBar
        beta = thetaStar(k).beta;
        eta = [1, x(i, :)]*beta;
        indx = [];
        indx = find(J==k);
        xOld = [];
        xOld = x(indx,:);
        q1(k) = getLogLike(x(i, :), y(i, :), beta, eta, xOld);
        if isnan(q1(k))
            q1(k) = -1*10^(100);
        end
    end
    [aa ab] = size(nj);
    if (aa > 1)
        nj = nj';
    end
    q1 = q1 + log(nj) - log(n-1+alpha);
    
    q2 = zeros(1,M);
    for k = 1:M
        beta = thetaStar(kBar+k).beta;
        eta = [1, x(i, :)]*beta;
        q2(k) = getLogLike0(x(i, :), y(i, :), beta, eta);
        if isnan(q2(k))
            q2(k) = -1*10^(100);
        end
    end
    
    q2 = q2+(log(alpha) - log(M))-log(n-1+alpha);
    q = [q1, q2];

    qMax = max(q);
    qRel = q - qMax;
    
    q = exp(qRel);
    
    q = q./sum(q);

    qCumSum = repmat(0, length(q), 1);
    qCumSum = cumsum (q); 

    u = rand;
    k0 = find ( qCumSum >= u);    
    picked = k0(1);
    
    if picked <= kBar
        J(i) = picked;
        nj(picked) = nj(picked)+1;
        thetaStar(kBar+1:end) = [];        
    else
        J(i) = kBar+1;
        [aa ab] = size(nj);
        if (aa >1)
            nj = [nj', 1];
        else
            nj = [nj, 1];
        end
        phi = thetaStar(picked);
        thetaStar(kBar+1:end) = [];
        thetaStar(kBar+1) = phi;
    end

end    
    
updatedNj = nj;
updatedJ = J;



function result = getLogLike(x, y, beta, eta, xOld)
global n d nLeaf;
global abNuA abNuB sigBeta n d nLeaf nCov;

priorB = abNuB(1)*ones(d,1);
priorB(1) = abNuA(1);
priorBS = abNuB(2)*ones(d,1);
priorBS(1) = abNuA(2);

logLikeX = getLogLikeX(x,xOld);
%logLikeX = 0;
% This is for GLM bit
logLikeY = -exp(eta)+ y*eta - gammaln(y+1)-sum(((beta-priorB).^2)./(2*priorBS));
result = logLikeX + logLikeY;

function result = getLogLike0(x, y, beta, eta)
global n d nLeaf;
global abNuA abNuB sigBeta n d nLeaf nCov;

priorB = abNuB(1)*ones(d,1);
priorB(1) = abNuA(1);
priorBS = abNuB(2)*ones(d,1);
priorBS(1) = abNuA(2);

logLikeX = getLogLikeX0(x);
%logLikeX = 0;
% This is for GLM bit
logLikeY = -exp(eta)+ y*eta - gammaln(y+1)-sum(((beta-priorB).^2)./(2*priorBS));
result = logLikeX + logLikeY;


function result = getLogLikeX(x,xOld)
global p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11
result = 0;
n = length(xOld(:,1));
for i = 1:11
    yOld = [];
    y = [];
    p = [];
    if (i==1)
        y = [1 - x(1), x(1)];
        yOld = [n - sum(xOld(:,1)), sum(xOld(:,1))];
        p = p1;
    elseif (i==2)
        y = [1 - sum(x(2:6)), x(2:6)];
        yOld = [n - sum(sum(xOld(:,2:6))), sum(xOld(:,2:6),1)];
        p = p2;
    elseif (i==3)
        y = [1 - sum(x(7:11)), x(7:11)];
        yOld = [n - sum(sum(xOld(:,7:11))), sum(xOld(:,7:11),1)];
        p = p3;
    elseif (i==4)
        y = [1 - sum(x(12:14)), x(12:14)];
        yOld = [n - sum(sum(xOld(:,12:14))), sum(xOld(:,12:14),1)];
        p = p4;
    elseif (i==5)
        y = [1 - x(15), x(15)];
        yOld = [n - sum(xOld(:,15)), sum(xOld(:,15))];
        p = p5;
    elseif (i==6)
        y = [1 - sum(x(16:17)), x(16:17)];
        yOld = [n - sum(sum(xOld(:,16:17))), sum(xOld(:,16:17),1)];
        p = p6;
    elseif (i==7)
        y = [1 - x(18), x(18)];
        yOld = [n - sum(xOld(:,18)), sum(xOld(:,18))];
        p = p7;
    elseif (i==8)
        y = [1 - x(19), x(19)];
        yOld = [n - sum(xOld(:,19)), sum(xOld(:,19))];
        p = p8;
    elseif (i==9)
        y = [1 - x(20), x(20)];
        yOld = [n - sum(xOld(:,20)), sum(xOld(:,20))];
        p = p9;
    elseif (i==10)
        y = [1 - x(21), x(21)];
        yOld = [n - sum(xOld(:,21)), sum(xOld(:,21))];
        p = p10;
    else
        y = [1 - x(22), x(22)];
        yOld = [n - sum(xOld(:,22)), sum(xOld(:,22))];
        p = p11;
    end
        
    % N choose K bit
    %result = result + gammaln(n+2)-sum(gammaln(y+1));
    % Top beta function
    result = result + sum(gammaln(y+p+yOld))-gammaln(n+sum(p)+1);
    % Botton beta function
    result = result - sum(gammaln(p+yOld)) + gammaln(sum(p)+n);
end

function result = getLogLikeX0(x)
global p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11
result = 0;
for i = 1:11
    y = [];
    p = [];
    if (i==1)
        y = [1 - x(1), x(1)];
        p = p1;
    elseif (i==2)
        y = [1 - sum(x(2:6)), x(2:6)];
        p = p2;
    elseif (i==3)
        y = [1 - sum(x(7:11)), x(7:11)];
        p = p3;
    elseif (i==4)
        y = [1 - sum(x(12:14)), x(12:14)];
        p = p4;
    elseif (i==5)
        y = [1 - x(15), x(15)];
        p = p5;
    elseif (i==6)
        y = [1 - sum(x(16:17)), x(16:17)];
        p = p6;
    elseif (i==7)
        y = [1 - x(18), x(18)];
        p = p7;
    elseif (i==8)
        y = [1 - x(19), x(19)];
        p = p8;
    elseif (i==9)
        y = [1 - x(20), x(20)];
        p = p9;
    elseif (i==10)
        y = [1 - x(21), x(21)];
        p = p10;
    else
        y = [1 - x(22), x(22)];
        p = p11;
    end
        
    % N choose K bit
    %result = result + gammaln(n+2)-sum(gammaln(y+1));
    % Top beta function
    result = result + sum(gammaln(y+p))-gammaln(sum(p)+1);
    % Botton beta function
    result = result - sum(gammaln(p)) + gammaln(sum(p));
end


function [remixedThetaStar, acceptProb] = remix(x, y, J, thetaStar)
 
global abNuA abNuB sigBeta n d nLeaf nCov;
global eps ctCov;

u1 = sort(unique(J));
for i = 1:length(u1)
    X = x(J == i, :);
    Y = y(J == i, :);
    %i
    nY = length(J(J==i));
    indList = find(J==i);

    nuA = thetaStar(i).nuA;
    nuB = thetaStar(i).nuB;
    sigComp = thetaStar(i).sigComp;
    beta=thetaStar(i).beta;

    relatedBeta = reshape(beta(1, :), 1, nLeaf);
    nuA = sqrt(getSigBeta(relatedBeta, nuA^2, abNuA(1), abNuA(2)));
    relatedBeta = reshape(beta(2:end, :), 1, nLeaf*ctCov);
    for repHype = 1:5
    nuB = sqrt(getSigBeta(relatedBeta, nuB^2, abNuB(1), abNuB(2)));
    end
    sigComp = sigBeta; 
    sigComp(1, :) = nuA*sigComp(1, :);
    sigComp(2:end, :) = nuB*sigComp(2:end, :);

    
    e = eps*( 1 ./ sqrt( 1 ./ (sigComp .^ 2) + nY/4) );

    [beta, acceptProb] = getBeta([ones(nY, 1), X], Y, beta);
    
    remixedThetaStar(i).beta = beta;
    remixedThetaStar(i).nuA = nuA;
    remixedThetaStar(i).nuB = nuB;
    remixedThetaStar(i).sigComp = sigComp;
end




%******************* PickAlpha ******************************

% This function updates "alpha: based on the old alpha and number of unique
% theta's
    

function newAlpha = pickAlpha(oldAlpha, iStar, a0, b0)
m = 40; w = 5;

x = log(oldAlpha+0.001); 
z = getLogPostAlpha(x, iStar, a0, b0) - exprnd(1);

u = rand;
L = x - w*u;
R = L + w;
v = rand;
J = floor(m*v);
K = (m-1) - J;

while J>0 && z < getLogPostAlpha(L, iStar, a0, b0)
    L = L - w;
    J = J - 1;
end

while K>0 && z < getLogPostAlpha(R, iStar, a0, b0)
    R = R+w;
    K = K-1;
end

u = rand;
newX = L + u*(R-L);

while z > getLogPostAlpha(newX, iStar, a0, b0)
    if newX < x
        L = newX;
    else
        R = newX;
    end    
    u = rand();
    newX = L + u*(R-L);
end

newAlpha = exp(newX);

    
function logPost = getLogPostAlpha(x, iStar, a0, b0)
global n;
alpha = (exp(x));

logLike = (alpha^iStar)*(exp(gammaln(alpha)-gammaln(alpha+n))); 

diffSig = x - a0;
logPrior = sum(-.5*log(2*pi)-log(b0)-( (( diffSig ).^2)./(2*(b0.^2)) ));

logPost = logLike + logPrior;


function mewAlpha = pickAlpha0 (oldAlpha, iStar)


global a0 b0 n;

nu = betarnd(oldAlpha+1, n);

proportion = (a0 + iStar - 1)/(n*(b0-log(nu)));
piNu = proportion /(proportion +1);

u = rand;
if u <= piNu
    mewAlpha = gamrnd(a0+iStar, 1./(b0-log(nu)));
else
    mewAlpha = gamrnd(a0+iStar-1, 1./(b0-log(nu)));
end


function [newB, acceptProb] = getBeta(x,y,beta)
global eps leapFrog;
global abNuA abNuB;

% Start at initial beta, then use derivatives and sampling

maxI = 500;
oldBeta = beta;
yx = x'*y; % dim = d x 1
d = length(x(1,:));
n = length(y);
xb = [];
xb1 = [];
xb2 = [];
priorB = abNuB(1)*ones(d,1);
priorB(1) = abNuA(1);
priorBS = abNuB(2)*ones(d,1);
priorBS(1) = abNuA(2);

for i = 1:maxI
    newBeta = oldBeta;
    for j = 1:1 % Make a move
        xb = [];
        xb = x*newBeta; % dim = n x 1
        %jmp = - x'*exp(xb) + yx; % dim = d x 1
        jmp = zeros(d,1);
        rdir = randn(d,1);
        newBeta = newBeta + eps*jmp + eps*rdir;
    end
    % Get acceptance probs
    xb1 = x*oldBeta;
    xb2 = x*newBeta;
    pOld = -sum(exp(xb1))+y'*xb1-sum(((oldBeta-priorB).^2)./(2*priorBS));
    pNew = -sum(exp(xb2))+y'*xb2-sum(((newBeta-priorB).^2)./(2*priorBS));
    acceptProb = min(1, exp(pNew-pOld));
    u = rand;
    if u < acceptProb
        oldBeta = newBeta;
    end
end
newB = newBeta;
    
    
    

% function [updateB, acceptProb] = getBeta(d, r, b, sigma0, e)
% global leapFrog;
% % d = [ones, X];
% % r is the y stuff
% % b is beta
% 
% [dim1, dim2] = size(b);
% 
% oldB = b;
% 
%     E = getE(d, r, oldB, sigma0);
%     g = getG(d, r, oldB, sigma0);
%     p = randn(size(b));
%     % p(1) = p(1) + mean(r);
%     H = sum(sum( .5*(p.^2) ))+ E;
%     newB = b; 
%     newG = g; 
% 
%     for leap=1:leapFrog
%         p = p - e.*newG/2;
%         newB = newB + e.*p;
%         tempG = getG(d, r, newB, sigma0);
%         if isfinite(tempG)
%             newG = tempG;
%         end
% 
%         p = p - e.*newG/2;
%     end
% 
%     newE = getE(d, r, newB, sigma0);
%     newH = sum(sum( .5*(p.^2) )) + newE;
% 
%     acceptProb = min(1, exp(H - newH)); 
% 
%     if (rand < acceptProb)
%         updateB = newB;
%     else
%         updateB = oldB;
%     end

function [newB, acceptProb] = updateBeta(x,y,indList)
% Use conjugate priors
global kk

n = length(y);
d = length(x(1,:));
% Initialize gamma
gammaNew = ones(d,1);
gammaOld = ones(d,1);
yPlus = zeros(d,1);
muOld = ones(n,1);
muNew = ones(n,1);
iMax = 5000;

for i = 1:iMax
    yPlus = [];
    
    yPlus = (x'*(y+kk(indList)))';
    for j = 1:d
        if sum(x(:,j)>0)
            gammaNew(j) = yPlus(j)/(x(:,j)'*muOld)*gammaOld(j);
            muNew = muOld.*(gammaNew(j)/gammaOld(j)).^(x(:,j));
            muOld = muNew;
        else % If we have no counts, keep beta = 0
            gammaNew(j) = 1;
        end
    end
    err = sum(abs(log(gammaOld)-log(gammaNew)));
    if err < 0.01
        break
    end
    if i == iMax
        disp('Last Betas')
        log(gammaOld)'
        disp('New Betas')
        log(gammaNew)'
    end
    gammaOld = gammaNew;
end
newB = log(gammaOld);
acceptProb = 1;
    
    
    



% This calculates the energy function for the posterior distribution. 

function E = getE(d, r, beta, sigma)
global nLeaf;

eta = d*beta;
q = length(eta);

logLike = -q*exp(eta) - sum(r)*eta;

% Prior is fine
logPrior =  sum(sum( ( -0.5*(beta).^2 ) ./ (sigma.^2) ));

E = -(logLike + logPrior);



% This part calculates the derivatives for all parameters. 
% Note that the calculation is completely vectorized and quite fast. 
% Moreover, the code is written in a way that avoids overflow.

function g = getG(d, r, beta, sigma)
global nLeaf;
eta = d*beta;
n = length(r);
m = max(eta,2);
dLogLike = (-n+sum(r)/eta)*m;

dLogPrior =  -(beta) ./ (sigma.^2);
 
g = -(dLogLike + dLogPrior);





function newSigma = getSigBeta(beta, oldSigma2, mu, tau)
m = 40; w = 5;

x = log(oldSigma2); 
z = getLogPostSigma(beta, x, mu, tau) - exprnd(1);

u = rand();
L = x - w*u;
R = L + w;
v = rand();
J = floor(m*v);
K = (m-1) - J;

while J>0 && z < getLogPostSigma(beta, L, mu, tau)
    L = L - w;
    J = J - 1;
end

while K>0 && z < getLogPostSigma(beta, R, mu, tau) 
    R = R+w;
    K = K-1;
end

u = rand();
newX = L + u*(R-L);

while z > getLogPostSigma(beta, newX, mu, tau)
    if newX < x
        L = newX;
    else
        R = newX;
    end
    
    u = rand();
    newX = L + u*(R-L);
end

newSigma = exp(newX);


function logPost = getLogPostSigma(beta, x, mu, tau)

sigma = sqrt(exp(x));
n = length(beta);

logPost = -n*(log(sqrt(2*pi)) + log(sigma)) - sum( (beta).^2 )/(2*(sigma^2)) - (log(sqrt(2*pi)) + log(tau)) - ( (x - mu)^2 )/(2*(tau^2));


