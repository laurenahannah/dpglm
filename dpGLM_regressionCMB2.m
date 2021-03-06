% This is for producing outputs for my paper.

function dpGLM_regressionCMB2(numberOfIterations, burnIn);
global muY sdY muX sdX;
global rX;

dataSize = 250;
testSize = 649;

load cmb.dat;
rodeoX = cmb(:,1);
rodeoY = cmb(:,2);
WassY = cmb(:,5);
%indList = randperm(1030);
load cmbperm.dat; % A list of permutations for comparison
indList = cmbperm(51,:);
%indList = 1:899;
xTrain = rodeoX(indList(1:dataSize),:);
yTrain = rodeoY(indList(1:dataSize));
xTest = rodeoX(indList(dataSize+1:dataSize+testSize),:);
[xTest, IX] = sort(xTest);
yTest = rodeoY(indList(dataSize+1:dataSize+testSize));
yTest = yTest(IX);
%xTest = rodeoX;
%yTest = rodeoY;
rX = xTest;

[nTrain, nn] = size(xTrain);
nTest = size(xTest, 1);
data = [xTrain(:,:); xTest(:,:)];
muX = mean(data,1);
data = data - repmat(mean(data), nTrain+nTest, 1); % This part centers the data;
sdX = std(data);
data = data./repmat(std(data),nTrain+nTest,1);

xTrain = data(1:nTrain, :);
xTest = data(nTrain+1:end, :);
[aaa, IX] = sort(xTest);
data2 = [yTrain; yTest];
muY = mean(data2);
sdY = std(data2);
data2 = data2 - muY;
data2 = data2./sdY;
yTrain = data2(1:nTrain);
yTest = data2(nTrain+1:end);
%yTest = rodeoY;


[newJ, etaBar, mHat, treeHat] = dpMnl(xTrain, yTrain, xTest, yTest, numberOfIterations, burnIn);

%mBar = mHat*sdY;
%mBar = mBar+muY;

%treeBar = treeHat*sdY;
%treeBar = treeBar+muY;

%ySm = kSmooth(rodeoX,rodeoY,20);

%figure
% subplot(2,1,2)
% hold on
% plot(xTrain(:,1),yTrain,'.')
% %plot(x1,etaT1,'k.')
%plot(x2,etaT2,'k.')

covfunc = {'covSum', {'covSEiso','covNoise'}};
logtheta0 = [log(1.0); log(sqrt(1.0)); log(sqrt(0.01))];
[logtheta, fvals, iter] = minimize(logtheta0, 'gpr', -10, covfunc, xTrain, yTrain);
exp(logtheta) 
%xGP = linspace(min(xTrain),max(xTrain),1000)';
[mmm, sss2] = gpr(logtheta, covfunc, xTrain, yTrain, xTest);
%yGP = mmm*sdY;
%yGP = yGP+muY;
%xGP2 = xGP*sdX;
%xGP2 = xGP2+muX;
gpErr = mmm-yTest;
disp('L1 GP error:')
l1GP = mean(abs(gpErr))
disp('L2 GP error:')
l2GP = mean(gpErr.*gpErr)

figure
hold on
plot(rX,yTest,'b.')
%plot(rodeoX,ySm,'g')
plot(rX,treeHat,'r')
plot(rX,mmm,'c')
plot(rX,mHat,'k')
plot(indList(1:dataSize),yTrain,'m+')

%plot(rodeoX,WassY,'r')

save ClusterNumber.dat newJ -ascii;
save OutValue.dat etaBar -ascii;

% This part calculates the accuracy rate and the F1 measure on the test
% set.
%result = getResults(yTest, p) 

% This part saves the results in a file.
%dlmwrite('dpGLM_results.dat', result, '-append');


function [newJ, etaBar, mHat, yTree] = dpMnl(xTrain, yTrain, xTest, yTest, numberOfIterations, burnIn);

global nTrain nTest n d nLeaf nCov;
global mu0 Sigma0 mu00 Sigma00 a0 b0;
global aSigma0 bSigma0 muMu sigMu;
global aSigma00 bSigma00 muSig sigSig;
global abNuA abNuB sigBeta;
global muEps sigEps;
global muY sdY muX sdX;
global leapFrog eps;
global rX;

leapFrog = 200; % Number of steps for the Hamiltonian dynamics
eps = 0.2; % This is the constant multiplier for the step size 

[nTrain, nCov] = size(xTrain);
nTest = size(xTest, 1);
data = [xTrain; xTest];
data = data - repmat(mean(data), nTrain+nTest, 1); % This part centers the data;
xTrain = data(1:nTrain, :);
xTest  = data(nTrain+1:end, :);

[n, nCov] = size(xTrain);
nLeaf = 1;

source(1).ind = 1:nCov;
% These are the parameters of the gamma prior for scale parameter, alpha.
a0 = -3; b0 = 2; 

% mu(j) ~ N(mu0(j), sigma0(j)), where sigma0 is the standard deviation 
% mu0(j) ~ N(muMu, sigMu) 
% log(sigma0(j).^2) ~ N(aSigma0, bSigma0)
mu0 = zeros(1, nCov);
muMu = zeros(1, nCov); sigMu = 2*ones(1, nCov);
Sigma0 = 1*ones(1, nCov); % Let's give epsilon the same prior as the sigmas
aSigma0 = -1; bSigma0 = 1; % for small samples
%aSigma0 = -1; bSigma0 = 1;

% log(Sigma(j)^2) ~ N(mu00(j), sigma00(j)), where Sigma00 is the standard
% deviation
% mu00(j) ~ N(muSig, sigSig)
% log(Sigma00(j).^2) ~ N(aSigma00, bSigma00)
mu00 = zeros(1, nCov);
muSig = 0; sigSig = 1;
Sigma00 = 1*ones(1, nCov);
aSigma00 = -2.5; bSigma00 = 1.5;
muEps = -3.5;
sigEps = 2;

abNuA  = [0, 1]; % Paramters of the prior for the intercept
abNuB  = [0, 1]; % Parameters of the prior for the coefficients

% Initial values
nuA = 1;
nuB = 1;
sigBeta = ones(nCov+1, nLeaf);

sigComp = sigBeta;
sigComp(1, :) = nuA*sigBeta(1, :);
sigComp(2:end, :) = nuB*sigBeta(2:end, :);

% An initial value for the Scale parameter of the Drichlet process
alpha(1) = 1;
mBar = [];
partInd = 1;
scrsz = get(0,'ScreenSize');

partit = .1:.1:.9;
np = length(partit)
yyy = xTrain + randn(nTrain,1)*.01;
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
%newNj

%newJ(1:50)

for i = 1:np+1
    vecs = [];
    vecs = ind(i).vec;
    thetaStar(i).mu = mean(xTrain(vecs,:),1);

    
%thetaStar(i).sd = std(xTrain(vecs,1:4));
if (newNj(i) > 1)
thetaStar(i).sd = std(xTrain(vecs,:),1)+.1;
else
    thetaStar(i).sd = 2*ones(1, nCov);
end

beta = regress(yTrain(vecs),[ones(length(vecs),1),xTrain(vecs,:)]);
thetaStar(i).eps = 1; % This is the error term for the GLM
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
if mod(iter,20)==0
    iter
end

% This part calls the MCMC algorithm
[thetaStar, newJ, newNj] = main_MCMC(xTrain, yTrain, thetaStar, newJ, newNj, alpha(iter));
% This prints the new frequncy of each cluster on the screen 
%disp('Number of samples in each component:')
%disp(newNj')
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
    eps = 0.4;
else
    eps = 0.2;
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


% This part samples from the posterior distribution of hyperparameters. You
% can remove this part if you do not want to use hyperparamters.
if iter >= 5

    uniqueMu = cat(1, thetaStar.mu);
    uniqueSd = log( (cat(1, thetaStar.sd)).^2 );
    
    for j = 1:nCov
        mu0(1, j) = getMu0(uniqueMu(:, j), mu0(1, j), muMu, sigMu, Sigma0(1, j));   
        Sigma0(1, j) = sqrt(getSig0(uniqueMu(:, j), Sigma0(1, j).^2, aSigma0, bSigma0, mu0(1, j)));
        
        mu00(1, j) = getMu0(uniqueSd(:, j), mu00(1, j), muSig, sigSig, Sigma00(1, j));   
        Sigma00(1, j) = sqrt(getSig0(uniqueSd(:, j), Sigma00(1, j).^2, aSigma00, bSigma00, mu00(1, j)));
    end
        
end


dpMnlSamp(iter).mu = cat(1, thetaStar.mu);
dpMnlSamp(iter).sd = cat(1, thetaStar.sd);
dpMnlSamp(iter).nuA = cat(1, thetaStar.nuA);
dpMnlSamp(iter).nuB = cat(1, thetaStar.nuB);
dpMnlSamp(iter).beta = cat(3, thetaStar.beta);
dpMnlSamp(iter).eps = cat(1, thetaStar.eps);

%save dpMnlSamp2 dpMnlSamp;
if (iter >= burnIn)&&(mod(iter,10)==0)
yOut = zeros(nTest,1);
err = zeros(nTest,1);
dims = max(newJ);
nn = sum(newNj);
for i = 1:length(xTest)
    % Get probabilities for each cluster
    pVec = zeros(dims,1);
    yVecs = zeros(dims,1);
    for j = 1:dims
        mu = thetaStar(j).mu;
        sd = thetaStar(j).sd;
        pp = getLogLikeX(xTest(i,:),mu,sd,mu0,Sigma0,mu00,Sigma00) + log(newNj(j))-log(nn);
        pVec(j) = pp;
        beta = thetaStar(j).beta;
        yVecs(j) = [1 xTest(i,:)]*beta;
    end
    pVec = pVec-max(pVec);
    pVec = exp(pVec);
    pVec = pVec/sum(pVec);
    yOut(i) = pVec'*yVecs;
    %yOut(i) = yOut(i)*sdY;
    %yOut(i) = yOut(i)+muY;
    err(i) = yTest(i) - yOut(i);
end
mBar = [mBar; yOut'];
end
end
mHat = mean(mBar,1);
metaErr = yTest-mHat';
%save dpRegressionXCluster.dat xClusterVec -ascii
%save dpRegressionTheta.mat -struct thetaVec
for i = 1:length(xTrain)
    beta = [];
    index = newJ(i);
    beta = thetaStar(index).beta;
    etaBar(i) = beta(1) + xTrain(i,:)*beta(2:length(beta));
    etaBar(i) = etaBar(i)*sdY;
    etaBar(i) = etaBar(i)+muY;
end

dims = max(newJ);
nn = sum(newNj);
cent = mean(xTest);

% figure
% subplot(2,1,1)
% hold on
% plot(xTest(:,1)-cent(1),yTest,'.')
% plot(xTest(:,1)-cent(1),yOut,'k.')

regInd = 1;

xVec = [];
xVec = zeros(nComp,2);
yyVec = [];
yyVec = zeros(nComp,2);
for i = 1:nComp
    mu = thetaStar(i).mu;
    sd = thetaStar(i).sd;
    [aa ab] = size(mu);
    % Make everything row vectors
    if aa > ab
        mu = mu';
    end
    [ba bb] = size(sd);
    if ba > bb
        sd = sd';
    end
    x1 = [];
    x2 = [];
    %size(mu(regInd))
    %size(sd(regInd))
    x1 = mu(regInd)-sd(regInd);
    x11 = x1*sdX(regInd);
    x11 = x11+muX(regInd);
    x2 = mu(regInd)+sd(regInd);
    x21 = x2*sdX(regInd);
    x21 = x21+muX(regInd);
    xVec(i,:) = [x11(regInd(1)),x21(regInd(1))];
    beta = thetaStar(i).beta;
    eta1 = [];
    eta2 = [];
    eta1 = beta(1) + sum(beta(2:end)'.*x1);
    eta2 = beta(1) + sum(beta(2:end)'.*x2);
    yyVec(i,:) = [eta1, eta2];
    yyVec(i,:) = yyVec(i,:)*sdY;
    yyVec(i,:) = yyVec(i,:)+muY;
end
timePassed = cputime - startTime
figure
hold on
plot(yTest(1:50),mHat(1:50),'.')
xlabel('Y Actual')
ylabel('Y Predicted')
title('DP-GLM Regression')

% Make a regression tree for comparison
concreteTree = [];
concreteTree = classregtree(xTrain,yTrain);
yTree = [];
yTree = eval(concreteTree,xTest);
%yTree = yTree*sdY;
%yTree = yTree+muY;
treeErr = [];
treeErr = yTest-yTree;

xxxx = [ones(nTrain,1), xTrain];
xxxxTest = [ones(nTest,1), xTest];
betaReg = regress(yTrain,xxxx)
yReg = xxxxTest*betaReg;
%yReg = yReg*sdY;
%yReg = yReg+muY;
plot(yTest(1:50),yReg(1:50),'g.')
plot(yTest(1:50),yTree(1:50),'c.')
%plot([-2,3],[-2,3])

legend('DP-GLM','Linear Regression','Regression Tree')
regErr = yTest - yReg;

figure
hold on
plot(yTest(1:50),metaErr(1:50),'r.')
plot(yTest(1:50),regErr(1:50),'m.')
plot(yTest(1:50),treeErr(1:50),'c.')
xlabel('Y Actual')
ylabel('Error')
title('DP-GLM Regression')
legend('DP Error','Linear Error','Tree Error')

disp('DP-GLM L1')
L1dp = sum(abs(metaErr))/nTest
disp('DP-GLM L2')
L2dp = metaErr'*metaErr/nTest

disp('Linear L1')
L1lin = sum(abs(regErr))/nTest
disp('Linear L2')
L2lin = regErr'*regErr/nTest

disp('Tree L1')
L1tree = sum(abs(treeErr))/nTest
disp('Tree L2')
L2tree = treeErr'*treeErr/nTest

%for i = 1:dims
%    beta = thetaStar(i).beta
%end
figure('Position',[scrsz(3)/8, scrsz(4)/8, scrsz(3)*3/4, scrsz(4)*3/4]);

plotInd = 1;
subplot(3,1,1)
hold on
frame = [];
%axis([-2,2,-5,5]);
plot(rX,yTest,'.')
plot(rX,yOut,'k.')
%h = gca;
%frame = getframe(h);
%dpTest = addframe(dpTest,frame);

%fig2 = figure;
subplot(3,1,2)
hold on
frame = [];
%axis([-2,2,-5,5]);
plot(rX,yTest,'.')
%plot(,etaBar,'r.')
%h = gca;
%frame = getframe(h);
%dpTrain = addframe(dpTrain,frame);
%cMap = colormap(jet(maxClusters));
%fig3 = figure;
subplot(3,1,3)
hold on
frame = [];
%axis([-2,2,-5,5]);
%plot(rX,etaBar,'r.')
for i = 1:nComp
    myHandle = plot(xVec(i,:),yyVec(i,:));
    %set(myHandle,'Color',cMap(i,:),'LineWidth',2);
end
xTestNew = [xTest(:,plotInd); 3];
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






function [thetaStar, updatedJ, updatedNj] = main_MCMC(x, y, thetaStar, J, nj, alpha);

global a0 b0; 
global mu0 Sigma0 mu00 Sigma00; 
global aSigma00 bSigma00 muSig sigSig;
global abNuA abNuB sigBeta;
global n nTrain nTest nLeaf;
global muEps sigEps;
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
            sd = sqrt(exp(normrnd(mu00, Sigma00)));
            mu = normrnd(mu0, Sigma0);
            nuA = sqrt(exp(normrnd(abNuA(1), abNuA(2))));
            nuB = sqrt(exp(normrnd(abNuB(1), abNuB(2))));
            sigComp = sigBeta;
            sigComp(1, :) = nuA*sigBeta(1, :);
            sigComp(2:end, :) = nuB*sigBeta(2:end, :);
            beta = normrnd(0, sigComp);
            epsilon = sqrt(exp(normrnd(muEps,sigEps)));
            thetaStar(kBar+1+m).mu = mu;
            thetaStar(kBar+1+m).sd = sd;
            thetaStar(kBar+1+m).nuA = nuA;
            thetaStar(kBar+1+m).nuB = nuB;
            thetaStar(kBar+1+m).sigComp = sigComp;
            thetaStar(kBar+1+m).beta = beta;
            thetaStar(kBar+1+m).eps = epsilon;
        end
    else
        kBar = length(nj);
        for m=1:M
            sd = sqrt(exp(normrnd(mu00, Sigma00)));
            mu = normrnd(mu0, Sigma0);
            nuA = sqrt(exp(normrnd(abNuA(1), abNuA(2))));
            nuB = sqrt(exp(normrnd(abNuB(1), abNuB(2))));
            sigComp = sigBeta;
            sigComp(1, :) = nuA*sigBeta(1, :);
            sigComp(2:end, :) = nuB*sigBeta(2:end, :);
            beta = normrnd(0, sigComp);
            epsilon = sqrt(exp(normrnd(muEps, sigEps)));
            thetaStar(kBar+m).mu = mu;
            thetaStar(kBar+m).sd = sd;
            thetaStar(kBar+m).nuA = nuA;
            thetaStar(kBar+m).nuB = nuB;
            thetaStar(kBar+m).sigComp = sigComp;
            thetaStar(kBar+m).beta = beta;
            thetaStar(kBar+m).eps = epsilon;
         end
    end

    q1 = zeros(1,kBar);
        
    for k = 1:kBar
        mu = thetaStar(k).mu;
        beta = thetaStar(k).beta;
        eta = [1, x(i, :)]*beta;
        sd = thetaStar(k).sd;
        sigComp = thetaStar(k).sigComp;
        epsilon = thetaStar(k).eps;
        q1(k) = getLogLike(x(i, :), y(i, :), mu, sd, eta, epsilon);
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
        mu = thetaStar(kBar+k).mu;
        beta = thetaStar(kBar+k).beta;
        eta = [1, x(i, :)]*beta;
        sd = thetaStar(kBar+k).sd;
        sigComp = thetaStar(kBar+k).sigComp;
        epsilon = thetaStar(kBar+k).eps;
        q2(k) = getLogLike(x(i, :), y(i, :), mu, sd, eta, epsilon);
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

function output = getLogLikeX(x, mu, sd, mu0, Sigma0, mu00,Sigma00)
sd = sd+.001;
diffX = (x-mu);
diffMu = mu - mu0;
likeX = sum( - log(sd) - ( ((diffX).^2)./(2*(sd.^2)) ) );
likeMu = sum(-.5*log(2*pi)-log(Sigma0)-( (( diffMu ).^2)./(2*(Sigma0.^2)) ) );
diffSig = x - mu00;
logPriorSd = sum(-.5*log(2*pi)-log(Sigma00 )-( (( diffSig ).^2)./(2*(Sigma00.^2)) ));
output = likeX + likeMu + logPriorSd;



function result = getLogLike(x, y, mu, sd, eta, epsilon);
global n d nLeaf;

sd = sd + .001;
diffX = (x - mu);
logLikeX = sum( - log(sd) - ( ((diffX).^2)./(2*(sd.^2)) ) ); 

% This is for GLM bit
logLikeY = -log(epsilon) - ( (eta - y)^2/(2*epsilon^2));

result = logLikeX + logLikeY;


function [remixedThetaStar, acceptProb] = remix(x, y, J, thetaStar)
 
global mu0 Sigma0 mu00 Sigma00 abNuA abNuB sigBeta n d nLeaf nCov;
global muEps sigEps;
global eps;

u1 = sort(unique(J));
for i = 1:length(u1)
    X = x(J == i, :);
    Y = y(J == i, :);
    %i
    nY = length(J(J==i));

    mu = thetaStar(i).mu;
    sd = thetaStar(i).sd;
    nuA = thetaStar(i).nuA;
    nuB = thetaStar(i).nuB;
    sigComp = thetaStar(i).sigComp;
    beta=thetaStar(i).beta;
    epsilon = thetaStar(i).eps;

    for j = 1:nCov
        mu(j) = getMu0(X(:, j), mu(j), mu0(j), Sigma0(j), sd(j));   
        sd(j) = sqrt(getSig0(X(:, j), sd(j).^2, mu00(j), Sigma00(j), mu(j)));
    end
    
    epsilon = sqrt(getEps0(X, Y, beta, epsilon^2, muEps, sigEps));

    relatedBeta = reshape(beta(1, :), 1, nLeaf);
    nuA = sqrt(getSigBeta(relatedBeta, nuA^2, abNuA(1), abNuA(2)));
    relatedBeta = reshape(beta(2:end, :), 1, nLeaf*nCov);
    for repHype = 1:50
    nuB = sqrt(getSigBeta(relatedBeta, nuB^2, abNuB(1), abNuB(2)));
    end
    sigComp = sigBeta; 
    sigComp(1, :) = nuA*sigComp(1, :);
    sigComp(2:end, :) = nuB*sigComp(2:end, :);


    e = eps*( 1 ./ sqrt( 1 ./ (sigComp .^ 2) + nY/4) );

    [beta, acceptProb(1, i)] = getBeta([ones(nY, 1), X], Y, beta, sigComp, e, epsilon);
    
    
    remixedThetaStar(i).mu = mu;
    remixedThetaStar(i).beta = beta;
    remixedThetaStar(i).sd = sd;
    remixedThetaStar(i).nuA = nuA;
    remixedThetaStar(i).nuB = nuB;
    remixedThetaStar(i).sigComp = sigComp;
    remixedThetaStar(i).eps = epsilon;
end


% I am not using this right now
function [yPred] = getPredProb(x, thetaStar, nj, alpha, mu0, Sigma0, mu00, Sigma00, sigBeta, yBar);

global a0 b0; 
global aSigma00 bSigma00 muSig sigSig abNuA abNuB;

global n nTrain nTest nLeaf nCov;

iStar = length(nj);
q = zeros(nTest, iStar+1);

% This part is sample from G_0, which is used to get the predictive
% probability
sd0 = repmat(sqrt(exp(normrnd(mu00, Sigma00))), nTest, 1);
mu0 = repmat(normrnd(mu0, Sigma0), nTest, 1);
nuA0 = sqrt(exp(normrnd(abNuA(1), abNuA(2))));
nuB0 = sqrt(exp(normrnd(abNuB(1), abNuB(2))));
sigComp0 = sigBeta;
sigComp0(1, :) = nuA0*sigBeta(1, :);
sigComp0(2:end, :) = nuB0*sigBeta(2:end, :);
beta0 = normrnd(0, sigComp0);
thetaStar(iStar+1).beta = beta0;

% This part uses the unique component to get the predictive probability
for k = 1:iStar
    mu = repmat(thetaStar(k).mu, nTest, 1);
    sd = repmat(thetaStar(k).sd, nTest, 1);
    diffX = (x - mu);
    q(:, k) = sum( - log(sd) - ( ((diffX).^2)./(2*(sd.^2)) ), 2);     
    q(:, k) = q(:, k) + log(nj(k)) - log(n+alpha);    
end

diffX = (x - mu0);
q(:, iStar+1) = sum( - log(sd0) - ( ((diffX).^2)./(2*(sd0.^2)) ), 2);     
q(:, iStar+1) = q(:, iStar+1) + log(alpha) - log(n+alpha);

% This calculates P(x) for the current iteration
%m = max(q');
%postX(:, 1) = exp(m'+log(sum(exp(q - repmat(m', 1, iStar+1) ), 2)));


% This part gets P(y) for the current iteration
qRel = q - repmat(m', 1, iStar+1);
q = exp(qRel);
q = q./repmat(sum(q, 2), 1, iStar+1);


for k = 1:iStar+1
    etaT = [ones(nTest, 1), x]*thetaStar(k).beta;
    m = max(etaT');        
    %predProb(:, :, k) = repmat(q(:, k), 1, nLeaf).*exp(etaT - repmat(m' + log(sum(exp(etaT - repmat(m', 1, nLeaf) ), 2)), [1, nLeaf, 1] ) );
    
end
    
predProb = sum(predProb, 3);




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



function newMu = getMu0(y, oldMu, mu0, sigma0, sigma)
m = 40; w = 5;

x = oldMu; 
z = getLogPostMu0(y, x, mu0, sigma0, sigma) - exprnd(1);

u = rand();
L = x - w*u;
R = L + w;
v = rand();
J = floor(m*v);
K = (m-1) - J;

while J>0 && z < getLogPostMu0(y, L, mu0, sigma0, sigma)
    L = L - w;
    J = J - 1;
end

while K>0 && z < getLogPostMu0(y, R, mu0, sigma0, sigma) 
    R = R+w;
    K = K-1;
end

u = rand();
newX = L + u*(R-L);

while z > getLogPostMu0(y, newX, mu0, sigma0, sigma)
    if newX < x
        L = newX;
    else
        R = newX;
    end
    
    u = rand();
    newX = L + u*(R-L);
end

newMu = newX;

    
function logPost = getLogPostMu0(y, mu, mu0, Sigma0, sd)

sd = sd+0.001;
Sigma0 = Sigma0+0.001;
diffY = (y - mu);
logLike = sum( -.5*log(2*pi) - log(sd) - ( ((diffY).^2)./(2*(sd.^2)) ) ); 

diffMu = mu - mu0;
logPriorMu = sum(-.5*log(2*pi)-log(Sigma0)-( (( diffMu ).^2)./(2*(Sigma0.^2)) ) );

logPost = logLike + logPriorMu; 




function newSigma2 = getSig0(y, oldSigma2, mu00, sigma00, muY)
m = 40; w = 5;

x = log(oldSigma2+0.001); 
z = getLogPostSigma0(y, x, mu00, sigma00, muY) - exprnd(1);

u = rand();
L = x - w*u;
R = L + w;
v = rand();
J = floor(m*v);
K = (m-1) - J;

while J>0 && z < getLogPostSigma0(y, L, mu00, sigma00, muY)
    L = L - w;
    J = J - 1;
end

while K>0 && z < getLogPostSigma0(y, R, mu00, sigma00, muY)
    R = R+w;
    K = K-1;
end

u = rand();
newX = L + u*(R-L);

while z > getLogPostSigma0(y, newX, mu00, sigma00, muY)
    if newX < x
        L = newX;
    else
        R = newX;
    end
    
    u = rand();
    newX = L + u*(R-L);
end

newSigma2 = exp(newX);

    
function logPost = getLogPostSigma0(y, x, mu00, Sigma00, mu)

sd = sqrt(exp(x));
sd = sd+0.001;
Sigma00 = Sigma00+0.001;

diffY = (y - mu);
logLike = sum( -.5*log(2*pi) - log(sd) - ( ((diffY).^2)./(2*(sd.^2)) ) ); 

diffSig = x - mu00;
logPriorSd = sum(-.5*log(2*pi)-log(Sigma00 )-( (( diffSig ).^2)./(2*(Sigma00.^2)) ));

logPost = logLike + logPriorSd;


function newSigma2 = getEps0(a,y, beta, oldSigma2, mu00, sigma00)
m = 40; w = 5;

d = [ones(length(a(:,1)),1), a];
eta = d*beta;

x = log(oldSigma2+0.001); 
z = getLogPostEps0(y, eta, x, mu00, sigma00) - exprnd(1);

u = rand();
L = x - w*u;
R = L + w;
v = rand();
J = floor(m*v);
K = (m-1) - J;

while J>0 && z < getLogPostEps0(y, eta, L, mu00, sigma00)
    L = L - w;
    J = J - 1;
end

while K>0 && z < getLogPostEps0(y, eta, R, mu00, sigma00)
    R = R+w;
    K = K-1;
end

u = rand();
newX = L + u*(R-L);

while z > getLogPostEps0(y, eta, newX, mu00, sigma00)
    if newX < x
        L = newX;
    else
        R = newX;
    end
    
    u = rand();
    newX = L + u*(R-L);
end

newSigma2 = exp(newX);

function logPost = getLogPostEps0(y, eta, x, mu00, Sigma00)

sd = sqrt(exp(x));
sd = sd+0.001;
Sigma00 = Sigma00+0.001;

diffY = (y - eta);
logLike = sum( -.5*log(2*pi) - log(sd) - ( ((diffY).^2)/(2*(sd^2)) ) ); 

diffSig = x - mu00;
logPriorSd = sum(-.5*log(2*pi)-log(Sigma00 )-( (( diffSig ).^2)./(2*(Sigma00.^2)) ));

logPost = logLike + logPriorSd;



function [updateB, acceptProb] = getBeta(d, r, b, sigma0, e, epsilon)
global leapFrog;
% d = [ones, X];
% r is the y stuff
% b is beta

[dim1, dim2] = size(b);

oldB = b;

    E = getE(d, r, oldB, sigma0, epsilon);
    g = getG(d, r, oldB, sigma0, epsilon);
    p = randn(size(b));
    % p(1) = p(1) + mean(r);
    H = sum(sum( .5*(p.^2) ))+ E;
    newB = b; 
    newG = g; 

    for leap=1:leapFrog
        p = p - e.*newG/2;
        newB = newB + e.*p;
        tempG = getG(d, r, newB, sigma0, epsilon);
        if isfinite(tempG)
            newG = tempG;
        end

        p = p - e.*newG/2;
    end

    newE = getE(d, r, newB, sigma0, epsilon);
    newH = sum(sum( .5*(p.^2) )) + newE;

    acceptProb = min(1, exp(H - newH)); 

    if (rand < acceptProb)
        updateB = newB;
    else
        updateB = oldB;
    end


% This calculates the energy function for the posterior distribution. 

function E = getE(d, r, beta, sigma, epsilon)
global nLeaf;

eta = d*beta;
q = length(eta);

%m = max(eta');
%logLike = sum( (sum((r.*eta), 2) - (m' + log(sum(exp(eta - repmat(m', 1, nLeaf) ), 2)) ) ) );
logLike = -q*log(epsilon) - sum( (eta-r).^2)/(2*epsilon^2);

% Prior is fine
logPrior =  sum(sum( ( -0.5*(beta).^2 ) ./ (sigma.^2) ));

E = -(logLike + logPrior);



% This part calculates the derivatives for all parameters. 
% Note that the calculation is completely vectorized and quite fast. 
% Moreover, the code is written in a way that avoids overflow.

function g = getG(d, r, beta, sigma, epsilon)
global nLeaf;
eta = d*beta;

%m = max(eta');

%dLogLike = (d'*r - d'*exp( eta - repmat( (m' + log(sum(exp(eta - repmat(m', 1, nLeaf) ), 2)) ) , 1, nLeaf) ) );
dLogLike = -1/(epsilon^2)*d'*(eta-r);

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



function results = getResults(target, p);

n = length(target);
c = unique(target);

for i = 1:n
    classProb(i) = p(i, target(i));
end
avgLogProb = mean(log(classProb));

[maxPred, maxInd] = max(p'); 
predClass = maxInd';

accRateTest = mean(logical(predClass == target))

[m, indM] = max(p');
predClass = indM';


for j = 1:length(c)
    i = c(j);
    categA = length(find(target==i & predClass == i));
    categB = length(find(target~=i & predClass == i));
    categC = length(find(target==i & predClass ~= i));
    
    categF1(i) = 2*categA / (2*categA+ categB + categC);
end

f1 = mean(categF1);


for i = 1:n
    precisionI(i) = 1 / sum(logical(p(i, :) >= p(i, target(i))));
end

precision = mean(precisionI);


results = [avgLogProb, accRateTest, precision, f1];   




function [train, rTrain, test, rTest, x1, etaT1, x2, etaT2] = simulateData(n, state)

rand('state', state);
randn('state', state);

nClass = 1;
nVar = 1;

mu = normrnd(0, 1, 2, nVar);
sd = sqrt(exp(normrnd(0, 2, 2, nVar)));

sigmaInt = .1;
nu = sqrt(exp(normrnd(0, 2, 1, 2)));
nuEps = sqrt(exp(normrnd(0,2,1,2)));

beta1 = [normrnd(0, sigmaInt, 1, nClass); normrnd(0, nu(1), nVar, nClass)];
beta2 = [normrnd(0, sigmaInt, 1, nClass); normrnd(0, nu(2), nVar, nClass)];

x1 = normrnd(repmat(mu(1, :), n/2, 1), repmat(sd(1, :), n/2, 1), n/2, nVar);
etaT1 = [ones(n/2, 1), x1]*beta1;
%y1 = etaT1 + nuEps(1)*randn(n/2,1);
y1 = sin(x1(:,1)*5).*abs(x1(:,1))+ randn(n/2,1)*nuEps(1)/5;

x2 = normrnd(repmat(mu(2, :), n/2, 1), repmat(sd(2, :), n/2, 1), n/2, nVar);
etaT2 = [ones(n/2, 1), x2]*beta2;
%y2 = etaT2 + nuEps(2)*randn(n/2,1);
y2 = sin(x2(:,1)*5).*abs(x2(:,1)) + randn(n/2,1)*nuEps(2)/5;

x = [x1; x2];
y = [y1; y2];

samp = randsample(n, 300);
train = x(samp, :);
rTrain = y(samp, :);
test = x(setdiff(1:n, samp), :);
rTest = y(setdiff(1:n, samp), :);



% gscatter(train(:, 2), train(:, 3), rTrain, '', '+*>d');