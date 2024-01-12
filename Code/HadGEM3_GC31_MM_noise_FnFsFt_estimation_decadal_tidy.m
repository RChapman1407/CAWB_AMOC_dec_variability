%Chapman, Ashwin, Wood, Baker 2024
%Run noise estimation on HadGEM3MM
%3 box model for AMOC
%Nov 2023
%piControl HadGEM3-GC3.1-MM
clear all; clc;

HadGEM3MM = readmatrix('HadGEM3MM.txt');
SN = HadGEM3MM(:,2);
ST = HadGEM3MM(:,3);
time = HadGEM3MM(:,1);

%Move to decadal averages
tav=10;
temp=movmean(SN,tav,1);
sampletimes=1:tav:size(SN,1);
temp2=temp(sampletimes,:);
SN=temp2;
temp=movmean(ST,tav,1);
sampletimes=1:tav:size(ST,1);
temp2=temp(sampletimes,:);
ST=temp2;
temp=movmean(time,tav,1);
sampletimes=1:tav:size(time,1);
temp2=temp(sampletimes,:);
time=temp2;
%%
%plot the data
plot(time,SN,'b');
hold on;
plot(time,ST,'r');

N = size(SN,1);
Sdata = [SN,ST];
Sdata = Sdata/1000; %raw salinity

%define problem parameters
p = HadGEM3MMParameters();
Dt = 10;

%Likelihood estimation
sigma_s = [-20,-20,0,0,0]; % 5 parameters for search, starting points
options = optimset('MaxIter',5500,'Display','iter','TolX',1e-6,'TolFun',1e-6,'MaxFunEvals',2000);
likfn = @(sigma) approxloglfmulti(Sdata,Dt,sigma,p);

%minimise liklihood
sigma = fminsearch(likfn,sigma_s,options);
%print results
Q = covmat(sigma) %is positive definite
noise_matrix = chol(Q,'lower')
p.Fn = p.Fno + sigma(4)*1e6; 
p.Ft = p.Fto + sigma(5)*1e6;
p.Fs = p.Fso - (sigma(4)+sigma(5))*1e6;

%plot the model with noise estimate
t = [-200:10:300]; %allow for spin up of box model
n = size(t,2);
Sinput = [p.Sn p.St];
H = hosingfunc0(t);
I = eye(2);
Sx = 1000*ThreeboxEMvectors(noise_matrix,I,n,Dt,t,Sinput,p,H);%psu
plot(t,Sx(1,:),'b-.');%Northern
plot(t,Sx(2,:),'r-.');%Tropical

legend('$S_{N}\ HadGEM3-GC3.1-MM$','$S_{T}\ HadGEM3-GC3.1-MM$','$S_{N}\ \rm{box\ model\ with\ estimated\ noise}$','$S_{T}\ \rm{box\ model\ with\ estimated\ noise}$','interpreter','latex','FontSize',12);
set(gca,'LineWidth',1.2,'FontSize',12);
xlabel('\boldmath $\rm{Time}\ (years)$','Interpreter','latex','FontSize',12);
ylabel('\boldmath $S_{x}\ (psu)$','Interpreter','latex','fontsize',12);

%% Approx log likelihood function from Sarkka and Solin
%
function l = approxloglfmulti(x,Dt,sigma,p)
%input data set x (eg FAMOUS),
%sigma is a vector of parameters

N = size(x,1); %number of data points
total = 0;

L= eye(2);

% Q symmetric covariance matrix param by sigma
Q = covmat(sigma); 
% set any parameters here
p.Fn = p.Fno + sigma(4)*1e6;
p.Ft = p.Fto + sigma(5)*1e6;
p.Fs = p.Fso - (sigma(4)+sigma(5))*1e6;


for k=1:N-1
    %
    tk= k*Dt;

    %
    Hk=hosingfunc0(tk);
    [dxk,qk] = threeboxfunction(x(k,:),p,Hk);
    %x(tk)
    xk = x(k,:)';
    %x(tk+1)
    xk1 = x(k+1,:)';
    %f(x(tk))
    df = xk1 - xk - dxk*Dt; %column vector
    L = (1+p.qscale*(qk<0))*eye(2);
    ML = L*Q*L'*Dt;

    temp1 = 0.5*log(det(2*pi*ML));
    temp2 = 0.5*df'*inv(ML)*df;

    l = temp1 + temp2;
    total = total + l;

end
l = total/N;

end


%% parameterize covariance matrix
%
function QM=covmat(sigma)

% always produces pos definite matrix
var1=exp(sigma(1));
var2=exp(sigma(2));
cov=exp((sigma(1)+sigma(2))/2)*tanh(sigma(3));
QM = [var1,cov;cov,var2];%
end


%% Wood three box model
%
function [dydt,q] = threeboxfunction(Sinput,p,H)
% input is raw salinities, as is output
% dydt(1) is S_N and dydt(2) is S_T
dydt = [0;0];

SN=Sinput(1);%inputs
ST=Sinput(2);
SB=p.Sb;
SS=p.Ss;
S0=p.S0;
% include p.dV change in Vn/Vt

% seconds per year
Y = 3.15e7;

SIP = (p.C-(p.Vn+p.dV).*SN - (p.Vt-p.dV).*ST - p.Vs.*SS - p.Vb.*SB)/p.Vip;

% switch depending on sign of q:
q = (p.lambda.*(p.alpha.*(p.Ts-p.T0) +p.beta.*(SN-SS)))...
    ./(1+ p.lambda.*p.alpha.*p.mu);
aq =abs(q);

%H is hosing given as input
FN = p.Fn + p.An*H;
FT = p.Ft + p.At*H;

if q>=0
    dydt(1) = (aq*(ST-SN)+p.Kn*(ST-SN)-FN*S0).*(Y/(p.Vn+p.dV));
    dydt(2) = (aq*(p.gamma*SS+(1-p.gamma).*SIP-ST) ...
        +p.Ks*(SS-ST)+p.Kn*(SN-ST)-FT*S0).*(Y/(p.Vt-p.dV));
else
    dydt(1) = (aq*(SB-SN)+p.Kn*(ST-SN)-FN*S0).*(Y/(p.Vn+p.dV));
    dydt(2) = (aq*(SN-ST)+p.Ks*(SS-ST)+p.Kn*(SN-ST)-FT*S0).*(Y/(p.Vt-p.dV));
end

end


%% Euler Maruyama integration with given noise
%
function SxEM = ThreeboxEMvectors(Linput,Q,N,Dt,t,Sinput,p,H);

SxEM = zeros(3,N); %preallocate array
len = length(t);
rng('default');
r1 = normrnd(0,1,[1,len]);
r2 = normrnd(0,1,[1,len]);
SN = Sinput(1);
ST = Sinput(2);
SS = p.Ss;
SIP = p.Sip;
SB = p.Sb;
S0 = p.S0;

% Cholesky factorization such that SQ*SQ'=Q
SQ = chol(Q, 'lower');


for j = 1:N
    %L=L*q
    Winc1 = sqrt(Dt)*r1(j);
    Winc2 = sqrt(Dt)*r2(j);
    S = [SN,ST]; %updated each iteration
    [dSdt,q] = threeboxfunction(S,p,H(j));
    
    L = (1+p.qscale*(q<0))*Linput;
    ML = L*SQ;
    % the output of threeboxfunction is not in psu
    SN = SN + dSdt(1,:)*Dt + ML(1,1)*Winc1 + ML(1,2)*Winc2;
    SxEM(1,j) = SN;
    ST = ST + dSdt(2,:)*Dt + ML(2,1)*Winc1 + ML(2,2)*Winc2;
    SxEM(2,j) = ST; 
    SIP = (p.C-(p.Vn+p.dV).*SN - (p.Vt-p.dV).*ST - p.Vs.*p.Ss - p.Vb.*p.Sb)/p.Vip;
    SxEM(3,j) = SIP;
end

end

%% HadGEM3MMparameter set
%
function p = HadGEM3MMParameters()

p.Vn = 4.191e16; p.Vt = 4.192e16; p.Vs = 13.26e16; p.Vip = 16.95e16; p.Vb = 96.76e16;
p.Fn = 0.5996e6; p.Ft = -0.7527e6; p.Fs = 1.126e6; p.Fip = -0.6728e6; p.Fb = 0;
p.Fno = p.Fn; p.Fso = p.Fs; p.Fto = p.Ft;
p.Sn = 0.03488; p.St = 0.03563; p.Ss = 0.03444; p.Sip = 0.0345; p.Sb = 0.03475; p.S0 = 0.0350;

p.Ts = 5.349; p.T0 = 4.514;
p.Kn = 4.73e6; p.Ks = 7.68e6; p.Kip = 123.49e6;
p.eta = 4.821e6;
p.alpha = 0.12; p.beta = 790; %these still from FAMOUS A (no update)

%gamma still from FAMOUS A, no update
p.lambda = 2.328e7; p.gamma = 0.58; p.mu = 0;
p.C = p.Vn*p.Sn + p.Vt*p.St + p.Vs*p.Ss + p.Vip*p.Sip + p.Vb*p.Sb;
p.An = 0.9841; p.At = -0.1853; p.As = -0.1742; p.Aip = -0.6245;

% used for modifications
p.offset=0;
p.dV = 0;
p.qscale =0;


end

%%
function [h] = hosingfunc0(time)
%input t is time steps being used
%output h is in m^3 s^-1

len = length(time);
h = zeros(1,len);

end
