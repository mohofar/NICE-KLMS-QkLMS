%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Morteza Homayounfar                             %
% Implimentation of NICE-KLMS and the others      %
% Transfer learning                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 clc;clear;close all;
%% data prepartion1 Macky Glass
%time delay (embedding) length
TD = 10;
%kernel parameter
a = 1;%fixed
%noise std
np =.01;
%data size
N_tr = 4000;
N_te = 1500;%
disp('Learning curves are generating. Please wait...');

load data\MKG.mat   %MK30 5000*1
MK30 = MKG;
MK30 = MK30+np*randn(size(MK30));
MK30 = (MK30 - min(MK30))/(max(MK30)-min(MK30));

%3000 training data
train_set = MK30(501:6700);
%300 testing data
test_set = MK30(7001:9300);

%data embedding
X = zeros(TD,N_tr);
for k=1:N_tr
    X(:,k) = train_set(k:k+TD-1)';
end
T = train_set(TD+1:TD+N_tr);

X_te = zeros(TD,N_te);
for k=1:N_te
    X_te(:,k) = test_set(k:k+TD-1)';
end
T_te = test_set(TD+1:TD+N_te);

% X     ---> trian data
% T     ---> desire train
% X_te  ---> test data
% T_te  ---> desire test

%% algorithms
%macy
load Loranz_results\centers.mat
load Loranz_results\centers_q.mat
load Loranz_results\clusters_filter_weigth.mat
load Loranz_results\clusters_filter_weigth_q.mat
load Loranz_results\s_loranz.mat
load Loranz_results\sq_loranz.mat

lr_l = 0.005;
tic
%macy
d_c = .1;
[clusters_filter_weigth_l2m,centers_l2m,EW_NICE_KLMS_l2m,prediction_NICE_KLMS_l2m,mse_NICE_KLMS_l2m] = MH_NICE_KLMS_transferable(X,T,X_te,T_te,TD,N_tr,N_te,d_c,lr_l,centers,clusters_filter_weigth,s_loranz);
toc
disp('NICE KLMS is working...')
%macy
[s_macy,clusters_filter_weigth,centers,EW_NICE_KLMS,prediction_NICE_KLMS1,mse_NICE_KLMS1] = MH_NICE_KLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,lr_l);
toc

disp('NICE QKLMS is working...')
%macy
d_q = 0.1;
[clusters_filter_weigth_q_l2m,centers_q_l2m,EW_NICE_QKLMS_l2m,prediction_NICE_QKLMS_l2m,mse_NICE_QKLMS_l2m] = MH_NICE_QKLMS_transferable(X,T,X_te,T_te,TD,N_tr,N_te,d_c,d_q,lr_l,centers_q,clusters_filter_weigth_q,sq_loranz);
toc

disp('NICE QKLMS is working...')
[sq_macy,clusters_filter_weigth_q,centers_q,EW_NICE_QKLMS,prediction_NICE_QKLMS1,mse_NICE_QKLMS1] = MH_NICE_QKLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,d_q,lr_l);
toc



% plot(mse_NICE_KLMS_l2m,'DisplayName','mse NICE KLMS MG to lorenz');
% hold on;
% plot(mse_NICE_QKLMS_l2m,'DisplayName','mse NICE QKLMS MG to lorenz');
% plot(mse_NICE_KLMS1,'DisplayName','mse NICE KLMS1');
% plot(mse_NICE_QKLMS1,'DisplayName','mse NICE QKLMS1');
% hold off;
% xlabel('iteration')
% ylabel('MSE')


plot(T_te,'DisplayName','Original')
hold on;
plot(prediction_NICE_KLMS1,'DisplayName','prediction NICE KLMS');
plot(prediction_NICE_KLMS_l2m,'DisplayName','prediction NICE KLMS lorenz to MG');
plot(prediction_NICE_QKLMS1,'DisplayName','prediction NICE QKLMS');
plot(prediction_NICE_QKLMS_l2m,'DisplayName','prediction NICE QKLMS lorenz to MG');
hold off;