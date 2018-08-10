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
np =.02;
%data size
N_tr = 5000;
N_te = 1000;%
disp('Learning curves are generating. Please wait...');

load data\lorenz_time_series.mat   
MK30 = lorenz_time_series;

MK30 = MK30+np*randn(size(MK30));
MK30 = (MK30 - min(MK30))/(max(MK30)-min(MK30));

%3000 training data
train_set = MK30(501:6700);
% train_set = MK30(1:180);
%300 testing data
test_set = MK30(7001:8300);
% test_set = MK30(181:end);

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
lr_l = 0.0005;
tic
%macy
disp('NICE KLMS is working...')
%macy
% for d_c = [0.0005]
for d_c = [10]
[s_loranz,clusters_filter_weigth,centers,EW_NICE_KLMS,prediction_NICE_KLMS,mse_NICE_KLMS1] = MH_NICE_KLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,lr_l);
end
toc
disp('NICE QKLMS is working...')
%macy
%  d_q = 0.0001;
d_q = 10;
[sq_loranz,clusters_filter_weigth_q,centers_q,EW_NICE_QKLMS,prediction_NICE_QKLMS,mse_NICE_QKLMS1] = MH_NICE_QKLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,d_q,lr_l);
 toc

load Macy_Glass_results\s_macy.mat
load Macy_Glass_results\sq_macy.mat
load Macy_Glass_results\mse_NICE_KLMS.mat
load Macy_Glass_results\mse_NICE_QKLMS.mat
load Macy_Glass_results\clusters_filter_weigth.mat
load Macy_Glass_results\clusters_filter_weigth_q.mat
load Macy_Glass_results\centers.mat
load Macy_Glass_results\centers_q.mat

disp('NICE KLMS m2l is working...')
[clusters_filter_weigth_m2l,centers_m2l,EW_NICE_KLMS_m2l,prediction_NICE_KLMS_m2l,mse_NICE_KLMS_m2l] = MH_NICE_KLMS_transferable(X,T,X_te,T_te,TD,N_tr,N_te,d_c,lr_l,centers,clusters_filter_weigth,s_macy);
toc

disp('NICE QKLMS m2l is working...')
[clusters_filter_weigth_q_m2l,centers_q_m2l,EW_NICE_QKLMS_m2l,prediction_NICE_QKLMS_m2l,mse_NICE_QKLMS_m2l] = MH_NICE_QKLMS_transferable(X,T,X_te,T_te,TD,N_tr,N_te,d_c,d_q,lr_l,centers_q,clusters_filter_weigth_q,sq_macy);
toc

% plot(mse_NICE_KLMS_m2l,'DisplayName','mse NICE KLMS MG to lorenz');
% hold on;
% plot(mse_NICE_QKLMS_m2l,'DisplayName','mse NICE QKLMS MG to lorenz');
% plot(mse_NICE_KLMS1,'DisplayName','mse NICE KLMS1');
% plot(mse_NICE_QKLMS1,'DisplayName','mse NICE QKLMS1');
% hold off;
% xlabel('iteration')
% ylabel('MSE')

figure,
plot(prediction_NICE_KLMS,'DisplayName','prediction NICE KLMS');
hold on;
plot(prediction_NICE_KLMS_m2l,'DisplayName','prediction NICE KLMS MG to lorenz');
plot(prediction_NICE_QKLMS,'DisplayName','prediction NICE QKLMS');
plot(prediction_NICE_QKLMS_m2l,'DisplayName','prediction NICE QKLMS MG to lorenz');
plot(T_te,'DisplayName','Original');
hold off;
xlabel('Samples')
ylabel('Amplitude')
