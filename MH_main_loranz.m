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
%% data prepartion2 Lorenz



%% algorithms
%macy
lr_l = 0.0005;
tic
% disp('LMS is working...')
% [W_LMS,prediction_LMS,mse_LMS]=MH_LMS(X,T,X_te,T_te,N_tr,TD,lr_l);
% toc
disp('KLMS is working...')
[EW_KLMS,prediction_KLMS,mse_KLMS] = MH_KLMS(X,T,X_te,T_te,N_tr,N_te,lr_l);
toc

disp('NICE KLMS is working...')
%macy
for d_c = [0.0005]
[s_loranz,clusters_filter_weigth,centers,EW_NICE_KLMS,prediction_NICE_KLMS,mse_NICE_KLMS] = MH_NICE_KLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,lr_l);
end
toc
disp('NICE QKLMS is working...')
%macy
 d_q = 0.0001;
[sq_loranz,clusters_filter_weigth_q,centers_q,EW_NICE_QKLMS,prediction_NICE_QKLMS,mse_NICE_QKLMS] = MH_NICE_QKLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,d_q,lr_l);
 toc
% disp('NICE M-QKLMS is working...')
% [clusters_filter_weigth_mq,centers_mq,EW_NICE_MQKLMS,prediction_NICE_MQKLMS,mse_NICE_MQKLMS] = MH_NICE_QKLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,d_q,lr_l);
% toc

%% plots
figure,
plot(T_te,'k-','LineWidth',2)
hold on
% plot(prediction_LMS,'g--')
% hold on
plot(prediction_KLMS,'b--')
hold on
plot(prediction_NICE_KLMS,'r--')
hold on
plot(prediction_NICE_QKLMS,'y--')
% hold on
% plot(prediction_NICE_MQKLMS,'k--')

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

legend('Original','KLMS','NICE KLMS','NICE QKLMS')
xlabel('iteration')
ylabel('Amplitude')

figure,
% plot(mse_LMS,'g-','LineWidth',2);
% hold on
plot(mse_KLMS,'b-','LineWidth',2);
hold on
plot(mse_NICE_KLMS,'r-','LineWidth',2);
hold on
plot(mse_NICE_QKLMS,'y-','LineWidth',2);
% hold on
% plot(mse_NICE_MQKLMS,'k-','LineWidth',2);

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

legend('KLMS','NICE KLMS','NICE QKLMS')
xlabel('iteration')
ylabel('MSE')
