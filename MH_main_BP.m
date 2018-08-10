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
N_tr = 170;
N_te = 50;%
disp('Learning curves are generating. Please wait...');
load data_bp.mat
load data_last_hour.mat
person = 45;
MK30 = [s1(person,:),s2(person,:)];
MK30 = MK30';
MK30 = MK30+np*randn(size(MK30));
MK30 = (MK30 - min(MK30))/(max(MK30)-min(MK30));

%3000 training data
train_set = MK30(1:180);
%300 testing data
test_set = MK30(181:end);

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
lr_l = 0.2;
tic
disp('LMS is working...')
[W_LMS,prediction_LMS,mse_LMS]=MH_LMS(X,T,X_te,T_te,N_tr,TD,lr_l);
toc
disp('KLMS is working...')
[EW_KLMS,prediction_KLMS,mse_KLMS] = MH_KLMS(X,T,X_te,T_te,N_tr,N_te,lr_l);
toc
disp('NICE KLMS is working...')
%macy
d_c = .09;
[s_bp45,clusters_filter_weigth_bp45,centers_bp,EW_NICE_KLMS_bp45,prediction_NICE_KLMS_bp45,mse_NICE_KLMS_bp45] = MH_NICE_KLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,lr_l);
toc
disp('NICE QKLMS is working...')
%macy
d_q = 0.01;
[sq_bp45,clusters_filter_weigth_q_bp45,centers_q_bp45,EW_NICE_QKLMS,prediction_NICE_QKLMS_bp45,mse_NICE_QKLMS_bp45] = MH_NICE_QKLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,d_q,lr_l);
toc
% disp('NICE M-QKLMS is working...')
% [clusters_filter_weigth_mq,centers_mq,EW_NICE_MQKLMS,prediction_NICE_MQKLMS,mse_NICE_MQKLMS] = MH_NICE_QKLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,d_q,lr_l);
% toc

%% plots
figure,
plot(T_te,'k-','LineWidth',2)
hold on
plot(prediction_LMS,'g--')
hold on
plot(prediction_KLMS,'b--')
hold on
plot(prediction_NICE_KLMS_bp45,'r--')
hold on
plot(prediction_NICE_QKLMS_bp45,'y--')
% hold on
% plot(prediction_NICE_MQKLMS,'k--')

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

legend('Original','LMS','KLMS','NICE KLMS','NICE QKLMS')
xlabel('iteration')
ylabel('Amplitude')

figure,
plot(mse_LMS,'g-','LineWidth',2);
hold on
plot(mse_KLMS,'b-','LineWidth',2);
hold on
plot(mse_NICE_KLMS_bp45,'r-','LineWidth',2);
hold on
plot(mse_NICE_QKLMS_bp45,'y-','LineWidth',2);
% hold on
% plot(mse_NICE_MQKLMS,'k-','LineWidth',2);

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

legend( 'LMS','KLMS','NICE KLMS','NICE QKLMS')
xlabel('iteration')
ylabel('MSE')
