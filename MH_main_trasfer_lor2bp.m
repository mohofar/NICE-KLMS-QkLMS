%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Morteza Homayounfar                             %
% Implimentation of NICE-KLMS and the others      %
% Transfer learning                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 clc;clear;close all
%% data prepartion1 Macky Glass
%time delay (embedding) length
TD = 10;
%kernel parameter
a = 1;%fixed
%noise std
np =.01;
%data size
N_tr = 170;
N_te = 50;%
disp('Learning curves are generating. Please wait...');
load data\data_bp.mat
load data\data_last_hour.mat
% person = 26;
for person = [1:60]
disp('person:')
disp(person)

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
load Loranz_results\s_loranz.mat
load Loranz_results\sq_loranz.mat
load Loranz_results\clusters_filter_weigth.mat
load Loranz_results\clusters_filter_weigth_q.mat
load Loranz_results\centers.mat
load Loranz_results\centers_q.mat

lr_l = .08;
tic

disp('NICE KLMS is working...')
d_c = .02;
[s_bp,clusters_filter_weigth_bp,centers_bp,EW_NICE_KLMS_bp,prediction_NICE_KLMS_bp,mse_NICE_KLMS_bp] = MH_NICE_KLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,lr_l);
[clusters_filter_weigth_l2bp,centers_l2bp,EW_NICE_KLMS_l2bp,prediction_NICE_KLMS_l2bp,mse_NICE_KLMS_l2bp] = MH_NICE_KLMS_transferable(X,T,X_te,T_te,TD,N_tr,N_te,d_c,lr_l,centers,clusters_filter_weigth,s_loranz);
toc
disp('NICE QKLMS is working...')
%macy
d_q = 0.001;
[sq_bp,clusters_filter_weigth_q_bp,centers_q_bp,EW_NICE_QKLMS_bp,prediction_NICE_QKLMS_bp,mse_NICE_QKLMS_bp] = MH_NICE_QKLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,d_q,lr_l);
[clusters_filter_weigth_q_l2bp,centers_q_l2bp,EW_NICE_QKLMS,prediction_NICE_QKLMS_l2bp,mse_NICE_QKLMS_l2bp] = MH_NICE_QKLMS_transferable(X,T,X_te,T_te,TD,N_tr,N_te,d_c,d_q,lr_l,centers_q,clusters_filter_weigth_q,sq_loranz);
toc

figure,
subplot 211
plot(mse_NICE_KLMS_bp,'r-','DisplayName','mse NICE_KLMS bp');
hold on;
plot(mse_NICE_KLMS_l2bp,'r--','DisplayName','mse NICE KLMS lorenz to bp');
plot(mse_NICE_QKLMS_bp,'b-','DisplayName','mse NICE QKLMS bp');
plot(mse_NICE_QKLMS_l2bp,'b--','DisplayName','mse NICE QKLMS lorenz to bp');
hold off;
title(person)
xlabel('iteration')
ylabel('MSE')

subplot 212
plot(prediction_NICE_KLMS_bp,'r-','DisplayName','prediction NICE KLMS bp');
hold on;
plot(prediction_NICE_KLMS_l2bp,'r--','DisplayName','prediction NICE KLMS lorenz to bp');
plot(prediction_NICE_QKLMS_bp,'b-','DisplayName','prediction NICE QKLMS bp');
plot(prediction_NICE_QKLMS_l2bp,'b--','DisplayName','prediction NICE QKLMS lorenz to bp');
plot(T_te,'k-','DisplayName','Original');
hold off;
ylabel('prediction')
xlabel('samples')
end
