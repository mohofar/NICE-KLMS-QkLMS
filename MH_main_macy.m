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
N_tr = 5000;
N_te = 1;%
disp('Learning curves are generating. Please wait...');

load MKG   %MK30 5000*1
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
lr_l = 0.0005;
tic
disp('LMS is working...')
[W_LMS,prediction_LMS,mse_LMS]=MH_LMS(X,T,X_te,T_te,N_tr,TD,lr_l);
toc
disp('KLMS is working...')
[EW_KLMS,prediction_KLMS,mse_KLMS] = MH_KLMS(X,T,X_te,T_te,N_tr,N_te,lr_l);
toc
disp('NICE KLMS is working...')
%macy
% d_c = 0.2

for d_c = [0.4]
% disp(d_c)
[s_macy,clusters_filter_weigth,centers,EW_NICE_KLMS,prediction_NICE_KLMS,mse_NICE_KLMS] = MH_NICE_KLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,lr_l);
% disp(length(centers))
% plot(mse_NICE_KLMS)
% hold on
end
% legend('0.2','0.4')
% 
% toc
disp('NICE QKLMS is working...')
% %macy
for d_q = [0.001]
[sq_macy,clusters_filter_weigth_q,centers_q,EW_NICE_QKLMS,prediction_NICE_QKLMS,mse_NICE_QKLMS] = MH_NICE_QKLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,d_q,lr_l);
% disp(length(centers_q))
% disp(d_q)
% plot(mse_NICE_QKLMS)
end
% legend('KLMS','0.0001','0.001','0.01','0.02','0.03','0.04','0.05','0.1')
toc
% % disp('NICE M-QKLMS is working...')
% % [clusters_filter_weigth_mq,centers_mq,EW_NICE_MQKLMS,prediction_NICE_MQKLMS,mse_NICE_MQKLMS] = MH_NICE_QKLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,d_q,lr_l);
% % toc
% 
% plots
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

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

legend('Original','KLMS','NICE KLMS','NICE QKLMS')
xlabel('iteration')
ylabel('Amplitude')

figure,
plot(mse_LMS,'g-','LineWidth',2);
hold on
plot(mse_KLMS,'b-','LineWidth',2);
hold on
plot(mse_NICE_KLMS,'r-','LineWidth',2);
hold on
plot(mse_NICE_QKLMS,'y-','LineWidth',2);

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

legend('LMS','KLMS','NICE KLMS','NICE QKLMS')
xlabel('iteration')
ylabel('MSE')
