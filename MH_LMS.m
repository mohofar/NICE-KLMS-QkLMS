function [w1,final_LMS_prediction,mse] = MH_LMS(X,T,X_te,T_te,N_tr,TD,lr_l)
mse = zeros(N_tr,1);

%lr_l = .004;%learning rate
w1 = zeros(1,TD);
e_l = zeros(N_tr,1);

for n=1:N_tr
    y = w1*X(:,n);
    e_l(n) = T(n) - y;
    w1 = w1 + lr_l*e_l(n)*X(:,n)';

    %testing MSE for learning curve
    err_te = T_te'-(w1*X_te);
    mse(n) = mean(err_te.^2);
end
final_LMS_prediction = (w1*X_te)';

end
