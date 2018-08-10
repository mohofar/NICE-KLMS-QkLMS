function [e_k1,y_te1,mse] = MH_KLMS(X,T,X_te,T_te,N_tr,N_te,lr_k)

%init
e_k1 = zeros(N_tr,1);
y1 = zeros(N_tr,1);
mse = zeros(N_tr,1);

% n=1 init
e_k1(1) = T(1);
y(1) = 0;
mse(1) = mean(T_te.^2);
% start
for n=2:N_tr
    %training
    ii = 1:n-1;
    y1(n) = lr_k*e_k1(ii)'*(exp(-sum((X(:,n)*ones(1,n-1)-X(:,ii)).^2)))';
    e_k1(n) = T(n) - y1(n);
    
    %testing MSE
    y_te1 = zeros(N_te,1);
    for jj = 1:N_te
        y_te1(jj) = lr_k*e_k1(1:n)'*(exp(-sum((X_te(:,jj)*ones(1,n)-X(:,1:n)).^2)))';
    end
    err1 = T_te - y_te1;
    mse(n) = mean(err1.^2);
    
end
end