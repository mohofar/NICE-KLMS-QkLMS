function [s,C,c,e_k,y_te,mse_te_k] = MH_NICE_QKLMS(X,T,X_te,T_te,TD,N_tr,N_te,d_c,d_q,lr_k)

%learning rate (step size)
w1_N = zeros(1,TD);
e_l_N = zeros(N_tr,1);

%centroid distance threshold
% d_c = 1.5;                                

%initial weigth

%%%%%omega(1,1) = eta*y(1,1)*phi(u(1,1));    
%center
c = [];
c(1,1) = mean(X(:,1));
%effective size of cluster1 for cerntroid update
s = [];
s(1,1) = 1;                             

% lr_k = 0.004;

%centroid
C = {};
C{1,1} = X(:,1);
C{2,1} = lr_k*T(1)*(exp(-sum((X(:,1)).^2)))';


%init
e_k = zeros(N_tr,1);
y = zeros(N_tr,1);
mse_te_k = zeros(N_tr,1);

% n=1 init
e_k(1) = T(1);
alpha(1) = lr_k*e_k(1);
y(1) = 0;
mse_te_k(1) = mean(T_te.^2);
% start
for n=2:N_tr
   % compute minimum cetntroid distance 
   ds = sum((abs(X(:,n) - c)).^2);
   d_min = min(ds); 
   d_arg = find(ds == d_min);
   % compute output of d_arg cluster then error
   d_arg = d_arg(1,1);

   y(n) = alpha(1:n-1)*(exp(-sum((C{1,d_arg}*ones(size(C{1,d_arg},2),n-1)-X(:,1:n-1)).^2)))';
   e_k(n) = T(n) - y(n);
   alpha(n) = lr_k*e_k(n)';
    %------------------------------------------
    % N I C E 
   if(d_min < d_c)
     %.....................................................................
   ds_q = sum((abs(X(:,n) - C{1,d_arg})).^2);
   d_min_q = min(ds_q); 
   d_arg_q = find(ds == d_min_q);
    %------------------------------------------
    if(d_min_q < d_q )
       alpha(n) = alpha(n) + lr_k*e_k(n);
    elseif(d_min_q > d_q)
     C{1,d_arg} =[C{1,d_arg},X(:,n)];
     C{2,d_arg} = C{2,d_arg} + lr_k*e_k(n)*(exp(-sum((X(:,n)).^2)))';                                                    
     %update cluster d_arg                            
     c(1,d_arg) = (s(1,d_arg)*(c(1,d_arg)+T(n))/(s(1,d_arg)+1));
     %update effective size
     s(1,d_arg) = s(1,d_arg) + 1;
    end
     %.....................................................................
   elseif(d_min > d_c)
     % new cluster
     cc = length(c) + 1;
     C{1,cc} = X(:,n);
     C{2,cc} = C{2,d_arg} + lr_k*e_k(n)*(exp(-sum((X(:,n)).^2)))';
     c(1,cc) = mean(X(:,n));
     s(1,cc) = 1;
                                       
    end
    %------------------------------------------
  
    %testing MSE
    y_te = zeros(N_te,1);
    for jj = 1:N_te
        y_te(jj) = lr_k*e_k(1:n)'*(exp(-sum((X_te(:,jj)*ones(1,n)-X(:,1:n)).^2)))';
    end
    err = T_te - y_te;
    mse_te_k(n) = mean(err.^2);
%     

end
end