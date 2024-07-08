%main ML assignment1

load A1_data.mat

%% 4: lasso 
close all

for lambda = [0.1, 1.9639, 10]

    w = lasso_ccd(t,X,lambda);
    y = X*w;
    yinterp = Xinterp*w;
    
    figure
    hold on
    plot(n, t, 'o', 'DisplayName', 'Noisy targets')
    plot(n, y, '*', 'DisplayName', 'Model Predictions')
    plot(ninterp, yinterp, 'DisplayName', 'Interpolated Data')
    hold off
    legend('Location', 'best') % Place the legend at the best possible location
    title(['Plot for \lambda = ', num2str(lambda)])
    xlabel('time indices n')
    
    lambda
    num_non_zero = nnz(w)
end

%% 4.3
close all

lambda = 0.8;
w = lasso_ccd(t,X,lambda);
y = X*w;
yinterp = Xinterp*w;

figure
hold on
plot(n, t, 'o', 'DisplayName', 'Noisy targets')
plot(ninterp, yinterp, 'DisplayName', 'Interpolated Data')
legend('Location', 'best') % Place the legend at the best possible location

num_non_zero = nnz(w)



%% 5: k-lasso

%lambda_vec 
lambda_max = max(abs(X'*t));
lambda_min = 10^(-5);
N_lambda = 100;
lambdavec = exp(linspace(log(lambda_min),log(lambda_max), N_lambda));

%k-cross lasso
K = 5;
[wopt,lambdaopt,RMSEval,RMSEest] = lasso_cv(t,X,lambdavec,K);

%% plot error
figure 
hold on
plot(lambdavec, RMSEval, '-o', 'DisplayName', 'RMSE_{val}')
plot(lambdavec, RMSEest, '-*', 'DisplayName', 'RMSE_{est}')
xline(lambdaopt, 'DisplayName', ['\lambda_{opt} = ', num2str(lambdaopt)])
hold off
legend('Location', 'best') % Place the legend at the best possible location
xlabel('\lambda')
title(['Estimation and validation error against \lambda for K = ', num2str(K)])    

%% plot interpolation
y = X*wopt;
yinterp = Xinterp*wopt;

figure
hold on
plot(n, t, 'o', 'DisplayName', 'Noisy targets')
plot(n, y, '*', 'DisplayName', 'Model Predictions')
plot(ninterp, yinterp, 'DisplayName', 'Interpolated Data')
hold off
legend('Location', 'best') % Place the legend at the best possible location
title(['Plot for \lambda = ', num2str(lambdaopt)])
xlabel('time indices n')



%% 6: multiframe lasso 

%lambda_vec 
lambda_max = max(abs(X'*t));
lambda_min = 10^(-5);
N_lambda = 100;
lambdavec = exp(linspace(log(lambda_min),log(lambda_max), N_lambda));

%k-cross lasso
K = 5;
[Wopt,lambdaopt,RMSEval,RMSEest] = multiframe_lasso_cv(Ttrain,Xaudio,lambdavec,K);

%% plot error
figure 
hold on
plot(lambdavec, RMSEval, '-o', 'DisplayName', 'RMSE_{val}')
plot(lambdavec, RMSEest, '-*', 'DisplayName', 'RMSE_{est}')
xline(lambdaopt, 'DisplayName', ['\lambda_{opt} = ', num2str(lambdaopt)])
hold off
legend('Location', 'best') % Place the legend at the best possible location
xlabel('\lambda')
title(['Estimation and validation error against \lambda for K = ', num2str(K)])   

%% load
load A1_data.mat

%% 7: denoise audio
lambdaopt = 0.0044;
lambda = 0.02; %better

[Ytest] = lasso_denoise(Ttest,Xaudio,lambdaopt);
save('denoised_audio', 'Ytest', 'fs')

%% play
load denoised_audio.mat

%soundsc(Ttrain,fs) %noisy
soundsc(Ytest,fs) % clean



