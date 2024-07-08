%% main
load A2_data.mat

%% 7
X = train_data_01;
X = X - mean(X,2); %make data zero mean

[U,S,V] = svd(X); 

PC = U(:,[1,2]); %%principal components
projection = PC'*X;

%% 
label_0 = find(train_labels_01 == 0);
label_1 = find(train_labels_01 == 1);

projection_0 = projection(:, label_0);
projection_1 = projection(:, label_1);

%% plot
figure 
hold on
plot(projection_0(1,:), projection_0(2,:), 'bo')
plot(projection_1(1,:), projection_1(2,:), 'r*')
legend('0', '1')
title('PCA with \it{d} = 2')
xlabel('First principal component')
ylabel('Second principal component')
hold off



%% 8 
load A2_data.mat
load PC.mat
load projection.mat

K = 2;
[y, C] = K_means_clustering(train_data_01, K);
C_plot = PC'*C;

%plotting
symbol = ['o','+','*','.','x'];
color = ['b', 'g', 'r', 'c', 'm'];
legend_str = cell(1, K);

figure
hold on
for k = 1:K
    y_k = projection(:, y == k);
    plot(y_k(1,:), y_k(2,:), 'Color', color(k), 'Marker', symbol(k), 'LineStyle','none')
end

for k = 1:K
    plot(C_plot(1, k), C_plot(2,k), 'Color', 'k', 'Marker', 'p', 'MarkerSize', 30, 'MarkerFaceColor',color(k))
end
hold off
title(['K-means clustering on data projected on 2D with PCA with \it{K} = ', num2str(K)])
xlabel('First principal component')
ylabel('Second principal component')


%% 9

% K = 5;
% [y, C] = K_means_clustering(train_data_01, K);

figure
hold on
sgtitle(['Centroids from K-means clustering with  \it{K} = ', num2str(K)])
for k = 1:K
    subplot(1,K,k)

    centroid = reshape(C(:,k), 28, 28);
    imshow(centroid, 'InitialMagnification','fit')
    title([num2str(k)])
end

hold off

%% 10 / 11 labels 
load A2_data.mat

K = 8;
[y, C] = K_means_clustering(train_data_01, K);
labels_centroids = zeros(1,K);

for k = 1:K
    %since 0 or 1 we can sum up and divide to get average 
    labels_centroids(k) = round(sum(train_labels_01(find(y == k))) / length(train_labels_01)); 
end

%% training data

[~, N_train] = size(train_data_01);

[num_ones, num_zeros, misclassified, sum_misclassified, misclassification_rate] = ...
    K_means_binary_classifier(train_data_01, train_labels_01, labels_centroids, C, K);


%% test data

[~, N_test] = size(test_data_01);

[num_ones, num_zeros, misclassified, sum_misclassified, misclassification_rate] = ...
    K_means_binary_classifier(test_data_01, test_labels_01, labels_centroids, C, K);


%% 12 MODEL
load A2_data.mat
MODEL = fitcsvm(train_data_01', train_labels_01);

%% training
predicted_train = predict(MODEL,train_data_01');

[true_zeros, true_ones, false_zeros, false_ones, sum_misclassified, misclassification_rate] = ...
    binary_classification(predicted_train, train_labels_01);

%% test
predicted_test = predict(MODEL,test_data_01');

[true_zeros, true_ones, false_zeros, false_ones, sum_misclassified, misclassification_rate] = ...
    binary_classification(predicted_test, test_labels_01);


%% 13 Guassian 
load A2_data.mat

beta = 6;
MODEL = fitcsvm(train_data_01', train_labels_01, 'KernelFunction','gaussian', 'KernelScale',beta);

%% training
predicted_train = predict(MODEL,train_data_01');

[true_zeros, true_ones, false_zeros, false_ones, sum_misclassified, misclassification_rate] = ...
    binary_classification(predicted_train, train_labels_01);

%% test
predicted_test = predict(MODEL,test_data_01');

[true_zeros, true_ones, false_zeros, false_ones, sum_misclassified, misclassification_rate] = ...
    binary_classification(predicted_test, test_labels_01);


