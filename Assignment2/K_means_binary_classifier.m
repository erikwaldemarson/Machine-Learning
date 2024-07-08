function [num_ones, num_zeros, misclassified, sum_misclassified, misclassification_rate] = K_means_binary_classifier(data, true_labels, labels_centroids, C, K)
    
    [~,N] = size(data); %number of examples
    y = zeros(N,1); %assign each example to a centroid
    pred_labels = zeros(N,1); %assign each example to a class

    for i = 1:N
        x = data(:,i);
        y(i) = K_means_classifier(x, C); %assign example to centroid
        pred_labels(i) = labels_centroids(y(i)); %assign example to label
    end

    misclassified = zeros(1,K); %number of missclassified
    num_ones = zeros(1, K); %number of ones for each cluster
    num_zeros = zeros(1, K); %number of zeros for each cluster
     
    %i use find(y==k) to compare each cluster
    for k = 1:K
        num_ones(k) = sum(true_labels(find(y == k)));
        num_zeros(k) = length(y(y == k)) - num_ones(k);

        misclassified(k) = sum(pred_labels(find(y == k)) ~= true_labels(find(y == k)));
    end
    
    sum_misclassified = sum(misclassified); %total number of misclassified

    misclassification_rate = sum_misclassified / N;

    
end