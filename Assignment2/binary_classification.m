function [true_zeros, true_ones, false_zeros, false_ones, sum_misclassified, misclassification_rate] = binary_classification(predicted_label,true_label)
    
    N = length(predicted_label); %assuming true_label has same length

    
    %basically like FP, TP, FF, TF classification
    true_zeros = 0;
    false_zeros = 0;
    true_ones = 0;
    false_ones = 0;
    
    for i = 1:N
        if predicted_label(i) == 0
            if true_label(i) == 0
                true_zeros = true_zeros + 1;
            else
                false_zeros = false_zeros + 1;
            end
        elseif predicted_label(i) == 1
            if true_label(i) == 1
                true_ones = true_ones + 1;
            else
                false_ones = false_ones + 1;
            end
        end
    
    end

    sum_misclassified = false_ones + false_zeros;
    misclassification_rate = sum_misclassified / N;

    

end