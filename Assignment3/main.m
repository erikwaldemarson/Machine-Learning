%% mnist
    addpath(genpath('./'));

    x_train = loadMNISTImages('data/mnist/train-images.idx3-ubyte');
    y_train = loadMNISTLabels('data/mnist/train-labels.idx1-ubyte');
    perm = randperm(numel(y_train));
    x_train = x_train(:,perm);
    y_train = y_train(perm);

    x_test = loadMNISTImages('data/mnist/t10k-images.idx3-ubyte');
    y_test = loadMNISTLabels('data/mnist/t10k-labels.idx1-ubyte');
    y_test(y_test==0) = 10;
    x_test = reshape(x_test, [28, 28, 1, 10000]);
    classes = [1:9 0]; 
    
    data_mean = mean(x_train(:));
    x_test = bsxfun(@minus, x_test, data_mean);

    training_opts = struct('learning_rate', 1e-3,...
        'iterations', 1000,...
        'batch_size', 64,...
        'momentum', 0.99,...
        'weight_decay', 0.0001);

   load('C:\Users\Erik\OneDrive\Documents\Git-repos\Machine Learning\Assignment3\code&data\models\network_trained_with_momentum.mat')
   

%% cifar10
    addpath(genpath('./'));

    % argument=2 is how many 10000 images that are loaded. 20000 in this
    % example. Load as much as your RAM can handle.
    [x_train, y_train, x_test, y_test, classes] = load_cifar10(2);
    data_mean = mean(mean(mean(x_train, 1), 2), 4); % mean RGB triplet
    x_test = bsxfun(@minus, x_test, data_mean);

    load('C:\Users\Erik\OneDrive\Documents\Git-repos\Machine Learning\Assignment3\code&data\models\cifar10_4.mat')
    % evaluate on the test set

    training_opts = struct('learning_rate', 1e-4,...
    'iterations', 1000,...
    'batch_size', 64,...
    'momentum', 0.99,...
    'weight_decay', 0.0001);


%% test accuracy
    pred = zeros(numel(y_test),1);
    batch = training_opts.batch_size;
    for i=1:batch:size(y_test)
        idx = i:min(i+batch-1, numel(y_test));
        % note that y_test is only used for the loss and not the prediction
        y = evaluate(net, x_test(:,:,:,idx), y_test(idx));
        [~, p] = max(y{end-1}, [], 1);
        pred(idx) = p;
    end
    
    fprintf('Accuracy on the test set: %f\n', mean(vec(pred) == vec(y_test)));

 %% confusion matrix & precision and recall
 predicted = double(vec(pred));
 actual = double(vec(y_test));
 C = confusionmat(actual, predicted, 'Order', 1:10);
 [sz, ~] = size(C);

 recall = zeros(1,10);
 precision = zeros(1,10);

 for i = 1:10
    TP = C(i,i);
    FN = sum(C(i,:)) - TP;
    FP = sum(C(:,i)) - TP;
    TN = sum(C, 'all') - TP - FN - FP;
    
    recall(i) = TP / (TP + FP);
    precision(i) = TP / (TP + FN);
 end

 recall
 precision
%% plot kernels mnist

    kernels = net.layers{2}.params.weights;
    [~,~,nrKernels] = size(kernels);

    figure
    for i = 1:nrKernels
        subplot(4, 4, i);
        imagesc(kernels(:,:,1));
        title(num2str(i))
        colorbar
    end
    sgtitle('Kernels MNIST', 'FontSize', 16)

    %% plot kernels cifar10

    kernels = net.layers{2}.params.weights;
    [~,~,nrKernels] = size(kernels);

    figure
    for i = 1:nrKernels
        subplot(6, 8, i);
        imagesc(kernels(:,:,1));
        title(num2str(i))
        colorbar
    end
    sgtitle('Kernels CIFAR-10', 'FontSize', 16)

    %% PLOT MISCLASSED IMAGES 
    % Store the predicted/actual labels for misclassed examples in truth
    % table, and the corresponding data in x_miss
    predicted = vec(pred);
    actual = vec(y_test);
    idx_wrong = predicted ~= actual;
    truth_table = [predicted(idx_wrong) actual(idx_wrong)];

    x_miss = x_test(:,:,:,idx_wrong);

    % NR of misclassified examples
    nrMisclassed = size(truth_table,1);

    % Random examples we want
    nrExamples = 8;
    randomExamples = randperm(nrMisclassed, nrExamples);

    % Select 8 matrices and 8x2 labels
    selectedMatrices = x_miss(:, :, :, randomExamples);
    selectedLabels = truth_table(randomExamples, :);


    figure
    tiledlayout(2, 4)
    if true
        for i=1:nrExamples
            nexttile
            imagesc(selectedMatrices(:, :, :, i) / 255); %cifar10
            %imagesc(selectedMatrices(:, :, :, i));  %mnist
            colormap(gray);

            %cifar10
            formattedTitle = sprintf('Predicted: %s\nActual: %s', classes{selectedLabels(i, 1)}, classes{selectedLabels(i, 2)});

            %mnist
            %formattedTitle = sprintf('Predicted: %s\nActual: %s', num2str(classes(selectedLabels(i, 1))), num2str(classes(selectedLabels(i, 2))));
            
            title(formattedTitle, 'FontSize',10);
        end
    end

    sgtitle('Misclassified images CIFAR-10')