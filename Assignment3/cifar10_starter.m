function cifar10_starter()
    addpath(genpath('./'));

    % argument=2 is how many 10000 images that are loaded. 20000 in this
    % example. Load as much as your RAM can handle.
    [x_train, y_train, x_test, y_test, classes] = load_cifar10(2);
    
    % visualize the images?
    if false
        for i=1:6
            for j=1:6
                subplot(6,6,6*(i-1)+j);
                imagesc(x_train(:,:,:,6*(i-1)+j)/255);
                colormap(gray);
                title(classes(y_train(6*(i-1)+j)));
                axis off;
            end
        end
        return;
    end
    
    % Always subtract the mean. Optimization will work much better if you do.
    data_mean = mean(mean(mean(x_train, 1), 2), 4); % mean RGB triplet
    x_train = bsxfun(@minus, x_train, data_mean);
    x_test = bsxfun(@minus, x_test, data_mean);
    % and shuffle the examples. Some datasets are stored so that all elements of class 1 are consecutive
    % training will not work on those datasets if you don't shuffle
    perm = randperm(numel(y_train));
    x_train = x_train(:,:,:,perm);
    y_train = y_train(perm);

    % we use 2000 validation images
    x_val = x_train(:,:,:,end-2000:end);
    y_val = y_train(end-2000:end);
    x_train = x_train(:,:,:,1:end-2001);
    y_train = y_train(1:end-2001);
    
    % net.layers = {};
    % net.layers{end+1} = struct('type', 'input', ...
    %     'params', struct('size', [32, 32, 3]));
    % 
    % net.layers{end+1} = struct('type', 'convolution',...
    %     'params', struct('weights', 0.1*randn(5,5,3,16)/sqrt(5*5*3/2), 'biases', zeros(16,1)),...
    %     'padding', [2 2]);
    % 
    % net.layers{end+1} = struct('type', 'relu');
    % 
    % net.layers{end+1} = struct('type', 'maxpooling');
    % 
    % net.layers{end+1} = struct('type', 'convolution',...
    %     'params', struct('weights', 0.1*randn(5,5,16,16)/sqrt(5*5*16/2), 'biases', zeros(16,1)),...
    %     'padding', [2 2]);
    % 
    % net.layers{end+1} = struct('type', 'relu');
    % 
    % net.layers{end+1} = struct('type', 'fully_connected',...
    %     'params', struct('weights', randn(10,4096)/sqrt(4096/2), 'biases', zeros(10,1)));
    % 
    % net.layers{end+1} = struct('type', 'softmaxloss');

    load('C:\Users\Erik\OneDrive\Documents\Git-repos\Machine Learning\Assignment3\code&data\models\cifar10_3.mat')

    % see the layer sizes
    [a, b] = evaluate(net, x_train(:,:,:,1:8), y_train(1:8), true);
    
    training_opts = struct('learning_rate', 1e-3,...
        'iterations', 1000,...
        'batch_size', 64,...
        'momentum', 0.99,...
        'weight_decay', 0.0001);
    
    net = training(net, x_train, y_train, x_val, y_val, training_opts);

    % since the training takes a lot of time, consider refining rather than
    % retraining the net. Add layers to a net where the parameters already
    % are good at the other layers.
    save('models/cifar10_4.mat', 'net');
    
    % evaluate on the test set
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
end