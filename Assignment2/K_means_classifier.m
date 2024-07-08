function label = K_means_classifier(x, C)
    d = vecnorm(C-x);
    [~, label] = min(d);
end
