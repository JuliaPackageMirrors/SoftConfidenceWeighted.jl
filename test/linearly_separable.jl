import Base: size, convert

using SoftConfidenceWeighted: SCW, init, fit!, predict, SCW1, SCW2
using SVMLightLoader: SVMLightFile


function split_dataset(X, y, training_ratio=0.8)
    assert(0.0 <= training_ratio <= 1.0)

    N = convert(Int64, size(X, 2)*training_ratio)
    training = X[:, 1:N-1], y[1:N-1]
    test = X[:, N:end], y[N:end]
    return training, test
end


function calc_accuracy(y_pred, y_true)
    n_correct = 0
    for (p, t) in zip(y_pred, y_true)
        if p == t
            n_correct += 1
        end
    end

    return n_correct / length(y_pred)
end


function test_batch(scw, X, y; training_ratio=0.8)
    println(size(X))
    training, test = split_dataset(X, y, training_ratio)

    X, labels = training
    scw = fit!(scw, X, labels)

    X, y_true = test
    y_pred = predict(scw, X)

    accuracy = calc_accuracy(y_pred, y_true)
    assert(accuracy == 1.0)

    println("BATCH")
    println("\tModel: $(typeof(scw))")
    println("\taccuracy: $accuracy")
    println("")
end


function test_online(scw, X, y; training_ratio=0.8)
    training, test = split_dataset(X, y, training_ratio)

    X, labels = training
    for i in 1:size(X, 2)
        scw = fit!(scw, slice(X, :, i), [labels[i]])
    end

    X, y_true = test

    y_pred = Int64[]
    for i in 1:size(X, 2)
        x = slice(X, :, i)
        label = predict(scw, x)
        append!(y_pred, label)
    end

    accuracy = calc_accuracy(y_pred, y_true)
    assert(accuracy == 1.0)

    println("ONLINE")
    println("\tModel: $(typeof(scw))")
    println("\taccuracy: $accuracy")
    println("")
end


function test_svmlight(scw, training_file, test_file, ndim ;
                       training_ratio=0.8)
    scw = fit!(scw, training_file, ndim)

    y_pred = predict(scw, test_file)
    y_true = [label for (_, label) in SVMLightFile(test_file)]
    accuracy = calc_accuracy(y_pred, y_true)
    assert(accuracy == 1.0)
end


X = readdlm("data/julia_array/digitsX.txt")
y = readdlm("data/julia_array/digitsy.txt")

println("TEST DIGITS\n")

# Dense matrix
test_batch(SCW{SCW1}(1.0, 1.0), X, y, training_ratio=0.8)
test_batch(SCW{SCW2}(1.0, 1.0), X, y, training_ratio=0.8)

test_online(SCW{SCW1}(1.0, 1.0), X, y, training_ratio=0.8)
test_online(SCW{SCW2}(1.0, 1.0), X, y, training_ratio=0.8)

X = sparse(X)
y = sparse(y)

# Sparse matrix
test_batch(SCW{SCW1}(1.0, 1.0), X, y, training_ratio=0.8)
test_batch(SCW{SCW2}(1.0, 1.0), X, y, training_ratio=0.8)

test_online(SCW{SCW1}(1.0, 1.0), X, y, training_ratio=0.8)
test_online(SCW{SCW2}(1.0, 1.0), X, y, training_ratio=0.8)

training_file = "data/svmlight/digits.train.txt"
test_file = "data/svmlight/digits.test.txt"
ndim = 64
test_svmlight(SCW{SCW1}(1.0, 1.0), training_file, test_file, ndim)
test_svmlight(SCW{SCW2}(1.0, 1.0), training_file, test_file, ndim)
