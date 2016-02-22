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


function test_batch(X, y, algorithm; training_ratio=0.8, C=1.0, ETA=1.0)
    model = init(C, ETA)

    training, test = split_dataset(X, y, training_ratio)

    X, labels = training
    model = fit!(model, X, labels)

    X, y_true = test
    y_pred = predict(model, X)

    accuracy = calc_accuracy(y_pred, y_true)
    assert(accuracy == 1.0)

    println("BATCH")
    println("\tModel: $algorithm")
    println("\taccuracy: $accuracy")
    println("")
end


function test_online(X, y, algorithm; training_ratio=0.8, C=1.0, ETA=1.0)
    model = init(C, ETA)

    training, test = split_dataset(X, y, training_ratio)

    X, labels = training
    for i in 1:size(X, 2)
        model = fit!(model, slice(X, :, i), [labels[i]])
    end

    X, y_true = test

    y_pred = Int64[]
    for i in 1:size(X, 2)
        x = slice(X, :, i)
        label = predict(model, x)
        append!(y_pred, label)
    end

    accuracy = calc_accuracy(y_pred, y_true)
    assert(accuracy == 1.0)

    println("ONLINE")
    println("\talgorithm: $algorithm")
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
test_batch(X, y, "SCW1")
test_batch(X, y, "SCW2")

test_online(X, y, "SCW1")
test_online(X, y, "SCW2")

X = sparse(X)
y = sparse(y)

# Sparse matrix
test_batch(X, y, SCW1, training_ratio=0.8)
test_batch(X, y, SCW2, training_ratio=0.8)

test_online(X, y, SCW1, training_ratio=0.8)
test_online(X, y, SCW2, training_ratio=0.8)

training_file = "data/svmlight/digits.train.txt"
test_file = "data/svmlight/digits.test.txt"
ndim = 64
test_svmlight(SCW{SCW1}(1.0, 1.0), training_file, test_file, ndim)
test_svmlight(SCW{SCW2}(1.0, 1.0), training_file, test_file, ndim)
