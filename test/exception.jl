using Base.Test

using SoftConfidenceWeighted: init, predict

# predict with a unfitted model
@test_throws ErrorException predict(SCW{SCW1}(1.0, 1.0), X)
@test_throws ArgumentError init(1.0, 1.0, algorithm = "SCW3")
