module SoftConfidenceWeighted

using Devectorize

import Distributions: Normal, cdf
import SVMLightLoader: SVMLightFile

typealias AA AbstractArray
export init, fit!, predict, SCW, SCW1, SCW2

abstract Model
abstract SCW1 <: Model
abstract SCW2 <: Model


type CDF
    ϕ
    ψ
    ζ

    function CDF(ETA)
        ϕ = cdf(Normal(0, 1), ETA)
        ψ = 1 + ϕ^2 / 2
        ζ = 1 + ϕ^2
        new(ϕ, ψ, ζ)
    end
end


#calc cdf in a constructor
type SCW{T<:Model}
    C::Float64
    cdf::CDF
    ndim::Int64
    weights::Array{Float64, 1}
    covariance::Array{Float64, 1}
    has_fitted::Bool

    function SCW(C, ETA)
        new(C, CDF(ETA), -1, [], [], false)
    end
end


function set_dimension!(scw::SCW, ndim::Int)
    assert(!scw.has_fitted)

    scw.ndim = ndim
    scw.weights = zeros(ndim)
    scw.covariance = ones(ndim)
    scw.has_fitted = true
    return scw
end


function calc_margin{T<:AA,R<:Real}(scw::SCW, x::T, label::R)
    #Devectorize.jl requires assignment
    @devec t = label .* dot(scw.weights, x)
end


function calc_confidence{T<:AA}(scw::SCW, x::T)
    @devec t = dot(x, scw.covariance .* x)
end

function calc_alpha{T<:AA,R<:Real}(scw::SCW{SCW1}, x::T, label::R)
    v = calc_confidence(scw, x)
    m = calc_margin(scw, x, label)
    cdf = scw.cdf
    (ϕ, ψ, ζ) = (cdf.ϕ, cdf.ψ, cdf.ζ)

    j = m^2 * ϕ^4 / 4
    k = v * ζ * ϕ^2
    t = (-m*ψ + sqrt(j+k)) / (v*ζ)
    return min(scw.C, max(0, t))
end


function calc_alpha{T<:AA,R<:Real}(scw::SCW{SCW2}, x::T, label::R)
    v = calc_confidence(scw, x)
    m = calc_margin(scw, x, label)
    cdf = scw.cdf
    ϕ, ψ, ζ = cdf.ϕ, cdf.ψ, cdf.ζ

    n = v + 1/2scw.C
    a = (ϕ*m*v)^2
    b = 4*n*v * (n + v*ϕ^2)
    gamma = ϕ * sqrt(a+b)

    c = -(2*m*n + m*v*ϕ^2) + gamma
    d = n^2 + n*v*ϕ^2
    t = c / 2d
    return max(0, t)
end


function init(C, ETA; algorithm = "SCW1")
    if algorithm == "SCW1"
        return SCW{SCW1}(C, ETA)
    elseif algorithm == "SCW2"
        return SCW{SCW2}(C, ETA)
    end

    throw(ArgumentError("Unexpected algorithm."))
end


function loss{T<:AA,R<:Real}(scw::SCW, x::T, label::R)
    t = calc_margin(scw, x, label)
    if t >= 1
        return 0
    end
    return 1-t
end


function calc_beta{T<:AA,R<:Real}(scw::SCW, x::T, label::R)
    alpha = calc_alpha(scw, x, label)
    v = calc_confidence(scw, x)
    m = calc_margin(scw, x, label)
    cdf = scw.cdf
    (ϕ, ψ, ζ) = (cdf.ϕ, cdf.ψ, cdf.ζ)

    j = -alpha * v * ϕ
    k = sqrt((alpha*v*ϕ)^2 + 4v)
    u = (j+k)^2 / 4
    return (alpha * ϕ) / (sqrt(u) + v*alpha*ϕ)
end


function update_covariance!{S<:AA,T<:AA,R<:Real}(t::S, scw::SCW, x::T, label::R)
    beta = calc_beta(scw, x, label)
    c = scw.covariance

    # same as
    # scw.covariance -= beta * (c .* x) .* (c .* x)
    @devec t[:] = (c .* x) .* (c .* x)
    BLAS.axpy!(-beta, t, scw.covariance)
    return scw
end


function update_weights!{S<:AA,T<:AA,R<:Real}(t::S, scw::SCW, x::T, label::R)
    alpha = calc_alpha(scw, x, label)

    # same as
    # scw.weights += alpha * label * (scw.covariance .* x)
    @devec t[:] = scw.covariance .* x
    BLAS.axpy!(alpha * label, t, scw.weights)
    return scw
end


function update!{S<:AA,T<:AA,R<:Real}(t::S, scw::SCW, x::T, label::R)
    if label != 1 && label != -1
        throw(ArgumentError("Data label must be 1 or -1."))
    end

    x = vec(full(x))
    if loss(scw, x, label) > 0
        update_weights!(t, scw, x, label)
        update_covariance!(t, scw, x, label)
    end
    return scw
end


function train!{T<:AA,R<:AA}(scw::SCW, X::T, labels::R)
    # preallocate for performance optimization
    t = Array(Float64, size(X, 1))
    for i in 1:size(X, 2)
        update!(t, scw, slice(X, :, i), labels[i])
    end
    return scw
end


function fit!{T<:AA,R<:AA}(scw::SCW, X::T, labels::R)
    if ndims(X) > 2
        throw(ArgumentError("Estimator expects 2 dim array."))
    end

    if ndims(labels) > 2
        throw(ArgumentError("Bad input size $(size(labels))"))
    end

    if !scw.has_fitted
        ndim = size(X, 1)
        set_dimension!(scw, ndim)
    end

    train!(scw, X, labels)
    return scw
end


function fit!(scw::SCW, filename::AbstractString, ndim::Int64)
    if !scw.has_fitted
        set_dimension!(scw, ndim)
    end

    t = Array(Float64, ndim)
    for (x, label) in SVMLightFile(filename, ndim)
        update!(t, scw, x, label)
    end
    return scw
end


@deprecate fit fit!


function compute{T<:AA}(scw::SCW, x::T)
    x = vec(full(x))
    if dot(x, scw.weights) > 0
        return 1
    else
        return -1
    end
end


function throw_error_if_not_fitted(scw)
    if !scw.has_fitted
        error("The model is not fitted yet")
    end
end


function predict{T<:AA}(scw::SCW, X::T)
    throw_error_if_not_fitted(scw)
    return [compute(scw, slice(X, :, i)) for i in 1:size(X, 2)]
end


function predict(scw::SCW, filename::AbstractString)
    throw_error_if_not_fitted(scw)

    labels = Int64[]
    for (x, _) in SVMLightFile(filename, scw.ndim)
        push!(labels, compute(scw, x))
    end
    return labels
end

end # module
