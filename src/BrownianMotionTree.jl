module BrownianMotionTree

export MLE, rand_pos_def, n_vec_to_sym, vec_to_sym, sym_to_vec, n_sym_to_vec, solve,
    star_tree_boundary

import DynamicPolynomials: @polyvar, Polynomial, differentiate, subs
import HomotopyContinuation
const HC = HomotopyContinuation
using LinearAlgebra


function build_system(tree::Matrix{<:Integer})
    n, m = size(tree)
    N = binomial(n+1,2)
    @polyvar θ[1:m] k[1:N] λ[1:N] s[1:N]

    K = Matrix{eltype(k)}(undef, n, n)
    S = Matrix{eltype(s)}(undef, n, n)
    l = 1
    for i in 1:n, j in i:n
        K[i,j] = K[j,i] = k[l]
        S[i,j] = S[j,i] = s[l]
        l += 1
    end

    Σ = (θ[1]+0) .* (view(tree, 1:n, 1) * view(tree, 1:n, 1)')
    for i in 2:m
        Σ .+= θ[i] .* (view(tree, 1:n, i) * view(tree, 1:n, i)')
    end

    KΣ = K * Σ
    KΣ_I = [KΣ[i,j] - (i == j) for i in 1:n for j in i:n]

    ∇_logdet_K = [
        [i == j ? 1Σ[i,i] : 2Σ[i,j] for i in 1:n for j in i:n];
        zeros(Int, m)
        ]
    ∇_trace_SK = [
        [i == j ? 1S[i,i] : 2S[i,j] for i in 1:n for j in i:n];
        zeros(Int, m)
        ]

    # create lagrange multipliers
    L_λ = sum(1:length(KΣ_I)) do i
        λ[i] .* differentiate(KΣ_I[i], [k; θ])
    end

    F = [∇_logdet_K - ∇_trace_SK - L_λ; KΣ_I]

    θ₀ = randn(ComplexF64, length(θ))
    Σ₀ = map(s -> s(θ => θ₀), Σ)
    Σ₀⁻¹ = inv(Σ₀)
    k₀ = [Σ₀⁻¹[j,i] for i in 1:n for j in i:n]
    G = map(1:N+m) do i
        subs(F[i], k => k₀, θ => θ₀)
    end

    res = HC.solve([G;randn(ComplexF64, N-m, 2N+1) * [s; λ; 1]],
                    system=HC.FPSystem,
                    affine_tracking=true)

    sol = HC.solution(first(res))
    λ₀ = sol[1:N]
    s₀ = sol[N+1:end]

    x₀ = [θ₀; k₀; λ₀]
    (F=F, generic_solution=x₀, generic_parameters=s₀, vars=[θ;k;λ], parameters=s)
end


function logdet(T, θ, S::AbstractMatrix)
    Σ = sum(θ[i] * T[:,i] * T[:,i]' for i in 1:size(T, 2))

    -log(det(Σ)) - tr(S*inv(Σ))
end

function gradient_logdet(T, θ, S::AbstractMatrix)
    Σ = sum(θ[i] * T[:,i] * T[:,i]' for i in 1:size(T, 2))
    Σ⁻¹ = inv(Σ)
    map(1:size(T,2)) do i
        -tr(Σ⁻¹ * T[:,i] * T[:,i]') + tr(S *  Σ⁻¹ * T[:,i] * T[:,i]' * Σ⁻¹)
    end
end

function hessian_logdet(T, θ, S::AbstractMatrix)
    Σ = sum(θ[i] * T[:,i] * T[:,i]' for i in 1:size(T, 2))
    Σ⁻¹ = inv(Σ)
    m = size(T,2)
    H = zeros(m,m)
    for i in 1:m, j in i:m
        kernel = Σ⁻¹ * T[:,i] * T[:,i]' * Σ⁻¹ * T[:,j] * T[:,j]'
        H[i,j] = H[j,i] = tr(kernel) - 2tr(S * kernel * Σ⁻¹)
    end
    Symmetric(H)
end

struct MLE{PT<:HC.PathTracker}
    tree::Matrix{Int}
    generic_parameters::Vector{ComplexF64}
    generic_solutions::Vector{Vector{ComplexF64}}
    pathtracker::PT
end

function MLE(tree::Matrix)
    F, x₀, s₀, vars, params = build_system(tree)
    result = HC.monodromy_solve(F, x₀, s₀; parameters=params, affine_tracking=true)
    generic_solutions = Vector.(HC.solutions(result))
    generic_parameters = s₀
    tracker = HC.pathtracker(F, generic_solutions; parameters=params, generic_parameters=s₀)
    MLE(tree, generic_parameters, generic_solutions, tracker)
end

struct CriticalPoint
    θ::Vector{Float64}
    Σ::Symmetric{Float64, Matrix{Float64}}
    crit_type::Symbol
    obj_value::Float64
end

function Base.show(io::IO, C::CriticalPoint)
    print(io, "CriticalPoint ($(C.crit_type)):\n")
    print(io, " * obj_value = $(C.obj_value)\n")
    print(io, " * θ = ", C.θ)
end

function solve(MLE::MLE, S::AbstractMatrix)
    thetas = trackall(MLE.tree, MLE.pathtracker, MLE.generic_solutions, S)
    map(thetas) do θ
        Σ = construct_Σ(θ, MLE.tree)

        if all(x -> x ≥ 0, θ)
            H = hessian_logdet(MLE.tree, θ, S)
            crit_type = :saddle_point
            if isposdef(H)
                crit_type = :local_minimum
            elseif isposdef(-H)
                crit_type = :local_maximum
            end
        else
            crit_type = :invalid
        end

        if crit_type != :invalid
            obj_value = logdet(MLE.tree, θ, S)
        else
            obj_value = NaN
        end
        CriticalPoint(θ, Σ, crit_type, obj_value)
    end
end


function trackall(tree::Matrix, pathtracker::HC.PathTracker, generic_solutions, S::AbstractMatrix)
    s = sym_to_vec(S)
    n, m = size(tree)
    n, N = size(S, 1), length(S)
    HC.set_parameters!(pathtracker, target_parameters=s)

    thetas = Vector{Float64}[]
    for x in generic_solutions
        result = HC.track(pathtracker, x)
        if HC.issuccess(result) && HC.isrealvector((@view HC.solution(result)[1:m]), 1e-8)
            θ = real.(HC.solution(result)[1:m])
            push!(thetas, θ)
        end
    end
    thetas
end

function construct_Σ(θ, tree)
    n, m = size(tree)
    Σ = (θ[1]+0) .* (view(tree, 1:n, 1) * view(tree, 1:n, 1)')
    for i in 2:m
        Σ .+= θ[i] .* (view(tree, 1:n, i) * view(tree, 1:n, i)')
    end
    Symmetric(Σ)
end


function rand_pos_def(n)
    S = randn(n, n)
    S *= S'
    Symmetric(S)
end

n_vec_to_sym(k) = div(-1 + round(Int, sqrt(1+8k)), 2)
n_sym_to_vec(n) = binomial(n+1,2)

function vec_to_sym(s)
    n = n_vec_to_sym(length(s))
    S = zeros(n, n)
    l = 1
    for i in 1:n, j in i:n
        S[i,j] = S[j,i] = s[l]
        l += 1
    end
    Symmetric(S)
end

function sym_to_vec(S)
    n = size(S, 1)
    [S[i,j] for i in 1:n for j in i:n]
end

function star_tree_boundary(tree::Matrix, S)
    if !all(isone, @view tree[:,1])
        error("The all 1 vector has to be the first column of the tree matrix.")
    end
    D̃ = distance_matrix(S)
    map(1:size(D̃, 2)) do i
        θ = D̃[:,i]
        Σ = construct_Σ(θ, tree)
        obj_value = logdet(tree, θ, S)
        CriticalPoint(θ, Σ, :boundary, obj_value)
    end
end

function distance_matrix(S::AbstractMatrix)
    n = size(S, 1)
    D = ones(n, n) .* diag(S)' + diag(S) .* ones(n,n) - 2*S
    vcat([0; diag(S)]', [diag(S) D])
end

end # module
