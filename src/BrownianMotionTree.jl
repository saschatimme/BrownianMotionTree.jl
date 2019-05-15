module BrownianMotionTree

import DynamicPolynomials: @polyvar, Polynomial, differentiate, subs
import HomotopyContinuation
const HC = HomotopyContinuation

include("system.jl")


function construct_Σ(tree::Matrix{<:Integer})
    m, n = size(tree)
    @polyvar θ[0:n]

    Σ = fill(θ[1]+0, m, m)
    for i in 1:n
        Σ .+= θ[i+1] .* (view(tree, 1:m, i) * view(tree, 1:m, i)')
    end

    Σ
end

function construct_equations(tree::Matrix{<:Integer})
    m, n = size(tree)
    N = binomial(n+1,2)
    @polyvar k[1:N] θ[0:n] λ[1:N] s[1:N]

    K = Matrix{eltype(k)}(undef, n, n)
    S = Matrix{eltype(s)}(undef, n, n)
    l = 1
    for i in 1:n, j in i:n
        K[i,j] = K[j,i] = k[l]
        S[i,j] = S[j,i] = s[l]
        l += 1
    end

    Σ = fill(θ[1]+0, m, m)
    for i in 1:n
        Σ .+= θ[i+1] .* (view(tree, 1:m, i) * view(tree, 1:m, i)')
    end

    KΣ = K * Σ
    KΣ_I = [KΣ[i,j] - (i == j) for i in 1:n for j in i:n]
    # LPS = [KΣ[i,j] - (i == j) for i in 1:n for j in i:n]

    ∇_logdet_K = [
        [i == j ? 1Σ[i,i] : 2Σ[i,j] for i in 1:n for j in i:n];
        #[-tr(K * differentiate.(Σ, Ref(θ[i]))) for i in 1:n+1]
        zeros(Int, n+1)
        ]
    ∇_trace_SK = [
        [i == j ? 1S[i,i] : 2S[i,j] for i in 1:n for j in i:n];
        #[-tr(D*K*differentiate.(Σ, Ref(θ[i]))*K) for i in 1:n+1]
        zeros(Int, n+1)
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
    G = map(1:N+n+1) do i
        subs(F[i], k => k₀, θ => θ₀)
    end

    res = HC.solve([G;randn(ComplexF64, N-(n+1), 2N+1) * [s; λ; 1]],
                    affine_tracking=true)

    sol = HC.solution(res[1])
    λ₀ = sol[1:N]
    s₀ = sol[N+1:end]

    x₀ = [k₀; θ₀; λ₀]
    (F=F, generic_solution=x₀, generic_parameters=s₀, vars=[k;θ;λ], parameters=s)
end

end # module
