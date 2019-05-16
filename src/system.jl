# not used at this point

export BMTSystem

struct BMTSystem <: HC.AbstractSystem
    T::BitMatrix
    N::Int
end

function BMTSystem(tree::AbstractMatrix)
    T = BitMatrix(tree)
    N = binomial(size(T,1)+1,2)
    BMTSystem(T,N)
end

Base.size(F::BMTSystem) = (2*F.N + size(F.T,2), 2*F.N + size(F.T,2))

struct BMTSystemCache <: HC.AbstractSystemCache
    outer::BitArray{3}
    K::Matrix{ComplexF64}
    Σ::Matrix{ComplexF64}
    λ::Vector{ComplexF64}
    KΣ_I::Vector{ComplexF64}
    J_KΣ_I::Matrix{ComplexF64}
    H_KΣ_I::BitArray{3}
    S::Matrix{ComplexF64}
end

function HC.cache(F::BMTSystem, x, s)
    n, m = size(F.T)
    N = F.N
    outer = trues(n, n, m)
    for i = 1:m
        outer[:,:,i] .= (view(F.T, 1:n, i) * view(F.T, 1:n, i)')
    end
    K = zeros(ComplexF64, n, n)
    Σ = zeros(ComplexF64, n, n)
    λ = zeros(ComplexF64, N)
    KΣ_I = zeros(ComplexF64, N)
    J_KΣ_I = zeros(ComplexF64, N, N+m+N)
    H_KΣ_I = falses(N, N+m, N+m)
    S = zeros(ComplexF64, n, n)
    BMTSystemCache(outer, K, Σ, λ, KΣ_I, J_KΣ_I, H_KΣ_I, S)
end

function fill_cache!(cache::BMTSystemCache, F::BMTSystem, x, s)
    Σ, K, S, λ = cache.Σ, cache.K, cache.S, cache.λ
    n, m = size(F.T)
    N = F.N
    Σ .= x[N+1]
    @inbounds for k in 2:m
        θ = x[N+k]
        for j in 1:n, i in j:n
            Σ[i,j] = Σ[j,i] = Σ[j,i] + θ * cache.outer[i,j,k]
        end
    end

    l = 1
    @inbounds for i in 1:n, j in i:n
        K[i,j] = K[j,i] = x[l]
        S[i,j] = S[j,i] = s[l]
        l += 1
    end

    @inbounds for (i,l) in enumerate((N+m+1):(2N+m))
        λ[i] = x[l]
    end
    cache
end

function fill_partial_values!(cache, S)
    KΣ_I = cache.KΣ_I
    l = 1
    @inbounds for i in 1:n, j in i:n
        KΣ_I[l] = -(i == j)
        for k in 1:n
            KΣ_I[l] += K[i,k] * Σ[k,j]
        end
        l += 1
    end
end

function compute_KΣ_I!(cache::BMTSystemCache, F::BMTSystem)
    outer, Σ, K, S = cache.outer, cache.Σ, cache.K, cache.S
    KΣ_I, J_KΣ_I, H_KΣ_I = cache.KΣ_I, cache.J_KΣ_I, cache.H_KΣ_I
    n, m = size(F.T)
    N = F.N

    KΣ_I = cache.KΣ_I
    l = 1
    @inbounds for i in 1:n, j in i:n
        KΣ_I[l] = -(i == j)
        for k in 1:n
            KΣ_I[l] += K[i,k] * Σ[k,j]
        end
        l += 1
    end

    # J_KΣ_I and H_KΣ_I
    fill!(H_KΣ_I, false)
    s = 1
    @inbounds for i in 1:n, j in i:n
        # derivative with respect to k
        t = 1
        for k in 1:n, l in k:n
            J_KΣ_I[s,t] = 0
            # derivative wrt K[k,l] = K[l,k]
            if k == i
                # k = i
                J_KΣ_I[s,t] = Σ[l,j]
                for v in 1:m
                    H_KΣ_I[s, t, N+v] = outer[l,j,v]
                end
            elseif l == i
                J_KΣ_I[s, t] = Σ[k,j]
                for v in 1:m
                    H_KΣ_I[s, t, N+v] = outer[k,j,v]
                end
            end

            t += 1
        end
        for k in 1:m
            J_KΣ_I[s,t] = 0
            for l in 1:n
                J_KΣ_I[s,t] += K[i,l] * outer[l,j,k]
            end

            w = 1
            for r in 1:n, v in r:n
                # derivative wrt K[k,l] = K[l,k]
                if r == i
                    H_KΣ_I[s, t, w] = outer[v,j,k]
                elseif v == i
                    H_KΣ_I[s, t, w] = outer[r,j,k]
                end
                w += 1
            end
            t += 1
        end
        s += 1
    end
    cache
end

function HC.evaluate!(u, F::BMTSystem, x, s, cache::BMTSystemCache, fill=true)
    u .= 0.0
    if fill
        fill_cache!(cache, F, x, s)
        compute_KΣ_I!(cache, F)
    end

    outer, Σ, K, S, λ = cache.outer, cache.Σ, cache.K, cache.S, cache.λ
    KΣ_I, J_KΣ_I, H_KΣ_I = cache.KΣ_I, cache.J_KΣ_I, cache.H_KΣ_I
    n, m = size(F.T)
    N = F.N

    k=1
    @inbounds for i in 1:n, j in i:n
        u[k] = (2 - (i == j)) * (Σ[i,j] - S[i,j])
        for s in 1:N
            u[k] -= λ[s] * J_KΣ_I[s, k]
        end
        k += 1
    end
    @inbounds for i in 1:m
        u[k] = 0
        for s in 1:N
            u[k] -= λ[s] * J_KΣ_I[s, k]
        end
        k += 1
    end
    # constraints
    @inbounds for i in 1:N
        u[k] = KΣ_I[i]
        k += 1
    end

    u
end

function HC.jacobian!(U, F::BMTSystem, x, s, cache::BMTSystemCache, fill=true)
    U .= 0.0
    if fill
        fill_cache!(cache, F, x, s)
        compute_KΣ_I!(cache, F)
    end

    outer, Σ, K, S, λ = cache.outer, cache.Σ, cache.K, cache.S, cache.λ
    KΣ_I, J_KΣ_I, H_KΣ_I = cache.KΣ_I, cache.J_KΣ_I, cache.H_KΣ_I
    n, m = size(F.T)
    N = F.N

    s=1
    @inbounds for i in 1:n, j in i:n
        # derivative of L[1:N] wrt K_ij is 0
        t = N+1
        # derivative wrt θ
        for k in 1:m
            # J_L[s,t] = 0
            U[s,t] = (2 - (i == j)) * outer[i,j, k]
            for v in 1:N
                U[s,t] -= λ[v] * H_KΣ_I[v, s, t]
            end
            t += 1
        end

        # derivative wrt λ
        for k in 1:N
            U[s,t] = -J_KΣ_I[k,s]
            t += 1
        end

        s += 1
    end
    @inbounds for i in 1:m
        t = 1

        while t ≤ N
            for v in 1:N
                U[s,t] -= λ[v] * H_KΣ_I[v, s, t]
            end
            t += 1
        end
        # derivative wrt θ is 0
        t += m
        # derivative wrt λ
        for v in 1:N
            U[s, t] = -J_KΣ_I[v, s]
            t += 1
        end
        s += 1
    end

    @inbounds for j in 1:size(U,2), i in 1:N
        U[s+i-1,j] = J_KΣ_I[i, j]
    end

    U
end


function HC.evaluate_and_jacobian!(u, U, F::BMTSystem, x, s, cache::BMTSystemCache)
    fill_cache!(cache, F, x, s)
    compute_KΣ_I!(cache, F)

    HC.evaluate!(u, F, x, s, cache, false)
    HC.jacobian!(U, F, x, s, cache, false)
end

function HC.evaluate(F::BMTSystem, x, s, cache::BMTSystemCache)
    u = zeros(ComplexF64, size(F)[1])
    HC.evaluate!(u, F, x, s, cache)
end

function HC.jacobian(F::BMTSystem, x, s, cache::BMTSystemCache)
    U = zeros(ComplexF64, size(F))
    HC.jacobian!(U, F, x, s, cache)
end

function HC.differentiate_parameters!(U, F::BMTSystem, x, s, cache::BMTSystemCache)
    n = size(F.T, 1)
    k=1
    U .= zero(eltype(U))
    @inbounds for i in 1:n, j in i:n
        U[k,k] = ((i == j) - 2)
        k += 1
    end

    U
end

function HC.differentiate_parameters(F::BMTSystem, x, s, cache::BMTSystemCache)
    U = zeros(ComplexF64, size(F)[1], F.N)
    HC.differentiate_parameters!(U, F, x, s, cache)
end
