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


Base.size(F::BMTSystem) = (2*F.N + size(F.T,1) + 1, 2*F.N + size(F.T,1) + 1)

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
    n, N = size(F.T, 1), F.N
    outer = trues(n, n, n+1)
    outer[:,:,1] .= true
    for i = 1:n
        outer[:,:,i+1] .= (view(F.T, 1:n, i) * view(F.T, 1:n, i)')
    end
    K = zeros(ComplexF64, n, n)
    Σ = zeros(ComplexF64, n, n)
    λ = zeros(ComplexF64, N)
    KΣ_I = zeros(ComplexF64, N)
    J_KΣ_I = zeros(ComplexF64, N, N+n+1+N)
    H_KΣ_I = falses(N, N+n+1, N+n+1)
    S = zeros(ComplexF64, n, n)
    BMTSystemCache(outer, K, Σ, λ, KΣ_I, J_KΣ_I, H_KΣ_I, S)
end

function fill_cache!(cache::BMTSystemCache, F::BMTSystem, x, s)
    Σ, K, S, λ = cache.Σ, cache.K, cache.S, cache.λ
    n, N = size(F.T, 1), F.N
    Σ .= x[N+1]
    for k in 1:n
        θ = x[N+1+k]
        for j in 1:n, i in j:n
            Σ[i,j] = Σ[j,i] += θ * cache.outer[i,j,k+1]
        end
    end

    l = 1
    for i in 1:n, j in i:n
        K[i,j] = K[j,i] = x[l]
        S[i,j] = S[j,i] = s[l]
        l += 1
    end

    for (i,l) in enumerate((N+n+2):(2N+n+1))
        λ[i] = x[l]
    end
    cache
end

function fill_partial_values!(cache, S)
    KΣ_I = cache.KΣ_I
    l = 1
    for i in 1:n, j in i:n
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
    n, N = size(F.T, 1), F.N

    KΣ_I = cache.KΣ_I
    l = 1
    for i in 1:n, j in i:n
        KΣ_I[l] = -(i == j)
        for k in 1:n
            KΣ_I[l] += K[i,k] * Σ[k,j]
        end
        l += 1
    end

    # J_KΣ_I and H_KΣ_I
    fill!(H_KΣ_I, false)
    s = 1
    for i in 1:n, j in i:n
        # derivative with respect to k
        t = 1
        for k in 1:n, l in k:n
            J_KΣ_I[s,t] = 0
            # derivative wrt K[k,l] = K[l,k]
            if k == i
                # k = i
                J_KΣ_I[s,t] = Σ[l,j]
                for v in 1:n+1
                    H_KΣ_I[s, t, N+v] = outer[l,j,v]
                end
            elseif l == i
                J_KΣ_I[s, t] = Σ[k,j]
                for v in 1:n+1
                    H_KΣ_I[s, t, N+v] = outer[k,j,v]
                end
            end

            t += 1
        end
        for k in 1:(n+1)
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
    if fill
        fill_cache!(cache, F, x, s)
        compute_KΣ_I!(cache, F)
    end

    outer, Σ, K, S, λ = cache.outer, cache.Σ, cache.K, cache.S, cache.λ
    KΣ_I, J_KΣ_I, H_KΣ_I = cache.KΣ_I, cache.J_KΣ_I, cache.H_KΣ_I
    n, N = size(F.T, 1), F.N

    # evaluate
    # L
    k=1
    for i in 1:n, j in i:n
        u[k] = (2 - (i == j)) * (Σ[i,j] - S[i,j])
        for s in 1:N
            u[k] -= λ[s] * J_KΣ_I[s, k]
        end
        k += 1
    end
    for i in 1:n+1
        u[k] = 0
        for s in 1:N
            u[k] -= λ[s] * J_KΣ_I[s, k]
        end
        k += 1
    end
    # constraints
    for i in 1:N
        u[k] = KΣ_I[i]
        k += 1
    end

    u
end

function HC.jacobian!(U, F::BMTSystem, x, s, cache::BMTSystemCache, fill=true)
    if fill
        fill_cache!(cache, F, x, s)
        compute_KΣ_I!(cache, F)
    end

    outer, Σ, K, S, λ = cache.outer, cache.Σ, cache.K, cache.S, cache.λ
    KΣ_I, J_KΣ_I, H_KΣ_I = cache.KΣ_I, cache.J_KΣ_I, cache.H_KΣ_I
    n, N = size(F.T, 1),F.N

    s=1
    for i in 1:n, j in i:n
        # derivative of L[1:N] wrt K_ij is 0
        t = N+1
        # derivative wrt θ
        for k in 1:n+1
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
    for i in 1:n+1
        t = 1

        while t ≤ N
            for v in 1:N
                U[s,t] -= λ[v] * H_KΣ_I[v, s, t]
            end
            t += 1
        end
        # derivative wrt θ is 0
        t += n+1
        # derivative wrt λ
        for v in 1:N
            U[s, t] = -J_KΣ_I[v, s]
            t += 1
        end
        s += 1
    end

    for j in 1:size(U,2), i in 1:N
        U[s+i-1,j] = J_KΣ_I[i, j]
    end

    U
end


function HC.evaluate_and_jacobian!(u, U, F::BMTSystem, x, s, cache::BMTSystemCache)
    fill_cache!(cache, F, x, s)
    compute_KΣ_I!(cache, F)

    evaluate!(u, F, x, s, cache, false)
    jacobian!(U, F, x, s, cache, false)
end

function HC.evaluate(F::BMTSystem, x, s, cache::BMTSystemCache)
    u = zeros(ComplexF64, size(F)[1])
    HC.evaluate!(u, F, x, s, cache)
end

function HC.jacobian(F::BMTSystem, x, s, cache::BMTSystemCache)
    U = zeros(ComplexF64, size(F))
    HC.jacobian!(U, F, x, s, cache)
end
