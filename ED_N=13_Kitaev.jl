#=
Exact-diagonalization ground state of a small (N=13, trigonal) Kitaev honeycomb
cluster, used as a ground-truth check on the CAMPS + Pauli-propagation pipeline
(Correlations/kitaev_correlation.jl) at a system size small enough for exact
methods (2^13 = 8192-dim Hilbert space).

The Hamiltonian is built as a genuine SparseMatrixCSC (each 2-site bond term is
a Kronecker product of single-site Pauli/identity matrices, which is exactly
sparse -- one nonzero per row/column -- never densified), and only the ground
state is extracted via a sparse Lanczos-type solver (KrylovKit.eigsolve), not a
full dense diagonalization.

Run from this directory so the project-local environment (Project.toml here,
pinning KrylovKit) is picked up:
    julia --project=. "ED_N=13_Kitaev.jl"
=#

using SparseArrays
using LinearAlgebra
using KrylovKit
using Printf

N = 22
# actual 13-site trigonal cluster bonds here -----------
# Same convention as CAMPS/Kitaev_honeycomb_CAMPS.py's X_BONDS/Y_BONDS/Z_BONDS:
# 1-indexed site-pair tuples, one list per bond color. Each site should appear
# in at most one bond of each color (a real honeycomb/trigonal vertex has one
# neighbor per color in the bulk, fewer at the boundary).
# Z_BONDS = [(3,4),(5,6),(10,11),(1,12),(13,8)]
# X_BONDS = [(1,2),(13,4),(7,6),(8,9),(11,12)]
# Y_BONDS = [(2,3),(4,5),(7,8),(9,10),(12,13)]   # PLACEHOLDER

Z_BONDS = [(1,18),(3,4),(5,6),(7,8),(10,22),(12,21),(14,15),(16,17),(19,20)]
X_BONDS = [(1,2),(4,19),(6,22),(8,9),(10,11),(12,13),(15,16),(17,18),(20,21)]
Y_BONDS = [(2,3),(4,5),(6,7),(9,10),(11,12),(13,14),(16,21),(18,19),(20,22)]

# -----------------------------------------------------------------------------

K = 1.0       # overall Kitaev bond coupling; H_K = J * sum_<ij> P_i P_j per bond color (pauli normalization)
V = 0.3       # effective-field three-body term strength (H_V), matches Kitaev_honeycomb_CAMPS.py's --V default
h_field = 0.1 # boundary field strength (H_V), matches Kitaev_honeycomb_CAMPS.py's --h_field default

const I2 = sparse(ComplexF64[1 0; 0 1])
const SX = sparse(ComplexF64[0 1; 1 0])
const SY = sparse(ComplexF64[0 -im; im 0])
const SZ = sparse(ComplexF64[1 0; 0 -1])
const PAULI_MAT = Dict(:X => SX, :Y => SY, :Z => SZ)
const TYPES = (:X, :Y, :Z)

# Sparse operator for `op` acting on `site` (1-indexed), identity elsewhere,
# built as a chain of Kronecker products -- stays exactly sparse throughout.
function embed_two_site(op::SparseMatrixCSC{ComplexF64}, i::Int, j::Int, N::Int)
    mats = [k == i || k == j ? op : I2 for k in 1:N]
    return foldl(kron, mats)
end

function embed_one_site(op::SparseMatrixCSC{ComplexF64}, site::Int, N::Int)
    mats = [k == site ? op : I2 for k in 1:N]
    return foldl(kron, mats)
end

# Sparse operator for `op_i`,`op_j`,`op_k` acting on three DISTINCT sites
# `i`,`j`,`k` (1-indexed), identity elsewhere -- same Kronecker-chain
# construction as embed_two_site, just with a third active site. Still stays
# exactly sparse (one nonzero per row/column of the full N-qubit matrix),
# since this is still a plain tensor product of single-qubit operators.
function embed_three_site(op_i::SparseMatrixCSC{ComplexF64}, i::Int,
                           op_j::SparseMatrixCSC{ComplexF64}, j::Int,
                           op_k::SparseMatrixCSC{ComplexF64}, k::Int, N::Int)
    mats = Vector{SparseMatrixCSC{ComplexF64,Int}}(undef, N)
    for s in 1:N
        mats[s] = s == i ? op_i : s == j ? op_j : s == k ? op_k : I2
    end
    return foldl(kron, mats)
end

# bonds_at[site][type] = neighbor, for whichever bond directions actually
# touch that site -- same role as CAMPS/Kitaev_honeycomb_CAMPS.py's bonds_at
# dict, built here directly from X_BONDS/Y_BONDS/Z_BONDS.
function build_bonds_at(N::Int, X_BONDS, Y_BONDS, Z_BONDS)
    bonds_at = [Dict{Symbol,Int}() for _ in 1:N]
    for (i, j) in X_BONDS
        bonds_at[i][:X] = j; bonds_at[j][:X] = i
    end
    for (i, j) in Y_BONDS
        bonds_at[i][:Y] = j; bonds_at[j][:Y] = i
    end
    for (i, j) in Z_BONDS
        bonds_at[i][:Z] = j; bonds_at[j][:Z] = i
    end
    return bonds_at
end

# H = H_K + H_V, matching CAMPS/Kitaev_honeycomb_CAMPS.py's
# Kitaev_Honeycomb_Pauli_terms exactly:
#   H_K = J * sum_bonds P_i P_j                                  (pure Kitaev bonds)
#   H_V = V * sum_{adjacent bond pairs} (three-body term)
#       + h * sum_{boundary sites} sigma_j^alpha                  (alpha = that
#                                                                    site's MISSING
#                                                                    bond direction)
# Three-body term convention: at a vertex j connected to neighbor i via an
# alpha-bond and to neighbor k via a beta-bond (alpha != beta), the term is
# sigma_i^alpha sigma_j^gamma sigma_k^beta, where gamma is the THIRD Pauli
# type. A bulk (degree-3) site contributes all three such pairs (xy, xz, yz);
# a boundary (degree-2) site contributes exactly one.
function build_hamiltonian(N::Int, X_BONDS, Y_BONDS, Z_BONDS, K::Float64, V::Float64 = 0.3, h::Float64 = 0.1)
    dim = 2^N
    H = spzeros(ComplexF64, dim, dim)
    for (i, j) in X_BONDS
        H += -K .* embed_two_site(SX, i, j, N)
    end
    for (i, j) in Y_BONDS
        H += -K .* embed_two_site(SY, i, j, N)
    end
    for (i, j) in Z_BONDS
        H += -K .* embed_two_site(SZ, i, j, N)
    end

    bonds_at = build_bonds_at(N, X_BONDS, Y_BONDS, Z_BONDS)
    n_threebody = 0
    n_boundary = 0
    for j in 1:N
        present = [t for t in TYPES if haskey(bonds_at[j], t)]

        for a in 1:length(present), b in (a+1):length(present)
            α, β = present[a], present[b]
            γ = only(setdiff(collect(TYPES), (α, β)))
            i_site = bonds_at[j][α]
            k_site = bonds_at[j][β]
            H += V .* embed_three_site(PAULI_MAT[α], i_site, PAULI_MAT[γ], j, PAULI_MAT[β], k_site, N)
            n_threebody += 1
        end

        if length(present) == 2
            missing_type = only(setdiff(collect(TYPES), present))
            H += h .* embed_one_site(PAULI_MAT[missing_type], j, N)
            n_boundary += 1
        end
    end
    println("  H_V: $n_threebody three-body terms, $n_boundary boundary-field terms")

    return H
end

# Bipartite von Neumann entanglement entropy of a statevector across the cut
# {1,...,N1} | {N1+1,...,N} -- same diagnostic CAMPS reports (entropy_summary),
# so the ED reference can be compared directly against the CAMPS bond entropy.
function entanglement_entropy(psi::Vector{ComplexF64}, N::Int, N1::Int)
    dimA, dimB = 2^N1, 2^(N - N1)
    M = reshape(psi, dimB, dimA)   # Julia is column-major: fastest index = site 1
    s = svdvals(Matrix(M))
    p = s .^ 2
    p = p[p .> 1e-14]
    return -sum(p .* log.(p))
end

println("="^60)
println("Sparse ED, Kitaev honeycomb trigonal cluster, N=$N")
println("="^60)
println("Hilbert space dimension: 2^$N = $(2^N)")
println("Bonds: $(length(X_BONDS)) X, $(length(Y_BONDS)) Y, $(length(Z_BONDS)) Z")
println("K=$K  V=$V  h_field=$h_field")

t_build = @elapsed H = build_hamiltonian(N, X_BONDS, Y_BONDS, Z_BONDS, K, V, h_field)
@printf("Built sparse Hamiltonian in %.2f s -- nnz = %d (dense would be %d)\n",
        t_build, nnz(H), (2^N)^2)

@assert ishermitian(H) "Hamiltonian is not Hermitian -- check bond lists / operator construction"

x0 = normalize(randn(ComplexF64, 2^N))
t_diag = @elapsed vals, vecs, info = eigsolve(H, x0, 1, :SR; ishermitian = true, tol = 1e-12)
@printf("Sparse Lanczos (KrylovKit) ground-state solve in %.2f s (converged=%s)\n",
        t_diag, info.converged >= 1)

E0 = real(vals[1])
psi0 = vecs[1]

@printf("\nGround-state energy: E0 = %.12f\n", E0)
@printf("Energy per site:     E0/N = %.12f\n", E0 / N)

N1 = N ÷ 2
S = entanglement_entropy(psi0, N, N1)
@printf("Bipartite entanglement entropy (cut at site %d|%d): S = %.6f\n", N1, N1 + 1, S)
