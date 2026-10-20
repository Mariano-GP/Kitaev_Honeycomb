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

N = 13
# actual 13-site trigonal cluster bonds here -----------
# Same convention as CAMPS/Kitaev_honeycomb_CAMPS.py's X_BONDS/Y_BONDS/Z_BONDS:
# 1-indexed site-pair tuples, one list per bond color. Each site should appear
# in at most one bond of each color (a real honeycomb/trigonal vertex has one
# neighbor per color in the bulk, fewer at the boundary).
Z_BONDS = [(3,4),(5,6),(10,11),(1,12),(13,8)]
X_BONDS = [(1,2),(13,4),(7,6),(8,9),(11,12)]
Y_BONDS = [(2,3),(4,5),(7,8),(9,10),(12,13)]   # PLACEHOLDER
# -----------------------------------------------------------------------------

const J = 1.0   # overall coupling; H = J * sum_<ij> P_i P_j per bond color (pauli normalization)

const I2 = sparse(ComplexF64[1 0; 0 1])
const SX = sparse(ComplexF64[0 1; 1 0])
const SY = sparse(ComplexF64[0 -im; im 0])
const SZ = sparse(ComplexF64[1 0; 0 -1])

# Sparse operator for `op` acting on `site` (1-indexed), identity elsewhere,
# built as a chain of Kronecker products -- stays exactly sparse throughout.
function embed_two_site(op::SparseMatrixCSC{ComplexF64}, i::Int, j::Int, N::Int)
    mats = [k == i || k == j ? op : I2 for k in 1:N]
    return foldl(kron, mats)
end

function build_hamiltonian(N::Int, X_BONDS, Y_BONDS, Z_BONDS, J::Float64)
    dim = 2^N
    H = spzeros(ComplexF64, dim, dim)
    for (i, j) in X_BONDS
        H += J .* embed_two_site(SX, i, j, N)
    end
    for (i, j) in Y_BONDS
        H += J .* embed_two_site(SY, i, j, N)
    end
    for (i, j) in Z_BONDS
        H += J .* embed_two_site(SZ, i, j, N)
    end
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

t_build = @elapsed H = build_hamiltonian(N, X_BONDS, Y_BONDS, Z_BONDS, J)
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
