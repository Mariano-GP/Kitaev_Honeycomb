#=
Bridge between CAMPS (Python/TeNPy) and Pauli propagation (Julia/PauliOperators+DBF).

Use case: you ran CAMPS on the Kitaev honeycomb cluster and want to compute
Pauli-propagation correlators C(t) = <psi_GS| W(t) V_j |psi_GS> where psi_GS
is the CAMPS ground state, staying in the Clifford-disentangled frame (i.e.
you already conjugated the Hamiltonian by the same circuit CAMPS applied to
the state, e.g. by exporting `current_terms` from `run_camps`, not the
original physical-frame Hamiltonian).

On the Python side, call:
    mps_path, terms_path = export_camps_state_for_julia(psi, current_terms, outdir, tag="camps_state")

This file provides the Julia-side counterpart: load those two files back in
and evaluate <psi_GS| P |psi_GS> for an arbitrary PauliOperators.jl PauliBasis,
using plain Julia arrays (no ITensors dependency).
=#

import PauliOperators as PO
using NPZ

"""
    load_mps_from_npz(path, N) -> Vector{Array{ComplexF64,3}}

Load the N site tensors exported by `export_camps_state_for_julia`. Each
tensor has shape (chi_left, d, chi_right) and is TeNPy's right-canonical
('B') form. Because the underlying MPS is finite with open boundaries (bond
dimension 1 at both ends), sequentially contracting B_1 B_2 ... B_N over the
shared bond legs reproduces the exact state -- no singular values need to be
reinserted. (Verified numerically against TeNPy's own full wavefunction
before writing this bridge.)
"""
function load_mps_from_npz(path::String, N::Int)
    data = npzread(path)
    Bs = Vector{Array{ComplexF64,3}}(undef, N)
    for i in 1:N
        B = data["B$(i-1)"]  # Python export is 0-indexed
        Bs[i] = ComplexF64.(B)
    end
    return Bs
end

"""
    load_pauli_terms_csv(path, N) -> PO.PauliSum{N, ComplexF64}

Load the Hamiltonian terms exported by `export_camps_state_for_julia`.
Each line is: coeff_re,coeff_im,site,pauli,site,pauli,...
site is 0-indexed (converted to 1-indexed here); pauli labels are X=1,Y=2,Z=3.
"""
function load_pauli_terms_csv(path::String, N::Int)
    H_sum = PO.PauliSum(N, ComplexF64)
    for line in eachline(path)
        line = strip(line)
        isempty(line) && continue
        parts = split(line, ",")
        coeff = parse(Float64, parts[1]) + im * parse(Float64, parts[2])

        xs = Int[]; ys = Int[]; zs = Int[]
        i = 3
        while i <= length(parts)
            site = parse(Int, parts[i]) + 1   # 0-indexed (Python) -> 1-indexed (Julia)
            label = parse(Int, parts[i+1])
            if label == 1
                push!(xs, site)
            elseif label == 2
                push!(ys, site)
            elseif label == 3
                push!(zs, site)
            end
            i += 2
        end
        H_sum += coeff * PO.Pauli(N, X=xs, Y=ys, Z=zs)
    end
    return H_sum
end

"""
    site_pauli_matrix(xb::Bool, zb::Bool) -> Matrix{ComplexF64}

Standard 2x2 Pauli matrix for a single site given its (x,z) bits, matching
PauliOperators.jl's own convention exactly (verified against
`Base.Matrix(::PauliBasis)` in the installed package source):
(0,0)->I, (1,0)->X, (1,1)->Y, (0,1)->Z. No extra phase correction needed --
`coeff(::PauliBasis) == 1` in PauliOperators.jl, i.e. a PauliBasis is already
the canonically-signed Hermitian operator.
"""
function site_pauli_matrix(xb::Bool, zb::Bool)
    if !xb && !zb
        return ComplexF64[1 0; 0 1]
    elseif xb && !zb
        return ComplexF64[0 1; 1 0]
    elseif xb && zb
        return ComplexF64[0 -1im; 1im 0]
    else
        return ComplexF64[1 0; 0 -1]
    end
end

"""
    expectation_value_mps(Bs, basis::PO.PauliBasis{N}) where N -> Float64

Compute <psi|P|psi> for the MPS represented by `Bs` (as returned by
`load_mps_from_npz`) and the Pauli string `basis`, via a standard left-to-right
transfer-matrix sweep (O(L*chi^3*d) using BLAS matrix multiplies, not a naive
O(chi^4) contraction). `psi` must already be normalized (true after
`psi.canonical_form()` on the Python side).
"""
function expectation_value_mps(Bs::Vector{Array{ComplexF64,3}}, basis::PO.PauliBasis{N}) where N
    length(Bs) == N || throw(DimensionMismatch("Bs has $(length(Bs)) sites, basis expects $N"))

    E = ones(ComplexF64, 1, 1)  # (bra_bond, ket_bond); boundary bond dim is 1
    for i in 1:N
        B = Bs[i]                     # (chiL, d, chiR)
        chiL, d, chiR = size(B)
        xb = (basis.x >> (i - 1)) & 1 != 0
        zb = (basis.z >> (i - 1)) & 1 != 0
        Opi = site_pauli_matrix(xb, zb)

        # 1. apply the single-site operator to the ket tensor's physical leg
        Bp = permutedims(B, (2, 1, 3))                  # (d, chiL, chiR)
        Bop = Opi * reshape(Bp, d, chiL * chiR)          # (d, chiL*chiR)
        Bop = permutedims(reshape(Bop, d, chiL, chiR), (2, 1, 3))  # (chiL, d, chiR)

        # 2. contract the running left environment into the (operator-dressed) ket
        T1 = E * reshape(Bop, chiL, d * chiR)             # (bra_bond, d*chiR)
        T1 = reshape(T1, size(E, 1) * d, chiR)            # (bra_bond*d, chiR)

        # 3. contract with the conjugated bra tensor to advance the environment
        Bc_mat = reshape(conj(B), chiL * d, chiR)          # (chiL*d, chiR)
        E = transpose(Bc_mat) * T1                         # (chiR, chiR)
    end
    val = E[1, 1]
    return real(val)
end

"""
    correlator_from_W_GS(W_sum::PO.PauliSum{N,T}, Bs) where {N,T} -> ComplexF64

Ground-state analogue of `correlator_from_W_ferro_complex`/
`correlator_vector_from_W_ferro_complex` from the 1D Heisenberg code, but for
a genuinely entangled state (loaded from CAMPS) instead of a classical
product state. Since there's no closed-form shortcut for an entangled state,
this sums coeff * <psi_GS|basis|psi_GS> over every term currently present in
W_sum -- expect this to be markedly slower per Trotter step than the
ferro/Neel bit-trick, since each term now costs an O(chi^3) contraction
instead of O(1).

Note this returns a COMPLEX number, not just its real part: each individual
`basis` is Hermitian, so `<psi|basis|psi>` is always real, but `coeff` is
generally complex here because `W_sum` is typically `W(t)*V_j`, a product of
two Hermitian operators that is itself non-Hermitian in general (exactly the
same reason `correlator_vector_from_W_ferro_complex` returns ComplexF64
rather than Float64).
"""
function correlator_from_W_GS(W_sum::PO.PauliSum{N,T}, Bs::Vector{Array{ComplexF64,3}}) where {N,T}
    total = zero(ComplexF64)
    for (basis, c) in W_sum
        total += c * expectation_value_mps(Bs, basis)
    end
    return total
end

"""
    single_site_pauli(N, site, pauli::Symbol)

Build a single-site PO.Pauli(N, ...) for `pauli` in (:X, :Y, :Z) at `site`.
Shared helper so the operator choice (XX vs YY vs ZZ correlators) is defined
in exactly one place.
"""
function single_site_pauli(N::Int, site::Int, pauli::Symbol)
    if pauli === :X
        return PO.Pauli(N, X=[site])
    elseif pauli === :Y
        return PO.Pauli(N, Y=[site])
    elseif pauli === :Z
        return PO.Pauli(N, Z=[site])
    else
        throw(ArgumentError("pauli must be :X, :Y, or :Z, got $pauli"))
    end
end

"""
    correlator_vector_from_W_GS(W_sum::PO.PauliSum{N,T}, Bs, N; pauli=:Z) where {N,T} -> Vector{ComplexF64}

Ground-state analogue of `correlator_vector_from_W_ferro_complex`: returns
C(t,j) = <psi_GS| W(t) P_j |psi_GS> for every site j = 1..N, where P is `pauli`
(:X, :Y, or :Z -- default :Z for ZZ correlators; pass :X for XX, :Y for YY).

Unlike the FM/Neel version, there is no shortcut that gets all N sites for the
price of one pass over W_sum -- each site requires forming the operator
product W_sum*P_j (via PauliSum*PauliSum) and then a full
`correlator_from_W_GS` evaluation, so this costs O(N) times what a single
`correlator_from_W_GS` call costs. For a large W_sum and/or large N this can
be the dominant cost of the whole Trotter loop -- profile before committing
to a long run.
"""
function correlator_vector_from_W_GS(W_sum::PO.PauliSum{N,T}, Bs::Vector{Array{ComplexF64,3}}; pauli::Symbol=:Z) where {N,T}
    C = zeros(ComplexF64, N)
    for j in 1:N
        Vj = PO.PauliSum(N, ComplexF64)
        Vj[PO.PauliBasis(single_site_pauli(N, j, pauli))] = 1.0 + 0im
        C[j] = correlator_from_W_GS(W_sum * Vj, Bs)
    end
    return C
end