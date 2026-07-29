#=
Exact dynamical correlator C_{i_site,j}(t) = <GS| P_{i_site}(t) P_j(0) |GS> for
the 13-site trigonal Kitaev cluster, computed EXACTLY (to Krylov-subspace
solver tolerance -- no Trotter/truncation error at all) using the sparse
Hamiltonian H and ground state |GS> already built by ED_N=13_Kitaev.jl. This is
the ground-truth reference for validating kitaev_correlation.jl's CAMPS +
Pauli-propagation dynamical correlators, in the exact same C_{i,j}(t) sense
(same operator ordering, same normalization) -- just on this smaller cluster
where exact methods are feasible (2^13 = 8192-dim Hilbert space).

The key simplification that keeps this "exact" tractable without ever forming
H's full spectrum: |GS> is an EXACT eigenstate of H (H|GS> = E0|GS>), so the
bra-side time evolution collapses to a trivial overall phase:

    <GS| e^{iHt} = (e^{-iHt}|GS>)^dagger = (e^{-iE0 t}|GS>)^dagger = e^{iE0 t} <GS|

Therefore
    C_j(t) = <GS| e^{iHt} P_i e^{-iHt} P_j |GS>
           = e^{iE0 t} <GS| P_i | a_j(t)>,     a_j(t) := e^{-iHt} P_j |GS>.

Only the KET a_j(t) needs to be time-propagated, via KrylovKit.exponentiate (a
Krylov-subspace matrix-exponential-action solver: it computes exp(t*A)*v using
only sparse matrix-vector products, never densifying H or diagonalizing it) --
so this stays a genuinely sparse computation, consistent with ED_N=13_Kitaev.jl.

Run from this directory so the project-local environment (KrylovKit pinned,
no DBF) is picked up:
    julia --project=. "ED_Kitaev_Correlation.jl"
=#

using SparseArrays
using LinearAlgebra
using KrylovKit
using NPZ
using Printf

include("ED_N=13_Kitaev.jl")   # brings in N, H, psi0, E0, V, h_field, SX/SY/SZ, I2, and the bond lists
# NOTE: H/psi0/E0 are built ONCE here, from whatever K/V/h_field are set at
# the top of ED_N=13_Kitaev.jl -- the sweep below only varies (i_site, pauli,
# dt, t_max), which is cheap to change per-config. Sweeping over DIFFERENT
# K/V/h_field values would mean rebuilding H and re-solving the ground state
# for each one (the expensive part, ~7s here but much more at larger N), so
# that's intentionally NOT part of this loop -- edit ED_N=13_Kitaev.jl's own
# K/V/h_field and rerun this whole file again if you need a different point.

out_dir = joinpath(@__DIR__, "correlation_outputs")
mkpath(out_dir)

pauli_mats = Dict(:X => SX, :Y => SY, :Z => SZ)

# Sparse single-site operator P_site, identity elsewhere -- same Kronecker-chain
# construction as embed_two_site (from ED_N=13_Kitaev.jl), just for one site.
function embed_one_site(op::SparseMatrixCSC{ComplexF64}, site::Int, N::Int)
    mats = [k == site ? op : I2 for k in 1:N]
    return foldl(kron, mats)
end

configurations = [
	# (i_site, pauli, dt, t_max)
	(8, :X, 0.05, 5.00)
	(11, :X, 0.05, 5.00)
	# (13, :Y, 0.05, 5.00)
    # (4, :Z, 0.05, 5.00)
	# (4, :X, 0.05, 5.00)
	# (4, :Y, 0.05, 5.00)
    # (2, :Z, 0.05, 5.00)
	# (2, :X, 0.05, 5.00)
	# (2, :Y, 0.05, 5.00)
    # (7, :Z, 0.05, 5.00)
	# (7, :X, 0.05, 5.00)
	# (7, :Y, 0.05, 5.00)

]

# Run every configuration in the list above against the SAME (already-built)
# H/psi0/E0. Wrapped in try/catch so one config's failure (e.g. Krylov
# non-convergence at some site) doesn't abort the whole sweep -- failures are
# collected and reported in the summary at the end instead.
failures = Tuple{Int,Any}[]
for (idx, config) in enumerate(configurations)
    i_site, pauli, dt, t_max = config
    pauli_label = Dict(:X => "XX", :Y => "YY", :Z => "ZZ")[pauli]

    println("\n" * "#"^60)
    println("# Configuration $idx / $(length(configurations)): " *
            "i_site=$i_site  pauli=$pauli  dt=$dt  t_max=$t_max")
    println("#"^60)
    println("Exact (Krylov) dynamical correlator, N=$N trigonal Kitaev cluster")
    println("  reference site = $i_site, pauli = $pauli, dt = $dt, t_max = $t_max")

    try
        P_i    = embed_one_site(pauli_mats[pauli], i_site, N)
        Pi_psi = P_i * psi0          # so <GS|P_i|.> = dot(Pi_psi, .), since P_i is Hermitian

        # Initial kets P_j|GS> for every target site j -- one Krylov propagation per site.
        a0 = [embed_one_site(pauli_mats[pauli], j, N) * psi0 for j in 1:N]

        steps      = floor(Int, t_max / dt)
        time_data  = collect(0:steps) .* dt
        C_mat      = Matrix{ComplexF64}(undef, steps + 1, N)

        t_start = time()
        for (step, t) in enumerate(time_data)
            phase = exp(im * E0 * t)
            for j in 1:N
                if t == 0.0
                    a_t = a0[j]
                else
                    a_t, einfo = exponentiate(H, -im * t, a0[j]; ishermitian = true, tol = 1e-12)
                    einfo.converged >= 1 || @warn "Krylov exponentiation did not converge to tol at t=$t, site j=$j (normres=$(einfo.normres))"
                end
                C_mat[step, j] = phase * dot(Pi_psi, a_t)
            end
            @printf("    t = %5.2f  C_%d%d(%s%s) = %+.5f %+.5fi\n",
                    t, i_site, i_site, pauli_label, pauli_label, real(C_mat[step, i_site]), imag(C_mat[step, i_site]))
            flush(stdout)
        end
        wall_time = time() - t_start
        @printf("\nDone in %.1f s\n", wall_time)

        @printf("Sanity check: C_%d%d(t=0) = %+.5f %+.5fi (expect 1.0, 0.0)\n",
                i_site, i_site, real(C_mat[1, i_site]), imag(C_mat[1, i_site]))

        out_path = joinpath(out_dir, "ED_kitaev_corr_N$(N)_$(pauli_label)_site$(i_site)_V$(V)_h$(h_field)_dt$(dt)_tmax$(t_max).npz")
        npzwrite(out_path, Dict(
            "time" => time_data,
            "C_re" => real.(C_mat),
            "C_im" => imag.(C_mat)))
        @printf("Saved: %s\n", out_path)
    catch e
        @warn "Configuration $idx failed -- continuing with the rest of the sweep" exception=e
        push!(failures, (idx, e))
    end
end

println("\n" * "="^60)
if isempty(failures)
    println("All $(length(configurations)) configurations completed successfully.")
else
    println("$(length(failures)) of $(length(configurations)) configurations FAILED:")
    for (idx, e) in failures
        println("  Configuration $idx: $(sprint(showerror, e))")
    end
end
println("="^60)
