# EXACT infinite-temperature OTOCs for the Kitaev honeycomb model
#
# Reference calculation for kitaev_OTOC_infinite_temp.jl (Pauli propagation).
# Same lattice, same Hamiltonian, same quantity -- but computed without any
# operator truncation, so the difference between the two IS Pauli propagation's
# truncation error. Both solvers include kitaev_model.jl, so the model they
# solve is guaranteed identical; nothing about the physics is redefined here.
#
#     julia --project=. ED_OTOC_infinite_temp.jl
#
# METHOD: STOCHASTIC TRACE ESTIMATION, NOT FULL DIAGONALIZATION
# -------------------------------------------------------------
# The quantity wanted is a trace,
#     F_ij(t) = Tr[W_i(t) V_j W_i(t) V_j] / 2^N ,      W_i(t) = U(t)^dag W_i U(t)
# and forming W_i(t) as an explicit matrix is hopeless: at N = 16 a dense
# 2^16 x 2^16 complex matrix is 68 GB. Full diagonalization is worse. Summing
# <b|A|b> over all 2^N basis states would be exact but needs 65536 separate time
# evolutions per output time, which is far slower than it is worth.
#
# So the trace is estimated stochastically (Hutchinson's estimator; in the
# physics literature the same thing is called quantum typicality or the
# thermal-pure-state method). For a random vector |r> with E[|r><r|] = I/2^N,
#     Tr[A]/2^N = E_r[ <r|A|r> ]
# exactly -- it is an unbiased estimator, not an approximation of the dynamics.
# Everything then acts on a single 2^N STATE VECTOR (1 MB at N = 16) instead of
# an operator. The Hamiltonian is never built as a matrix either: it is applied
# term by term straight from the PauliSum (apply_H!), so memory is O(2^N) rather
# than O(4^N).
#
# ACCURACY. The estimator's standard error per sample is ~2^(-N/2) -- about
# 4e-3 at N = 16 -- and falls as 1/sqrt(n_samples). It is measured, not assumed:
# the per-sample spread is accumulated and reported as "F_err" on every number,
# saved into the .npz, and printed at the end. Two structural points:
#   * Self-averaging. The error shrinks EXPONENTIALLY in system size, so this
#     method gets more accurate as N grows, exactly opposite to how ED usually
#     behaves. At N = 20 a single sample is already good to ~1e-3.
#   * Correlated across time. All output times reuse the SAME random vectors, so
#     the sampling error is smooth in t rather than independent point to point.
#     The SHAPE of a curve is therefore more accurate than the pointwise error
#     bar suggests, though its overall offset is not.
# Raise n_samples to drive the error down; cost is linear in it, and samples run
# in parallel across Julia threads (start julia with -t auto to use them).
#
# The estimator is written as an inner product of two states, which is both
# numerically stabler than a raw expectation value and lets the W(t)|r> piece be
# shared across all probe sites:
#     F_ij = < V_j W_i(t) r | W_i(t) V_j r >
# because (W(t) V)^dag = V^dag W(t)^dag = V W(t) for Hermitian W, V.
#
# This is "exact" in the sense that matters here -- no operator truncation, no
# Trotter error (Krylov exponentiation is converged to `krylov_tol`), no
# approximation of the dynamics. The only error is the statistical trace-sampling
# error, which is reported explicitly as a standard error on every number and is
# far below the truncation errors it exists to measure.
#
# COST -- WHAT ACTUALLY DRIVES IT
# ------------------------------
# W is inserted in the MIDDLE of the time-evolution chain, so W(t)|psi> cannot be
# marched forward incrementally the way a plain |psi(t)> can: each output time
# needs its own back-evolution over the full interval [0,t]. The forward-marched
# states ARE reused (one short dt step each per iteration), so the back-
# evolutions are the whole cost, and the figure of merit is the total simulated
# time they cover:
#     sum_{k=0}^{S} t_k  =  dt * S(S+1)/2  ~=  t_max * S / 2 ,   S = t_max/dt
# giving
#     cost ~ n_samples * (1 + n_probes) * t_max * S / 2 * n_terms * 2^N .
#
# THAT FORMULA IS A LOWER BOUND, NOT A FORECAST. Measured on 8 threads, 4
# probes, 6 samples, both to t_max = 3.0:
#     dt = 0.25  (13 output points)  ->      361 s
#     dt = 0.05  (61 output points)  ->   62089 s
# The formula predicts 4.7x for those; the truth was 172x, an effective S^3.
# The gap is implementation, not algorithm: `evolve` allocates a fresh 2^N
# vector per matrix-vector product, so at large workloads with several threads
# the garbage collector dominates. Switching to preallocated per-thread buffers
# would recover most of it. Until then, BENCHMARK A SHORT WINDOW AND EXTRAPOLATE
# CAUTIOUSLY -- and prefer a coarse output grid, which costs nothing in
# accuracy: unlike the Pauli-propagation solver, where dt is a Trotter step that
# controls accuracy, dt here is purely an output grid and Krylov evolution is
# exact at any step size.
#
# Probe count enters linearly via (1 + n_probes), so adding sites is cheap.

# PARALLEL BY DEFAULT =========================================================
# Julia starts with ONE thread unless told otherwise, and the thread count is
# fixed at startup -- it cannot be raised once the process is running. Since the
# sample loop below is threaded, a plain `julia ED_OTOC_infinite_temp.jl` would
# silently run every sample in series.
#
# So when this file is run AS A SCRIPT on a single thread, it relaunches itself
# with `-t auto` and exits. The check sits above every `include`/`using` on
# purpose: relaunching after loading the packages would pay that startup cost
# twice. A sentinel environment variable makes the handoff one-shot, so a
# machine that genuinely reports one CPU cannot ping-pong forever.
#
# When the file is instead `include`d from a REPL there is no process to
# replace, so it warns and continues rather than killing the user's session.
if Threads.nthreads() == 1 && get(ENV, "KITAEV_OTOC_RELAUNCHED", "") != "1"
    if abspath(PROGRAM_FILE) == @__FILE__
        @info "Single-threaded; relaunching with -t auto so samples run in parallel."
        # ARGS are forwarded so a configuration selection (see the RUN block)
        # survives the relaunch instead of silently reverting to "run all".
        cmd = addenv(`$(Base.julia_cmd()) --project=$(Base.active_project()) -t auto $(@__FILE__) $ARGS`,
                     "KITAEV_OTOC_RELAUNCHED" => "1")
        p = run(cmd; wait = true)
        exit(p.exitcode)
    else
        @warn """
            Julia is running with 1 thread, so ED samples will run in SERIES.
            This session cannot be upgraded -- thread count is fixed at startup.
            Fix for the VS Code REPL: the workspace .vscode/settings.json now sets
            "julia.NumThreads": "auto"; restart the Julia REPL to pick it up.
            Fix for a terminal: julia --project=. -t auto ED_OTOC_infinite_temp.jl
            """
    end
end

include(joinpath(@__DIR__, "kitaev_model.jl"))

using LinearAlgebra
using Printf
using NPZ
using Random
import KrylovKit


# PAULI ACTION ON A STATE VECTOR ==============================================
# PauliOperators stores a Pauli string as two bitmasks (x, z), with the per-site
# operator read off the bit pair: (0,0)->I, (1,0)->X, (1,1)->Y, (0,1)->Z. That
# is the same convention the MPS contraction in the correlation code used. As a
# product of single-site matrices this equals
#     P = i^(number of Y sites) * X^x Z^z
# (check one site: i * X*Z = i*[0 -1; 1 0] = [0 -i; i 0] = Y), so its action on a
# computational basis state |b> is a single term -- one phase and one bit flip:
#     P|b> = i^popcount(x&z) * (-1)^popcount(b&z) * |b xor x>
# Every operator below is applied through this one identity, which is why no
# matrix is ever formed.

struct PauliTerms
    xs::Vector{UInt64}
    zs::Vector{UInt64}
    cs::Vector{ComplexF64}   # coefficient, with the i^n_Y phase already folded in
end

function pauli_terms(H_sum)
    xs = UInt64[]; zs = UInt64[]; cs = ComplexF64[]
    for (b, c) in H_sum
        x = UInt64(b.x); z = UInt64(b.z)
        push!(xs, x); push!(zs, z)
        push!(cs, ComplexF64(c) * im^count_ones(x & z))
    end
    return PauliTerms(xs, zs, cs)
end

# out .= H * v, applied term by term. Never forms H.
function apply_H!(out::Vector{ComplexF64}, T::PauliTerms, v::Vector{ComplexF64})
    fill!(out, zero(ComplexF64))
    dim = length(v)
    @inbounds for k in eachindex(T.cs)
        x = T.xs[k]; z = T.zs[k]; c = T.cs[k]
        for b in UInt64(0):UInt64(dim - 1)
            amp = v[b + 1]
            iszero(amp) && continue
            s = isodd(count_ones(b & z)) ? -c : c
            out[(b ⊻ x) + 1] += s * amp
        end
    end
    return out
end

# out .= P * v for a single Pauli string given by its (x, z) bitmasks.
function apply_pauli!(out::Vector{ComplexF64}, x::UInt64, z::UInt64, v::Vector{ComplexF64})
    dim = length(v)
    phase = ComplexF64(im^count_ones(x & z))
    @inbounds for b in UInt64(0):UInt64(dim - 1)
        s = isodd(count_ones(b & z)) ? -phase : phase
        out[(b ⊻ x) + 1] = s * v[b + 1]
    end
    return out
end

apply_pauli(x::UInt64, z::UInt64, v::Vector{ComplexF64}) =
    apply_pauli!(similar(v), x, z, v)

# Bitmasks of a single-site Pauli, in the same (x, z) encoding.
function pauli_masks(n::Int, site::Int, pauli::Symbol)
    b = PO.PauliBasis(site_pauli(n, site, pauli))
    return UInt64(b.x), UInt64(b.z)
end


# TIME EVOLUTION ==============================================================
# exp(-i H tau) |v>, by Krylov (Lanczos, since H is Hermitian). `tau` may be
# negative -- that is the backward evolution U(t)^dag = exp(+i H t) that the
# Heisenberg-picture operator W(t) = U(t)^dag W U(t) requires.
function evolve(T::PauliTerms, v::Vector{ComplexF64}, tau::Float64;
                krylov_tol::Float64 = 1e-10, krylovdim::Int = 30)
    iszero(tau) && return copy(v)
    scratch = similar(v)
    Hop = w -> apply_H!(similar(scratch), T, w)
    result, info = KrylovKit.exponentiate(Hop, -im * tau, v;
                                          ishermitian = true,
                                          tol = krylov_tol,
                                          krylovdim = krylovdim)
    info.converged == 0 && @warn "Krylov exponentiation did not converge at tau=$tau (residual $(info.normres))"
    return result
end


# OTOC BY TYPICALITY ==========================================================
# Returns (F, Csq, F_err) for one time, each a length-(n_probes) vector, where
#   F[p]     = Tr[W V_p W V_p]/2^N          -- the OTOC, exactly 1 at t = 0
#   Csq[p]   = 2*(1 - F[p])                 -- squared commutator, 0 at t = 0
#   F_err[p] = standard error of F[p] over the random-state samples
#
# The identity Csq = 2*(1 - F) holds because W and V are Hermitian AND unitary
# (W^2 = V^2 = I for a Pauli string), which makes
#   [W,V]^dag [W,V] = 2I - WVWV - VWVW  ->  Tr[.]/2^N = 2 - 2F.
# This is the exact analogue of the PP code's Csq = 2*(norm2 - F); there norm2
# is only 1 up to truncation loss, here it is 1 identically, which is precisely
# the difference the comparison is measuring.

function run_ed_otoc(; Kxx = 1.0, Kyy = 1.0, Kzz = 1.0, V = 0.3, J = 0.005,
                       dt::Float64 = 0.05, t_max::Float64 = 3.0,
                       i_site::Int = 2, pauli::Symbol = :Z,
                       probe_sites::Vector{Int} = collect(1:N),
                       n_samples::Int = 8,
                       rng_seed::Int = 1234,
                       krylov_tol::Float64 = 1e-10,
                       out_dir::String = joinpath(@__DIR__, "otoc_outputs"),
                       tag::Union{Nothing,String} = nothing)

    n = N
    dim = 2^n

    println("="^70)
    println("EXACT infinite-temperature OTOC (typicality + Krylov, no truncation)")
    println("="^70)
    @printf("  N=%d  dim=2^%d=%d  dt=%.3f  t_max=%.2f  samples=%d\n", n, n, dim, dt, t_max, n_samples)
    @printf("  Kxx=%s Kyy=%s Kzz=%s  V=%s  J=%s  (no boundary field)\n", Kxx, Kyy, Kzz, V, J)
    check_periodic(n)

    # --- 1. Hamiltonian, from the SHARED definition in kitaev_model.jl -----
    H_sum = build_hamiltonian(; Kxx = Kxx, Kyy = Kyy, Kzz = Kzz, V = V, J = J, n = n)
    T = pauli_terms(H_sum)
    println("Hamiltonian terms: $(length(H_sum))")

    # Every term of this Hamiltonian is a Hermitian Pauli string with a real
    # coefficient, so H is Hermitian -- which the Lanczos path in `evolve`
    # assumes. Check it rather than trust it: a stray imaginary coefficient
    # would silently produce non-unitary "evolution" and a nonsense OTOC.
    for (b, c) in H_sum
        abs(imag(c)) < 1e-12 ||
            error("Hamiltonian term $b has imaginary coefficient $c -- H is not Hermitian")
    end

    # --- 2. Operators ------------------------------------------------------
    wx, wz = pauli_masks(n, i_site, pauli)
    probes = [(p, pauli_masks(n, p, pauli)...) for p in probe_sites]
    n_probes = length(probes)
    @printf("  W(0) = %s_%d   probing %d site(s): %s\n",
            pauli, i_site, n_probes, join(probe_sites, ","))

    steps = floor(Int, t_max / dt)
    time_data = [step * dt for step in 0:steps]
    # Every sample is kept separately rather than accumulated, so the spread
    # across samples gives a MEASURED standard error per (time, probe) instead
    # of a theoretical estimate -- and so samples can run independently in
    # parallel without contending over shared accumulators.
    F_samples = zeros(Float64, n_samples, steps + 1, n_probes)

    @printf("  Stochastic trace estimation: %d sample(s) on %d thread(s)\n",
            n_samples, Threads.nthreads())
    if n_samples > 1 && Threads.nthreads() == 1
        println("  (start julia with -t auto to run samples in parallel)")
    end

    # Progress is only reported once per SAMPLE, and samples run concurrently,
    # so a big run prints nothing for a long time and looks hung. State the
    # workload up front instead. The figure of merit is the total simulated time
    # that has to be back-evolved, sum_over_steps (1 + n_probes) * t_step, which
    # is what makes the cost quadratic in t_max/dt.
    work = (1 + n_probes) * dt * steps * (steps + 1) / 2
    @printf("  Workload: %d output times x %d back-evolutions = %.0f units of simulated time\n",
            steps + 1, 1 + n_probes, work)
    println("  (progress prints once per sample, at its completion -- no output until then)")
    flush(stdout)
    t_start = time()

    Threads.@threads for sample in 1:n_samples
        # Each sample gets its own independently seeded RNG, so results are
        # reproducible regardless of how many threads happen to be available
        # and of the order the scheduler runs them in.
        rng = MersenneTwister(rng_seed + sample)

        # Random phase vector: each amplitude is a random phase over sqrt(dim),
        # giving E[|r><r|] = I/dim exactly -- the unbiasedness the estimator
        # rests on. (Rademacher +/-1 vectors would work identically here; random
        # phases are used because they have no preferred basis direction, which
        # keeps the variance uniform across the Pauli operators being probed.)
        r = Vector{ComplexF64}(undef, dim)
        @inbounds for b in 1:dim
            r[b] = cis(2pi * rand(rng)) / sqrt(dim)
        end

        # Forward-marched states, advanced by one dt per step (cheap). These are
        # the only pieces that can be reused across output times.
        #   a      = U(t)|r>
        #   bs[p]  = U(t) V_p |r>
        a = copy(r)
        bs = [apply_pauli(px, pz, r) for (_, px, pz) in probes]

        for step in 0:steps
            t = step * dt

            # W(t)|r> = U(t)^dag W U(t)|r>, shared across every probe.
            g = evolve(T, apply_pauli(wx, wz, a), -t; krylov_tol = krylov_tol)

            for (pi, (_, px, pz)) in enumerate(probes)
                # left  = V_p W(t)|r>      right = W(t) V_p |r>
                left  = apply_pauli(px, pz, g)
                right = evolve(T, apply_pauli(wx, wz, bs[pi]), -t; krylov_tol = krylov_tol)
                F_samples[sample, step+1, pi] = real(dot(left, right))
            end

            if step < steps
                a = evolve(T, a, dt; krylov_tol = krylov_tol)
                for pi in 1:n_probes
                    bs[pi] = evolve(T, bs[pi], dt; krylov_tol = krylov_tol)
                end
            end
        end
        @printf("  sample %d/%d done (%.1f s elapsed)\n", sample, n_samples, time() - t_start)
        flush(stdout)
    end
    wall_time = time() - t_start

    F_mat = dropdims(sum(F_samples, dims = 1), dims = 1) ./ n_samples
    # Standard error of the mean across samples. With a single sample there is
    # no spread to measure, so this is NaN rather than 0 -- the honest statement
    # is "unmeasured, expected ~2^(-N/2)", not "exact".
    F_err = if n_samples > 1
        var_mat = dropdims(sum((F_samples .- reshape(F_mat, 1, size(F_mat)...)) .^ 2, dims = 1), dims = 1)
        sqrt.(var_mat ./ (n_samples * (n_samples - 1)))
    else
        fill(NaN, size(F_mat))
    end
    Csq_mat = 2 .* (1 .- F_mat)

    # --- 3. Sanity checks --------------------------------------------------
    # At t = 0, W(0) = P_i and V_p = P_p are both the same Pauli type, so they
    # commute on every site and F must be exactly 1 / Csq exactly 0. Any
    # deviation here is a bug in the Pauli-action or inner-product code, not a
    # physics result, so fail loudly rather than save a bad file.
    t0_dev = maximum(abs.(F_mat[1, :] .- 1.0))
    @printf("\n  Check F(t=0) == 1 for all probes: max deviation %.2e\n", t0_dev)
    t0_dev < 1e-10 || error("F(t=0) deviates from 1 by $t0_dev -- operator construction is wrong")

    @printf("  Stochastic trace error: expected ~%.1e per sample (2^(-N/2))", 2.0^(-n / 2))
    if n_samples > 1
        @printf(", measured stderr over %d samples: max %.1e",
                n_samples, maximum(filter(isfinite, F_err)))
    else
        print(" -- unmeasured with a single sample; raise n_samples to quantify it")
    end
    println()

    # --- 4. Save, in the SAME format as the Pauli-propagation driver -------
    # Identical keys and shapes, so the two files can be compared column by
    # column with no reshaping. "norm2" is identically 1 here by construction
    # (no truncation), which is exactly the baseline the PP run's norm2 decays
    # away from. "F_err" is extra and has no PP counterpart.
    safe_tag = tag === nothing ?
        # Same layout as the Pauli-propagation solver's filenames, so a PP/ED
        # pair differs only by the leading prefix and one segment. ED has no
        # truncation to record, so where PP writes "coeff<eps>_w<weight>" this
        # writes "exact" -- naming the absence rather than inventing a
        # threshold, which would read as a real setting.
        "N$(n)_$(lattice_tag())_exact_V$(V)_J$(J)_$(pauli)$(pauli)_site$(i_site)_dt$(dt)_tmax$(t_max)" :
        tag
    mkpath(out_dir)
    out_path = joinpath(out_dir, "ED_kitaev_otoc_infT_$(safe_tag).npz")
    npzwrite(out_path, Dict(
        "time"   => time_data,
        "sites"  => Float64.(probe_sites),
        "F"      => F_mat,
        "F_err"  => F_err,
        "Csq"    => Csq_mat,
        "norm2"  => ones(Float64, length(time_data)),
        "C_re"   => Csq_mat,
        "C_im"   => zeros(Float64, size(Csq_mat))))
    @printf("  Saved: %s  (wall time: %.1f s)\n", basename(out_path), wall_time)

    return time_data, probe_sites, F_mat, Csq_mat, F_err
end


# COMPARISON AGAINST PAULI PROPAGATION ========================================
# Reads the matching PP output (written by kitaev_OTOC_infinite_temp.jl with the
# same parameters) and prints a per-time comparison. The PP file stores the full
# N-site matrix; ED may have probed a subset, so columns are matched by site
# index via the "sites" key rather than by position.
function compare_to_pp(time_data, probe_sites, F_ed, F_err;
                       Kxx = 1.0, Kyy = 1.0, Kzz = 1.0, V = 0.0, J = 0.05,
                       dt = 0.05, pp_t_max = 3.0, i_site = 2, pauli = :Z,
                       out_dir::String = joinpath(@__DIR__, "otoc_outputs"),
                       show_site::Union{Nothing,Int} = nothing)

    # N (the lattice constant from kitaev_model.jl), not a local `n` -- this
    # function takes no size argument of its own.
    # PP filenames now also carry a truncation tag, which ED has no counterpart
    # for, so an exact name match is no longer possible -- glob on the rest and
    # report what turned up.
    pp_stem = "N$(N)_$(lattice_tag())_"
    pp_tail = "_V$(V)_J$(J)_$(pauli)$(pauli)_site$(i_site)_dt$(dt)_tmax$(pp_t_max).npz"
    matches = isdir(out_dir) ? filter(f -> startswith(f, "PP_kitaev_otoc_infT_" * pp_stem) &&
                                           endswith(f, pp_tail), readdir(out_dir)) : String[]
    if isempty(matches)
        println("\n  (No PP file matching PP_kitaev_otoc_infT_$(pp_stem)*$(pp_tail) -- run " *
                "kitaev_OTOC_infinite_temp.jl with the same parameters to compare.)")
        return nothing
    end
    if length(matches) > 1
        println("\n  $(length(matches)) PP runs match these parameters, differing only in " *
                "truncation:")
        for f in matches; println("      ", f); end
        println("  Comparing against the first; pass a different out_dir or move the others " *
                "aside to pick deliberately.")
    end
    pp_path = joinpath(out_dir, first(matches))
    @printf("\n  PP file: %s\n", basename(pp_path))

    pp = npzread(pp_path)
    F_pp, pp_time, pp_norm2 = pp["F"], pp["time"], pp["norm2"]
    site = show_site === nothing ? i_site : show_site
    col_ed = findfirst(==(site), probe_sites)
    col_ed === nothing && (println("\n  (site $site not among the ED probes)"); return nothing)

    println("\n" * "="^70)
    println("ED vs Pauli propagation -- F_$(i_site)$(site)($(pauli)$(pauli))")
    println("="^70)
    @printf("%8s  %12s  %12s  %12s  %10s\n", "t", "ED", "PP", "|PP - ED|", "PP norm2")
    println("-"^70)
    max_dev = 0.0
    for (k, t) in enumerate(time_data)
        # PP and ED share dt, so match by nearest time rather than assuming
        # identical grid lengths (PP usually runs to a larger t_max).
        kp = findmin(abs.(pp_time .- t))[2]
        abs(pp_time[kp] - t) < 1e-9 || continue
        dev = abs(F_pp[kp, site] - F_ed[k, col_ed])
        max_dev = max(max_dev, dev)
        @printf("%8.2f  %12.6f  %12.6f  %12.2e  %10.6f\n",
                t, F_ed[k, col_ed], F_pp[kp, site], dev, pp_norm2[kp])
    end
    println("-"^70)
    @printf("  Max |PP - ED| over the window: %.3e\n", max_dev)
    @printf("  (ED statistical error ~%.1e; deviations above that are PP truncation error)\n",
            maximum(filter(isfinite, F_err)))
    return max_dev
end


# RUN =========================================================================
# Runs whenever this file is loaded, by EITHER route -- as a script
# (`julia --project=. ED_OTOC_infinite_temp.jl`) or via `include(...)` from the
# REPL / VS Code's "execute file in REPL". It deliberately carries no
# `if abspath(PROGRAM_FILE) == @__FILE__` guard: under that guard an `include`
# loads the definitions and then silently does nothing, which is
# indistinguishable from the script being broken. This matches how
# kitaev_OTOC_infinite_temp.jl behaves.
#
# To load the functions WITHOUT running anything, comment out the call below.
#
# THIS FILE RUNS ONLY THE EXACT SOLVER. It never performs Pauli propagation --
# that lives entirely in kitaev_OTOC_infinite_temp.jl and is run separately.
# The optional comparison below does not compute anything either: it just READS
# a PP .npz already on disk and prints a table against it. It is off by default
# so this file does one job.
COMPARE_TO_SAVED_PP = false

# Timing, measured on 8 threads with 4 probes and 6 samples, both to t_max=3.0:
#     dt = 0.25  (13 output points)  ->      361 s   (~6 min)
#     dt = 0.05  (61 output points)  ->   62089 s   (~17.2 h)
# That is 172x for 4.7x the output points -- far worse than the linear-in-S
# scaling the cost model in the header predicts, and worse than quadratic.
# Treat the header's formula as a LOWER BOUND, not a forecast; the measured
# exponent is closer to S^3. The likely culprit is allocation, not algorithm:
# `evolve` builds a fresh 2^N vector on every matrix-vector product, so with
# several threads running concurrently the garbage collector dominates at large
# workloads. That is fixable with in-place buffers if a fine grid is needed
# often; until then, BENCHMARK A SHORT WINDOW BEFORE COMMITTING to a long one.

# CONFIGURATIONS ==============================================================
# Each entry is one full ED run, the counterpart of `configurations` in
# kitaev_OTOC_infinite_temp.jl. Entries are NAMED tuples rather than positional
# ones: ED takes more knobs than PP (probe_sites, n_samples), and a positional
# tuple makes it easy to put a J value in the V slot without any error.
#
# To compare against PP, keep V, J, i_site, pauli, dt and t_max identical to the
# PP configuration -- the filenames only line up when every one of those match.
# PROBE SITES ARE LATTICE-DEPENDENT -- RE-DERIVE THEM AFTER EDITING THE BONDS.
# A probe set is chosen to span bond DISTANCES from i_site (one site per shell,
# so the OTOC front can be watched moving outward). Site NUMBERS carry none of
# that: relabelling the graph moves every distance, and nothing in this file or
# in the filename fingerprint will complain -- the run simply measures different
# geometry than intended.
#
# That already happened once. The earlier probe set [2, 16, 12, 5] was picked on
# the previous bond lists (tag L243535, diameter 4), where it gave distances
# 0, 4, 2, 3 from site 2. On the CURRENT lattice (L24e521, diameter 5) the same
# four numbers give 0, 1, 5, 3 -- site 16 became a nearest neighbour and site 12
# landed at the far end.
#
# So the ladder is COMPUTED, not written down: probe_ladder(i) (kitaev_model.jl)
# returns one site per distance shell, nearest first, out to the diameter of
# whichever lattice KITAEV_LATTICE selected. On torus16 that is
# [2, 16, 15, 14, 13, 12] (distances 0..5); on torus32 it is
# [2, 3, 4, 5, 6, 12, 23] (distances 0..6). Hardcoding either one makes the
# other silently wrong -- the old literal [2,16,15,14,13,12] becomes distances
# 0,3,2,3,4,5 on torus32, with two probes in one shell and none at distance 1.
#
# COST: probes enter as (1 + n_probes), so the torus32 ladder is ~1.1x the
# torus16 one. For a cheaper sweep pass max_shell to stop early, e.g.
# probe_ladder(2; max_shell = 3), and accept that the outer shells go unmeasured.
const LADDER2 = probe_ladder(2)
println("Probe ladder from site 2 on $(LATTICE): ", LADDER2,
        "  (distances ", join(bond_distances(2)[LADDER2], ","), ")")

configurations = [
    (Kxx = 1.0, Kyy = 1.0, Kzz = 1.0, V = 0.0, J = 0.05, i_site = 2, pauli = :Z,
     dt = 0.05, t_max = 5.00, probe_sites = LADDER2, n_samples = 1),
    # (Kxx = 1.0, Kyy = 1.0, Kzz = 1.0, V = 0.3, J = 0.0, i_site = 2, pauli = :Z,
    #  dt = 0.05, t_max = 3.00, probe_sites = LADDER2, n_samples = 8),
    # (Kxx = 1.0, Kyy = 1.0, Kzz = 1.0, V = 0.3, J = 0.05, i_site = 2, pauli = :Z,
    #  dt = 0.05, t_max = 3.00, probe_sites = LADDER2, n_samples = 8),

    # --- configs 2-5: exact reference for the OTOC-PP cluster runs ------------
    # RUN THESE WITH  KITAEV_LATTICE=otocpp4x2  (see kitaev_model.jl). On the
    # default torus they would silently produce a different physical system --
    # the filename fingerprint would differ, so the plots would refuse them, but
    # the compute would be wasted.
    #
    # Matched to OTOC-PP/new_data/honeycomb_mguerre3: V = 0 (their model has no
    # three-body term), K = 1, W on site 1, probes 2,3,4,5 (bond distances
    # 1,2,3,4 from site 1 on THEIR numbering), t_max = 7.
    #
    # dt = 0.25 is the OUTPUT grid, not a Trotter step -- Krylov evolution is
    # exact at any spacing, and ED cost is superlinear in the number of output
    # points (measured: dt 0.25 -> 361 s vs dt 0.05 -> 62089 s to t_max 3). At
    # 29 points to t = 7 that is roughly 20-30 min per configuration; the PP
    # curves are smooth, so a coarse reference grid loses nothing.
    (Kxx = 1.0, Kyy = 1.0, Kzz = 1.0, V = 0.0, J = 0.1,  i_site = 1, pauli = :Z,
     dt = 0.25, t_max = 5.00, probe_sites = LADDER2, n_samples = 1),
    (Kxx = 1.0, Kyy = 1.0, Kzz = 1.0, V = 0.0, J = 0.0, i_site = 1, pauli = :Z,
     dt = 0.25, t_max = 5.00, probe_sites = LADDER2, n_samples = 1),
    (Kxx = 1.0, Kyy = 1.0, Kzz = 1.0, V = 0.0, J = 0.01, i_site = 1, pauli = :Z,
     dt = 0.25, t_max = 5.00, probe_sites = LADDER2, n_samples = 1),
    (Kxx = 1.0, Kyy = 1.0, Kzz = 1.0, V = 0.0, J = 1.0,  i_site = 1, pauli = :Z,
     dt = 0.25, t_max = 5.00, probe_sites = LADDER2, n_samples = 1),
]

# WHICH CONFIGURATIONS RUN
#   julia --project=. -t auto ED_OTOC_infinite_temp.jl          -> all, one after another
#   julia --project=. -t auto ED_OTOC_infinite_temp.jl 2        -> only configuration 2
#   julia --project=. -t auto ED_OTOC_infinite_temp.jl 1,3      -> configurations 1 and 3
#   SLURM job array (#SBATCH --array=1-3)                       -> one per array task
#
# To run configurations AT THE SAME TIME, give each its own job, not one job
# with the whole list. A single ED run already uses every thread it is given
# (its samples run in parallel), so a list inside one job always runs in
# sequence. A job array gives each configuration its own node or cores:
# SLURM_ARRAY_TASK_ID is read automatically when no argument is passed.
# Do not start several at once on the same cores -- they would contend for
# the same threads and each would run slower, not faster.
function selected_configs(n_cfg::Int)
    spec = !isempty(ARGS)                          ? ARGS[1] :
           haskey(ENV, "SLURM_ARRAY_TASK_ID")      ? ENV["SLURM_ARRAY_TASK_ID"] :
           nothing
    spec === nothing && return collect(1:n_cfg)
    idx = try
        parse.(Int, split(spec, ','; keepempty = false))
    catch
        error("Could not read configuration selection \"$spec\". Pass a number (2) " *
              "or a comma-separated list (1,3).")
    end
    bad = filter(i -> !(1 <= i <= n_cfg), idx)
    isempty(bad) || error("Configuration(s) $(join(bad, ", ")) do not exist; there are " *
                          "$n_cfg configuration(s) in the list.")
    return idx
end

to_run = selected_configs(length(configurations))
println("Running configuration(s) $(join(to_run, ", ")) of $(length(configurations)) " *
        "on $(Threads.nthreads()) thread(s).")

# Wrapped in try/catch so one configuration's failure does not abort the rest,
# matching the PP script. Failures are reported in the summary at the end.
failures = Tuple{Int,Any}[]
completed = NamedTuple[]
for idx in to_run
    cfg = configurations[idx]
    phys = (Kxx = cfg.Kxx, Kyy = cfg.Kyy, Kzz = cfg.Kzz, V = cfg.V, J = cfg.J,
            i_site = cfg.i_site, pauli = cfg.pauli)

    println("\n" * "#"^70)
    println("# ED configuration $idx / $(length(configurations)): " *
            "V=$(cfg.V)  J=$(cfg.J)  i_site=$(cfg.i_site)  pauli=$(cfg.pauli)  " *
            "dt=$(cfg.dt)  t_max=$(cfg.t_max)")
    println("# probes=$(join(cfg.probe_sites, ","))  samples=$(cfg.n_samples)")
    println("#"^70)

    try
        t0 = time()
        time_data, probe_sites, F_ed, Csq_ed, F_err = run_ed_otoc(;
            phys...,
            dt = cfg.dt, t_max = cfg.t_max,
            probe_sites = cfg.probe_sites,
            n_samples = cfg.n_samples)
        wall = time() - t0

        if COMPARE_TO_SAVED_PP
            compare_to_pp(time_data, probe_sites, F_ed, F_err;
                          phys..., dt = cfg.dt, pp_t_max = cfg.t_max)
        end

        errs = filter(isfinite, F_err)
        push!(completed, (idx = idx, cfg = cfg, wall = wall,
                          max_err = isempty(errs) ? NaN : maximum(errs)))
    catch e
        @warn "ED configuration $idx failed -- continuing with the rest" exception = (e, catch_backtrace())
        push!(failures, (idx, e))
    end
end

# End-of-sweep summary: what each run cost and how precise it came out, so the
# numbers don't have to be reconstructed by scrolling back through the log.
println("\n" * "="^86)
if !isempty(completed)
    println("ED runs completed:")
    @printf("%4s %6s %6s %5s %6s %6s %8s %13s %12s\n",
            "cfg", "V", "J", "site", "dt", "t_max", "samples", "max stderr", "wall time")
    println("-"^86)
    for r in completed
        c = r.cfg
        @printf("%4d %6.3g %6.3g %5d %6.3g %6.3g %8d %13.2e %10.1f s\n",
                r.idx, c.V, c.J, c.i_site, c.dt, c.t_max, c.n_samples, r.max_err, r.wall)
    end
    println("-"^86)
    println("max stderr is the stochastic-trace error; ED has no truncation or Trotter error.")
end
if isempty(failures)
    println("\nAll $(length(to_run)) selected ED configuration(s) completed successfully.")
else
    println("\n$(length(failures)) of $(length(to_run)) selected ED configuration(s) FAILED:")
    for (idx, e) in failures
        println("  Configuration $idx: $(sprint(showerror, e))")
    end
end
println("="^86)


