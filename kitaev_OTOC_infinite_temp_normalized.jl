# Infinite-temperature OTOCs for the Kitaev 2d Honeycomb model
#
# Reads no input data files. Run it directly:
#     julia --project=. kitaev_OTOC_infinite_temp.jl
#
# The lattice and Hamiltonian come from kitaev_model.jl, shared verbatim with
# the exact reference solver ED_OTOC_infinite_temp.jl. Edit the MODEL there.
#
# WHY NO CAMPS, NO MPS, NO EXPORTED FILES
# ---------------------------------------
# An infinite-temperature OTOC is a TRACE over the full Hilbert space, so it is
# independent of any state and of any choice of basis. Two consequences:
#
#   1. There is no ground state to load. The CAMPS MPS export, the MPS
#      contraction machinery, and the O(chi^3) expectation values that
#      Correlations/kitaev_correlation.jl needs are all irrelevant here.
#   2. There is no Clifford frame to respect. Conjugating both W and V by the
#      same Clifford leaves Tr[W V W V]/2^N unchanged, so the exported
#      "rotated" Hamiltonian and "rotated" single-site operators are
#      unnecessary -- we build the PHYSICAL Hamiltonian directly from the bond
#      lists in kitaev_model.jl and probe it with PLAIN single-site Paulis. The
#      numbers are identical to what a rotated-frame calculation would give.
#
# The only thing inherited from the correlation code is the Heisenberg-picture
# Trotter propagation of W(t), which is unchanged.
#
# WHY THE MEASUREMENT IS ALMOST FREE AT BETA = 0
# ----------------------------------------------
# Write the evolved operator in the Pauli basis, W_i(t) = sum_k c_k P_k, and let
# V_j be a single Pauli string. Every Pauli either commutes or anticommutes with
# V_j, so P_k V_j P_k V_j = s_k * I with s_k = +1 (commute) / -1 (anticommute),
# and the infinite-temperature trace kills every k != l cross term:
#
#     F_ij(t) = Tr[W_i(t) V_j W_i(t) V_j] / 2^N  =  sum_k |c_k|^2 s_k
#             =  norm2 - 2 * sum_{k anticommuting with V_j} |c_k|^2
#
#     C_ij(t) = Tr[ [W_i(t),V_j]^dag [W_i(t),V_j] ] / 2^N
#             =  4 * sum_{k anticommuting with V_j} |c_k|^2   =  2*(norm2 - F_ij)
#
# where norm2 = sum_k |c_k|^2. The entire measurement is a commutation BIT-TEST
# per Pauli term -- O(N * |W|) integer operations per time step. No state, no
# MPS contraction, no operator products, nothing quadratic in |W|. This is
# strictly CHEAPER than the two-point function in kitaev_correlation.jl, which
# pays an O(N * |W| * chi^3) MPS contraction at every step.
#
# F starts at 1 and C starts at 0 (at t=0, W = V_i commutes with every V_j for
# j != i, and with itself). C_ij(t) rising above 0 is the operator front
# reaching site j -- that is the light cone.
#
# RENORMALIZED VARIANT -- HOW THIS DIFFERS FROM kitaev_OTOC_infinite_temp.jl
# ---------------------------------------------------------------------------
# This is a copy of that solver with ONE change: after every Trotter timestep
# W(t) is rescaled to unit norm,
#
#     W  <-  W / sqrt(sum_k |c_k|^2)
#
# so norm2 == 1 at every output time, as it would be under exact evolution.
#
# WHY THAT COSTS YOU THE ERROR DIAGNOSTIC
# ---------------------------------------
# In the original solver norm2 IS the error measure: exact evolution preserves
# it, so any shortfall is exactly the operator weight truncation discarded, and
# "norm2 has fallen to 0.7" means "this curve is 30% worth of wrong". After
# renormalization norm2 is 1 BY CONSTRUCTION. Printing it verifies the rescaling
# fired; it says nothing whatever about accuracy. Do not read a flat norm2 = 1
# here as evidence that a run is converged -- it is true of every run, including
# the worst.
#
# WHAT REPLACES IT. The discarded weight is still measured, just not from the
# post-step norm:
#
#   "retention"     -- the norm2 measured immediately AFTER each Trotter step
#                      and BEFORE rescaling. The fraction of weight that step
#                      kept. This is the per-step analogue of the old norm2.
#   "cum_retention" -- the running product of those factors. The closest
#                      analogue of the old norm2(t) curve, and what to plot
#                      beside the OTOC in place of it.
#
# cum_retention is NOT identical to the norm2 an un-normalized run would report:
# rescaling changes which terms survive the next step's |c| threshold, so the
# two runs follow different trajectories. It is the retained fraction along THIS
# trajectory, which is the honest statement.
#
# RELATION TO THE ÷norm2 CORRECTION. Dividing C by norm2 after the fact assumes
# the discarded weight split between commuting and anticommuting strings in the
# same ratio as the weight that survived. Renormalizing every step makes that
# same assumption, but applies it DURING propagation, so it also feeds back into
# the dynamics. The two are therefore related but not equal, and a run from this
# file should not be compared against a ÷norm2-corrected run as though it were
# the same scheme.

using Printf   # kitaev_model.jl's check_periodic uses @printf

# The lattice (N, X/Y/Z_BONDS), the Hamiltonian, and the single-site Pauli
# helper all live in kitaev_model.jl, shared verbatim with the exact-
# diagonalization reference solver ED_OTOC_infinite_temp.jl. Edit the MODEL
# there, not here -- the whole value of the ED comparison rests on both solvers
# reading one definition, so that any disagreement between them is truncation
# error rather than a bond list that was updated in only one file.
include(joinpath(@__DIR__, "kitaev_model.jl"))

import DBF as DBF
using LinearAlgebra
using NPZ
using Random



# TRUNCATION STRATEGIES =====================================================
# Unchanged from the correlation code: these control how aggressively W(t) is
# pruned as it grows under Heisenberg evolution. Whatever they discard shows up
# directly as a drop in norm2 (see the header).

#Basic coefficient truncation strategy: remove Pauli terms with |c| <= thresh.
function coeff_clip(ps::PO.PauliSum{N}; thresh=1e-16) where {N}
	ps_=deepcopy(ps)
    return filter(p->abs(p.second) > thresh, ps), filter(p->abs(p.second) <= thresh, ps_)
end

function weight(p::PO.PauliBasis)
	# PauliOperators stores X and Z parts as bit-integers.
	# OR-ing them gives a bitmask of sites that are non-identity.
	return count_ones(p.x | p.z)
end

#In-place filter to remove terms with Pauli weight > max_weight.
function weight_clip!(ps::PO.PauliSum{N}, max_weight::Int) where {N}
	# ps is a Dictionary (PauliBasis => Coefficient).
	# We use filter! to keep only keys (p.first) that satisfy the condition.
	return filter!(p -> weight(p.first) <= max_weight, ps)
end

# Majorana weight is defined as the number of Majorana operators in the
# Jordan-Wigner representation of a Pauli operator.
# This function truncates Pauli terms based on their Majorana weight.
function majorana_weight(Pb::Union{PO.PauliBasis{N}, PO.Pauli{N}}) where N
	w = 0
	control = true # Tracks the parity of the Jordan-Wigner string (True=Outside, False=Inside)

	# Extract bitmasks for I, Z, and X components
	# (Note: In PauliOperators, internal storage is typically UInt128 )
	Ibits = ~(Pb.z | Pb.x)
	Zbits = Pb.z & ~Pb.x

	# We iterate from site N down to 1 (mirroring the JW string definition)
	for i in reverse(1:N)
		xbit = (Pb.x >> (i - 1)) & 1 != 0      # True if op at i is X or Y
		Zbit = (Zbits >> (i - 1)) & 1 != 0     # True if op at i is Z
		Ibit = (Ibits >> (i - 1)) & 1 != 0     # True if op at i is I

		# Logic: If we see a Z but we are "outside" the string (control=true), it's a new fermion.
		# If we see an I but we are "inside" the string (control=false), it's a hole (costly).
		if (Zbit && control) || (Ibit && !control)
			w += 2
		elseif xbit
			# X or Y operators toggle the string requirement
			control = !control
			w += 1
		end
	end
	return w
end

#Same but for Majorana weight, which is more relevant for fermionic systems.
function majorana_weight_clip!(ps::PO.PauliSum{N}, max_weight::Int) where {N}
	return filter!(p -> majorana_weight(p.first) <= max_weight, ps)
end



#     WeightDampedTruncation(alpha, thresh)

# Truncation strategy: remove Pauli terms where |c|·exp(-alpha·weight) ≤ thresh.
# - alpha=0  →  plain coefficient clip at thresh
# - large alpha  →  approaches a hard weight cutoff

# `kind` selects WHICH weight is damped:
#
#   :pauli    -- weight(p), the number of non-identity sites. Damps how far the
#                string reaches across the lattice.
#   :majorana -- majorana_weight(p), the number of Majorana operators in the
#                JORDAN-WIGNER image. Damps how much JW string the operator
#                drags, which is not the same thing as how far it reaches.
#
# THE TWO SCALES DIFFER, so equal alpha does NOT mean equal severity. A
# single-site Z has Pauli weight 1 but JW Majorana weight 2, and the gap widens
# with reach -- so at the same alpha the Majorana form damps much harder.
# Measured at alpha=0.5, thresh=1e-4, t=0.3: Pauli lost 0.08% of the norm,
# Majorana lost 3.15%. Calibrate to equal norm2 before comparing them, never to
# equal alpha.
#
# NOTE: the argument that "K and V are quadratic in Majoranas so only J grows
# Majorana weight" belongs to Kitaev's FOUR-Majorana representation in a fixed
# Z2 gauge sector. It does NOT transfer to this Jordan-Wigner counting, and the
# numbers above are what that mismatch looks like. Do not expect :majorana to be
# the J-adapted knob on the strength of that argument alone.
#
# The two are DIFFERENT approximations. They get different filename tags and a
# different code in the .npz so they can never be silently mixed on one axis.
const DAMP_WEIGHT_KINDS = (:pauli, :majorana)

struct WeightDampedTruncation
    alpha::Float64
    thresh::Float64
    kind::Symbol
    # Inner constructor so a typo (:majorna) fails HERE rather than falling
    # through to the wrong weight function and quietly producing a run that
    # looks fine and means something else.
    function WeightDampedTruncation(alpha::Real, thresh::Real, kind::Symbol)
        kind in DAMP_WEIGHT_KINDS ||
            error("WeightDampedTruncation kind must be one of " *
                  "$(DAMP_WEIGHT_KINDS); got :$kind")
        return new(Float64(alpha), Float64(thresh), kind)
    end
end
# Defaulting to :pauli keeps the plain two-argument form meaning what it did
# before this field existed.
WeightDampedTruncation(alpha::Real, thresh::Real) =
	WeightDampedTruncation(alpha, thresh, :pauli)
WeightDampedTruncation(alpha::Real) = WeightDampedTruncation(alpha, 1e-3, :pauli)

# Same (kept, clipped) contract as CombinedTruncation -- the Trotter step does
# `W, _ = truncate(W)`, so anything callable here must return BOTH halves.
#
# The test is on |c|*exp(-alpha*w), NOT on |c| alone: a high-weight string must
# carry exponentially more amplitude to survive. This is a SOFT version of
# weight_clip! / majorana_weight_clip!, and the point is that it has no cliff --
# a hard cap discards a weight-(k+1) string of amplitude 0.9 while keeping a
# weight-k string of amplitude 1e-9, which is exactly backwards.
function (t::WeightDampedTruncation)(W)
	W_ = deepcopy(W)
	# Branch ONCE here, not once per term: the kind is fixed for the whole run.
	wf = t.kind === :majorana ? majorana_weight : weight
	keep(p) = abs(p.second) * exp(-t.alpha * wf(p.first)) > t.thresh
	kept    = filter(keep, W)
	clipped = filter(!keep, W_)
	return kept, clipped
end

# How it prints in logs and string interpolation, matching CombinedTruncation so
# the sweep summary table stays readable. The kind is named because two runs
# differing only in it are otherwise indistinguishable in the log.
Base.show(io::IO, t::WeightDampedTruncation) =
	print(io, "|c|*exp(-", t.alpha, "*", t.kind, "_weight) > ", t.thresh)

make_weight_damped_truncation(alpha::Real, thresh::Real; kind::Symbol = :pauli) =
	WeightDampedTruncation(alpha, thresh, kind)
# Explicit aliases, for config lists where naming the kind at the call site is
# clearer than a trailing keyword.
make_pauli_damped_truncation(alpha::Real, thresh::Real) =
	WeightDampedTruncation(alpha, thresh, :pauli)
make_majorana_damped_truncation(alpha::Real, thresh::Real) =
	WeightDampedTruncation(alpha, thresh, :majorana)

#Combined truncation strategy: remove Pauli terms based on both pauli and coefficient criteria.
#
# A callable STRUCT rather than an anonymous closure, so a run can report the
# settings it actually used. A closure captures thresh/max_weight where nothing
# can read them back, so every log line and filename described its truncation as
# "#make_combined_truncation##0" -- useless for the one thing this study varies.
# Call syntax is unchanged: `truncate(W)` still returns (kept, clipped).
struct CombinedTruncation
	thresh::Float64
	max_weight::Int
end

function (t::CombinedTruncation)(W)
	W_ = deepcopy(W)
	keep(p) = abs(p.second) > t.thresh && weight(p.first) <= t.max_weight
	kept    = filter(keep, W)
	clipped = filter(!keep, W_)
	return kept, clipped
end

# How it prints in logs and string interpolation.
Base.show(io::IO, t::CombinedTruncation) =
	print(io, "|c| > ", t.thresh, " and Pauli weight <= ", t.max_weight)

# Filename-safe rendering of a coefficient threshold: 1e-3 -> "1e3",
# 5e-3 -> "5e3", 1e-12 -> "1e12". The exponent's SIGN is dropped, which is
# unambiguous here only because every threshold is below 1 -- a threshold above
# 1 would collide with its reciprocal. The mantissa/exponent split is computed
# rather than taken from log10 alone, because log10(1e-3) can land a hair below
# -3 and floor() would then produce "10e4" instead of "1e3".
function coeff_tag(x::Real)
    x == 0 && return "0"
    e = floor(Int, log10(abs(x)))
    m = abs(x) / 10.0^e
    if m >= 10 - 1e-9        # guard the floating-point edge described above
        m /= 10; e += 1
    end
    mstr = isapprox(m, round(m); atol = 1e-9) ? string(Int(round(m))) :
                                                string(round(m, digits = 3))
    return "$(mstr)e$(abs(e))"
end

# Compact filename-safe form: "coeff<threshold>_w<max Pauli weight>", e.g.
# "coeff1e3_w12". Falls back for any other callable so a hand-written truncation
# still produces a valid (if vague) filename rather than crashing the save.
trunc_tag(t::CombinedTruncation) = "coeff$(coeff_tag(t.thresh))_w$(t.max_weight)"
# Damped runs need BOTH knobs in the name. The "customtrunc" fallback below is
# the same string for every alpha, so two damped runs differing only in alpha
# would collide and one would be silently overwritten -- the exact failure the
# comment above this block warns about.
# "wdamp" = Pauli-weight damping, "mwdamp" = Majorana-weight damping. The two
# are different approximations, so they must not share a filename either.
trunc_tag(t::WeightDampedTruncation) =
	"$(t.kind === :majorana ? "mwdamp" : "wdamp")$(t.alpha)_c$(coeff_tag(t.thresh))"
trunc_tag(t) = "customtrunc"

make_combined_truncation(thresh::Float64, max_weight::Int) =
	CombinedTruncation(thresh, max_weight)


# This function order the Hamiltonian terms in different ways, depending on the order_type argument.
function build_H_terms(H_total_sum; order_type::String = "lexical", rng_seed::Int = 1234)
	H_vec = collect(H_total_sum)

	site_pair(basis) = begin
		zbits = basis.z
		xbits = basis.x
		mask = zbits | xbits
		sites = Int[]
		i = 1
		while mask != 0
			if (mask & 0x1) == 1
				push!(sites, i)
			end
			mask >>= 1
			i += 1
		end
		if isempty(sites)
			(typemax(Int), typemax(Int))
		elseif length(sites) == 1
			(sites[1], sites[1])
		else
			(minimum(sites), maximum(sites))
		end
	end

	if order_type == "lexical"
		H_terms = sort(H_vec, by = p -> (p.first.z, p.first.x))

	elseif order_type == "by_site"
		H_terms = sort(H_vec, by = p -> site_pair(p.first))

	elseif order_type == "by_kind"
		function kind_code(basis)
			has_x = (basis.x != 0)
			has_z = (basis.z != 0)
			if has_z && !has_x
				return 1
			elseif has_x && !has_z
				return 2
			elseif has_x && has_z
				return 3
			else
				return 4
			end
		end
		H_terms = sort(H_vec, by = p -> (kind_code(p.first), site_pair(p.first)))

	elseif order_type == "random"
		rng = MersenneTwister(rng_seed)
		H_terms = shuffle(rng, H_vec)

	else
		error("Unknown order_type = $order_type")
	end

	return H_terms
end


# TROTTER EVOLUTION ===========================================================
# Heisenberg-picture propagation of W(t), Hamiltonian-agnostic and identical to
# the correlation code. Second-order (forward half-step then reversed backward
# half-step), applying one Hamiltonian term at a time and truncating after each.
#
# The `V_ops` / `error_accumulation` arguments are vestigial: they existed to
# accumulate per-step truncation error against a ground-state two-point
# measurement, which is not the quantity this script computes. Leave
# error_accumulation=false (the default). The norm2 diagnostic described in the
# file header is the correct truncation-error measure here, and it is always
# computed for free by otoc_vector_infT.
function trotter_step_2nd_order_DBf_commutationwise!(
						W::PO.PauliSum{N, T},
						H_terms::Vector{Pair{PO.PauliBasis{N}, T}},
						dt::Float64;
						cutoff = 1e-12,
						truncate = nothing,
) where {N, T}

	# --- 1. Forward half-step ---
	for (H_basis, h_coeff) in H_terms
		h_real = real(h_coeff)
		Î¸ = h_real * dt
		W = DBF.evolve(W, H_basis, Î¸)
		DBF.coeff_clip!(W, thresh = 1e-12)
		W, _ = coeff_clip(W, thresh = cutoff)

		if !isnothing(truncate)
			W, _ = truncate(W)
		end
	end

	# --- 2. Backward half-step ---
	for (H_basis, h_coeff) in Iterators.reverse(H_terms)
		h_real = real(h_coeff)
		Î¸ = h_real * dt
		W = DBF.evolve(W, H_basis, Î¸)

		DBF.coeff_clip!(W, thresh = 1e-12)
		W, _ = coeff_clip(W, thresh = cutoff)

		if !isnothing(truncate)
			W, _ = truncate(W)
		end
	end

	return W
end


# INFINITE-TEMPERATURE OTOC MEASUREMENT =======================================
# Replaces the ground-state two-point measurement used by
# Correlations/kitaev_correlation.jl. See the derivation in the file header.

# otoc_vector_infT computes, for every site j, the infinite-temperature OTOC and
# squared commutator of W_sum = W_i(t) against the probe V_j, plus the shared
# norm2 truncation diagnostic. Cost: one PO.commute bit-test per (Pauli term,
# site) pair -- O(N * |W|), no state and no MPS contraction.
#
# The beta = 0 sign trick requires each probe V_j to be a SINGLE Pauli string:
# only then does every term of W(t) either commute or anticommute with it,
# giving the s_k = +/-1 that makes the trace collapse to a diagonal sum. Passing
# V_bases as PauliBasis (not PauliSum) makes that requirement structural.
#
# Returns (F, Csq, norm2) with
#   F[j]   = Tr[W V_j W V_j]/2^N                          -- the OTOC, starts at 1
#   Csq[j] = Tr[[W,V_j]^dag [W,V_j]]/2^N = 2*(norm2-F[j]) -- squared commutator,
#            starts at 0 and grows as the operator front reaches site j
#   norm2  = 1 under exact evolution; its decay measures truncation error
function otoc_vector_infT(W_sum::PO.PauliSum{N,T},
                          V_bases::Vector{PO.PauliBasis{N}}) where {N,T}
    n_sites = length(V_bases)
    anti = zeros(Float64, n_sites)   # anti[j] = sum of |c_k|^2 over terms anticommuting with V_j
    norm2 = 0.0

    #this is the main loop that computes the anti-commutation and norm2 values
    for (basis, c) in W_sum
        w = abs2(c)
        norm2 += w
        for j in 1:n_sites
            if !PO.commute(basis, V_bases[j])
                anti[j] += w
            end
        end
    end

    F   = norm2 .- 2 .* anti
    Csq = 4 .* anti
    return F, Csq, norm2
end



# DRIVER ======================================================================
# Builds the Hamiltonian from the bond lists at the top of this file, evolves
# W(0) = P_i_site in the Heisenberg picture, and measures the beta = 0 OTOC
# against every site at every step.
function run_otoc_infT(n::Int;
                       Kxx::Float64 = 1.0, Kyy::Float64 = 1.0, Kzz::Float64 = 1.0,
                       V::Float64 = 0.3, J::Float64 = 0.0,
                       dt::Float64 = 0.05, t_max::Float64 = 5.0,
                       cutoff::Float64 = 1e-12,
                       truncation_strategy = make_combined_truncation(1e-5, 22),
                       order_type::String = "lexical",
                       i_site::Int = 1,
                       pauli::Symbol = :Z,
                       out_dir::String = joinpath(@__DIR__, "otoc_outputs"),
                       tag::Union{Nothing,String} = nothing)

    println("="^60)
    println("Infinite-temperature OTOC (Trotter Order: 2, commutation-wise)")
    println("="^60)
    println("  N=$n  dt=$dt  t_max=$t_max")
    println("  Kxx=$Kxx Kyy=$Kyy Kzz=$Kzz  V=$V  J=$J  (no boundary field)")
    # BOTH truncation knobs, because they are what this study varies and a run
    # is not reproducible without them. They act in sequence on every Trotter
    # sub-step: `cutoff` first (a plain coefficient clip inside
    # trotter_step_2nd_order_DBf_commutationwise!), then `truncation_strategy`.
    # Whatever they discard shows up as the norm2 shortfall reported below, so
    # these two lines and the final norm2 are what a curve should be read
    # against.
    println("  Truncation:")
    println("    per-step coefficient cutoff : |c| > $cutoff")
    println("    strategy                    : $truncation_strategy")

    # --- 1. Build the Hamiltonian from the SHARED definition ---
    # Periodic boundary conditions mean every site carries exactly one bond of
    # each color. Since the boundary field that used to compensate for missing
    # edge bonds is gone, a site short a bond is now genuinely dangling rather
    # than field-corrected -- check_periodic names any such site, and the usual
    # cause is bond lists not yet updated from the open cluster to the wrapped
    # one. Both this solver and the ED reference call the same check on the same
    # lists, so they can never disagree about the lattice.
    open_sites = check_periodic(n)
    if !isempty(open_sites)
        @warn "Lattice is not periodic ($(length(open_sites)) site(s) short a bond -- listed above). " *
              "Those sites have free, uncompensated boundaries now that the boundary field is gone. " *
              "Fix X_BONDS/Y_BONDS/Z_BONDS in kitaev_model.jl."
    end

    H_sum = build_hamiltonian(; Kxx = Kxx, Kyy = Kyy, Kzz = Kzz, V = V, J = J, n = n)
    H_terms = build_H_terms(H_sum; order_type = order_type)
    println("System Size: N = $n  |  Hamiltonian terms: $(length(H_sum))")

    # --- 2. Define operators ---
    println("\nInitializing W(0) = $(pauli)_$(i_site)...")
    W0 = site_pauli(n, i_site, pauli)
    W_sum = PO.PauliSum(n, ComplexF64)
    W_sum += 1.0 * W0

    # Probe operators V_j for every site j, extracted once so the per-step
    # measurement is a pure bit-test loop with no allocation.
    V_bases = [PO.PauliBasis(site_pauli(n, j, pauli)) for j in 1:n]

    # If W(0) commutes with every term of H it is an exact conserved quantity
    # and never evolves: W(t) = W(0) for all t, so Csq stays identically 0 at
    # every site and every time and F stays identically 1. That is a degenerate
    # (and uninformative) run, so catch it up front rather than after 100 steps.
    W0_basis = PO.PauliBasis(W0)
    if all(PO.commute(W0_basis, h_basis) for (h_basis, _) in H_sum)
        @warn "W(0) = $(pauli)_$i_site commutes with every term of the Hamiltonian -- " *
              "W(t) will stay exactly W(0) forever and the OTOC will be trivially flat. " *
              "Pick a different i_site or a different pauli."
    end

    # --- 3. Simulation constants ---
    steps = floor(Int, t_max / dt)
    time_data = Float64[]
    terms_data = Int[]
    norm2_data = Float64[]          # post-rescaling: 1.0 at every step, by construction
    # The real truncation diagnostics now that norm2 is pinned to 1 -- see the
    # header. retention[k] is the fraction of weight the step INTO time k kept;
    # cum_retention[k] is the running product.
    retention_data = Float64[]
    cum_retention_data = Float64[]
    last_retention = 1.0            # no Trotter step has run yet at t = 0
    cum_retention  = 1.0
    F_mat   = Matrix{Float64}(undef, steps + 1, n)   # OTOC              Tr[W V W V]/2^N
    Csq_mat = Matrix{Float64}(undef, steps + 1, n)   # squared commutator

    println("\n" * "-"^70)
    t_start = time()

    # --- 4. Time evolution loop ---
    for step in 0:steps
        t = step * dt
        curr_terms = length(W_sum)

        push!(time_data, t)
        push!(terms_data, curr_terms)

        # Infinite-temperature OTOC of W(t) = P_i_site(t) against every probe
        # V_j, plus the norm2 truncation diagnostic -- all from one pass of
        # commutation bit-tests over W_sum's terms (see otoc_vector_infT).
        F_row, Csq_row, norm2 = otoc_vector_infT(W_sum, V_bases)
        F_mat[step+1, :]   = F_row
        Csq_mat[step+1, :] = Csq_row
        push!(norm2_data, norm2)
        push!(retention_data, last_retention)
        push!(cum_retention_data, cum_retention)

        # norm2 here is measured on the ALREADY-RESCALED operator, so it is 1 to
        # floating-point and is printed as a CHECK that the rescaling fired --
        # not as an accuracy readout (see the header). The two retention columns
        # are what actually carries the truncation information.
        # One literal format string on one line: @printf requires a literal, so
        # "..." * "..." across two lines is a compile-time error, not a style nit.
        @printf("    t = %5.2f  |W| = %6d terms  F_%d%d(%s%s) = %+.5f  Csq = %+.5f  norm2 = %.8f  (step kept %.5f, cumulative %.5f)\n",
                t, curr_terms, i_site, i_site, pauli, pauli,
                F_mat[step+1, i_site], Csq_mat[step+1, i_site], norm2,
                last_retention, cum_retention)
        flush(stdout)

        step == steps && break

        W_sum = trotter_step_2nd_order_DBf_commutationwise!(
            W_sum, H_terms, dt;
            cutoff = cutoff,
            truncate = truncation_strategy,
        )

        # --- RENORMALIZE: the one change from kitaev_OTOC_infinite_temp.jl ---
        # Measure the surviving weight, record it, then rescale to unit norm.
        # Scaling by a scalar is the verified PauliSum idiom (`ps * s`); there is
        # no in-place scale! in this version of PauliOperators.
        n2_after = sum(abs2(c) for (_, c) in W_sum)
        if !(n2_after > 0)
            error("operator norm collapsed to $(n2_after) at t = $(t + dt): truncation " *
                  "discarded every term, so there is nothing left to renormalize. " *
                  "Loosen the truncation or shorten t_max.")
        end
        last_retention = n2_after
        cum_retention *= n2_after
        W_sum = W_sum * (1 / sqrt(n2_after))
    end
    wall_time = time() - t_start

    # Trim to however many steps actually completed before any early exit
    n_done = length(time_data)
    F_mat   = F_mat[1:n_done, :]
    Csq_mat = Csq_mat[1:n_done, :]

    # norm2 is 1 by construction here, so reporting it as a quality measure would
    # be meaningless. The cumulative retention is the quantity that plays the old
    # norm2's role, and it is what the warning below is keyed on.
    @printf("\n  Final norm2 = %.8f (pinned to 1 by renormalization -- NOT an accuracy measure)\n",
            norm2_data[end])
    @printf("  Cumulative retention = %.6f (product of per-step surviving fractions)\n",
            cum_retention_data[end])
    println("    -- this is what the un-normalized solver would have reported as norm2")
    if cum_retention_data[end] < 0.9
        @warn "cumulative retention has dropped to $(round(cum_retention_data[end], digits=4)) " *
              "by t = $(time_data[end]) -- truncation has discarded a large fraction of the " *
              "operator and renormalization has been papering over it, so the late-time OTOC " *
              "is unreliable. A flat norm2 = 1 does NOT contradict this. Loosen the truncation " *
              "or shorten t_max."
    end

    # The output tag records everything that changes the answer -- couplings,
    # reference site, pauli, time grid, system size, and the TRUNCATION -- so
    # runs with different parameters can never silently overwrite each other,
    # and so downstream plotting can read those straight off the filename
    # instead of having to match them by hand. Truncation belongs here above
    # all: it is the variable this study sweeps, and two runs differing only in
    # it would otherwise collide and one would be lost.
    #
    # Layout: N, lattice fingerprint, truncation, V, J, pauli, site, dt, tmax.
    # The Kitaev couplings are NOT in the name -- they are written into the .npz
    # payload as "K" instead, so a K sweep would still be distinguishable after
    # the fact, but two runs differing only in K DO collide by filename. Add
    # them back here if you ever sweep K.
    # "_norm" marks this as a RENORMALIZED run. Without it the tag is identical
    # to the un-normalized solver's and a run here would silently OVERWRITE that
    # file -- same N, lattice, truncation, V, J, site, dt and t_max. The marker
    # sits after the truncation segment so parse_meta's regexes still match:
    # it looks for "_coeff..._w<n>_" / "_m?wdamp..._" and for _V / _J / _dt /
    # _tmax / _site / _(XX|YY|ZZ) independently, none of which this disturbs.
    safe_tag = tag === nothing ?
        "N$(n)_$(lattice_tag())_$(trunc_tag(truncation_strategy))_norm_V$(V)_J$(J)_$(pauli)$(pauli)_site$(i_site)_dt$(dt)_tmax$(t_max)" :
        tag
    mkpath(out_dir)
    out_path = joinpath(out_dir, "PP_kitaev_otoc_infT_$(safe_tag).npz")
    # "F", "Csq" and "norm2" are the real content. "C_re"/"C_im" are ALIASES
    # kept so the existing plotting scripts (Correlations/operator_spread.jl,
    # time_trace.jl) can read this file unmodified: C_re carries the squared
    # commutator Csq -- the natural light-cone quantity, 0 before the operator
    # front arrives -- and C_im is identically zero, since every quantity here
    # is real by construction. Read "F"/"Csq" directly in new code; the file
    # prefix is PP_kitaev_otoc_infT_ (not PP_kitaev_corr_camps_) precisely so
    # an OTOC run can never be mistaken for a correlation run.
    #
    # "lattice_id" and "trunc" travel INSIDE the file rather than in its name.
    # lattice_id is the bond-list fingerprint (see lattice_tag in
    # kitaev_model.jl): N alone does not identify a lattice, and an open cluster
    # compared against a torus of the same size produces a large, plausible
    # disagreement that reads as solver error. Keeping it in the payload gets
    # that check without lengthening every filename.
    npzwrite(out_path, Dict(
        "time"       => time_data,
        "terms"      => Float64.(terms_data),
        "F"          => F_mat,
        "Csq"        => Csq_mat,
        # 1.0 everywhere. Kept under this key so existing readers (load_run and
        # every plotting script) still work -- but a plot of it is a flat line at
        # 1 that means only "renormalization was on". Plot "cum_retention"
        # instead wherever the old norm2 curve was wanted.
        "norm2"      => norm2_data,
        "retention"     => retention_data,
        "cum_retention" => cum_retention_data,
        # Flag so a script can tell a renormalized run from an ordinary one
        # without parsing the filename. 0.0 would be an ordinary run.
        "normalized" => [1.0],
        "C_re"       => Csq_mat,
        "C_im"       => zeros(Float64, size(Csq_mat)),
        "lattice_id" => [Float64(lattice_id())],
        "K"          => [Kxx, Kyy, Kzz],
        # [thresh, max_weight, cutoff, alpha]. The 4th slot is APPENDED, so old
        # readers that index [1] and [2] are unaffected and old files still load.
        #
        # max_weight is never NaN now: every plotting script does
        # Int(d["trunc"][2]), and Int(NaN) throws InexactError -- a non-Combined
        # strategy used to write a file that crashed the plotters on load.
        # A damped run has no hard cap, so it reports N, which can never bind.
        #
        # alpha = 0 means "not damped". It is what distinguishes a damped run
        # from a plain coefficient run of the same thresh; see the note in
        # otoc_truncation_sweep.jl about not mixing the two on one axis.
        # Slot 5 is the damping KIND: 0 = not damped, 1 = Pauli weight,
        # 2 = Majorana weight. Machine-readable so a sweep can refuse to mix the
        # three on one axis without having to parse the filename.
        "trunc"      => (truncation_strategy isa CombinedTruncation ?
                            [truncation_strategy.thresh,
                             Float64(truncation_strategy.max_weight), cutoff, 0.0, 0.0] :
                         truncation_strategy isa WeightDampedTruncation ?
                            [truncation_strategy.thresh, Float64(n), cutoff,
                             truncation_strategy.alpha,
                             truncation_strategy.kind === :majorana ? 2.0 : 1.0] :
                            [NaN, Float64(n), cutoff, NaN, NaN])))
    @printf("  Saved: %s  (wall time: %.1f s)\n", basename(out_path), wall_time)

    # cum_retention_data is APPENDED as a 5th element: Julia's tuple destructuring
    # takes the first N targets and ignores the rest, so existing callers written
    # as `_, _, _, norm2 = run_otoc_infT(...)` keep working unchanged.
    return time_data, F_mat, Csq_mat, norm2_data, cum_retention_data
end


# CONFIGURATIONS ==============================================================
# Each entry is one full run. Unlike the correlation code, nothing here depends
# on a matching CAMPS export, so V/J/K can be swept freely -- the Hamiltonian
# is rebuilt from scratch for every configuration.
#
# J is the Heisenberg coupling: J = 0 leaves the integrable (free-fermion)
# Kitaev model, J != 0 breaks integrability. Comparing the two is the point of
# measuring OTOCs at all, so the sweep below does exactly that.
configurations = [
	 #(truncation, Kxx, Kyy, Kzz, V, J, i_site, pauli, dt, t_max)
        # Short run to verify the renormalization: norm2 must print as 1.00000000
        # at every step while "step kept" falls below 1. Lengthen t_max for real use.
        #(make_combined_truncation(1e-4, 16), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.05, 0.50),
        # (make_combined_truncation(1e-4, 16), 1.0, 1.0, 1.0, 0.3, 0.05, 2, :Z, 0.05, 3.00),
        # (make_combined_truncation(1e-4, 16), 1.0, 1.0, 1.0, 0.3, 0.05, 16, :Z, 0.05, 3.00),
        # (make_combined_truncation(1e-4, 16), 1.0, 1.0, 1.0, 0.3, 0.05, 12, :Z, 0.05, 3.00),
        # (make_combined_truncation(1e-4, 16), 1.0, 1.0, 1.0, 0.0, 0.00, 2, :Z, 0.05, 3.00),
        # (make_combined_truncation(5e-4, 10), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.05, 3.00),
        # (make_combined_truncation(5e-4, 10), 1.0, 1.0, 1.0, 0.3, 0.00, 2, :Z, 0.05, 3.00),
        # (make_combined_truncation(5e-5, 16), 1.0, 1.0, 1.0, 0.3, 0.05, 2, :Z, 0.05, 3.00),
        # (make_combined_truncation(1e-4, 10), 1.0, 1.0, 1.0, 0.3, 0.05, 2, :Z, 0.05, 3.00),
        # (make_combined_truncation(1e-5, 8), 1.0, 1.0, 1.0, 0.3, 0.05, 2, :Z, 0.05, 3.00),
        # (make_combined_truncation(5e-6, 6), 1.0, 1.0, 1.0, 0.3, 0.05, 2, :Z, 0.05, 3.00),

        # Verified smoke test of both damping kinds (t_max 0.30, ~1 s each):
        # (make_pauli_damped_truncation(0.5, 1e-4),    1.0, 1.0, 1.0, 0.3, 0.05, 2, :Z, 0.05, 0.30),
        # (make_majorana_damped_truncation(0.5, 1e-4), 1.0, 1.0, 1.0, 0.3, 0.05, 2, :Z, 0.05, 0.30),

        # WEIGHT-DAMPED: keep a term if |c|*exp(-alpha*w) > thresh, where w is
        # either the Pauli weight or the Majorana weight -- pick with the helper.
        # alpha = 0 reduces to a plain coefficient clip; large alpha approaches a
        # hard weight cap, but WITHOUT the cliff: a high-amplitude long string
        # survives here where weight_clip! would discard it outright.
        #
        # Sweep alpha at FIXED thresh, and do NOT mix the two kinds in one sweep
        # -- they penalise different things, so a shared axis means nothing.
        # Moving alpha and thresh together likewise leaves the resulting norm2
        # unattributable to either knob.
        #
        # Pauli-weight damping -- penalises how far the string reaches:
        (make_pauli_damped_truncation(0.1, 1e-4), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.05, 3.00),
        (make_pauli_damped_truncation(0.25,  1e-4), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.05, 3.00),
        # (make_pauli_damped_truncation(1.0,  1e-4), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.05, 5.00),
        # (make_pauli_damped_truncation(1.25, 1e-4), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.05, 5.00),
        # (make_pauli_damped_truncation(2.0,  1e-4), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.05, 5.00),
        
        
        # Majorana-weight (Jordan-Wigner) damping. NOT comparable to the Pauli
        # form at equal alpha -- the JW weight scale is ~2x larger, so the same
        # alpha damps much harder (0.08% vs 3.15% norm loss at alpha=0.5). Pick
        # alpha to match a target norm2, not to match the list above:
        (make_majorana_damped_truncation(0.1, 1e-4), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.05, 3.00),
        (make_majorana_damped_truncation(0.25,  1e-4), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.05, 0.00),
        # (make_majorana_damped_truncation(1.0,  1e-4), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.05, 5.00),
        # (make_majorana_damped_truncation(0.25, 1e-4), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.05, 5.00),
        # (make_majorana_damped_truncation(2.0,  1e-4), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.05, 5.00),
        
        # (make_pauli_damped_truncation(1.0,  1e-5), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.05, 5.00),
        # (make_majorana_damped_truncation(1.0,  1e-5), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.05, 5.00),
        # (make_pauli_damped_truncation(1.0,  1e-5), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.0, 5.00),
        # (make_majorana_damped_truncation(1.0,  1e-5), 1.0, 1.0, 1.0, 0.0, 0.05, 2, :Z, 0.0, 5.00),
]
# Wrapped in try/catch so one config's failure doesn't abort the whole sweep --
# failures are collected and reported in the summary at the end instead.
failures = Tuple{Int,Any}[]
completed = Tuple{Int,Any,Float64,Float64}[]   # (cfg, truncation, norm2_end, cum_retention_end)
for (idx, config) in enumerate(configurations)
    truncation_strategy, Kxx, Kyy, Kzz, cfg_V, cfg_J, i_site, pauli, dt, t_max = config

    println("\n" * "#"^60)
    println("# Configuration $idx / $(length(configurations)): " *
            "Kxx=$Kxx Kyy=$Kyy Kzz=$Kzz  V=$cfg_V  J=$cfg_J  " *
            "i_site=$i_site  pauli=$pauli  dt=$dt  t_max=$t_max")
    println("# Truncation: $truncation_strategy")
    println("#"^60)

    try
        _, _, _, norm2, cumret = run_otoc_infT(N;
                      Kxx = Kxx, Kyy = Kyy, Kzz = Kzz,
                      V = cfg_V, J = cfg_J,
                      dt = dt, t_max = t_max, i_site = i_site, pauli = pauli,
                      truncation_strategy = truncation_strategy)
        push!(completed, (idx, truncation_strategy, norm2[end], cumret[end]))
    catch e
        @warn "Configuration $idx failed -- continuing with the rest of the sweep" exception=e
        push!(failures, (idx, e))
    end
end

# End-of-sweep summary pairing each truncation with what it COST in discarded
# operator weight. In the un-normalized solver that cost is the final norm2; here
# norm2 is pinned to 1, so the cost is the CUMULATIVE RETENTION instead. norm2 is
# still shown, as a check that the rescaling fired on every run -- but reporting
# "lost" from it would print ~0% for every configuration however badly truncated,
# which is exactly the misreading the file header warns about.
println("\n" * "="^78)
if !isempty(completed)
    println("Truncation vs operator weight retained:")
    @printf("%5s  %-40s %10s %12s %10s\n",
            "cfg", "truncation", "norm2", "cum.retained", "lost")
    println("-"^78)
    for (idx, tr, n2, cr) in completed
        @printf("%5d  %-40s %10.6f %12.6f %9.2f%%\n",
                idx, string(tr), n2, cr, 100 * (1 - cr))
    end
    println("-"^78)
    println("norm2 is pinned to 1 by renormalization and is only a check that it fired.")
    println("\"cum.retained\" is the product of the per-step surviving fractions -- the")
    println("quantity the un-normalized solver reports as norm2, and the one that bounds")
    println("how far the OTOC from that run can be trusted.")
end

if isempty(failures)
    println("\nAll $(length(configurations)) configurations completed successfully.")
else
    println("\n$(length(failures)) of $(length(configurations)) configurations FAILED:")
    for (idx, e) in failures
        println("  Configuration $idx: $(sprint(showerror, e))")
    end
end
println("="^78)
