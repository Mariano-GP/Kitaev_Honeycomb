#Correlation functions for Kitaev 2d Honeycomb model 

import PauliOperators as PO
using LinearAlgebra
using Printf
using NPZ
using DelimitedFiles
import DBF as DBF
using Plots

# CAMPS STATE BRIDGE ==============================================================================
# We start our code by defining the bridge between CAMPS and Pauli propagation. we need this piece of conde
# In order to load the CAMPS ground state and the Hamiltonian terms from the files exported by the Python side.
# This is necesarry because the CAMPS ground state is a genuinely entangled state, 
#and we need to be able to compute expectation values of Pauli operators in this state. 

#The following load mps_from_npz function loads the MPS tensors from the .npz file exported by CAMPS. 
#Each tensor has shape (chi_left, d, chi_right) and is in TeNPy's right-canonical ('B') form.
# The function returns a vector of tensors, one for each site in the MPS.
function load_mps_from_npz(path::String, N::Int)
    data = npzread(path)
    Bs = Vector{Array{ComplexF64,3}}(undef, N)
    for i in 1:N
        B = data["B$(i-1)"]  # Python export is 0-indexed
        Bs[i] = ComplexF64.(B)
    end
    return Bs
end

#The following load_pauli_terms_csv function loads the Hamiltonian terms from the .csv file exported by CAMPS.
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

# load_rotated_ops_csv loads the Clifford-rotated single-site X/Y/Z operators
# exported by export_rotated_single_site_ops (Python) -- one properly-rotated
# PauliSum per (site, pauli) pair, in the SAME disentangled frame as the CAMPS
# ground state and the rotated Hamiltonian loaded above. This matters because
# if H' = C H C^dagger and |phi> = C|psi>, then <phi|Z_i|phi> = <psi|(C^dagger
# Z_i C)|psi>, NOT <psi|Z_i|psi>, unless Z_i is first rotated FORWARD the same
# way (Z_i' = C Z_i C^dagger) before being used in this frame -- a plain,
# un-rotated single-site Pauli here measures a different (generally
# unphysical) quantity, not the correlator of the original spin operators.
function load_rotated_ops_csv(path::String, N::Int)
    ops = Dict{Tuple{Int,Symbol},PO.PauliSum{N,ComplexF64}}()
    label_to_pauli = Dict(1 => :X, 2 => :Y, 3 => :Z)
    for line in eachline(path)
        line = strip(line)
        isempty(line) && continue
        parts = split(line, ",")
        origin_site = parse(Int, parts[1]) + 1   # 0-indexed (Python) -> 1-indexed (Julia)
        origin_pauli = label_to_pauli[parse(Int, parts[2])]
        coeff = parse(Float64, parts[3]) + im * parse(Float64, parts[4])

        xs = Int[]; ys = Int[]; zs = Int[]
        i = 5
        while i <= length(parts)
            site = parse(Int, parts[i]) + 1
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
        key = (origin_site, origin_pauli)
        if !haskey(ops, key)
            ops[key] = PO.PauliSum(N, ComplexF64)
        end
        ops[key] += coeff * PO.Pauli(N, X=xs, Y=ys, Z=zs)
    end
    return ops
end


#This is crucial helper function to identify which correlation function we are computing,
# and to build the corresponding single-site Pauli operator. if we want the XX, YY, or ZZ correlator,
# we need to build the corresponding single-site Pauli operator.
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


# This computes the expectation value of a Pauli operator in the CAMPS ground state, 
#given the MPS tensors and the Pauli basis.
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


# correlator_from_W_GS is the Ground-state analogue of `correlator_from_W_ferro_complex`
# `correlator_vector_from_W_ferro_complex` from the 1D Heisenberg code, but for
# a genuinely entangled state (loaded from CAMPS) instead of a classical
# product state. Since there's no closed-form shortcut for an entangled state,
# this sums coeff * <psi_GS|basis|psi_GS> over every term currently present in W_sum 
function correlator_from_W_GS(W_sum::PO.PauliSum{N,T}, Bs::Vector{Array{ComplexF64,3}}) where {N,T}
    total = zero(ComplexF64)
    for (basis, c) in W_sum
        total += c * expectation_value_mps(Bs, basis)
    end
    return total
end

# correlator_vector_from_W_GS computes the full N-site vector of correlators C_j = <psi_GS| W_sum * V_j |psi_GS>
# where V_j is the CAMPS-ROTATED single-site Pauli operator at site j, looked
# up from `rotated_ops` (see load_rotated_ops_csv) -- NOT a plain single-site
# Pauli, which would measure a different, unphysical quantity in this frame.
function correlator_vector_from_W_GS(W_sum::PO.PauliSum{N,T}, Bs::Vector{Array{ComplexF64,3}},
                                      rotated_ops::Dict{Tuple{Int,Symbol},PO.PauliSum{N,ComplexF64}};
                                      pauli::Symbol=:Z) where {N,T}
    C = zeros(ComplexF64, N)
    for j in 1:N
        Vj = rotated_ops[(j, pauli)]
        C[j] = correlator_from_W_GS(W_sum * Vj, Bs)
    end
    return C
end


# TRUNCATION STRATEGIES =====================================================
# We will start our main code by defining our possible truncation strategies.

#Basic coefficient truncation strategy: remove Pauli terms with |c| ≤ thresh.
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

#Same but for Majorana weight, which is more relevant for fermionic systems. 
function majorana_weight_clip!(ps::PO.PauliSum{N}, max_weight::Int) where {N}
	return filter!(p -> majorana_weight(p.first) <= max_weight, ps)
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

#Combined truncation strategy: remove Pauli terms based on both pauli and coefficient criteria. 
function make_combined_truncation(thresh::Float64, max_weight::Int)
	return function(W)
		W_ = deepcopy(W)
		kept    = filter(p -> abs(p.second) > thresh && weight(p.first) <= max_weight, W)
		clipped = filter(p -> !(abs(p.second) > thresh && weight(p.first) <= max_weight), W_)
		return kept, clipped
	end
end

# Weight Damped Truncation strategy: remove Pauli terms where |c|·exp(-alpha·weight) ≤ thresh.
# alpha = 0  →  plain coefficient clip at thresh
# large alpha  →  approaches a hard weight cutoff
struct WeightDampedTruncation
    alpha::Float64
    thresh::Float64
end
WeightDampedTruncation(alpha::Real) = WeightDampedTruncation(Float64(alpha), 1e-3)





# HAMILTONIAN ===========================================================================
# Now we will bulid the hamiltionian for the Kitaev model. 
# The Hamiltonian is defined on a honeycomb lattice with three types of bonds (x, y, z) 
# corresponding to the three Pauli matrices and making a hexagonal lattice.

# H = H_K + H_V, matching CAMPS/Kitaev_honeycomb_CAMPS.py's
# Kitaev_Honeycomb_Pauli_terms, just built here from adjacency matrices
# instead of bond-pair lists:
#   H_K = -Jxx sum_x-bonds XX - Jyy sum_y-bonds YY - Jzz sum_z-bonds ZZ
#         (pre-existing sign convention for this function; unrelated to V/h)
#   H_V = V * sum_{adjacent bond pairs} (three-body term)
#       + h * sum_{boundary sites} sigma_j^alpha  (alpha = that site's MISSING
#                                                   bond direction)
# Three-body term convention (Kitaev's original perturbative field expansion,
# also used in arXiv:2507.08939 Eq. 3): at a vertex j connected to neighbor i
# via an alpha-bond and to neighbor k via a beta-bond (alpha != beta), the
# term is sigma_i^alpha sigma_j^gamma sigma_k^beta, where gamma is the THIRD
# Pauli type. A bulk (degree-3) site contributes all three such pairs (xy, xz,
# yz); a boundary (degree-2) site contributes exactly one. V/h follow the
# paper's own (positive) sign convention directly, matching the Python-side
# implementation -- they are NOT negated the way Jxx/Jyy/Jzz are above.
function Kitaev_Honeycomb_Hamiltionian(adj_x, adj_y, adj_z, Kxx, Kyy, Kzz, V=0.3, h=0.1)
    am_X = adj_x
    am_Y = adj_y
    am_Z = adj_z
    @assert size(am_X) == size(am_Y) == size(am_Z) #Adjacency matrices must all be the same size

    N = size(am_X, 1) # Total number of sites in the honeycomb lattice
    H_sum = PO.PauliSum(N)

    for i in 1:N
        for j in 1:N
            if am_X[i, j] == 1 && i < j
                # interaction term in XX
                H_sum += (-Kxx) * PO.Pauli(N, X=[i, j])
            end
        end
    end

    for i in 1:N
        for j in 1:N
            if am_Y[i, j] == 1 && i < j
                # interaction term in YY
                H_sum += (-Kyy) * PO.Pauli(N, Y=[i, j])
            end
        end
    end

    for i in 1:N
        for j in 1:N
            if am_Z[i, j] == 1 && i < j
                # interaction term in ZZ
                H_sum += (-Kzz) * PO.Pauli(N, Z=[i, j])
            end
        end
    end

    # --- H_V: three-body effective-field terms + boundary field -----------
    TYPES = (:X, :Y, :Z)
    adj_by_type = Dict(:X => am_X, :Y => am_Y, :Z => am_Z)

    # bonds_at[site][type] = neighbor, read directly off the adjacency
    # matrices -- same role as CAMPS/Kitaev_honeycomb_CAMPS.py's bonds_at
    # dict, just built from am_X/am_Y/am_Z instead of X_BONDS/Y_BONDS/Z_BONDS.
    bonds_at = [Dict{Symbol,Int}() for _ in 1:N]
    for t in TYPES
        am = adj_by_type[t]
        for i in 1:N, j in 1:N
            if am[i, j] == 1
                bonds_at[i][t] = j
            end
        end
    end

    single_pauli(site, ptype) = ptype === :X ? PO.Pauli(N, X=[site]) :
                                 ptype === :Y ? PO.Pauli(N, Y=[site]) :
                                                PO.Pauli(N, Z=[site])

    for j in 1:N
        present = [t for t in TYPES if haskey(bonds_at[j], t)]

        for a in 1:length(present), b in (a+1):length(present)
            α, β = present[a], present[b]
            γ = only(setdiff(collect(TYPES), (α, β)))
            i_site = bonds_at[j][α]
            k_site = bonds_at[j][β]

            xs = Int[]; ys = Int[]; zs = Int[]
            for (site, ptype) in ((i_site, α), (j, γ), (k_site, β))
                if ptype === :X
                    push!(xs, site)
                elseif ptype === :Y
                    push!(ys, site)
                else
                    push!(zs, site)
                end
            end
            H_sum += V * PO.Pauli(N, X=xs, Y=ys, Z=zs)
        end

        if length(present) == 2
            missing_type = only(setdiff(collect(TYPES), present))
            H_sum += h * single_pauli(j, missing_type)
        end
    end

    return H_sum
end


# This function order the Hamiltonian terms in different ways, depending on the order_type argument.
function build_H_terms(H_total_sum; order_type::String = "lexical", rng_seed::Int = 1234)
	H_vec = collect(H_total_sum)

	if order_type == "lexical"
		H_terms = sort(H_vec, by = p -> (p.first.z, p.first.x))

	elseif order_type == "by_site"
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
		H_terms = sort(H_vec, by = p -> site_pair(p.first))

	elseif order_type == "by_kind"
		function kind_code(basis)
			zbits = basis.z
			xbits = basis.x
			mask = zbits | xbits
			has_x = (xbits != 0)
			has_z = (zbits != 0)
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
		H_terms = sort(H_vec, by = p -> (kind_code(p.first), site_pair(p.first)))

	elseif order_type == "random"
		rng = MersenneTwister(rng_seed)
		H_terms = shuffle(rng, H_vec)

	else
		error("Unknown order_type = $order_type")
	end

	return H_terms
end

# TROTTER EVOLUTION ===========================================================================
# These are the functions that will be used to perform the time evolution 
# of the system using Trotter decomposition, they are Hamiltonian agnostic
# And so they are the same as in the 1D Heisenberg model.

# Needed for the Trotter evolution functions below
function anti_commutes(p1::PO.Pauli, p2::PO.Pauli)
	return !PO.commute(PO.PauliBasis(p1), PO.PauliBasis(p2))
end

function trotter_step_1st_order_DBf!(W::PO.PauliSum{N, T},
	H_terms::Vector{Pair{PO.PauliBasis{N}, T}},
	dt::Float64,
    V_ops::Vector{PO.Pauli{N}};
	cutoff = 1e-12,
	error_accumulation = false,
) where {N, T}
	
    step_errors = zeros(Float64, length(V_ops))
	corr_values = zeros(Float64, length(V_ops))

	function measure_all_corrs(W_curr)
		return [correlator_from_W(W_curr, v) for v in V_ops]
	end

	for (H_basis, h_coeff) in H_terms
		h_real = real(h_coeff)
		θ = 2 * h_real * dt

		W = DBF.evolve(W, H_basis, θ)
		DBF.coeff_clip!(W, thresh = 1e-12)
		
        if error_accumulation==true
			vals_pre = measure_all_corrs(W)
		end
		
        DBF.coeff_clip!(W, thresh = cutoff)
		vals_post = measure_all_corrs(W)
		
        if error_accumulation==true
			step_errors .+= (vals_pre .- vals_post)
		end
		corr_values .+= vals_post
	end
	return W, corr_values, step_errors
end

function trotter_step_2nd_order_DBf!(
						W::PO.PauliSum{N, T},
						H_terms::Vector{Pair{PO.PauliBasis{N}, T}},
						dt::Float64,
						V_ops::Vector{PO.Pauli{N}};
						cutoff = 1e-12,
						error_accumulation = false,
) where {N, T}

	step_errors = zeros(Float64, length(V_ops))
	corr_values = zeros(Float64, length(V_ops))

	function measure_all_corrs(W_curr)
		return [correlator_from_W(W_curr, v) for v in V_ops]
	end

	# --- 1. Forward half-step ---
	for (H_basis, h_coeff) in H_terms
		h_real = real(h_coeff)
		θ = h_real * dt
		W = DBF.evolve(W, H_basis, θ)
		DBF.coeff_clip!(W, thresh = 1e-12)
		
        if error_accumulation==true
			vals_pre = measure_all_corrs(W)
		end
		
        DBF.coeff_clip!(W, thresh = cutoff)
		vals_post = measure_all_corrs(W)
		
        if error_accumulation==true
			step_errors .+= (vals_pre .- vals_post)
		end
		corr_values .+= vals_post
	end

	# --- 2. Backward half-step ---
	for (H_basis, h_coeff) in Iterators.reverse(H_terms)
		h_real = real(h_coeff)
		θ = h_real * dt
		W = DBF.evolve(W, H_basis, θ)

		DBF.coeff_clip!(W, thresh = 1e-12)
		
        if error_accumulation==true
			vals_pre = measure_all_corrs(W)
		end
		
        DBF.coeff_clip!(W, thresh = cutoff)
		vals_post = measure_all_corrs(W)
		
        if error_accumulation==true
			step_errors .+= (vals_pre .- vals_post)
		end
		corr_values .+= vals_post
	end

	return W, corr_values, step_errors
end

# this is the function we use to compute the exact diagonalization reference for C_{1,1}(t)
function trotter_step_2nd_order_DBf_commutationwise!(
						W::PO.PauliSum{N, T},
						H_terms::Vector{Pair{PO.PauliBasis{N}, T}},
						dt::Float64,
						V_ops::Vector{PO.Pauli{N}};
						cutoff = 1e-12,
						truncate = nothing,
						error_accumulation = false,
) where {N, T}

	step_errors = zeros(Float64, length(V_ops))

	function measure_all_corrs(W_curr)
		return [correlator_from_W(W_curr, v) for v in V_ops]
	end

	# --- 1. Forward half-step ---
	for (H_basis, h_coeff) in H_terms
		h_real = real(h_coeff)
		θ = h_real * dt
		W = DBF.evolve(W, H_basis, θ)
		DBF.coeff_clip!(W, thresh = 1e-12)
		W, clipped_W = coeff_clip(W, thresh = cutoff)

		if error_accumulation==true
			step_errors .+= measure_all_corrs(clipped_W)
		end

		if !isnothing(truncate)
			W, weight_clipped = truncate(W)
			if error_accumulation==true
				step_errors .+= measure_all_corrs(weight_clipped)
			end
		end
	end

	# --- 2. Backward half-step ---
	for (H_basis, h_coeff) in Iterators.reverse(H_terms)
		h_real = real(h_coeff)
		θ = h_real * dt
		W = DBF.evolve(W, H_basis, θ)

		DBF.coeff_clip!(W, thresh = 1e-12)
		W, clipped_W = coeff_clip(W, thresh = cutoff)

		if error_accumulation==true
			step_errors .+= measure_all_corrs(clipped_W)
		end

		if !isnothing(truncate)
			W, weight_clipped = truncate(W)
			if error_accumulation==true
				step_errors .+= measure_all_corrs(weight_clipped)
			end
		end
	end

	return W, step_errors
end

# Save correlation data to CSV file
function save_corr_csv(filename::String,
	time_data::Vector{Float64},
	corr_data::Dict{Int, Vector{Float64}},
	V_sites::Vector{Int})

	nT = length(time_data)
	nJ = length(V_sites)

	M = Array{Float64}(undef, nT, nJ + 1)

	for t_idx in 1:nT
		M[t_idx, 1] = time_data[t_idx]
	end

	for (col_idx, j) in enumerate(V_sites)
		vals = corr_data[j]
		@assert length(vals) == nT
		for t_idx in 1:nT
			M[t_idx, col_idx+1] = vals[t_idx]
		end
	end

	writedlm(filename, M, ',')
end


# CAMPS GROUND-STATE DRIVER ===================================================
# Same Pauli-propagation Trotter loop as run_corr_configs above, but the
# reference state is the genuinely entangled CAMPS ground state (loaded via
# camps_state_bridge.jl) instead of a classical FM/Neel product state. Output
# format is deliberately identical to run_corr_configs's .npz file (same keys:
# "time","terms","C_re","C_im", same (steps+1, N) shape for the full N-site
# correlator matrix) so downstream tools that already read that format don't
# need to change. There is no closed-form bit trick for an entangled state, so
# every measured correlator costs a real O(chi^3) MPS contraction per
# Pauli-string term, rather than O(1) bit arithmetic -- expect this to be
# markedly slower per Trotter step than the FM/Neel-based driver, especially
# since computing the FULL N-site vector (to match run_corr_configs's output
# shape) costs N times what a single-site query costs.
#
# `mps_path`/`terms_path` must come from the SAME CAMPS run and be in the SAME
# frame: on the Python side, call
#     run_camps_and_export_for_julia(N, D, terms, cliffords, outdir; ...)
# (or, if you already have psi/circuits from a completed run_camps call,
# flatten circuits and call transform_hamiltonian yourself -- see the
# "--export_for_julia" flag in Kitaev_honeycomb_CAMPS.py for the exact snippet)
# so that terms_path holds the ORIGINAL Hamiltonian conjugated by the full
# CAMPS Clifford circuit ("the Hamiltonian rotation"), matching psi's frame.
#
# KNOWN LIMITATION: pass error_accumulation=false (the default). Setting it to
# true would make trotter_step_2nd_order_DBf_commutationwise! call the OLD
# bit-based correlator_from_W internally for its step-error bookkeeping, which
# is only valid for classical FM/Neel states, not this entangled ground state.
function run_corr_from_CAMPS(mps_path::String, terms_path::String, rotated_ops_path::String, N::Int;
                             dt::Float64 = 0.05, t_max::Float64 = 5.0,
                             cutoff::Float64 = 1e-12,
                             truncation_strategy = make_combined_truncation(1e-4, 13),
                             order_type::String = "lexical",
                             i_site::Int = 7,
                             pauli::Symbol = :Z,
                             error_accumulation::Bool = false,
                             out_dir::String = joinpath(@__DIR__, "correlation_outputs"),
                             tag::Union{Nothing,String} = nothing)

    println("="^60)
    println("Correlation Simulation from CAMPS ground state (Trotter Order: 2, commutation-wise)")
    println("="^60)
    println("  N=$N  dt=$dt  t_max=$t_max  cutoff=$cutoff")

    # --- 1. Load the ground state, the (already-rotated) Hamiltonian, and the
    #     (already-rotated) single-site operators used to build W(0)/V_j ---
    println("Loading CAMPS ground state from: $mps_path")
    Bs = load_mps_from_npz(mps_path, N)
    println("Loading rotated Hamiltonian from: $terms_path")
    H_sum = load_pauli_terms_csv(terms_path, N)
    H_terms = build_H_terms(H_sum; order_type = order_type)
    println("Loading rotated single-site operators from: $rotated_ops_path")
    rotated_ops = load_rotated_ops_csv(rotated_ops_path, N)
    println("System Size: N = $N  |  Hamiltonian terms: $(length(H_sum))")
	 println("Truncation strategy: $(truncation_strategy)")
    # --- 2. Define operators ---
    # W(0) is the CAMPS-rotated version of P_i_site (see load_rotated_ops_csv),
    # NOT a plain single-site Pauli -- a plain Pauli here would measure a
    # different, unphysical quantity in this disentangled frame (see the
    # Python export_rotated_single_site_ops docstring for the derivation).
    println("\nInitializing W(0) = rotated $(pauli)_$(i_site) (CAMPS ground state)...")
    W_sum = deepcopy(rotated_ops[(i_site, pauli)])

    # Check whether W(0), as a WHOLE rotated operator, is an exact conserved
    # quantity of the rotated Hamiltonian (every term of W(0) commutes with
    # every term of H_sum) -- if so it will never evolve, W(t) = W(0) for all
    # t. This is the correct, general version of "is i_site inert for this
    # pauli": since W(0) is now a rotated (generally multi-term) operator
    # rather than a single plain Pauli, checking only one term's (x,z) bits
    # against H_sum is no longer sufficient.
    is_inert = all(PO.commute(w_basis, h_basis) for (w_basis, _) in W_sum, (h_basis, _) in H_sum)
    if is_inert
        @warn "W(0) = rotated $(pauli)_$i_site commutes with every term of the rotated " *
              "Hamiltonian -- W(t) will stay exactly W(0) forever. Pick a different i_site."
    end

    # V_ops_vec is only read by trotter_step_2nd_order_DBf_commutationwise!'s
    # internal error-accumulation bookkeeping, which is skipped entirely
    # whenever error_accumulation=false (the required default here -- see the
    # KNOWN LIMITATION note above); its content is otherwise unused, so a
    # single placeholder Pauli is enough to satisfy the type signature.
    V_ops_vec = [PO.Pauli(N, Z=[i_site])]

    # --- 3. Simulation constants ---
    steps = floor(Int, t_max / dt)
    time_data = Float64[]
    terms_data = Int[]
    C_mat = Matrix{ComplexF64}(undef, steps + 1, N)

    println("\n" * "-"^50)
    println(@sprintf("%8s  %10s", "Time", "N_terms"))
    println("-"^50)

    t_start = time()
    # --- 4. Time evolution loop ---
    for step in 0:steps
        t = step * dt
        curr_terms = length(W_sum)

        push!(time_data, t)
        push!(terms_data, curr_terms)
        # <psi_GS| W(t) P_j |psi_GS> for every site j (P = pauli) -- matches
        # run_corr_configs's correlator_vector_from_W_ferro_complex(W_sum) call,
        # just against the CAMPS ground state instead of the FM product state.
        C_mat[step+1, :] = correlator_vector_from_W_GS(W_sum, Bs, rotated_ops; pauli = pauli)

        @printf("    t = %5.2f  |W| = %6d terms  C_%d%d(%s%s) = %+.5f %+.5fi\n",
                t, curr_terms, i_site, i_site, pauli, pauli,
                real(C_mat[step+1, i_site]), imag(C_mat[step+1, i_site]))
        flush(stdout)

        step == steps && break

        W_sum, _ = trotter_step_2nd_order_DBf_commutationwise!(
            W_sum, H_terms, dt, V_ops_vec;
            cutoff = cutoff,
            truncate = truncation_strategy,
            error_accumulation = error_accumulation,
        )
    end
    wall_time = time() - t_start

    # Trim C_mat to however many steps actually completed before any early exit
    n_done = length(time_data)
    C_mat = C_mat[1:n_done, :]

    # Default tag is derived from the input mps_path's own filename (e.g.
    # "camps_D100_mps.npz" -> "camps_D100"), so the bond dimension you exported
    # with is automatically carried into the output filename -- different D
    # (or otherwise differently-tagged) runs can't silently overwrite each
    # other just because you forgot to pass `tag=` explicitly. `pauli` and
    # `i_site` are also included so an XX run and a ZZ run, or two runs
    # anchored at different reference sites, at the same D/dt/t_max don't
    # collide -- and so downstream plotting scripts can read the reference
    # site straight off the filename instead of having to match it by hand
    # (a mismatch here silently produces cross-site correlators mislabeled as
    # autocorrelators, e.g. C_44(t=0) != 1.0).
    input_tag = replace(basename(mps_path), r"_mps\.npz$" => "")
    safe_tag = tag === nothing ? "$(input_tag)_$(pauli)$(pauli)_site$(i_site)_V$(V)_h$(h_field)_dt$(dt)_tmax$(t_max)_N=$(N)" : tag
    mkpath(out_dir)
    out_path = joinpath(out_dir, "PP_kitaev_corr_camps_$(safe_tag).npz")
    npzwrite(out_path, Dict(
        "time"  => time_data,
        "terms" => Float64.(terms_data),
        "C_re"  => real.(C_mat),
        "C_im"  => imag.(C_mat)))
    @printf("  Saved: %s  (wall time: %.1f s)\n", basename(out_path), wall_time)

    return time_data, C_mat
end


# mps_path/terms_path are the files written by export_camps_state_for_julia /
# run_camps_and_export_for_julia (Python, --export_for_julia), which land in
# the Python script's --outdir, e.g.:
#     python CAMPS/Kitaev_honeycomb_CAMPS.py --Dlist_camps 100 --export_for_julia \
#            --outdir ./Kitaev_Honeycomb_camps_outputs
# NOT next to this .jl file -- point camps_outdir below at wherever you passed
# to --outdir. Their filenames already carry the bond dimension (e.g.
# "camps_D100_mps.npz" for D=100), and run_corr_from_CAMPS's output tag is
# derived from mps_path's own name (see above), so this D=100 run's output
# can't collide with a D=60/D=250/etc. run.
N = 22
camps_outdir = joinpath(@__DIR__, "..", "CAMPS","Kitaev_Honeycomb_camps_outputs")
# Matches Kitaev_honeycomb_CAMPS.py's Julia-export tag
# ("camps_N{N}_D{D}_V{V}_h{h_field}") -- V/h_field must match whatever you
# actually passed to Kitaev_honeycomb_CAMPS.py's --V/--h_field (defaults there
# are V=0.3, h_field=0.1). These select which EFFECTIVE-FIELD STRENGTH run to
# load, not a reference site -- that's still a downstream choice made only by
# run_corr_from_CAMPS's `i_site` argument below, so it never belongs here.
# No other code in this file needs to change for the three-body/boundary-field
# terms: load_pauli_terms_csv, expectation_value_mps, weight, build_H_terms,
# and the Trotter evolution functions all already operate on PauliBasis
# bitmasks / arbitrary-length site,label CSV rows, with no assumption that a
# term touches exactly 2 sites -- they pick up 1-body and 3-body terms
# exactly the same way as the existing 2-body Kitaev bonds.
V = -0.3
h_field = 0.1
mps_path = joinpath(camps_outdir, "camps_N$(N)_D100_V$(V)_h$(h_field)_mps.npz")
terms_path = joinpath(camps_outdir, "camps_N$(N)_D100_V$(V)_h$(h_field)_terms.csv")
rotated_ops_path = joinpath(camps_outdir, "camps_N$(N)_D100_V$(V)_h$(h_field)_rotated_ops.csv")
# Which site is "inert" is pauli-dependent, not just site-dependent, and is
# specific to THIS circuit's rotated Hamiltonian -- adding the three-body and
# boundary-field terms changes which (if any) sites stay inert compared to the
# pure-Kitaev (V=0, h_field=0) case, since those terms can touch sites that
# were previously untouched by the given pauli type. Check the @warn output
# from run_corr_from_CAMPS after rerunning; it's pauli-aware and will name
# which sites are safe to pick for i_site on this specific run.
configurations = [
	# (Truncation Strategy, Kxx, Kyy, Kzz, V, h_field, i_site, pauli, dt, t_max)
	# Kxx/Kyy/Kzz are recorded here for BOOKKEEPING only -- run_corr_from_CAMPS
	# loads the Hamiltonian pre-baked from terms.csv (whatever K values were
	# used when Kitaev_honeycomb_CAMPS.py exported it), so it has no K
	# parameter of its own. These fields exist so each configuration documents
	# which CAMPS export (--K/--Kx/--Ky/--Kz at export time) it actually
	# requires -- e.g. isotropic Kxx=Kyy=Kzz=1.0 (non-Abelian B phase) here.
	# (make_combined_truncation(1e-4, 22), 1.0, 1.0, 1.0, 0.3, 0.1, 11, :Z, 0.05, 5.00)
	(make_combined_truncation(1e-3, 12), 1.0, 1.0, 1.0, -0.3, 0.1, 12, :Y, 0.05, 5.00)
	#(make_combined_truncation(1e-3, 12), 1.0, 1.0, 1.0, 0.3, 0.1, 20, :Z, 0.05, 5.00)
	# (make_combined_truncation(1e-4, 22), 1.0, 1.0, 1.0, 0.3, 0.1, 4, :Z, 0.05, 5.00)
	# (make_combined_truncation(1e-4, 22), 1.0, 1.0, 1.0, 0.3, 0.1, 4, :X, 0.05, 5.00)
	# (make_combined_truncation(1e-4, 22), 1.0, 1.0, 1.0, 0.3, 0.1,
	# (make_combined_truncation(1e-4, 13), 1.0, 1.0, 1.0, 0.3, 0.1, 2, :X, 0.05, 5.00)
	# (make_combined_truncation(1e-4, 13), 1.0, 1.0, 1.0, 0.3, 0.1, 2, :Y, 0.05, 5.00)
	# (make_combined_truncation(1e-4, 13), 1.0, 1.0, 1.0, 0.3, 0.1, 7, :Z, 0.05, 5.00)
	# (make_combined_truncation(1e-4, 13), 1.0, 1.0, 1.0, 0.3, 0.1, 7, :X, 0.05, 5.00)
	# (make_combined_truncation(1e-4, 13), 1.0, 1.0, 1.0, 0.3, 0.1, 7, :Y, 0.05, 5.00)
	# (make_combined_truncation(1e-4, 13), 1.0, 1.0, 1.0, 0.3, -0.1, 7, :Z, 0.05, 5.00)
	# (make_combined_truncation(1e-4, 13), 1.0, 1.0, 1.0, 0.3, -0.1, 7, :X, 0.05, 5.00)
	# (make_combined_truncation(1e-4, 13), 1.0, 1.0, 1.0, 0.3, -0.1, 7, :Y, 0.05, 5.00)
]

# Run every configuration in the list above. Each config carries its OWN
# (V, h_field), since a different effective-field strength means a genuinely
# different CAMPS ground state/rotated Hamiltonian -- mps_path/terms_path/
# rotated_ops_path are rebuilt per-config from that config's own V/h_field,
# not reused from the single set defined above (which is now only a fallback
# default for interactive/one-off use, e.g. testing a single call by hand).
# Kxx/Kyy/Kzz are NOT part of the export filename (Kitaev_honeycomb_CAMPS.py's
# tag doesn't encode them) -- if you sweep configs with DIFFERENT K values,
# make sure you've actually exported a matching CAMPS run for each one, since
# nothing here will detect a K mismatch automatically.
# Wrapped in try/catch so one config's failure (missing export file, an
# i_site that turns out fully inert, etc.) doesn't abort the whole sweep --
# failures are collected and reported in the summary at the end instead.
failures = Tuple{Int,Any}[]
for (idx, config) in enumerate(configurations)
    truncation_strategy, Kxx, Kyy, Kzz, cfg_V, cfg_h_field, i_site, pauli, dt, t_max = config

    println("\n" * "#"^60)
    println("# Configuration $idx / $(length(configurations)): " *
            "Kxx=$Kxx Kyy=$Kyy Kzz=$Kzz  V=$cfg_V  h_field=$cfg_h_field  " *
            "i_site=$i_site  pauli=$pauli  dt=$dt  t_max=$t_max")
    println("#"^60)

    cfg_mps_path = joinpath(camps_outdir, "camps_N$(N)_D100_V$(cfg_V)_h$(cfg_h_field)_mps.npz")
    cfg_terms_path = joinpath(camps_outdir, "camps_N$(N)_D100_V$(cfg_V)_h$(cfg_h_field)_terms.csv")
    cfg_rotated_ops_path = joinpath(camps_outdir, "camps_N$(N)_D100_V$(cfg_V)_h$(cfg_h_field)_rotated_ops.csv")

    try
        run_corr_from_CAMPS(cfg_mps_path, cfg_terms_path, cfg_rotated_ops_path, N;
                             dt = dt, t_max = t_max, i_site = i_site, pauli = pauli,
                             truncation_strategy = truncation_strategy)
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
