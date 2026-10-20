# Shared lattice + Hamiltonian definition for the Kitaev honeycomb model.
#
# This file is included by BOTH solvers:
#   kitaev_OTOC_infinite_temp.jl  -- Pauli propagation (truncated, approximate)
#   ED_OTOC_infinite_temp.jl      -- exact diagonalization (exact, small N)
#
# It exists so those two can never drift apart. The whole point of running them
# side by side is to measure Pauli propagation's truncation error, and that
# measurement is worthless if the two are quietly solving different models --
# a single edited bond or a sign convention changed in one file but not the
# other would show up as "truncation error" that is really a modelling bug.
# There is exactly one lattice and one Hamiltonian here, and both solvers read
# it. Edit the model ONLY in this file.

import PauliOperators as PO
using Printf


# LATTICE =====================================================================
# 1-indexed site-pair tuples, one list per bond color -- same convention as
# CAMPS/Kitaev_honeycomb_CAMPS.py's X_BONDS/Y_BONDS/Z_BONDS and as
# Correlations/operator_spread.jl.
#
# PERIODIC BOUNDARY CONDITIONS: every site must appear in EXACTLY ONE bond of
# each color (degree 3, so 3N/2 bonds total). The Hamiltonian carries no
# boundary field, so a site missing a bond is a genuinely dangling site with
# nothing compensating for it. Call check_periodic() to verify -- both solvers
# do this automatically at startup and warn, naming the offending sites.

# WHICH LATTICE. Default is the 16-site torus; set the environment variable to
# select another WITHOUT editing this file, so one run can use a different
# cluster while everything else stays put:
#
#     KITAEV_LATTICE=torus16     (default)  16-site torus,  N = 16
#     KITAEV_LATTICE=torus32               32-site torus,  N = 32
#
#     $env:KITAEV_LATTICE = "torus32"; julia --project=. kitaev_OTOC_infinite_temp.jl
#
# N IS DERIVED FROM THE LATTICE, not set independently -- see below. A hardcoded
# N that disagrees with the bond lists produces a Hamiltonian with dangling
# sites and is the single easiest way to get silently wrong physics here.
#
# Mixing outputs from the two is prevented automatically, not by discipline:
# lattice_tag() is a checksum OF THE BOND LISTS, so each lattice stamps a
# different fingerprint into every filename, and the plotting scripts refuse to
# compare across fingerprints.
#     KITAEV_LATTICE=torus16old            the PREVIOUS 16-site torus, N = 16
#
# torus16old is kept selectable for one reason: almost every ED reference in
# otoc_outputs/ was generated on it (fingerprint L243535), including the only
# long ones -- t_max = 5 and 7 at V=0, J=0.05, all 16 probes, with a measured
# error bar. Scoring a new truncation scheme against exact results needs a run
# on the SAME graph, and regenerating those references on torus16 would cost
# hours of ED per physics point. It is a valid honeycomb torus in its own right
# (it passes the same four tests as torus16), just a different labelling with
# 8 four-cycles where torus16 has none.
const LATTICE = get(ENV, "KITAEV_LATTICE", "torus32")
LATTICE in ("torus16", "torus16old", "torus32") ||
    error("KITAEV_LATTICE=\"$LATTICE\" is not a known lattice; " *
          "use torus16, torus16old or torus32")

# --- torus16: 16-site periodic (torus) cluster -------------------------------
# 24 bonds, 8 per color, every site degree (1,1,1). Verified by check_periodic()
# and by the plaquette count below.
#
# The last entry on each line is a WRAP-AROUND bond closing the boundary. Those
# five were not free to choose: given the 19 bulk bonds, the sublattice
# structure (odd sites = A, even sites = B, so every bond must join an odd to an
# even site) leaves only four possible completions, and exactly one of them
# yields a proper honeycomb -- 8 hexagonal plaquettes each carrying two bonds of
# every color. The other three produce 6 plaquettes plus malformed loops. The
# y-bond (9,2) was forced outright, being the only y-deficient pair.
#
# Consistency: V - E + F = 16 - 24 + 8 = 0, the Euler characteristic of a torus,
# as periodic boundary conditions require.
const TORUS16_Z = [(1,9),(3,15),(5,13),(7,11),(10,8),(16,2),(14,4),(12,6)]
const TORUS16_X = [(2,3),(4,5),(6,7),(8,1),(9,16),(15,14),(13,12),(11,10)]
const TORUS16_Y = [(1,2),(3,4),(5,6),(7,8),(10,9),(16,15),(14,13), (12,11)]

# --- torus32: 32-site periodic (torus) cluster -------------------------------
# 48 bonds, 16 per colour, every site degree (1,1,1). Verified to be a GENUINE
# honeycomb torus, by the same four tests torus16 passes:
#   * one x, one y, one z bond at every site, connected, bipartite
#   * 16 properly-coloured hexagons (2x/2y/2z each) double-covering every edge,
#     so V - E + F = 32 - 48 + 16 = 0, the Euler characteristic of a torus
#   * a colour-preserving automorphism subgroup of order 16 acting REGULARLY on
#     the 16 A-sites -- i.e. the honeycomb modulo two translations
#
# It is in fact a CLEANER honeycomb than torus16: it has ZERO 4-cycles, where
# torus16 has 8 (non-contractible loops that wrap its short direction). Diameter
# 6 against torus16's 5, so the operator front has more room before it wraps.
const TORUS32_Z = [(1,9),(3,15),(5,13),(7,11),(10,17),(16,19),(14,21),(12,23),
(24,25),(18,31),(20,29),(22,27),(26,6),(32,8),(30,2),(28,4)]
const TORUS32_X = [(2,3),(4,5),(6,7),(8,1),(9,16),(15,14),(13,12),(11,10),
(17,18),(19,20),(21,22),(23,24),(25,32),(31,30),(29,28),(27,26)]
const TORUS32_Y = [(1,2),(3,4),(5,6),(7,8),(10,9),(16,15),(14,13),(12,11),
(24,17),(18,19),(20,21),(22,23),(26,25),(32,31),(30,29),(28,27)]


# NOTE: the otocpp4x2 lattice (OTOC-PP's honeycomb:4,2,periodic, fingerprint
# L242ad9) has been removed. Output files generated on it are still in
# otoc_outputs/, and otocpp_csv_grid.jl / otocpp_J_compare.jl still reference
# that tag -- those comparisons can no longer be regenerated from here.

# --- torus16old: the 16-site torus these bond lists held BEFORE the relabelling
# Fingerprint L243535. A genuine honeycomb torus (one bond of each colour at
# every site, bipartite, 8 properly-coloured hexagons double-covering every
# edge, translation group of order 8 acting regularly) -- it differs from
# torus16 only in labelling and in having 8 non-contractible 4-cycles. Retained
# because the project's ED references live on it; see the note above.
const TORUS16OLD_Z = [(2,1),(4,15),(6,7),(14,13),(16,11),(8,9),(3,12),(5,10)]
const TORUS16OLD_X = [(2,3),(4,5),(14,15),(16,7),(12,11),(10,9),(1,6),(8,13)]
const TORUS16OLD_Y = [(3,4),(5,6),(1,14),(15,16),(7,8),(13,12),(11,10),(9,2)]

const Z_BONDS = LATTICE == "torus32" ? TORUS32_Z :
                LATTICE == "torus16old" ? TORUS16OLD_Z : TORUS16_Z
const X_BONDS = LATTICE == "torus32" ? TORUS32_X :
                LATTICE == "torus16old" ? TORUS16OLD_X : TORUS16_X
const Y_BONDS = LATTICE == "torus32" ? TORUS32_Y :
                LATTICE == "torus16old" ? TORUS16OLD_Y : TORUS16_Y

# N FOLLOWS THE BOND LISTS. Derived, never typed in by hand: every site must
# carry one bond of each colour, so N is fixed by the lattice and a mismatch
# would leave sites with no bonds at all -- a Hamiltonian missing terms, which
# check_periodic() would flag but only after the run had started.
const N = LATTICE == "torus32" ? 32 : 16

# Cross-check the two against each other rather than trusting the line above:
# the highest site index appearing in any bond must be exactly N.
let hi = maximum(max(a, b) for bl in (X_BONDS, Y_BONDS, Z_BONDS) for (a, b) in bl)
    hi == N || error("lattice \"$LATTICE\": bonds reach site $hi but N = $N")
    nb = length(X_BONDS) + length(Y_BONDS) + length(Z_BONDS)
    nb == 3N ÷ 2 || error("lattice \"$LATTICE\": $nb bonds, but a 3-regular " *
                          "N=$N lattice needs $(3N ÷ 2)")
end

# --- 13-site trigonal cluster (from ED_N=13_Kitaev.jl); swap in to cross-check
#     against the exact-diagonalization reference in this same folder ---
# const N = 13
# const Z_BONDS = [(3,4),(5,6),(10,11),(1,12),(13,8)]
# const X_BONDS = [(1,2),(13,4),(7,6),(8,9),(11,12)]
# const Y_BONDS = [(2,3),(4,5),(7,8),(9,10),(12,13)]

# Adjacency matrices in the form Kitaev_Honeycomb_Hamiltionian expects
# (symmetric 0/1 matrices, one per bond color).
function adjacency_from_bonds(bonds, n::Int)
    am = zeros(Int, n, n)
    for (i, j) in bonds
        (1 <= i <= n && 1 <= j <= n) ||
            error("bond ($i,$j) is out of range for an $n-site lattice")
        i == j && error("bond ($i,$j) is a self-loop")
        am[i, j] == 1 && error("bond ($i,$j) is listed twice")
        am[i, j] = 1
        am[j, i] = 1
    end
    return am
end

# check_periodic verifies the defining property of a periodic (wrapped) Kitaev
# cluster: every site carries exactly one x-, one y- and one z-bond. Returns the
# vector of offending sites (empty when the lattice is periodic) and, unless
# quiet=true, prints exactly which bond color each bad site is missing -- which
# is the information you need to write the wrap-around bonds.
function check_periodic(n::Int = N; quiet::Bool = false)
    color_deg(bonds, j) = count(b -> b[1] == j || b[2] == j, bonds)
    bad = Int[]
    for j in 1:n
        d = (color_deg(X_BONDS, j), color_deg(Y_BONDS, j), color_deg(Z_BONDS, j))
        d == (1, 1, 1) || push!(bad, j)
    end

    if !quiet
        nb = length(X_BONDS) + length(Y_BONDS) + length(Z_BONDS)
        # Name the active lattice in every run's log: with more than one
        # selectable, "which cluster was this?" must not require guessing from
        # the filename after the fact.
        @printf("Lattice: %s (%s)\n", LATTICE, lattice_tag())
        @printf("Lattice: N=%d, bonds = %d X + %d Y + %d Z = %d (3-regular needs %d)\n",
                n, length(X_BONDS), length(Y_BONDS), length(Z_BONDS), nb, 3n ÷ 2)
        if isempty(bad)
            println("Lattice is PERIODIC: every site has exactly one bond of each color.")
        else
            println("Lattice is NOT PERIODIC -- $(length(bad)) of $n sites are short a bond:")
            for j in bad
                d = (color_deg(X_BONDS, j), color_deg(Y_BONDS, j), color_deg(Z_BONDS, j))
                missing_colors = join([c for (c, dd) in zip(("x", "y", "z"), d) if dd == 0], ",")
                @printf("    site %2d: (x,y,z) degrees = (%d,%d,%d)%s\n", j, d[1], d[2], d[3],
                        isempty(missing_colors) ? "  [duplicated bond]" : "  missing $missing_colors")
            end
        end
    end
    return bad
end


# PROBE GEOMETRY ==============================================================
# Bond distance from `origin` to every site, over the CURRENTLY SELECTED lattice.
# Site NUMBERS carry no geometry: which site sits at which distance changes
# completely between torus16 and torus32, and changed once already when the
# torus16 bond lists were rewritten. Anything that wants "a near site" or "a far
# site" must ask the lattice rather than hold a number.
function bond_distances(origin::Int; n::Int = N)
    adj = [Int[] for _ in 1:n]
    for (i, j) in vcat(X_BONDS, Y_BONDS, Z_BONDS)
        push!(adj[i], j); push!(adj[j], i)
    end
    d = fill(-1, n); d[origin] = 0; q = [origin]
    while !isempty(q)
        u = popfirst!(q)
        for v in adj[u]
            d[v] == -1 && (d[v] = d[u] + 1; push!(q, v))
        end
    end
    any(==(-1), d) && @warn "sites unreachable from $origin: $(findall(==(-1), d))"
    return d
end

# One site per distance shell, nearest first: the natural OTOC probe ladder.
# Returns [origin, a site at distance 1, one at distance 2, ...] out to the
# lattice diameter, so the front can be watched crossing each shell in turn.
#
# Computed, never typed in. The previous hardcoded ladder [2,16,15,14,13,12] was
# correct for torus16 and silently became distances 0,3,2,3,4,5 on torus32 --
# two probes in the same shell and no probe at distance 1.
function probe_ladder(origin::Int; n::Int = N, max_shell::Int = typemax(Int))
    d = bond_distances(origin; n = n)
    out = Int[]
    for r in 0:min(maximum(d), max_shell)
        idx = findfirst(==(r), d)
        idx === nothing || push!(out, idx)
    end
    return out
end


# HAMILTONIAN =================================================================
# The Hamiltonian is defined on a honeycomb lattice with three types of bonds
# (x, y, z) corresponding to the three Pauli matrices:
#
#   H_K = -Kxx sum_x-bonds XX - Kyy sum_y-bonds YY - Kzz sum_z-bonds ZZ
#   H_V = V * sum_{adjacent bond pairs} (three-body term)
#   H_J = -J * sum_{all bonds} (XX + YY + ZZ)     (isotropic Heisenberg)
#
# Three-body term convention (Kitaev's original perturbative field expansion,
# also used in arXiv:2507.08939 Eq. 3): at a vertex j connected to neighbor i
# via an alpha-bond and to neighbor k via a beta-bond (alpha != beta), the
# term is sigma_i^alpha sigma_j^gamma sigma_k^beta, where gamma is the THIRD
# Pauli type. A bulk (degree-3) site contributes all three such pairs (xy, xz,
# yz). V follows the paper's own (positive) sign convention directly -- it is
# NOT negated the way Kxx/Kyy/Kzz are above.
#
# NO BOUNDARY FIELD. The single-site term h * sigma_j^alpha that earlier
# versions added at degree-2 sites (alpha = that site's missing bond direction)
# existed only to stand in for the bonds an OPEN cluster is missing at its
# edges. Under periodic boundary conditions every site is degree 3, there are
# no missing bonds, and that term has nothing to compensate for -- so it is
# gone entirely rather than left behind as a parameter that should always be 0.
# If you switch back to an open cluster, note that this Hamiltonian then has a
# genuinely free boundary; it does not silently fall back to a field.
#
# Built as a ComplexF64 PauliSum so its coefficient type matches W(0)'s and
# DBF.evolve's Trotter loop sees a single consistent element type.
function Kitaev_Honeycomb_Hamiltionian(adj_x, adj_y, adj_z, Kxx, Kyy, Kzz, V=0.0, J=0.05)
    am_X = adj_x
    am_Y = adj_y
    am_Z = adj_z
    @assert size(am_X) == size(am_Y) == size(am_Z) #Adjacency matrices must all be the same size

    N = size(am_X, 1) # Total number of sites in the honeycomb lattice
    H_sum = PO.PauliSum(N, ComplexF64)

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

    # --- H_V: three-body effective-field terms ----------------------------
    # Skipped entirely at V == 0 rather than added with a zero coefficient: a
    # zero term still occupies a slot in the PauliSum, and the Trotter loop in
    # the PP solver would then perform a no-op rotation for each one (48 of them
    # at N = 16), costing time and inflating the reported term count.
    if V != 0
    TYPES = (:X, :Y, :Z)
    adj_by_type = Dict(:X => am_X, :Y => am_Y, :Z => am_Z)

    # bonds_at[site][type] = neighbor, read directly off the adjacency matrices.
    bonds_at = [Dict{Symbol,Int}() for _ in 1:N]
    for t in TYPES
        am = adj_by_type[t]
        for i in 1:N, j in 1:N
            if am[i, j] == 1
                bonds_at[i][t] = j
            end
        end
    end

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
    end
    end

    # --- H_J: isotropic Heisenberg interaction (J = 0 disables it) ---------
    # Tuning J away from 0 breaks the Kitaev model's integrability and is the
    # cleanest knob for seeing genuinely chaotic OTOC growth: at J = 0 the pure
    # Kitaev model is free-fermion solvable and its OTOC front spreads without
    # the exponential scrambling that the Heisenberg coupling introduces.
    # Unlike the bond-color-specific Kitaev terms above, a Heisenberg term puts
    # the FULL (XX + YY + ZZ) on EVERY bond regardless of that bond's color, so
    # this loop walks the union of the three adjacency matrices.
    if J != 0
        for i in 1:N
            for j in 1:N
                if i < j && (am_X[i, j] == 1 || am_Y[i, j] == 1 || am_Z[i, j] == 1)
                    H_sum += (-J) * PO.Pauli(N, X=[i, j])
                    H_sum += (-J) * PO.Pauli(N, Y=[i, j])
                    H_sum += (-J) * PO.Pauli(N, Z=[i, j])
                end
            end
        end
    end

    return H_sum
end

# Convenience wrapper: build the Hamiltonian straight from the bond lists above.
# Both solvers call THIS, so neither can accidentally assemble its own variant.
function build_hamiltonian(; Kxx = 1.0, Kyy = 1.0, Kzz = 1.0, V = 0.3, J = 0.0, n::Int = N)
    adj_x = adjacency_from_bonds(X_BONDS, n)
    adj_y = adjacency_from_bonds(Y_BONDS, n)
    adj_z = adjacency_from_bonds(Z_BONDS, n)
    return Kitaev_Honeycomb_Hamiltionian(adj_x, adj_y, adj_z, Kxx, Kyy, Kzz, V, J)
end

# lattice_tag fingerprints the CURRENT bond lists, e.g. "L24a1b2" (24 bonds,
# checksum a1b2). Both solvers stamp it into every output filename.
#
# Why this exists: N alone does not identify a lattice. An open 16-site cluster
# and a periodic 16-site torus are different physical systems that would
# otherwise produce identically-named files, so editing the bond lists silently
# invalidates every earlier output while leaving it looking current. Comparing
# across that boundary yields a large disagreement that is easy to misread as a
# solver bug -- it happened during development, which is why the tag is here.
#
# The checksum is a plain FNV-1a fold over canonicalized (color, min, max)
# triples, sorted so bond ORDER within a list cannot change the tag. Written out
# rather than using Julia's `hash` because `hash` is not guaranteed stable
# across Julia versions, and a filename must not change under you.
function lattice_id()
    triples = Vector{Int}[]
    for (c, bl) in ((1, X_BONDS), (2, Y_BONDS), (3, Z_BONDS))
        for (a, b) in bl
            push!(triples, [c, min(a, b), max(a, b)])
        end
    end
    sort!(triples)
    h = UInt32(2166136261)
    for t in triples, v in t
        h = (h ⊻ UInt32(v)) * UInt32(16777619)
    end
    return Int(length(triples)) * 100000 + Int(h % UInt32(0x10000))
end

# Display form used in filenames: bond count then a 4-digit HEX checksum, e.g.
# "L243535" = 24 bonds, checksum 3535. Kept hexadecimal because that is the form
# already baked into existing filenames -- switching to decimal silently renames
# every output and orphans the lot.
function lattice_tag()
    n_bonds = length(X_BONDS) + length(Y_BONDS) + length(Z_BONDS)
    h = lattice_id() % 100000
    return "L$(n_bonds)" * string(h, base = 16, pad = 4)
end

# Plain single-site Pauli at `site`. These are both W(0) and the probes V_j: no
# Clifford rotation is applied or needed, because the infinite-temperature trace
# is basis-independent.
function site_pauli(n::Int, site::Int, pauli::Symbol)
    pauli === :X && return PO.Pauli(n, X=[site])
    pauli === :Y && return PO.Pauli(n, Y=[site])
    pauli === :Z && return PO.Pauli(n, Z=[site])
    error("pauli must be :X, :Y or :Z, got $pauli")
end
