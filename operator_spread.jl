
# Operator_Spread.jl
#
# "Light cone" plot for the Kitaev honeycomb CAMPS ground state: a heatmap of
# Re C_{origin,j}(t) = Re<psi_GS| W(t) P_j |psi_GS> (W(0) = P_origin) averaged
# over graph-distance shells from a chosen origin site, versus time -- showing
# how the perturbation created at `origin_site` spreads through the lattice.
# Plotted on a diverging red/white/blue colormap centered at 0 (signed values),
# matching the style of the earlier 1D Heisenberg ED/PP light-cone panels.
#
# Uses the SAME data already saved by run_corr_from_CAMPS (Correlations/
# kitaev_correlation.jl) -- the full row C_{i_site,j}(t) for every j -- no PP
# propagation happens in this script, we just read the matrix back off disk
# (run kitaev_correlation.jl first if the .npz file is missing).
#
# No ED panel: unlike the 1D Heisenberg 1-magnon case, there is no cheap
# exact-diagonalization shortcut for the Kitaev honeycomb model.
#
# No circshift/site-offset recentering: that convention assumed a periodic 1D
# ring (site offset j-1 modulo N). The honeycomb cluster here is not a ring, and
# the MPS chain's 1..N site ordering does not follow the physical lattice's
# nearest-neighbor structure. Instead we reconstruct the ORIGINAL (pre-Clifford
# -rotation) physical bond structure straight from the bond lists in
# CAMPS/Kitaev_honeycomb_CAMPS.py (X_BONDS/Y_BONDS/Z_BONDS, 1-indexed, matching
# Julia's convention directly) and do a breadth-first search from the origin
# site to get the graph distance to every other site -- the physically
# meaningful distance for a Lieb-Robinson-style light cone. (The Clifford
# rotation is just a change of computational frame/basis; by construction --
# conjugating the ORIGINAL physical Hamiltonian by the full CAMPS circuit
# before exporting terms.csv -- correlators measured against the disentangled
# CAMPS state equal the ORIGINAL physical model's correlators, so propagation
# should track the ORIGINAL lattice's bond distance, not the rotated
# Hamiltonian's term structure or the MPS chain's 1D site ordering.)
#
# Sites at the same graph distance from the origin are averaged together into
# one row (there are, in general, multiple sites per distance shell on the
# honeycomb lattice).

using NPZ
using Printf
using Plots

# --- Edit these to match the run you want to plot ---
npz_path = joinpath(@__DIR__, "correlation_outputs", "PP_kitaev_corr_camps_camps_D100_ZZ_dt0.05_tmax5.0.npz")
origin_site = 4   # only used as a FALLBACK for older files saved before run_corr_from_CAMPS
                  # tagged its output with the reference site -- must match the i_site the npz
                  # was generated with, exactly like ref_site in time_trace.jl.

plots_dir = joinpath(@__DIR__, "plots")
mkpath(plots_dir)

# Newer files carry their reference site in the filename ("..._site<N>_..."); trust that over
# the manually-set fallback above so this can't silently mismatch (see time_trace.jl).
site_match = match(r"_site(\d+)_", basename(npz_path))
if site_match !== nothing
    origin_site = parse(Int, site_match.captures[1])
else
    println("Note: no _site<N>_ tag found in filename (older file?) -- using manually set " *
            "origin_site=$origin_site; verify this against the run_corr_from_CAMPS call that " *
            "produced this file.")
end

pauli_label = if occursin("_XX_", npz_path)
    "XX"
elseif occursin("_YY_", npz_path)
    "YY"
elseif occursin("_ZZ_", npz_path)
    "ZZ"
else
    println("Note: no _XX_/_YY_/_ZZ_ tag found in filename (older file?) -- assuming XX.")
    "XX"
end

# --- Original (pre-Clifford-rotation) physical honeycomb bonds, 1-indexed --
# straight from CAMPS/Kitaev_honeycomb_CAMPS.py's X_BONDS/Y_BONDS/Z_BONDS.
N = 22
const Z_BONDS = [(1, 18), (3, 4), (5, 6), (7, 8), (10, 22), (12, 21), (14, 15), (16, 17), (19, 20)]
const X_BONDS = [(1, 2), (4, 19), (6, 22), (8, 9), (10, 11), (12, 13), (15, 16), (17, 18), (20, 21)]
const Y_BONDS = [(2, 3), (4, 5), (6, 7), (9, 10), (11, 12), (13, 14), (16, 21), (18, 19), (20, 22)]

adj = [Int[] for _ in 1:N]
for (i, j) in vcat(X_BONDS, Y_BONDS, Z_BONDS)
    push!(adj[i], j)
    push!(adj[j], i)
end

# Breadth-first search from origin_site -> graph distance to every site.
dist = fill(-1, N)
dist[origin_site] = 0
queue = [origin_site]
while !isempty(queue)
    u = popfirst!(queue)
    for v in adj[u]
        if dist[v] == -1
            dist[v] = dist[u] + 1
            push!(queue, v)
        end
    end
end
any(dist .== -1) && @warn "Some sites are unreachable from site $origin_site (lattice graph disconnected?): $(findall(dist .== -1))"

max_d = maximum(dist)
println("Graph distances from site $origin_site (BFS over the original honeycomb bonds):")
for d in 0:max_d
    println("  distance $d: sites $(findall(dist .== d))")
end

data = npzread(npz_path)
t = data["time"]
C_re = data["C_re"]   # (n_steps, N) -- signed, unlike |C|; C_re[1, origin_site] = 1 at t=0

# Average Re(C_{origin,j}(t)) over all sites j at each graph distance -> (n_distances, n_steps)
M = zeros(Float64, max_d + 1, length(t))
for d in 0:max_d
    sites_at_d = findall(dist .== d)
    M[d+1, :] = sum(C_re[:, sites_at_d], dims=2)[:, 1] ./ length(sites_at_d)
end

# Symmetric color limits around 0, like the diverging ED/PP light-cone panels used
# for the 1D Heisenberg case, so positive and negative correlator values read on
# the same visual scale.
clim_val = maximum(abs.(M))
fig = heatmap(t, 0:max_d, M;
              xlabel        = "t",
              ylabel        = "graph distance from site $origin_site",
              title         = "Re<$(pauli_label)> operator-spreading light cone, CAMPS ground state (site $origin_site)",
              color         = :balance,
              clims         = (-clim_val, clim_val),
              titlefontsize = 13,
              guidefontsize = 12,
              tickfontsize  = 11,
              left_margin   = 10Plots.mm,
              bottom_margin = 8Plots.mm,
              size          = (800, 500),
              dpi           = 200)

display(fig)
out_path = joinpath(plots_dir, "$(pauli_label)_lightcone_site$(origin_site)_" * replace(basename(npz_path), ".npz" => "") * ".png")
savefig(fig, out_path)
println("\nSaved: $(out_path)")
