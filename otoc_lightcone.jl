# OTOC light cone -- the OTOC counterpart of Correlations/operator_spread.jl.
#
# Heatmap of the squared commutator C_{i,j}(t) = Tr|[W_i(t), V_j]|^2 / 2^N,
# averaged over graph-distance shells from the reference site i, versus time.
# This is THE canonical OTOC plot: the boundary where C rises off zero is the
# butterfly light cone, and its slope is the butterfly velocity.
#
#     julia --project=. otoc_lightcone.jl
#
# TWO DIFFERENCES FROM THE CORRELATION VERSION, both deliberate:
#
# 1. SEQUENTIAL colormap, not diverging. operator_spread.jl plots Re C_{ij}(t),
#    which is signed and oscillates about zero, so a red/white/blue diverging
#    map centered at 0 is right there. The squared commutator is a squared norm:
#    non-negative, starting at 0 and growing to at most 4. It has no polarity,
#    so it gets a one-hue light->dark ramp. A diverging map here would invent a
#    midpoint that means nothing and split one physical quantity across two hues.
#
# 2. The lattice comes from kitaev_model.jl rather than being pasted in. The
#    distance shells are only meaningful if they are computed on the SAME bonds
#    the dynamics ran on, and that lattice is now periodic -- a stale open-cluster
#    bond list would silently distort every shell.
#
# Set `quantity = :F` to plot the OTOC itself instead; that one IS signed
# (+1 unscrambled, through 0, negative on overshoot) so it switches to the
# diverging map automatically.

include(joinpath(@__DIR__, "kitaev_model.jl"))
include(joinpath(@__DIR__, "otoc_plotting.jl"))

# --- Edit these ---------------------------------------------------------
npz_path = joinpath(@__DIR__, "otoc_outputs",
                    "PP_kitaev_otoc_infT_N16_L243535_K1.0-1.0-1.0_V0.3_J0.0_ZZ_site2_dt0.05_tmax3.0.npz")
quantity = :Csq          # :Csq (squared commutator) or :F (the OTOC itself)
clim_mode = :fixed       # :fixed -> (0,4) for Csq / (-1,1) for F, so runs are
                         #           directly comparable to each other
                         # :auto  -> scale to this run's own range
# ------------------------------------------------------------------------

run = load_run(npz_path)
m = run.meta
m.N == N || error("""
    This file is an N=$(m.N) run, but kitaev_model.jl currently defines N=$N.
    The distance shells would be computed on the wrong lattice. Either point
    npz_path at an N=$N run, or switch kitaev_model.jl back to the cluster this
    file was generated on.""")
# Matching N is not enough -- same site count, different bonds gives different
# distance shells. The fingerprint catches that; untagged older files can only
# be warned about.
if isempty(m.lat)
    @warn "This run predates the lattice fingerprint, so its bonds cannot be verified against " *
          "the current lattice. If the bond lists changed since, every distance shell below is wrong."
elseif m.lat != lattice_tag()
    error("This run is on lattice $(m.lat) but kitaev_model.jl is now $(lattice_tag()). Same N, " *
          "different bonds -- the distance shells would be wrong. Regenerate the run, or restore " *
          "the bond lists it was made with.")
end

@printf("Run: %s\n", basename(npz_path))
@printf("  solver=%s  N=%d  reference site=%d  pauli=%s  V=%s  J=%s  dt=%s\n",
        m.solver, m.N, m.site, m.pauli, m.V, m.J, m.dt)

dist = graph_distances(m.site, N, X_BONDS, Y_BONDS, Z_BONDS)
max_d = maximum(dist)
println("Graph distances from site $(m.site) (BFS over the periodic honeycomb bonds):")
for d in 0:max_d
    println("  distance $d: sites $(findall(dist .== d))")
end

Q = quantity === :Csq ? run.Csq : run.F
t = run.time

# Average over all sites at each graph distance. On a honeycomb there are
# generally several sites per shell, and on a PERIODIC cluster the shells are
# more uniformly populated than on an open one -- which is exactly why the
# averaged front is cleaner to read here than it was on the open cluster.
M = zeros(Float64, max_d + 1, length(t))
shell_counts = Int[]
for d in 0:max_d
    cols = [run.col(s) for s in findall(dist .== d)]
    cols = [c for c in cols if c !== nothing]
    push!(shell_counts, length(cols))
    isempty(cols) && continue
    M[d+1, :] = vec(sum(Q[:, cols], dims = 2)) ./ length(cols)
end
println("sites per shell: ", shell_counts)

is_csq = quantity === :Csq
cmap = is_csq ? SEQ_BLUE : DIV_BLUE_RED
clims = if clim_mode === :fixed
    is_csq ? (0.0, 4.0) : (-1.0, 1.0)
else
    is_csq ? (0.0, maximum(M)) : (-maximum(abs.(M)), maximum(abs.(M)))
end
# Short colorbar label: the full definition belongs in the title/caption, not
# rotated down the side of the figure where it collides with the tick labels.
qlabel = is_csq ? "squared commutator" : "OTOC F"

fig = heatmap(t, 0:max_d, M;
              base_attrs()...,
              xlabel = "t",
              ylabel = "graph distance from site $(m.site)",
              title  = "$(m.pauli) OTOC light cone — $(m.solver), N=$N, site $(m.site), V=$(m.V), J=$(m.J)",
              colorbar_title = "  " * qlabel,
              color  = cmap,
              clims  = clims,
              yticks = 0:max_d,
              titlefontsize = 11,
              guidefontsize = 11,
              tickfontsize  = 10,
              left_margin   = 10Plots.mm,
              bottom_margin = 8Plots.mm,
              right_margin  = 6Plots.mm,
              size = (860, 460), dpi = 200)

display(fig)
out = joinpath(plots_dir(),
    "otoc_lightcone_$(quantity)_" * replace(basename(npz_path), ".npz" => "") * ".png")
savefig(fig, out)
println("\nSaved: $out")

# A truncation warning belongs next to the picture, not buried in a log: with
# Pauli propagation the whole heatmap is only as trustworthy as norm2, and a
# light cone drawn from a run that lost half its operator weight looks just as
# crisp as a good one.
if m.solver == "PP"
    @printf("\nPP norm2: starts %.6f, ends %.6f\n", run.norm2[1], run.norm2[end])
    run.norm2[end] < 0.99 && @warn "norm2 has fallen to $(round(run.norm2[end], digits=4)) by " *
        "t=$(t[end]) -- late-time rows of this heatmap are unreliable. Plot otoc_growth.jl " *
        "alongside it, and read the light cone only over the window where norm2 ≈ 1."
end
