# Operator growth and truncation loss -- the OTOC counterpart of
# Correlations/operator_growth.jl.
#
#     julia --project=. otoc_growth.jl
#
# TWO PANELS, NOT TWO Y-AXES. |W|(t) counts Pauli terms and runs over orders of
# magnitude; norm2(t) is a fidelity that sits just below 1. Putting them on one
# plot with two y-scales would let the vertical distance between the curves mean
# whatever the scaling happened to make it mean, which is the single most
# misleading thing a chart can do. They are stacked as two panels sharing one
# time axis instead, so every curve is read against exactly one scale.
#
# Read them together: |W| tells you what a run COST, norm2 tells you what it was
# WORTH. A configuration whose |W| stays flat because the truncation is throwing
# everything away is cheap and wrong, and only the norm2 panel reveals it --
# under exact evolution norm2 == 1 for all t, so any shortfall is precisely the
# operator weight the truncation discarded.
#
# ED runs have no "terms" (there is no Pauli sum) and norm2 == 1 identically, so
# they are skipped here; that constancy is the baseline the PP curves fall away
# from.

include(joinpath(@__DIR__, "otoc_plotting.jl"))

# --- Edit this: the runs to overlay. Empty => every PP file in otoc_outputs/ --
paths = String[
    # joinpath(@__DIR__, "otoc_outputs", "PP_..._J0.0_ZZ_site2_dt0.05_tmax3.0.npz"),
]
isempty(paths) && (paths = find_runs(prefix = "PP_"))
# ----------------------------------------------------------------------------

runs = [load_run(p) for p in paths]
filter!(r -> r.terms !== nothing, runs)
isempty(runs) && error("no PP runs with a \"terms\" array found in otoc_outputs/")

println("Plotting $(length(runs)) run(s):")
for r in runs; println("  ", basename(r.meta.path)); end

# Legend labels must distinguish the runs from each other, so build them from
# whichever parameters actually differ across this particular set -- a legend
# that repeats identical text for every entry is no legend at all.
function distinguishing_labels(runs)
    # Truncation first: it is what this plot exists to compare, so when runs
    # differ in it that is what the legend should lead with.
    fields = [(:trunc, r -> r.meta.trunc),
              (:N, r -> r.meta.N), (:V, r -> r.meta.V), (:J, r -> r.meta.J),
              (:dt, r -> r.meta.dt), (:tmax, r -> r.meta.tmax),
              (:site, r -> r.meta.site), (:pauli, r -> r.meta.pauli)]
    varying = [(name, f) for (name, f) in fields if length(unique(f(r) for r in runs)) > 1]
    isempty(varying) && return [basename(r.meta.path)[1:min(end-4, 34)] for r in runs]
    return [join(["$name=$(f(r))" for (name, f) in varying], "  ") for r in runs]
end
labels = distinguishing_labels(runs)

# Legend OUTSIDE the axes. Inside, it sat on top of the |W| curves in the
# top-left, which is where every run starts -- an overlay that hides the data it
# is labelling is worse than no legend.
common = base_attrs(legend = :outertopright, legendfontsize = 8, grid = true,
                    guidefontsize = 11, tickfontsize = 10, titlefontsize = 11)

p_terms = plot(; common..., ylabel = "|W|  (# Pauli terms)", yscale = :log10,
                 title = "Operator growth and truncation loss")
# norm2 shares the legend above it, so its own is suppressed -- but it keeps an
# empty outer-right slot so both panels have the same plotting width and their
# time axes line up vertically.
p_norm  = plot(; common..., xlabel = "t", ylabel = "norm2 = Σ|cₖ|²",
                 foreground_color_legend = nothing, background_color_legend = nothing)

for (i, (r, lab)) in enumerate(zip(runs, labels))
    c = series_color(i)
    plot!(p_terms, r.time, max.(r.terms, 1); label = lab, color = c, lw = 2)
    plot!(p_norm,  r.time, r.norm2;          label = "", color = c, lw = 2)
end

# The two panels are read against ONE time axis, so their plotting areas must be
# the same width. The top panel surrenders width to its outer legend, so the
# bottom one is given an invisible legend entry of matching text width -- the
# box is transparent (set above), leaving only the reserved space. Without this
# the panels have different pixel-per-unit-t scales and a feature at t=1 in one
# does not sit above the same t=1 in the other.
plot!(p_norm, [NaN], [NaN]; label = " "^maximum(length, labels), linealpha = 0)

# The exact-evolution baseline. norm2 == 1 is not a target to approach, it is
# what an untruncated run gives identically, so it is drawn as a reference rule
# rather than as another series competing for a legend slot.
hline!(p_norm, [1.0]; color = INK_MUTED, lw = 1, linestyle = :dash, label = "")
annotate!(p_norm, [(0.02, 1.0, text("  exact: norm2 = 1", 8, :left, :bottom, INK_MUTED))])

fig = plot(p_terms, p_norm; layout = grid(2, 1, heights = [0.6, 0.4]),
           size = (1000, 620), dpi = 200,
           left_margin = 10Plots.mm, bottom_margin = 8Plots.mm)
display(fig)
out = joinpath(plots_dir(), "otoc_operator_growth.png")
savefig(fig, out)
println("\nSaved: $out")

println("\n" * "-"^72)
@printf("%-40s %12s %12s\n", "run", "|W| final", "norm2 final")
println("-"^72)
for (r, lab) in zip(runs, labels)
    @printf("%-40s %12d %12.6f\n", lab, Int(r.terms[end]), r.norm2[end])
end
println("-"^72)
