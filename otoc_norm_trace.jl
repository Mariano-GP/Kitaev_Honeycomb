# Operator norm over time, for every Pauli-propagation run in otoc_outputs/.
#
#     julia --project=. otoc_norm_trace.jl
#
# WHAT IS PLOTTED, AND WHY IT IS NOT ALWAYS "norm2"
# -------------------------------------------------
# Under exact evolution W(t) keeps unit norm, so norm2 = sum_k |c_k|^2 == 1 and
# any shortfall is exactly the operator weight truncation threw away. That makes
# norm2(t) the error budget of an ordinary run.
#
# Runs from kitaev_OTOC_infinite_temp_normalized.jl rescale W to unit norm after
# every timestep, so their norm2 is 1 BY CONSTRUCTION -- plotting it would draw a
# flat line at 1 and read as "lossless", which is the opposite of the truth. For
# those runs the discarded weight lives in "cum_retention" (the running product
# of per-step surviving fractions), and that is what gets plotted instead.
#
# Both appear on the same axis because they measure the same thing -- the
# fraction of the operator still being propagated -- but they are NOT the same
# quantity: rescaling changes which terms clear the next |c| threshold, so a
# normalized run and its un-normalized twin follow different trajectories. They
# are drawn DASHED and labelled "(norm.)" so the two are never read as one
# family.
#
# ED runs are excluded: exact evolution gives norm2 == 1 identically, so they
# would add a flat line on top of the reference rule already drawn at 1.

include(joinpath(@__DIR__, "otoc_plotting.jl"))
using LaTeXStrings

# --- What to plot ------------------------------------------------------------
site_sel   = nothing    # nothing -> every reference site; or e.g. 2
pauli_sel  = "ZZ"
lat_sel    = nothing    # nothing -> every lattice; or e.g. "L243535"
V_values   = [0.0]      # nothing -> every V present; or e.g. [0.0]
J_values   = [0.05]     # nothing -> every J present; or e.g. [0.05]

# Which truncation strategies to include:
#   :damped      -- weight-damped runs only (wdamp / mwdamp)
#   :coefficient -- plain coefficient / weight-cap runs only
#   :all         -- every strategy on one axis
# Note this is NOT a lattice filter. Runs on different lattices should never
# share an axis, but that is what `lat_sel` is for -- the lattice fingerprint is
# in every filename and is the only thing that establishes two runs are on the
# same graph.
strategy   = :damped

# Renormalized runs (kitaev_OTOC_infinite_temp_normalized.jl) have norm2 pinned
# to 1, so the only thing they can contribute is their cum_retention curve. With
# this false they are dropped entirely rather than drawn as a second, dashed
# family -- which keeps ONE quantity on the axis instead of two that measure the
# same idea along different trajectories. Set true to bring them back.
include_normalized = false

logy        = false     # true -> log y axis, to separate runs that all collapse
show_sqrt   = false     # true -> plot the norm ||W|| = sqrt(norm2) instead of norm2
min_tmax    = 0.0       # raise to drop short smoke-test runs
# -----------------------------------------------------------------------------

# ---- load every PP run, picking the right norm series for its kind -----------
runs = NamedTuple[]
skipped = String[]
for path in find_runs(prefix = "PP_")
    local m
    try
        m = parse_meta(path)
    catch
        push!(skipped, basename(path))        # unparseable name: say so, never drop silently
        continue
    end
    (pauli_sel === nothing || m.pauli == pauli_sel) || continue
    (site_sel === nothing || m.site == site_sel) || continue
    (lat_sel === nothing || m.lat == lat_sel) || continue
    m.tmax >= min_tmax || continue

    d = npzread(path)
    # Damping kind from the payload (trunc[5]: 0 none, 1 Pauli, 2 Majorana),
    # falling back to the filename tag for files written before that slot
    # existed. Reading the payload rather than the name means a renamed file
    # cannot misclassify itself.
    damp_kind = (haskey(d, "trunc") && length(d["trunc"]) >= 5) ? d["trunc"][5] :
                (startswith(m.trunc, "wdamp") ? 1.25 :
                 startswith(m.trunc, "wdamp")  ? 1.0 : 0.0)
    is_damped = damp_kind != 0.0
    (strategy === :all ||
     (strategy === :damped      && is_damped) ||
     (strategy === :coefficient && !is_damped)) || continue

    # A run is renormalized if it says so, or (older files) if it carries the
    # cum_retention array. Fall back to norm2 for everything else.
    is_norm = (haskey(d, "normalized") && d["normalized"][1] == 1.0) || haskey(d, "cum_retention")
    (include_normalized || !is_norm) || continue
    series = is_norm && haskey(d, "cum_retention") ? d["cum_retention"] : d["norm2"]
    push!(runs, (meta = m, time = d["time"], series = Float64.(series),
                 normalized = is_norm, damp_kind = damp_kind,
                 terms = haskey(d, "terms") ? d["terms"] : nothing))
end
isempty(runs) && error("no PP runs matched in otoc_outputs/")

keep(vals, x) = vals === nothing || any(v -> isapprox(v, x; atol = 1e-12), vals)
filter!(r -> keep(V_values, r.meta.V) && keep(J_values, r.meta.J), runs)
isempty(runs) && error("no PP runs match V_values=$V_values, J_values=$J_values")

isempty(skipped) || println("skipped $(length(skipped)) file(s) whose name could not be parsed:\n   " *
                            join(skipped, "\n   "))

# One panel per physics point. Comparing norm decay ACROSS different V/J on one
# axis would put runs with different Hamiltonians -- and so different intrinsic
# operator growth -- in the same frame, where a difference in truncation cannot
# be told from a difference in the physics.
cells = sort(unique([(r.meta.V, r.meta.J) for r in runs]))
# +1 cell for a LEGEND-ONLY panel. Colour is assigned per truncation tag
# globally, so a legend built from any one data panel names only that panel's
# tags -- here the first cell holds 2 curves while another holds 25, leaving most
# of them unidentified. Giving the legend its own cell names every tag exactly
# once and occludes no data.
n_panels = length(cells) + 1
cols = ceil(Int, sqrt(n_panels))
rows = ceil(Int, n_panels / cols)

# One colour per TRUNCATION TAG, fixed across every panel, so the same strategy
# is the same hue wherever it appears.
tags = sort(unique([r.meta.trunc for r in runs]))
tag_color(t) = series_color(findfirst(==(t), tags))

yvals(r) = show_sqrt ? sqrt.(max.(r.series, 0.0)) : r.series
ylab = show_sqrt ? L"\|W(t)\|" : L"\mathrm{norm}^2 = \sum_k |c_k|^2"

# ---- report ------------------------------------------------------------------
println("\n$(length(runs)) PP run(s) over $(length(cells)) physics point(s)   [strategy = :$strategy]")
# Lattices present. Two runs on different lattices are different GRAPHS, and a
# shared axis would read a lattice difference as a truncation difference -- so
# say it out loud rather than leave it to be noticed.
lats_here = sort(unique([r.meta.lat for r in runs]))
if length(lats_here) > 1
    @warn "these runs span SEVERAL lattices ($(join(lats_here, ", "))) -- they are " *
          "different graphs and should not share an axis. Set lat_sel to one of them."
else
    println("lattice: $(only(lats_here))  (all runs on the same graph)")
end
println("-"^100)
@printf("%6s %6s  %-22s %6s %8s %12s  %s\n",
        "V", "J", "truncation", "site", "t_max", "final", "series")
println("-"^100)
for (V, J) in cells, r in sort([x for x in runs if (x.meta.V, x.meta.J) == (V, J)],
                               by = x -> x.meta.trunc)
    @printf("%6.3g %6.3g  %-22s %6d %8.2f %12.6f  %s\n",
            V, J, r.meta.trunc, r.meta.site, r.meta.tmax, r.series[end],
            r.normalized ? "cum_retention (renormalized run)" : "norm2")
end
println("-"^100)
if any(r -> r.normalized, runs)
    println("A renormalized run's norm2 is pinned to 1, so its CUM_RETENTION is plotted")
    println("instead -- dashed. The two are comparable in meaning but are not the same")
    println("trajectory.")
else
    println("Renormalized runs are excluded (include_normalized = false), so every curve")
    println("here is norm2 = sum_k |c_k|^2 -- one quantity, one meaning.")
end

# ---- figure -------------------------------------------------------------------
on_bottom(i) = i > length(cells) - cols
on_left(i)   = mod1(i, cols) == 1

panels = map(enumerate(cells)) do (idx, (V, J))
    fam = sort([r for r in runs if (r.meta.V, r.meta.J) == (V, J)], by = r -> r.meta.trunc)
    p = plot(; base_attrs()...,
             title = "V = $V,  J = $J", titlefontsize = 13, tickfontsize = 11,
             xlabel = on_bottom(idx) ? "time  t" : "",
             ylabel = on_left(idx) ? ylab : "",
             guidefontsize = 12,
             legend = false,          # the dedicated legend panel carries it
             yscale = logy ? :log10 : :identity,
             ylims = logy ? :auto : (0, 1.05),
             grid = true)
    # Exact evolution keeps the norm at 1. Drawn as a reference RULE rather than
    # as another series: it is not a run, it is the value every run falls away
    # from.
    hline!(p, [1.0]; color = INK_MUTED, lw = 1, linestyle = :dash, label = "")
    for r in fam
        plot!(p, r.time, yvals(r);
              color = tag_color(r.meta.trunc), lw = 2,
              linestyle = r.normalized ? :dash : :solid,
              label = "")
    end
    p
end

# The legend as its own panel: one key per truncation tag (colour), plus two
# explaining solid vs dashed. Keys carry the SAME colour and linestyle as the
# curves they stand for, or the legend misdescribes the plot.
legend_panel = plot(; base_attrs()..., framestyle = :none, grid = false,
                    xlims = (0, 1), ylims = (0, 1), xticks = false, yticks = false,
                    legend = :left, legendfontsize = 7)
for t in tags
    plot!(legend_panel, Float64[], Float64[]; color = tag_color(t), lw = 2, label = t)
end
# The solid/dashed style keys exist only to separate the two series types. With
# renormalized runs excluded there is one type on the axis, and a key explaining
# a distinction the figure does not contain is worse than no key.
if any(r -> r.normalized, runs)
    plot!(legend_panel, Float64[], Float64[]; color = INK_SECONDARY, lw = 2,
          linestyle = :solid, label = "norm2  (ordinary run)")
    plot!(legend_panel, Float64[], Float64[]; color = INK_SECONDARY, lw = 2,
          linestyle = :dash, label = "cum_retention  (renormalized)")
end

fig = plot(panels..., legend_panel; layout = (rows, cols),
           size = (max(520 * cols, 1100), 400 * rows + 110), dpi = 200,
           plot_title = L"\mathrm{Surviving\ operator\ weight\ vs\ time}",
           plot_titlefontsize = 17, top_margin = 6Plots.mm,
           left_margin = 12Plots.mm, bottom_margin = 10Plots.mm, right_margin = 5Plots.mm)

# Windows locks a PNG an image viewer has open; one locked file must not abort.
function save_resilient(fig, path)
    for attempt in 1:3
        try
            savefig(fig, path); return path
        catch
            attempt < 3 && (sleep(1.5); continue)
            alt = replace(path, ".png" => "_new.png")
            try
                savefig(fig, alt)
                @warn "could not overwrite $(basename(path)); wrote $(basename(alt))"
                return alt
            catch err
                @error "could not save $(basename(path))" exception = err
                return nothing
            end
        end
    end
end

# The V/J filter goes in the filename: a restricted figure and the all-points
# figure are different plots, and sharing a name would have one silently
# overwrite the other.
filt_tag = ((V_values === nothing && J_values === nothing) ? "_all" :
            "_V" * (V_values === nothing ? "all" : join(V_values, "-")) *
            "_J" * (J_values === nothing ? "all" : join(J_values, "-"))) *
           (strategy === :all ? "" : "_$(strategy)")
out = joinpath(plots_dir(),
    "otoc_surviving_weight_$(show_sqrt ? "normW" : "norm2")$(filt_tag)$(logy ? "_log" : "").png")
s = save_resilient(fig, out)
s === nothing || println("\nSaved: $(basename(s))")
