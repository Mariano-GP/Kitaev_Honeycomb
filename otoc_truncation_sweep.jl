# Squared-commutator convergence against truncation cutoff -- the Kitaev
# counterpart of the SYK "C_anti / C_anti_cor vs t" figure.
#
#     julia --project=. otoc_truncation_sweep.jl
#
# WHAT THE TWO SERIES ARE
# -----------------------
# Pauli propagation represents W_i(t) = sum_k c_k P_k. Split the strings by
# whether they commute with the probe V_j:
#
#     C_anti(t)  =  4 * sum_{k anticommuting with V_j} |c_k|^2
#
# That IS the squared commutator Tr[W(t),V]^dag[W(t),V] / 2^N -- "anti" names
# how it is computed, not a different quantity. Equivalently
# C_anti = 2*(norm2 - F), and this script asserts that identity on load.
#
#     C_anti_cor(t)  =  C_anti(t) / norm2(t)
#
# is the NORM-CORRECTED version. Truncation discards operator weight, so norm2
# falls below 1 and C_anti inherits the whole loss -- it decays toward zero at
# late t purely as an artifact. Dividing by the surviving norm restores the sum
# rule the discarded weight would have carried, on the assumption that what was
# thrown away was split between commuting and anticommuting strings in the same
# ratio as what survived. It is a correction, not a fix: it recovers most of the
# shortfall but still undershoots, and it cannot manufacture information about
# strings that were never propagated.
#
# Unlike the OTOC F, the squared commutator gets NO cancellation benefit from
# truncation (F = norm2 - 2*anti loses ~equally from both terms and stays
# accurate; C = 4*anti loses directly). That is exactly why this plot is the
# honest test of a cutoff and the F time-trace is not.
#
# HOW FILES ARE FOUND
# -------------------
# Nothing is hard-coded to a filename. Every .npz in otoc_outputs/ is read and
# grouped by physics -- N, lattice, reference site, Pauli, V, J. Each group that
# has an ED reference AND at least one eligible PP run gets its own figure, so
# adding runs (more cutoffs, longer t_max, other V/J) needs no edits here.
#
#   * ED reference: if several ED files exist for one group, the one reaching
#     the latest time is used.
#   * PP duplicates: several runs at the SAME cutoff (e.g. a t_max=3 and a
#     t_max=5 run) are one curve, not two -- the longest is kept and the others
#     are listed as skipped. Plotting both would draw identical overlapping
#     curves under the same legend label.
#   * Each PP curve is drawn to its own final time, so one shorter run does not
#     clip every other curve on the figure.

include(joinpath(@__DIR__, "otoc_plotting.jl"))

# --- What to plot ----------------------------------------------------------
probe_j = 2        # which target site's C_{i,j}(t) to show

# WHICH PHYSICS POINTS GET A FIGURE.
#   nothing                              -> every V/J that has ED + eligible PP data
#   [(V = 0.3, J = 0.05)]                -> just that one
#   [(V = 0.3, J = 0.05), (V = 0.0, J = 0.0)]
# An entry may also carry `site = 2` to restrict the reference site.
targets =  [(V = 0.0, J = 0.05)]   

show_corrected = true   # draw the dashed C_anti/norm2 series too

# `require_inactive_weight_cap` keeps the sweep one-dimensional: a cap of N can
# never bind, so those runs are truncated by coefficient ALONE. Mixing in
# weight-capped runs (w4, w10, ...) would put two knobs on one axis and the
# "convergence" would stop meaning anything.
require_inactive_weight_cap = true
# ---------------------------------------------------------------------------

# "5e-5", "2.5e-5", "1e-4" -- read as the cutoff was written, not "0.0001".
function cutoff_label(x::Real)
    e = floor(Int, log10(x))
    m = x / 10.0^e
    if m >= 10 - 1e-9; m /= 10; e += 1; end
    ms = isapprox(m, round(m); atol = 1e-6) ? string(Int(round(m))) : string(round(m, sigdigits = 3))
    return "$(ms)e$(e)"
end

physics_key(m::RunMeta) = (N = m.N, lat = m.lat, site = m.site, pauli = m.pauli, V = m.V, J = m.J)

function wanted(key)
    targets === nothing && return true
    any(targets) do tg
        tg.V == key.V && tg.J == key.J && (!haskey(tg, :site) || tg.site == key.site)
    end
end

# ---- read and classify every output file ------------------------------------
ed_files = Dict{NamedTuple,Vector{Tuple{String,Float64}}}()   # key => [(path, t_end)]
pp_files = Dict{NamedTuple,Vector{NamedTuple}}()
for path in find_runs()
    local m
    try
        m = parse_meta(path)
    catch
        continue                                  # not a solver output we recognise
    end
    isempty(m.lat) && continue                    # predates the lattice fingerprint
    key = physics_key(m)
    if m.solver == "ED"
        push!(get!(ed_files, key, Tuple{String,Float64}[]), (path, m.tmax))
    else
        d = npzread(path)
        haskey(d, "trunc") || continue            # cutoff unrecorded: cannot place on the axis
        cut, wmax = d["trunc"][1], Int(d["trunc"][2])
        push!(get!(pp_files, key, NamedTuple[]),
              (path = path, cut = cut, wmax = wmax, t_end = Float64(d["time"][end])))
    end
end

# One color per CUTOFF VALUE, fixed across every figure this script draws: the
# same cutoff is the same hue wherever it appears, and adding a new cutoff to
# one group never repaints the others. Ordered coarse -> fine.
cut_ladder = sort(unique([c.cut for (key, runs) in pp_files for c in runs
                          if !require_inactive_weight_cap || c.wmax >= key.N]);
                  rev = true)
cut_color(cut) = series_color(findfirst(==(cut), cut_ladder))

# ---- one figure per physics group -------------------------------------------
function plot_group(key, ed_path::String, pp_runs::Vector{NamedTuple})
    ed = load_run(ed_path)
    if ed.col(probe_j) === nothing
        println("  skipped: ED run does not probe site $probe_j (has $(join(ed.sites, ", ")))")
        return nothing
    end

    # Eligible = cutoff-only (if required), has the probe column, and satisfies
    # the identity the plot rests on.
    eligible = NamedTuple[]
    for c in pp_runs
        require_inactive_weight_cap && c.wmax < key.N && continue
        r = load_run(c.path)
        r.col(probe_j) === nothing && continue
        @assert maximum(abs.(r.Csq .- 2 .* (r.norm2 .- r.F))) < 1e-10 "Csq != 2(norm2-F) in $(basename(c.path))"
        push!(eligible, merge(c, (run = r,)))
    end

    # Collapse duplicates at the same cutoff to the longest run.
    by_cut = Dict{Float64,Vector{NamedTuple}}()
    for c in eligible
        push!(get!(by_cut, c.cut, NamedTuple[]), c)
    end
    chosen, skipped = NamedTuple[], NamedTuple[]
    for (_, group) in by_cut
        sort!(group, by = c -> -c.t_end)
        push!(chosen, group[1])
        append!(skipped, group[2:end])
    end
    if isempty(chosen)
        println("  skipped: no cutoff-only PP runs for this group")
        return nothing
    end
    sort!(chosen, by = c -> -c.cut)               # coarse first, as the curves converge

    @printf("  reference: %s  (t <= %.2f)\n", basename(ed_path), maximum(ed.time))
    @printf("  %d cutoff(s):\n", length(chosen))
    for c in chosen
        @printf("     cutoff %-8s w<=%2d  t_end=%.2f  norm2(end)=%.4f   %s\n",
                cutoff_label(c.cut), c.wmax, c.t_end, c.run.norm2[end], basename(c.path))
    end
    for c in skipped
        @printf("     (skipped duplicate, shorter run at cutoff %s: %s)\n",
                cutoff_label(c.cut), basename(c.path))
    end
    length(chosen) < 4 && println("  note: only $(length(chosen)) cutoff(s) -- a comparison, " *
                                  "not yet a convergence sweep")

    # Frame the time the exact reference covers, but no further than the
    # longest PP run reaches.
    t_win = min(maximum(ed.time), maximum(c.t_end for c in chosen))
    ed_i  = findall(t -> t <= t_win + 1e-9, ed.time)

    p = plot(; base_attrs()...,
             size = (1240, 820), dpi = 200,
             xlabel = "t", ylabel = L"C_{%$(key.site),%$(probe_j)}(t)",
             guidefontsize = 15, tickfontsize = 12,
             title = "Squared commutator vs truncation cutoff  ·  " *
                     "N=$(key.N) $(key.pauli) site $(key.site)→$(probe_j)  V=$(key.V) J=$(key.J)",
             titlefontsize = 15,
             legend = :bottomright, legendfontsize = 10,
             left_margin = 10Plots.mm, bottom_margin = 8Plots.mm,
             right_margin = 6Plots.mm, top_margin = 5Plots.mm,
             grid = true)

    # Style-only legend keys: ONE entry per cutoff (color) plus two explaining
    # solid vs dashed, rather than every cutoff listed twice.
    plot!(p, Float64[], Float64[]; color = INK_SECONDARY, lw = 2,
          linestyle = :solid, label = "C_anti  (raw)")
    show_corrected && plot!(p, Float64[], Float64[]; color = INK_SECONDARY, lw = 2,
          linestyle = :dash, label = "C_anti_cor  (÷ norm²)")

    for c in chosen
        r   = c.run
        idx = findall(t -> t <= t_win + 1e-9, r.time)
        col = r.col(probe_j)
        C   = r.Csq[idx, col]
        hue = cut_color(c.cut)
        plot!(p, r.time[idx], C; color = hue, lw = 2, linestyle = :solid,
              label = "cutoff " * cutoff_label(c.cut))
        show_corrected && plot!(p, r.time[idx], C ./ r.norm2[idx];
                                color = hue, lw = 2, linestyle = :dash, label = "")
    end

    # Exact last => drawn on top, in ink: it is the reference, not one more
    # member of the family.
    plot!(p, ed.time[ed_i], ed.Csq[ed_i, ed.col(probe_j)];
          color = INK_PRIMARY, lw = 3.5, label = "Exact (ED)")

    out = joinpath(plots_dir(),
        "otoc_truncation_sweep_C_N$(key.N)_$(key.lat)_$(key.pauli)_site$(key.site)_to$(probe_j)_" *
        "V$(key.V)_J$(key.J)_tmax$(@sprintf("%.1f", t_win)).png")
    savefig(p, out)
    println("  saved: $(basename(out))")

    # The numbers, at the latest time EVERY chosen run reaches, so the rows are
    # comparable with each other.
    t_cmp = min(maximum(ed.time), minimum(c.t_end for c in chosen))
    ke = findlast(t -> t <= t_cmp + 1e-9, ed.time)
    Cex = ed.Csq[ke, ed.col(probe_j)]
    println("  " * "-"^76)
    @printf("  %-9s %8s %12s %12s %12s %10s\n",
            "cutoff", "norm2", "C_anti", "C_anti_cor", "C_exact", "cor gain")
    println("  " * "-"^76)
    for c in chosen
        r = c.run
        k = findlast(t -> t <= t_cmp + 1e-9, r.time)
        Cr, n2 = r.Csq[k, r.col(probe_j)], r.norm2[k]
        Cc = Cr / n2
        gap = Cex - Cr
        gain = abs(gap) < 1e-6 ? "-" : @sprintf("%.1f%%", 100 * (Cc - Cr) / gap)
        @printf("  %-9s %8.4f %12.4f %12.4f %12.4f %10s\n",
                cutoff_label(c.cut), n2, Cr, Cc, Cex, gain)
    end
    println("  " * "-"^76)
    @printf("  at t = %.2f.  \"cor gain\" = share of the gap to exact that ÷norm2 recovers.\n", t_cmp)
    return out
end

saved = String[]
groups = sort(collect(intersect(keys(ed_files), keys(pp_files))),
              by = k -> (k.V, k.J, k.site, k.pauli))
for key in groups
    wanted(key) || continue
    println("\n" * "="^78)
    println("V=$(key.V)  J=$(key.J)  site $(key.site)  $(key.pauli)  N=$(key.N)  $(key.lat)  " *
            "probe j=$probe_j")
    println("="^78)
    ed_path = first(sort(ed_files[key], by = x -> -x[2]))[1]
    out = plot_group(key, ed_path, pp_files[key])
    out === nothing || push!(saved, out)
end

# Physics points with data on only one side, so a missing figure is explained
# rather than silently absent.
only_ed = sort([k for k in keys(ed_files) if !haskey(pp_files, k) && wanted(k)], by = k -> (k.V, k.J))
only_pp = sort([k for k in keys(pp_files) if !haskey(ed_files, k) && wanted(k)], by = k -> (k.V, k.J))
println("\n" * "="^78)
println("$(length(saved)) figure(s) saved to $(plots_dir())")
isempty(only_ed) || println("ED reference but no PP runs:  " *
                            join(["V=$(k.V) J=$(k.J) site$(k.site)" for k in only_ed], ";  "))
isempty(only_pp) || println("PP runs but no ED reference:  " *
                            join(["V=$(k.V) J=$(k.J) site$(k.site)" for k in only_pp], ";  "))
isempty(saved) && error("No figures produced. Each figure needs an ED file and at least one " *
                        "cutoff-only PP run (max_weight = N) for the same V, J, site and Pauli" *
                        (targets === nothing ? "." : ", among the requested targets."))
