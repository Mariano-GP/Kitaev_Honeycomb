# OTOC-PP cluster results (CSV): ONE FIGURE PER J, comparing all truncations.
#
#     julia --project=. otocpp_csv_grid.jl
#
# The cluster sweep has two axes -- J and the truncation cutoff -- so the two
# are split across the two levels that keep each readable:
#
#   * one FIGURE per J value (the physics),
#   * inside it one PANEL per probe site, and within a panel one CURVE per
#     cutoff (the numerics). Convergence is then read off directly: curves that
#     lie on top of each other are converged in the cutoff, curves that do not
#     are still truncation-limited.
#
# Two things differ from otoc_VJ_grid.jl, both forced by what OTOC-PP computes:
#
#   * NO V AXIS. Their kitaev_honeycomb model is
#     K*sigma^g sigma^g + J*(XX+YY+ZZ) + fields -- there is no three-body V
#     term, so V is not a knob there and the cutoff takes its place.
#   * THE QUANTITY IS C, NOT F. Their CSV column `otoc_j` is the squared
#     commutator ||[W(t),V_j]||^2 = 4*sum_anti |c|^2 -- the same definition as
#     `Csq` here. They do not save norm2, so F = norm2 - C/2 cannot be
#     reconstructed from these files.
#
# `corrected_j` (dashed) is THEIR truncation correction: the drop in C measured
# immediately before and after every truncation, accumulated and added back. It
# is a different scheme from this project's C_anti/norm2 and the two are not
# interchangeable.
#
# NOT COMPARABLE TO THIS PROJECT'S ED RUNS. Their lattice is generated from unit
# cells; at 16 sites (2x4 or 4x2) it is only two cells wide, so wrapping closes
# square plaquettes. It is not the same graph as this project's torus under any
# relabelling, so no ED curve belongs on these axes.

include(joinpath(@__DIR__, "otoc_plotting.jl"))
using DelimitedFiles
using LaTeXStrings

# --- What to plot ------------------------------------------------------------
data_dir = joinpath(@__DIR__, "..", "OTOC-PP", "new_data", "honeycomb_mguerre3")
out_dir  = joinpath(@__DIR__, "..", "OTOC-PP", "plots")

show_corrected = true    # dashed: their truncation-corrected curve

# EXACT REFERENCE (ED), drawn as a thick black line when one exists.
#
# This project's ED solver can run on OTOC-PP's lattice: set
# KITAEV_LATTICE=otocpp4x2 (see kitaev_model.jl) and it stamps THAT lattice's
# own fingerprint into the filename. Only files carrying that fingerprint are
# eligible here. An ED run on this project's torus has the same N, K, J, dt and
# t_max but is a DIFFERENT GRAPH -- 4 square plaquettes against 0, not
# isomorphic under any relabelling -- so drawing it as "exact" would show a gap
# that reads as PP truncation error while actually being the lattice. Such
# files are reported and skipped, never plotted.
show_ed = true
ed_dir  = joinpath(@__DIR__, "otoc_outputs")
ed_lattice_tag = "L242ad9"   # fingerprint of KITAEV_LATTICE=otocpp4x2; torus16 is L243535

# WHICH PROBE SITES GET A PANEL.
#   nothing   -> one representative site per bond-distance shell from W
#   [2, 8, 5] -> exactly these
#   :all      -> every probe site (many panels)
show_sites = nothing

J_values      = nothing  # nothing -> a figure for every J present; or e.g. [0.0, 0.05]
cutoff_values = nothing  # nothing -> every cutoff present; or e.g. [1e-4, 5e-5]
t_max_plot    = nothing  # nothing -> the longest run in the figure
# ------------------------------------------------------------------------------

function cutoff_label(x::Real)
    e = floor(Int, log10(x)); m = x / 10.0^e
    if m >= 10 - 1e-9; m /= 10; e += 1; end
    ms = isapprox(m, round(m); atol = 1e-6) ? string(Int(round(m))) : string(round(m, sigdigits = 3))
    return "$(ms)e$(e)"
end

# Their lattice, rebuilt from OTOC-PP's _honeycomb_bonds, so distances are real
# bond distances -- their site NUMBERS run cell by cell and carry no geometry.
function honeycomb_bonds(Lx, Ly, periodic)
    a(m, n) = 2 * ((n - 1) * Lx + (m - 1)) + 1
    b(m, n) = a(m, n) + 1
    out = Tuple{Int,Int}[]
    for n in 1:Ly, m in 1:Lx
        push!(out, (a(m, n), b(m, n)))
        (m < Lx) ? push!(out, (b(m, n), a(m + 1, n))) :
            (periodic && Lx > 1 && push!(out, (b(m, n), a(1, n))))
        (n < Ly) ? push!(out, (b(m, n), a(m, n + 1))) :
            (periodic && Ly > 1 && push!(out, (b(m, n), a(m, 1))))
    end
    return out
end

function bond_distances(Lx, Ly, periodic, origin, N)
    adj = [Int[] for _ in 1:N]
    for (i, j) in honeycomb_bonds(Lx, Ly, periodic)
        push!(adj[i], j); push!(adj[j], i)
    end
    d = fill(-1, N); d[origin] = 0; q = [origin]
    while !isempty(q)
        u = popfirst!(q)
        for v in adj[u]; d[v] == -1 && (d[v] = d[u] + 1; push!(q, v)); end
    end
    return d
end

# ---- load every CSV/meta pair ------------------------------------------------
partition_eq(s) = (i = findfirst(==('='), s); i === nothing ? (s, "", "") :
                   (s[1:prevind(s, i)], "=", s[nextind(s, i):end]))

function read_meta(path)
    m = Dict{String,String}()
    for line in eachline(path)
        k, sep, v = partition_eq(line)
        isempty(sep) && continue
        m[strip(k)] = strip(v)
    end
    return m
end

function load_otocpp_run(csv_path)
    meta_path = replace(csv_path, ".csv" => ".meta.txt")
    isfile(meta_path) || return nothing
    md = read_meta(meta_path)
    mm = match(r"honeycomb:(\d+),(\d+)(,periodic)?", get(md, "lattice", ""))
    mm === nothing && return nothing
    Lx, Ly, per = parse(Int, mm.captures[1]), parse(Int, mm.captures[2]), mm.captures[3] !== nothing
    pr = get(md, "params", "")
    getp(k, d) = (g = match(Regex("(?:^|,)$k=([-\\d.eE+]+)"), pr); g === nothing ? d : parse(Float64, g.captures[1]))

    raw, hdr = readdlm(csv_path, ','; header = true)
    names = vec(string.(hdr))
    time = Float64.(raw[:, findfirst(==("time"), names)])
    icol = findfirst(==("n_terms"), names)
    terms = icol === nothing ? nothing : Int.(raw[:, icol])
    sites = Int[]; C = Dict{Int,Vector{Float64}}(); Ccor = Dict{Int,Vector{Float64}}()
    for (k, nm) in enumerate(names)
        g = match(r"^otoc_(\d+)$", nm)
        g === nothing && continue
        j = parse(Int, g.captures[1]); push!(sites, j)
        C[j] = Float64.(raw[:, k])
        kc = findfirst(==("corrected_$j"), names)
        kc === nothing || (Ccor[j] = Float64.(raw[:, kc]))
    end
    return (path = csv_path, Lx = Lx, Ly = Ly, periodic = per, N = 2 * Lx * Ly,
            K = getp("K", 1.0), J = getp("J", 0.0),
            cut = parse(Float64, get(md, "cutoff", "NaN")),
            w_site = parse(Int, get(md, "W_site", "1")),
            dt = parse(Float64, get(md, "dt", "NaN")),
            t_max = parse(Float64, get(md, "t_max", "NaN")),
            elapsed = parse(Float64, get(md, "elapsed_s", "NaN")),
            max_terms = parse(Int, get(md, "max_n_terms", "0")),
            time = time, terms = terms, sites = sort(sites), C = C, Ccor = Ccor)
end

isdir(data_dir) || error("no such directory: $data_dir")
runs = filter(!isnothing, [load_otocpp_run(joinpath(data_dir, f))
                           for f in sort(readdir(data_dir)) if endswith(f, ".csv")])
isempty(runs) && error("no OTOC-PP CSV/meta pairs found in $data_dir")
mkpath(out_dir)

keep(vals, x) = vals === nothing || any(v -> isapprox(v, x; atol = 1e-12), vals)
runs = [r for r in runs if keep(J_values, r.J) && keep(cutoff_values, r.cut)]
isempty(runs) && error("no runs match J_values=$J_values, cutoff_values=$cutoff_values")

# One color per CUTOFF VALUE, fixed across every figure: the same cutoff is the
# same hue everywhere, and adding a cutoff to one J never repaints another.
cut_ladder = sort(unique([r.cut for r in runs]); rev = true)      # coarse -> fine
cut_color(c) = series_color(findfirst(==(c), cut_ladder))

# Find the exact reference for one (J, W site), if this project has run one on
# the matching lattice. Returns the loaded run plus any same-physics files that
# were REJECTED for being on another lattice, so the caller can say so out loud
# rather than leaving a silently missing curve.
function ed_for(J, w_site)
    (show_ed && isdir(ed_dir)) || return (run = nothing, wrong = String[])
    cands = NamedTuple[]
    for f in readdir(ed_dir)
        (startswith(f, "ED_") && endswith(f, ".npz")) || continue
        local m
        try
            m = parse_meta(joinpath(ed_dir, f))
        catch
            continue
        end
        # V is fixed at 0: OTOC-PP's model has no three-body term at all.
        (m.site == w_site && m.pauli == "ZZ" && m.V == 0.0 &&
         isapprox(m.J, J; atol = 1e-12)) || continue
        push!(cands, (meta = m, ok = m.lat == ed_lattice_tag))
    end
    good = [c for c in cands if c.ok]
    isempty(good) && return (run = nothing,
                             wrong = [basename(c.meta.path) for c in cands])
    best = argmax(c -> c.meta.tmax, good)       # the longest matching run
    return (run = load_run(best.meta.path), wrong = String[])
end

# A figure is one (lattice, W site, J): a 2x4 and a 4x2 cluster are different
# graphs with different site numbering and do not belong on shared axes.
figs = sort(unique([(r.Lx, r.Ly, r.periodic, r.w_site, r.J) for r in runs]))

# Windows keeps a lock on a PNG that an image viewer (or VS Code's preview) has
# open, and Julia surfaces that as SystemError "Invalid argument" when savefig
# tries to overwrite it. One locked file must not abort a whole sweep: the
# figures after it would never be written either. Retry briefly, then fall back
# to a suffixed filename and say so, rather than throwing.
function save_resilient(fig, path)
    for attempt in 1:3
        try
            savefig(fig, path)
            return path
        catch
            attempt < 3 && (sleep(1.5); continue)
            alt = replace(path, ".png" => "_new.png")
            try
                savefig(fig, alt)
                @warn "could not overwrite $(basename(path)) -- is it open in a viewer? " *
                      "wrote $(basename(alt)) instead"
                return alt
            catch err
                @error "could not save $(basename(path)) or a fallback" exception = err
                return nothing
            end
        end
    end
end

function plot_figure(key)
    Lx, Ly, per, w_site, J = key
    g = sort([r for r in runs if (r.Lx, r.Ly, r.periodic, r.w_site, r.J) == key], by = r -> -r.cut)
    N = first(g).N
    dist = bond_distances(Lx, Ly, per, w_site, N)
    all_sites = sort(reduce(intersect, [r.sites for r in g]))
    probes = if show_sites === :all
        all_sites
    elseif show_sites === nothing
        [minimum(s for s in all_sites if dist[s] == d)
         for d in sort(unique(dist[all_sites])) if d >= 0]
    else
        bad = setdiff(show_sites, all_sites)
        isempty(bad) || error("site(s) $(join(bad, ", ")) not measured for J=$J")
        collect(show_sites)
    end

    t_plot = t_max_plot === nothing ? maximum(r.time[end] for r in g) : Float64(t_max_plot)
    win(r) = findall(t -> t <= t_plot + 1e-9, r.time)

    # Exact reference on THIS lattice, if one has been run. `ed.Csq` is the same
    # quantity as their `otoc_j`: norm2 is 1 identically under exact evolution,
    # so Csq = 2(norm2 - F) = 4*sum_anti, which is what they report.
    edinfo = ed_for(J, w_site)
    ed = edinfo.run
    ed_i = ed === nothing ? Int[] : findall(t -> t <= t_plot + 1e-9, ed.time)

    vals = Float64[]
    for r in g, j in probes
        ii = win(r); append!(vals, r.C[j][ii])
        show_corrected && haskey(r.Ccor, j) && append!(vals, r.Ccor[j][ii])
    end
    if ed !== nothing
        for j in probes
            c = ed.col(j)
            c === nothing || append!(vals, ed.Csq[ed_i, c])
        end
    end
    lo, hi = extrema(vals); pad = max(0.06 * (hi - lo), 0.02)
    ylims = (lo - pad, hi + pad)

    @printf("\nJ = %-6.3g  lattice %dx%d %s  N=%d  W on site %d  (%d cutoff(s))\n",
            J, Lx, Ly, per ? "periodic" : "open", N, w_site, length(g))
    println("  panels: ", join(["site $j (d=$(dist[j]))" for j in probes], ", "))
    @printf("  %-9s %7s %9s %11s %11s %11s\n", "cutoff", "t_max", "wall s", "max terms", "C_end", "Ccor_end")
    for r in g
        j = probes[min(2, length(probes))]
        @printf("  %-9s %7.2f %9.0f %11d %11.4f %11.4f\n", cutoff_label(r.cut), r.t_max,
                r.elapsed, r.max_terms, r.C[j][end], get(r.Ccor, j, r.C[j])[end])
    end
    println("  (C_end / Ccor_end at site $(probes[min(2, length(probes))]))")
    if ed !== nothing
        @printf("  ED reference: %s  (t <= %.2f, dt = %.3g)\n",
                basename(ed.meta.path), maximum(ed.time), ed.meta.dt)
        # Deviation from exact, at the times the two grids SHARE. ED runs on a
        # coarser output grid (dt 0.25 against 0.05), and interpolating it onto
        # the PP grid would invent reference values that were never computed --
        # so the comparison is made only where ED actually has a point.
        # Matched PER RUN, not once for the group: cutoffs are free to have been
        # run on different grids, and reusing one run's indices for another
        # would compare different instants without any visible symptom.
        @printf("  %-9s %-6s %14s %14s\n", "cutoff", "site", "max|raw-ED|", "max|cor-ED|")
        any_shared = false
        for r in g
            sh = [(k, findfirst(t -> abs(t - te) < 1e-9, r.time))
                  for (k, te) in enumerate(ed.time) if te <= t_plot + 1e-9]
            sh = [(k, kp) for (k, kp) in sh if kp !== nothing]
            if isempty(sh)
                @printf("  %-9s %s\n", cutoff_label(r.cut), "(shares no output time with ED)")
                continue
            end
            any_shared = true
            ke = [k for (k, _) in sh]; kp = [k for (_, k) in sh]
            for j in probes
                c = ed.col(j); c === nothing && continue
                e  = ed.Csq[ke, c]
                dr = maximum(abs.(r.C[j][kp] .- e))
                dc = haskey(r.Ccor, j) ? maximum(abs.(r.Ccor[j][kp] .- e)) : NaN
                @printf("  %-9s %-6d %14.4f %14.4f\n", cutoff_label(r.cut), j, dr, dc)
            end
        end
        any_shared && println("  (ED sampling error is ~3e-3 here, so deviations far " *
                              "above that are PP error)")
    elseif !isempty(edinfo.wrong)
        println("  ED reference: NONE on this lattice. Same V/J/site but a different " *
                "lattice (NOT plotted):")
        for f in edinfo.wrong; println("      ", f); end
        println("      -> rerun ED with KITAEV_LATTICE=otocpp4x2 to get a usable reference")
    elseif show_ed
        println("  ED reference: none yet for J=$J on lattice $ed_lattice_tag " *
                "(run ED with KITAEV_LATTICE=otocpp4x2)")
    end

    cols = min(length(probes), 3)
    rows = ceil(Int, length(probes) / cols)
    on_bottom(i) = i > length(probes) - cols
    on_left(i)   = mod1(i, cols) == 1
    ylab = L"C_{%$(w_site),j}(t)"

    subplots = map(enumerate(probes)) do (idx, j)
        p = plot(; base_attrs()...,
                 title = "j = $j   (distance $(dist[j]))", titlefontsize = 13, tickfontsize = 11,
                 xlabel = on_bottom(idx) ? "time  t" : "",
                 ylabel = on_left(idx) ? ylab : "",
                 guidefontsize = 13,
                 legend = idx == 1 ? :bottomright : false, legendfontsize = 9,
                 xlims = (0, t_plot), ylims = ylims, grid = true)
        if idx == 1 && show_corrected
            plot!(p, Float64[], Float64[]; color = INK_SECONDARY, lw = 2, linestyle = :solid,
                  label = "raw")
            plot!(p, Float64[], Float64[]; color = INK_SECONDARY, lw = 2, linestyle = :dash,
                  label = "corrected")
        end
        for r in g
            ii = win(r); hue = cut_color(r.cut)
            show_corrected && haskey(r.Ccor, j) &&
                plot!(p, r.time[ii], r.Ccor[j][ii]; color = hue, lw = 1.5,
                      linestyle = :dash, label = "")
            plot!(p, r.time[ii], r.C[j][ii]; color = hue, lw = 2,
                  label = idx == 1 ? "cutoff $(cutoff_label(r.cut))" : "")
        end
        # Exact last => on top, and in ink rather than a series hue: it is the
        # reference the cutoffs are judged against, not another member of the
        # family. Its own (coarser) output grid is used, so a marker means a
        # value was actually computed there.
        if ed !== nothing && ed.col(j) !== nothing
            plot!(p, ed.time[ed_i], ed.Csq[ed_i, ed.col(j)];
                  color = INK_PRIMARY, lw = 3, label = idx == 1 ? "ED exact" : "",
                  marker = :circle, markersize = 2.6, markerstrokewidth = 0)
        end
        p
    end

    fig = plot(subplots...; layout = (rows, cols),
               size = (max(520 * cols, 1100), 400 * rows + 110), dpi = 200,
               plot_title = L"\mathrm{OTOC\!-\!PP}\ \ C_{%$(w_site),j}(t)\ \ \mathrm{-\ honeycomb\ %$(Lx)x%$(Ly),\ N=%$(N),\ K=%$(first(g).K),\ J=%$(J)}",
               plot_titlefontsize = 17, top_margin = 6Plots.mm,
               left_margin = 12Plots.mm, bottom_margin = 10Plots.mm, right_margin = 5Plots.mm)

    out = joinpath(out_dir,
        "otocpp_cutoffs_hc$(Lx)x$(Ly)_N$(N)_K$(first(g).K)_J$(J)_site$(w_site)" *
        "_tmax$(@sprintf("%.1f", t_plot)).png")
    saved_path = save_resilient(fig, out)
    saved_path === nothing || println("  saved: $(basename(saved_path))")
    return saved_path
end

saved = [plot_figure(k) for k in figs]
println("\n" * "="^92)
println("$(length(saved)) figure(s) saved to $(abspath(out_dir))")
println("""
Reminder: these are OTOC-PP runs on ITS generated lattice (unit-cell numbering,
square plaquettes at 2 cells wide). They are NOT on this project's 16-site
torus, so they cannot be overlaid on the ED results.""")
