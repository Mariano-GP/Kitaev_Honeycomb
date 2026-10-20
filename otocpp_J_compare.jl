# OTOC-PP cluster results (CSV): ONE FIGURE PER CUTOFF, comparing all J values.
#
#     julia --project=. otocpp_J_compare.jl
#
# This is the TRANSPOSE of otocpp_csv_grid.jl. That script fixes J and varies the
# cutoff (a numerics question: is it converged?). This one fixes the cutoff and
# varies J (a physics question: how does integrability breaking change the
# spreading?). Same loader, same lattice reconstruction, different grouping.
#
#   * one FIGURE per cutoff,
#   * inside it one PANEL per probe site (one per bond-distance shell from W),
#   * within a panel one CURVE per J, with a fixed hue per J across both figures.
#
# THE QUANTITY IS C, NOT F. Their CSV columns are
# `time, n_terms, otoc_j, corrected_j, err_j` -- `otoc_j` is the squared
# commutator ||[W(t),V_j]||^2 = 4*sum_anti|c_k|^2. They do NOT write norm2, and
# F = norm2 - C/2 needs it, so F is not recoverable from these files. Plotting C
# is not a substitution made for convenience; it is the only thing the data
# supports. (This project's own PP solver does save norm2 and does report F --
# but on the torus lattice, which is a different graph.)
#
# `corrected_j` (dashed) is THEIR truncation correction -- the drop in C measured
# immediately before and after every truncation, accumulated and added back. It
# is a different scheme from this project's C_anti/norm2.
#
# ED is drawn DOTTED in ink (black), not in a series hue: it is the reference the
# PP curves are judged against, not another member of the family, so it does not
# compete for a colour slot. The cost is that the several exact curves in a panel
# are not separable from each other by eye -- acceptable here because they nearly
# coincide (all saturate at 2), and because the per-J numbers are in the deviation
# table regardless. ED is always LOADED even when not drawn, so that table is
# reported either way.
#
# COVERAGE: the otocpp4x2 ED runs probe only sites [2,3,4,5]. A panel for any
# other site is PP-only -- that is a gap in the ED runs, not a plotting failure,
# and it is named in the console output rather than left as a silent absence.

include(joinpath(@__DIR__, "otoc_plotting.jl"))
using DelimitedFiles
using LaTeXStrings

# --- What to plot ------------------------------------------------------------
data_dir = joinpath(@__DIR__, "..", "OTOC-PP", "new_data", "honeycomb_mguerre3")
out_dir  = joinpath(@__DIR__, "..", "OTOC-PP", "plots")

cutoff_values = [5e-5, 1e-4]   # one figure each; nothing -> every cutoff present
J_values      = nothing        # nothing -> every J present at that cutoff

show_corrected = true    # dashed: their truncation-corrected curve
show_ed        = true    # dotted, in ink (black); see the header note
ed_dir         = joinpath(@__DIR__, "otoc_outputs")
ed_lattice_tag = "L242ad9"     # KITAEV_LATTICE=otocpp4x2; the torus is L243535

show_sites  = [2, 5, 16, 12]   # nothing -> one representative site per distance shell
t_max_plot  = nothing    # nothing -> the longest run in the figure

# A figure needs at least this many J curves to be worth drawing; groups with
# fewer are reported by name rather than silently dropped or emitted as a
# one-curve "comparison".
min_J_per_figure = 2
# ------------------------------------------------------------------------------

function cutoff_label(x::Real)
    e = floor(Int, log10(x)); m = x / 10.0^e
    if m >= 10 - 1e-9; m /= 10; e += 1; end
    ms = isapprox(m, round(m); atol = 1e-6) ? string(Int(round(m))) : string(round(m, sigdigits = 3))
    return "$(ms)e$(e)"
end

# Same number as LaTeX math for the figure title: 5e-5 -> 5\times 10^{-5}.
function cutoff_tex(x::Real)
    e = floor(Int, log10(x)); m = x / 10.0^e
    if m >= 10 - 1e-9; m /= 10; e += 1; end
    ms = isapprox(m, round(m); atol = 1e-6) ? string(Int(round(m))) : string(round(m, sigdigits = 3))
    return ms == "1" ? "10^{$e}" : "$(ms)\\times 10^{$e}"
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
    sites = Int[]; C = Dict{Int,Vector{Float64}}()
    Ccor = Dict{Int,Vector{Float64}}(); Cerr = Dict{Int,Vector{Float64}}()
    for (k, nm) in enumerate(names)
        g = match(r"^otoc_(\d+)$", nm)
        g === nothing && continue
        j = parse(Int, g.captures[1]); push!(sites, j)
        C[j] = Float64.(raw[:, k])
        kc = findfirst(==("corrected_$j"), names)
        kc === nothing || (Ccor[j] = Float64.(raw[:, kc]))
        ke = findfirst(==("err_$j"), names)
        ke === nothing || (Cerr[j] = Float64.(raw[:, ke]))
    end
    return (path = csv_path, Lx = Lx, Ly = Ly, periodic = per, N = 2 * Lx * Ly,
            K = getp("K", 1.0), J = getp("J", 0.0),
            cut = parse(Float64, get(md, "cutoff", "NaN")),
            w_site = parse(Int, get(md, "W_site", "1")),
            dt = parse(Float64, get(md, "dt", "NaN")),
            t_max = parse(Float64, get(md, "t_max", "NaN")),
            elapsed = parse(Float64, get(md, "elapsed_s", "NaN")),
            max_terms = parse(Int, get(md, "max_n_terms", "0")),
            time = time, terms = terms, sites = sort(sites),
            C = C, Ccor = Ccor, Cerr = Cerr)
end

isdir(data_dir) || error("no such directory: $data_dir")
all_runs = filter(!isnothing, [load_otocpp_run(joinpath(data_dir, f))
                               for f in sort(readdir(data_dir)) if endswith(f, ".csv")])
isempty(all_runs) && error("no OTOC-PP CSV/meta pairs found in $data_dir")
mkpath(out_dir)

keep(vals, x) = vals === nothing || any(v -> isapprox(v, x; atol = 1e-12), vals)
runs = [r for r in all_runs if keep(cutoff_values, r.cut) && keep(J_values, r.J)]
isempty(runs) && error("no runs match cutoff_values=$cutoff_values, J_values=$J_values")

# One colour per J VALUE, fixed across every figure this script draws: J=0.05 is
# the same hue at 5e-5 as at 1e-4, so the two figures can be read side by side
# and adding a J to one never repaints the other. Ordered weakest -> strongest
# integrability breaking.
J_ladder = sort(unique([r.J for r in runs]))
J_color(J) = series_color(findfirst(==(J), J_ladder))

# NOT gated on show_ed: the exact reference is always LOADED so the deviation
# table can report it, and show_ed only controls whether it is DRAWN.
function ed_for(J, w_site)
    isdir(ed_dir) || return (run = nothing, wrong = String[])
    cands = NamedTuple[]
    for f in readdir(ed_dir)
        (startswith(f, "ED_") && endswith(f, ".npz")) || continue
        local m
        try
            m = parse_meta(joinpath(ed_dir, f))
        catch
            continue
        end
        (m.site == w_site && m.pauli == "ZZ" && m.V == 0.0 &&
         isapprox(m.J, J; atol = 1e-12)) || continue
        push!(cands, (meta = m, ok = m.lat == ed_lattice_tag))
    end
    good = [c for c in cands if c.ok]
    isempty(good) && return (run = nothing, wrong = [basename(c.meta.path) for c in cands])
    best = argmax(c -> c.meta.tmax, good)
    return (run = load_run(best.meta.path), wrong = String[])
end

# Windows keeps a lock on a PNG an image viewer has open, surfaced as a
# SystemError when savefig overwrites. One locked file must not abort the sweep.
function save_resilient(fig, path)
    for attempt in 1:3
        try
            savefig(fig, path); return path
        catch
            attempt < 3 && (sleep(1.5); continue)
            alt = replace(path, ".png" => "_new.png")
            try
                savefig(fig, alt)
                @warn "could not overwrite $(basename(path)) -- open in a viewer? " *
                      "wrote $(basename(alt)) instead"
                return alt
            catch err
                @error "could not save $(basename(path)) or a fallback" exception = err
                return nothing
            end
        end
    end
end

# A figure is one (lattice, W site, cutoff): a 2x4 and a 4x2 cluster are
# different graphs with different site numbering and do not share axes.
figs = sort(unique([(r.Lx, r.Ly, r.periodic, r.w_site, r.cut) for r in runs]))

function plot_figure(key)
    Lx, Ly, per, w_site, cut = key
    g = sort([r for r in runs if (r.Lx, r.Ly, r.periodic, r.w_site, r.cut) == key], by = r -> r.J)
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
        isempty(bad) || error("site(s) $(join(bad, ", ")) not measured at cutoff $cut")
        collect(show_sites)
    end

    t_plot = t_max_plot === nothing ? maximum(r.time[end] for r in g) : Float64(t_max_plot)
    win(r) = findall(t -> t <= t_plot + 1e-9, r.time)

    vals = Float64[]
    for r in g, j in probes
        ii = win(r); append!(vals, r.C[j][ii])
        show_corrected && haskey(r.Ccor, j) && append!(vals, r.Ccor[j][ii])
    end
    lo, hi = extrema(vals); pad = max(0.06 * (hi - lo), 0.02)
    ylims = (lo - pad, hi + pad)

    @printf("\ncutoff %-8s lattice %dx%d %s  N=%d  W on site %d  (%d J value(s))\n",
            cutoff_label(cut), Lx, Ly, per ? "periodic" : "open", N, w_site, length(g))
    println("  panels: ", join(["site $j (d=$(dist[j]))" for j in probes], ", "))
    jref = probes[min(2, length(probes))]
    @printf("  %-9s %7s %9s %11s %11s %11s\n",
            "J", "t_max", "wall s", "max terms", "C_end", "Ccor_end")
    for r in g
        @printf("  %-9.4g %7.2f %9.0f %11d %11.4f %11.4f\n", r.J, r.t_max, r.elapsed,
                r.max_terms, r.C[jref][end], get(r.Ccor, jref, r.C[jref])[end])
    end
    println("  (C_end / Ccor_end at site $jref)")

    # Exact reference, as NUMBERS regardless of whether it is drawn -- matched
    # per run, because different J runs are free to have used different output
    # grids and reusing one run's indices for another compares different
    # instants with no visible symptom.
    for r in g
        info = ed_for(r.J, w_site)
        ed = info.run
        if ed === nothing
            isempty(info.wrong) || @printf("  J=%-6.4g ED on another lattice, not used: %s\n",
                                           r.J, join(info.wrong, ", "))
            continue
        end
        sh = [(k, findfirst(t -> abs(t - te) < 1e-9, r.time))
              for (k, te) in enumerate(ed.time) if te <= t_plot + 1e-9]
        sh = [(k, kp) for (k, kp) in sh if kp !== nothing]
        isempty(sh) && continue
        ke = [k for (k, _) in sh]; kp = [k for (_, k) in sh]
        for j in probes
            c = ed.col(j); c === nothing && continue
            e = ed.Csq[ke, c]
            dr = maximum(abs.(r.C[j][kp] .- e))
            dc = haskey(r.Ccor, j) ? maximum(abs.(r.Ccor[j][kp] .- e)) : NaN
            @printf("  J=%-6.4g site %-3d max|raw-ED|=%8.4f  max|cor-ED|=%8.4f\n", r.J, j, dr, dc)
        end
    end

    # Panels ED cannot cover. A panel with no dotted curve is a GAP IN THE ED
    # RUNS, not a plotting failure -- say which and why, rather than leaving an
    # absence to be noticed. Stated once: every ED run here probes the same sites.
    let probe_ref = nothing
        for r in g
            e = ed_for(r.J, w_site).run
            if e !== nothing
                probe_ref = e
                break
            end
        end
        if probe_ref !== nothing
            gaps = [j for j in probes if probe_ref.col(j) === nothing]
            isempty(gaps) || println("  NO ED at site(s) $(join(gaps, ", ")) -- those panels are " *
                "PP-only. These ED runs probe $(join(probe_ref.sites, ", ")); rerun ED with the " *
                "missing sites in probe_sites (KITAEV_LATTICE=otocpp4x2) to fill them.")
        end
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
        # Style keys once, in ink: one entry per LINE STYLE, plus one per J below
        # for colour. Listing every J three times over would be a legend nobody
        # reads.
        # Each style key must carry the SAME linestyle and colour as the curves it
        # labels, or the legend misdescribes the plot: raw solid in ink, corrected
        # dashed in ink, ED dotted in black to match the draw calls below.
        if idx == 1 && show_corrected
            plot!(p, Float64[], Float64[]; color = INK_SECONDARY, lw = 2,
                  linestyle = :solid, label = "raw")
            plot!(p, Float64[], Float64[]; color = INK_SECONDARY, lw = 2,
                  linestyle = :dash, label = "corrected")
        end
        if idx == 1 && show_ed
            plot!(p, Float64[], Float64[]; color = INK_PRIMARY, lw = 2.6,
                  linestyle = :dot, label = "ED exact")
        end
        for r in g
            ii = win(r); hue = J_color(r.J)
            show_corrected && haskey(r.Ccor, j) &&
                plot!(p, r.time[ii], r.Ccor[j][ii]; color = hue, lw = 1.5,
                      linestyle = :dash, label = "")
            plot!(p, r.time[ii], r.C[j][ii]; color = hue, lw = 2,
                  label = idx == 1 ? "J = $(r.J)" : "")
            if show_ed
                ed = ed_for(r.J, w_site).run
                # Unlabelled: the dotted style key above carries "ED exact" once.
                # Labelling each would add one legend entry per J for a set of
                # curves the legend cannot tell apart anyway.
                if ed !== nothing && ed.col(j) !== nothing
                    ei = findall(t -> t <= t_plot + 1e-9, ed.time)
                    plot!(p, ed.time[ei], ed.Csq[ei, ed.col(j)]; color = INK_PRIMARY,
                          lw = 2.6, linestyle = :dot, label = "")
                end
            end
        end
        p
    end

    fig = plot(subplots...; layout = (rows, cols),
               size = (max(520 * cols, 1100), 400 * rows + 110), dpi = 200,
               plot_title = L"\mathrm{OTOC\!-\!PP}\ \ C_{%$(w_site),j}(t)\ \ \mathrm{across\ J\ \cdot\ honeycomb\ %$(Lx)x%$(Ly),\ N=%$(N),\ K=%$(first(g).K)}\ \ \varepsilon=%$(cutoff_tex(cut))",
               plot_titlefontsize = 17, top_margin = 6Plots.mm,
               left_margin = 12Plots.mm, bottom_margin = 10Plots.mm, right_margin = 5Plots.mm)

    out = joinpath(out_dir,
        "otocpp_Jcompare_hc$(Lx)x$(Ly)_N$(N)_K$(first(g).K)_cut$(cutoff_label(cut))" *
        "_site$(w_site)_tmax$(@sprintf("%.1f", t_plot)).png")
    saved_path = save_resilient(fig, out)
    saved_path === nothing || println("  saved: $(basename(saved_path))")
    return saved_path
end

saved = String[]
skipped = String[]
for k in figs
    n_J = length(unique([r.J for r in runs
                         if (r.Lx, r.Ly, r.periodic, r.w_site, r.cut) == k]))
    if n_J < min_J_per_figure
        push!(skipped, "hc$(k[1])x$(k[2]) site$(k[4]) cutoff $(cutoff_label(k[5])): only $n_J J value(s)")
        continue
    end
    s = plot_figure(k)
    s === nothing || push!(saved, s)
end

println("\n" * "="^92)
println("$(length(saved)) figure(s) saved to $(abspath(out_dir))")
if !isempty(skipped)
    println("Not drawn (too few J values to be a comparison):")
    for s in skipped; println("   ", s); end
end
println("""
NOTE: the quantity is C (their `otoc_j` = squared commutator), NOT F. Their CSVs
carry no norm2 column, and F = norm2 - C/2 cannot be formed without it.""")
