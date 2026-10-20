# OTOC time traces across the (V, J) parameter plane.
#
#     julia --project=. otoc_VJ_grid.jl
#
# One panel per (V, J) combination that has an ED run -- combinations never run
# are not drawn. Panels are ordered by V, then J, and packed into a near-square
# grid; each panel's title names its V and J. Each panel shows F_{i,j}(t) for
# every probe site j, one color per j, with the SAME color meaning the same site
# in every panel, so a site can be followed across the grid.
#
# Set `show_missing_cells = true` for the full V x J matrix instead (rows = V,
# columns = J), where combinations without a run appear as empty, labelled
# cells. That layout makes gaps in the parameter scan visible.
#
# Reading it:
#   * J = 0 column: flux-conserving, free-Majorana (integrable) Kitaev model,
#     with or without the V gap.
#   * Moving right: J breaks integrability.
#   * V = 0 row: gapless Kitaev; moving down opens the chiral gap.
#
# ADDING PAULI PROPAGATION LATER
# ------------------------------
# Nothing in the file list needs editing. Set `include_pp = true` and every
# panel looks for a PP run with the same V, J, reference site, Pauli and
# lattice in otoc_outputs/, and draws it DASHED in the same site color as its
# ED curve. Panels with no PP run simply show ED alone.

include(joinpath(@__DIR__, "otoc_plotting.jl"))
using LaTeXStrings

# --- What to plot ------------------------------------------------------------
quantity   = :F          # :F (the OTOC) or :Csq (squared commutator)
i_site     = 2           # reference site of W(0); runs with another site are ignored
pauli      = "ZZ"

show_sites = nothing     # nothing -> every probe site the ED runs share; or e.g. [2, 16]
V_values   = nothing     # nothing -> every V with an ED run; or e.g. [0.0, 0.3]
J_values   = nothing     # nothing -> every J with an ED run; or e.g. [0.0, 0.05]
t_max_plot = nothing     # nothing -> the longest ED run; or e.g. 3.0
show_missing_cells = false   # true -> full V x J matrix, with empty cells for combinations never run
show_markers = false     # markers at every output time (crowded at dt=0.05, t=7)

# Pauli propagation overlay.
include_pp = true
pp_cutoff  = 8e-5        # nothing -> the least-truncated run available; or e.g. 5e-5
# Only use coefficient-truncated PP runs (max_weight = N). A weight-capped run
# is a different approximation, and would make panels not comparable.
pp_require_inactive_weight_cap = true
# ------------------------------------------------------------------------------

function cutoff_label(x::Real)
    e = floor(Int, log10(x))
    m = x / 10.0^e
    if m >= 10 - 1e-9; m /= 10; e += 1; end
    ms = isapprox(m, round(m); atol = 1e-6) ? string(Int(round(m))) : string(round(m, sigdigits = 3))
    return "$(ms)e$(e)"
end

# Same number as LaTeX math, for the figure title: "8e-5" -> 8\times 10^{-5}.
function cutoff_tex(x::Real)
    e = floor(Int, log10(x))
    m = x / 10.0^e
    if m >= 10 - 1e-9; m /= 10; e += 1; end
    ms = isapprox(m, round(m); atol = 1e-6) ? string(Int(round(m))) : string(round(m, sigdigits = 3))
    return ms == "1" ? "10^{$e}" : "$(ms)\\times 10^{$e}"
end

# ---- discover runs -------------------------------------------------------------
ed_by = Dict{Tuple{Float64,Float64},Tuple{String,RunMeta}}()     # (V,J) => longest ED run
pp_by = Dict{Tuple{Float64,Float64},Vector{NamedTuple}}()        # (V,J) => PP candidates
for path in find_runs()
    local m
    try
        m = parse_meta(path)
    catch
        continue
    end
    (isempty(m.lat) || m.site != i_site || m.pauli != pauli) && continue
    vj = (m.V, m.J)
    if m.solver == "ED"
        if !haskey(ed_by, vj) || m.tmax > ed_by[vj][2].tmax
            ed_by[vj] = (path, m)
        end
    elseif include_pp
        d = npzread(path)
        haskey(d, "trunc") || continue
        push!(get!(pp_by, vj, NamedTuple[]),
              (path = path, meta = m, cut = d["trunc"][1], wmax = Int(d["trunc"][2]),
               t_end = Float64(d["time"][end])))
    end
end
isempty(ed_by) && error("No ED runs found for reference site $i_site, $pauli in otoc_outputs/.")

# Every panel must be on the same lattice and system size, or the grid compares
# different systems while looking uniform.
systems = unique([(m.N, m.lat) for (_, m) in values(ed_by)])
length(systems) == 1 || error("ED runs span several systems $(systems); keep one N/lattice in " *
                              "otoc_outputs/ or filter by moving files aside.")
N_sys, lat_sys = only(systems)

keep(vals, x) = vals === nothing || any(v -> isapprox(v, x; atol = 1e-12), vals)
Vs = sort(unique([vj[1] for vj in keys(ed_by) if keep(V_values, vj[1])]))
Js = sort(unique([vj[2] for vj in keys(ed_by) if keep(J_values, vj[2])]))
(isempty(Vs) || isempty(Js)) && error("No ED runs match V_values=$V_values, J_values=$J_values.")

cells = show_missing_cells ?
    [(V, J) for V in Vs for J in Js] :
    sort([vj for vj in keys(ed_by) if vj[1] in Vs && vj[2] in Js], by = vj -> (vj[1], vj[2]))
if show_missing_cells
    rows, cols = length(Vs), length(Js)
else
    cols = ceil(Int, sqrt(length(cells))); rows = ceil(Int, length(cells) / cols)
end

ed = Dict(vj => load_run(ed_by[vj][1]) for vj in cells if haskey(ed_by, vj))

# Probe sites: those shared by every ED run in the grid, unless narrowed.
shared_sites = reduce(intersect, [r.sites for r in values(ed)])
probe_sites = if show_sites === nothing
    shared_sites
else
    missing_sites = setdiff(show_sites, shared_sites)
    isempty(missing_sites) || error("show_sites asks for $(join(missing_sites, ", ")), which not " *
                                    "every ED run probed. Shared sites: $(join(shared_sites, ", ")).")
    collect(show_sites)
end
isempty(probe_sites) && error("The ED runs share no probe sites.")
site_color = Dict(j => series_color(k) for (k, j) in enumerate(probe_sites))

# PP choice per panel: coefficient-only if required, the requested cutoff if
# given, otherwise the finest cutoff (ties -> longest run).
function pick_pp(vj)
    include_pp && haskey(pp_by, vj) || return nothing
    c = filter(pp_by[vj]) do r
        (!pp_require_inactive_weight_cap || r.wmax >= r.meta.N) &&
        (pp_cutoff === nothing || isapprox(r.cut, pp_cutoff; rtol = 1e-6))
    end
    isempty(c) && return nothing
    best = first(sort(c, by = r -> (r.cut, -r.t_end)))
    return merge(best, (run = load_run(best.path),))
end
pp = Dict(vj => pick_pp(vj) for vj in keys(ed))
any_pp = any(!isnothing, values(pp))

# Does ONE cutoff cover every panel? If so it is a property of the figure and
# belongs in the main title. If the panels were fed different cutoffs -- pick_pp
# takes the finest run each cell happens to have, so this is not hypothetical --
# a single headline cutoff would mislabel the odd panels out, and each panel
# names its own instead. The title must never claim a uniformity the data lacks.
pp_cuts = sort(unique([p_.cut for p_ in values(pp) if p_ !== nothing]); rev = true)
uniform_cut = length(pp_cuts) == 1 ? only(pp_cuts) : nothing
length(pp_cuts) > 1 && println("note: PP cutoffs differ across panels " *
    "($(join(cutoff_label.(pp_cuts), ", "))) -- naming each panel's own, not the figure's.")

t_plot = t_max_plot === nothing ? maximum(r.time[end] for r in values(ed)) : Float64(t_max_plot)
Q(r) = quantity === :F ? r.F : r.Csq
within(r) = findall(t -> t <= t_plot + 1e-9, r.time)

# One y-range for the whole grid, fitted to what is actually drawn, so panels
# are directly comparable.
vals = Float64[]
for (vj, r) in ed
    idx = within(r)
    for j in probe_sites
        append!(vals, Q(r)[idx, r.col(j)])
        p_ = pp[vj]
        p_ === nothing || append!(vals, Q(p_.run)[within(p_.run), p_.run.col(j)])
    end
end
lo, hi = extrema(vals)
pad = max(0.06 * (hi - lo), 0.02)
ylims = (lo - pad, hi + pad)

# ---- report what was found -------------------------------------------------------
println("Reference site $i_site, $pauli, N=$N_sys, lattice $lat_sys")
println("Grid: $(rows) x $(cols)  (V = $(join(Vs, ", ")) ; J = $(join(Js, ", ")))")
println("Probe sites: $(join(probe_sites, ", "))   t in [0, $t_plot]")
println("-"^88)
@printf("%6s %6s  %-8s %-28s\n", "V", "J", "ED t_max", "PP")
println("-"^88)
for vj in cells
    haskey(ed, vj) || (@printf("%6.3g %6.3g  %-8s\n", vj[1], vj[2], "(none)"); continue)
    p_ = pp[vj]
    # The table always names the cutoff, whatever the figure title does: it is
    # the per-run inventory, and it is what makes an odd cutoff visible at all.
    pp_txt = !include_pp ? "(overlay off)" :
             p_ === nothing ? "none" :
             "cutoff $(cutoff_label(p_.cut)) w<=$(p_.wmax) to t=$(p_.t_end)"
    @printf("%6.3g %6.3g  %-8.2f %s\n", vj[1], vj[2], ed[vj].time[end], pp_txt)
end
println("-"^88)

# ---- panels -------------------------------------------------------------------------
ylab = quantity === :F ?
    L"\mathrm{OTOC}\quad F_{%$(i_site),j}(t)" :
    L"\mathrm{squared\ commutator}\quad C_{%$(i_site),j}(t)"
n_cells = length(cells)
on_bottom(idx) = idx > n_cells - cols
on_left(idx)   = mod1(idx, cols) == 1
first_filled = findfirst(vj -> haskey(ed, vj), cells)

subplots = map(enumerate(cells)) do (idx, vj)
    V, J = vj
    if !haskey(ed, vj)
        p = plot(; base_attrs()..., framestyle = :none, grid = false, legend = false,
                 xlims = (0, 1), ylims = (0, 1), xticks = false, yticks = false,
                 title = "V = $V,  J = $J", titlefontsize = 14)
        annotate!(p, 0.5, 0.5, Plots.text("no ED run", 12, :gray))
        return p
    end
    r = ed[vj]
    p_ = pp[vj]
    ttl = "V = $V,  J = $J"
    # Only when the figure title cannot carry it -- see uniform_cut above.
    (p_ === nothing || uniform_cut !== nothing) || (ttl *= "   PP ε=$(cutoff_label(p_.cut))")
    r.time[end] < t_plot - 1e-9 && (ttl *= "   (ED to t=$(r.time[end]))")

    p = plot(; base_attrs()...,
             title = ttl, titlefontsize = 14, tickfontsize = 11,
             xlabel = on_bottom(idx) ? "time  t" : "",
             ylabel = on_left(idx) ? ylab : "",
             guidefontsize = 13,
             legend = idx == first_filled ? :topright : false, legendfontsize = 10,
             xlims = (0, t_plot), ylims = ylims, grid = true)

    # Legend: one key per probe site (color), plus solid/dashed keys once PP is on.
    if idx == first_filled && any_pp
        plot!(p, Float64[], Float64[]; color = INK_SECONDARY, lw = 2, linestyle = :solid, label = "ED")
        plot!(p, Float64[], Float64[]; color = INK_SECONDARY, lw = 2, linestyle = :dash, label = "PP")
    end

    ie = within(r)
    for j in probe_sites
        hue = site_color[j]
        if p_ !== nothing
            ip = within(p_.run)
            plot!(p, p_.run.time[ip], Q(p_.run)[ip, p_.run.col(j)];
                  color = hue, lw = 1.6, linestyle = :dash, label = "",
                  marker = show_markers ? :diamond : :none, markersize = 1.6, markerstrokewidth = 0)
        end
        plot!(p, r.time[ie], Q(r)[ie, r.col(j)];
              color = hue, lw = 2, label = idx == first_filled ? "j = $j" : "",
              marker = show_markers ? :circle : :none, markersize = 1.8, markerstrokewidth = 0)
    end
    p
end

qname = quantity === :F ? "F" : "C"

# Assembled as a plain String and wrapped with latexstring(), NOT written as an
# L"..." literal: inside L"..." a bare $(x) is not interpolated -- there $ is the
# math-mode delimiter and only %$(x) interpolates -- so a $(x) in a title silently
# prints as source. Building the string first keeps ordinary Julia interpolation.
ttl_math = "\\mathrm{$(pauli)\\ OTOC}\\ \\ $(qname)_{$(i_site),j}(t)" *
           "\\ \\ \\mathrm{across\\ V\\ and\\ J\\ \\cdot\\ N=$(N_sys)}"
uniform_cut === nothing ||
    (ttl_math *= "\\ \\ \\mathrm{\\cdot\\ PP\\ cutoff}\\ \\varepsilon=$(cutoff_tex(uniform_cut))")

fig = plot(subplots...; layout = (rows, cols),
           size = (max(560 * cols, 1100), 430 * rows + 100), dpi = 200,
           plot_title = latexstring(ttl_math),
           plot_titlefontsize = 20, top_margin = 6Plots.mm,
           left_margin = 12Plots.mm, bottom_margin = 10Plots.mm, right_margin = 5Plots.mm)

# The cutoff is in the filename too when it is uniform, so a rerun at another
# cutoff writes a new file instead of overwriting this one.
out = joinpath(plots_dir(),
    "otoc_VJ_grid_$(quantity)_N$(N_sys)_$(lat_sys)_$(pauli)_site$(i_site)_tmax$(@sprintf("%.1f", t_plot))" *
    (any_pp ? "_withPP" * (uniform_cut === nothing ? "" : "_cut$(cutoff_label(uniform_cut))") : "") * ".png")
savefig(fig, out)
println("\nSaved: $out")
