# Shared plotting helpers for the infinite-temperature OTOC results.
#
# Included by otoc_lightcone.jl, otoc_growth.jl, otoc_time_trace.jl and
# otoc_butterfly.jl so all four agree on the palette, the axis chrome, and how
# a run's parameters are recovered from its filename.
#
# WHAT THE .npz FILES CONTAIN
# ---------------------------
# Both solvers write the same keys (kitaev_OTOC_infinite_temp.jl for Pauli
# propagation, ED_OTOC_infinite_temp.jl for the exact reference):
#   "time"  (nt)            output times
#   "F"     (nt, ncol)      the OTOC, Tr[W_i(t) V_j W_i(t) V_j]/2^N; starts at 1
#   "Csq"   (nt, ncol)      squared commutator, 2*(norm2 - F); starts at 0
#   "norm2" (nt)            sum_k |c_k|^2 -- 1 exactly for ED, decaying with
#                           truncation loss for PP. THE error diagnostic.
#   "C_re"/"C_im"           aliases (C_re == Csq, C_im == 0) kept only so older
#                           correlation-era scripts can read these files
# PP files additionally carry "terms" (nt), the Pauli-term count |W|(t), and
# their columns run over ALL N sites. ED files instead carry "sites"
# (the probe sites actually computed, possibly a subset) and "F_err" (the
# stochastic-trace standard error), and their columns run over "sites" only --
# so ED columns must be matched to a site by looking the index up in "sites",
# never by assuming column j is site j.

using NPZ
using Printf
using Plots
# Math typesetting for axis/panel titles. GR renders LaTeXString labels with its
# OWN mathtext engine, so no TeX installation is required -- but that engine
# only covers basic math (sub/superscripts, greek, \mathrm, \langle\rangle).
# Keep labels to that subset; anything fancier silently renders as raw source.
using LaTeXStrings

gr()

# PALETTE =====================================================================
# From the data-viz reference palette, used in its documented slot order. That
# order is the colorblind-safety mechanism, not a cosmetic choice: it is
# validated for adjacent pairs (which is the pairlist line charts use), so
# series are assigned slots 1,2,3,... in order and never cycled or reordered.
const SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Chart chrome / ink. Text always wears ink tokens, never a series color -- a
# colored mark beside a label carries the identity instead.
const INK_PRIMARY   = "#0b0b0b"
const INK_SECONDARY = "#52514e"
const INK_MUTED     = "#898781"
const GRIDLINE      = "#e1e0d9"
const SURFACE       = "#fcfcfb"

# Sequential ramp (light -> dark, ONE hue) for magnitude heatmaps. The squared
# commutator is non-negative and has no meaningful midpoint, so it gets a
# sequential ramp; a diverging map would invent a polarity that isn't there.
const SEQ_BLUE = cgrad(["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                        "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                        "#184f95", "#104281", "#0d366b"])

# Diverging blue <-> gray <-> red, for the OTOC F itself, which IS signed and
# does have a meaningful zero (F > 0 = operators still largely commuting,
# F < 0 = overshoot past full scrambling).
const DIV_BLUE_RED = cgrad(["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec",
                            "#f0a6a6", "#d03b3b", "#7a1c1c"])

series_color(i::Int) = SERIES[mod1(i, length(SERIES))]

# Common axis chrome: recessive grid and axes, ink-colored text.
function base_attrs(; kwargs...)
    return (background_color = SURFACE,
            background_color_inside = SURFACE,
            foreground_color_axis = GRIDLINE,
            foreground_color_border = GRIDLINE,
            foreground_color_text = INK_SECONDARY,
            foreground_color_guide = INK_SECONDARY,
            gridalpha = 0.5,
            gridlinewidth = 0.6,
            gridcolor = GRIDLINE,
            titlefontcolor = INK_PRIMARY,
            kwargs...)
end


# FILENAME METADATA ===========================================================
# Every run's parameters are encoded in its own filename by the two solvers, so
# a plot can never silently mislabel itself the way a hand-set constant can. If
# a tag is missing the run is rejected rather than guessed at -- mislabelling a
# cross-site OTOC as an autocorrelator is exactly the kind of error that looks
# plausible on a plot.
struct RunMeta
    path::String
    solver::String     # "PP" or "ED"
    N::Int
    site::Int          # reference site i (where W(0) lives)
    pauli::String      # "XX" / "YY" / "ZZ"
    J::Float64
    V::Float64
    dt::Float64
    tmax::Float64
    lat::String        # lattice fingerprint, e.g. "L243535"; "" for older untagged files
    trunc::String      # "coeff1e3_w12" for PP, "exact" for ED, "" if absent
end

function parse_meta(path::String)
    b = basename(path)
    solver = startswith(b, "ED_") ? "ED" : startswith(b, "PP_") ? "PP" :
        error("$b: filename must start with PP_ or ED_")
    grab(re, what) = begin
        m = match(re, b)
        m === nothing && error("$b: no $what tag in filename")
        m.captures[1]
    end
    return RunMeta(path, solver,
                   parse(Int,     grab(r"_N(\d+)_", "_N<sites>_")),
                   parse(Int,     grab(r"_site(\d+)_", "_site<N>_")),
                                  grab(r"_(XX|YY|ZZ)_", "_XX_/_YY_/_ZZ_"),
                   parse(Float64, grab(r"_J(-?[\d.]+)_", "_J<value>_")),
                   parse(Float64, grab(r"_V(-?[\d.]+)_", "_V<value>_")),
                   parse(Float64, grab(r"_dt([\d.]+)_", "_dt<value>_")),
                   # bounded so the extension's dot isn't swallowed by a greedy [\d.]+
                   parse(Float64, grab(r"_tmax(\d+(?:\.\d+)?)", "_tmax<value>")),
                   # Optional: files written before lattice_tag() existed have none.
                   (mm = match(r"_N\d+_(L\w+?)_", b); mm === nothing ? "" : mm.captures[1]),
                   # PP carries "coeff<eps>_w<weight>", ED carries "exact".
                   # "wdamp<alpha>_c<thresh>" is the weight-damped strategy
                   # (kitaev_OTOC_infinite_temp.jl). Without it here, parse_meta
                   # returns "" for those runs and every script that filters on
                   # the truncation tag drops them silently.
                   # "wdamp" is Pauli-weight damping, "mwdamp" Majorana-weight.
                   (mt = match(r"_(coeff[\d.]+e\d+_w\d+|m?wdamp[\d.]+_c[\d.]+e\d+|exact|customtrunc)_", b);
                    mt === nothing ? "" : mt.captures[1]))
end

label_of(m::RunMeta) = "$(m.solver)  J=$(m.J)  dt=$(m.dt)"

# Load one run. Returns a NamedTuple with the raw arrays plus `col(site)`, which
# maps a physical site index to the right column -- the indirection that keeps
# ED's probe subset and PP's full-N matrix interchangeable at the call site.
function load_run(path::String)
    isfile(path) || error("no such file: $path\n(run the solver first)")
    m = parse_meta(path)
    d = npzread(path)
    sites = haskey(d, "sites") ? Int.(d["sites"]) : collect(1:size(d["F"], 2))
    colmap = Dict(s => k for (k, s) in enumerate(sites))
    return (meta = m, time = d["time"], F = d["F"], Csq = d["Csq"],
            norm2 = d["norm2"],
            terms = haskey(d, "terms") ? d["terms"] : nothing,
            F_err = haskey(d, "F_err") ? d["F_err"] : nothing,
            sites = sites,
            col = s -> get(colmap, s, nothing))
end

# All runs in otoc_outputs/ matching a prefix -- convenience for the scripts'
# "plot everything I have" default.
function find_runs(; prefix::String = "", dir::String = joinpath(@__DIR__, "otoc_outputs"))
    isdir(dir) || return String[]
    return sort([joinpath(dir, f) for f in readdir(dir)
                 if endswith(f, ".npz") && startswith(f, prefix)])
end

# Graph distance from `origin` to every site, over the CURRENT lattice in
# kitaev_model.jl. Physical bond distance is the meaningful x-axis for a
# Lieb-Robinson / butterfly light cone -- not the 1..N site numbering, which
# carries no geometry.
function graph_distances(origin::Int, n::Int, X_B, Y_B, Z_B)
    adj = [Int[] for _ in 1:n]
    for (i, j) in vcat(X_B, Y_B, Z_B)
        push!(adj[i], j); push!(adj[j], i)
    end
    dist = fill(-1, n)
    dist[origin] = 0
    q = [origin]
    while !isempty(q)
        u = popfirst!(q)
        for v in adj[u]
            dist[v] == -1 && (dist[v] = dist[u] + 1; push!(q, v))
        end
    end
    any(dist .== -1) && @warn "sites unreachable from $origin: $(findall(dist .== -1))"
    return dist
end

# FRONT-AWARE TRUNCATION CORRECTION ===========================================
# Truncation loses operator weight, delta(t) = 1 - norm2(t), and that loss
# corrupts F and C in different places:
#   ahead of the front  -- true F = 1, but the run reports F ~ norm2 = 1-delta
#   behind the front    -- true C = 2, but the run reports C ~ 2*norm2
# so the fix is applied where each one is damaged:
#     Csq_corr = Csq + 2*delta*Theta           (Theta = 1 once the front arrives)
#     F_corr   = F   +   delta*(1 - Theta)
#
# Substituting Csq = 2*(norm2 - F) shows that in BOTH regions the result obeys
#     Csq_corr = 2*(1 - F_corr),
# i.e. the scheme is exactly "assume norm2 = 1", with Theta choosing which member
# of the pair is trusted and which is re-derived. Callers should assert that
# identity as a self-test.
#
# Lives here, not in otoc_front_correction.jl, so the overlay scripts and the
# correction script cannot drift into two different definitions of "corrected".
#
# `thr` is in Csq units. The front index comes from the TRUNCATED Csq, so a site
# whose C never reaches thr is treated as permanently ahead of the front and its
# C is left uncorrected -- callers should report those sites rather than let the
# omission pass silently.
function front_correct(F, Csq, norm2; thr::Real = 0.2)
    δ = 1 .- norm2
    nT, nS = size(Csq)
    tj = [something(findfirst(>(thr), Csq[:, j]), nT + 1) for j in 1:nS]
    Θ = [k >= tj[j] ? 1.0 : 0.0 for k in 1:nT, j in 1:nS]
    Csq_c = Csq .+ 2 .* δ .* Θ          # δ is length nT, broadcast down columns
    F_c   = F   .+      δ .* (1 .- Θ)
    return F_c, Csq_c, Θ, tj
end


# LATTICES THAT OUTPUT FILES STILL REFERENCE ==================================
# kitaev_model.jl holds only the CURRENT lattice. Runs on disk carry their own
# fingerprint in the filename and were generated on whatever the lattice was
# then, so a script that needs bond DISTANCES for older data cannot read them
# from the live model -- it would label every shell wrongly and never say so.
# Keyed by that fingerprint, a run looks up its own geometry.
#
# Add an entry when a new lattice produces data that needs distances. An unknown
# tag is an error, never a silent fallback: a wrong distance table turns every
# front-based correction into nonsense that still looks plausible.
const KNOWN_LATTICES = Dict(
    # 16-site torus, the ORIGINAL bond lists. Most existing output was made on it.
    "L243535" => (
        z = [(2,1),(4,15),(6,7),(14,13),(16,11),(8,9),(3,12),(5,10)],
        x = [(2,3),(4,5),(14,15),(16,7),(12,11),(10,9),(1,6),(8,13)],
        y = [(3,4),(5,6),(1,14),(15,16),(7,8),(13,12),(11,10),(9,2)]),
    # 16-site torus, the CURRENT TORUS16 in kitaev_model.jl.
    "L24e521" => (
        z = [(1,9),(3,15),(5,13),(7,11),(10,8),(16,2),(14,4),(12,6)],
        x = [(2,3),(4,5),(6,7),(8,1),(9,16),(15,14),(13,12),(11,10)],
        y = [(1,2),(3,4),(5,6),(7,8),(10,9),(16,15),(14,13),(12,11)]),
    # 32-site torus (TORUS32). Diameter 6, zero 4-cycles.
    "L48439d" => (
        z = [(1,9),(3,15),(5,13),(7,11),(10,17),(16,19),(14,21),(12,23),
             (24,25),(18,31),(20,29),(22,27),(26,6),(32,8),(30,2),(28,4)],
        x = [(2,3),(4,5),(6,7),(8,1),(9,16),(15,14),(13,12),(11,10),
             (17,18),(19,20),(21,22),(23,24),(25,32),(31,30),(29,28),(27,26)],
        y = [(1,2),(3,4),(5,6),(7,8),(10,9),(16,15),(14,13),(12,11),
             (24,17),(18,19),(20,21),(22,23),(26,25),(32,31),(30,29),(28,27)]),
)

# `n` is DERIVED from the bond table, not passed in: a default of 16 silently
# truncated a 32-site lattice to its first 16 sites and reported half of them as
# unreachable.
function lattice_distances(lat::AbstractString, origin::Int)
    haskey(KNOWN_LATTICES, lat) || error(
        "no bond table for lattice \"$lat\". Add it to KNOWN_LATTICES in " *
        "otoc_plotting.jl. Distances must NOT be taken from kitaev_model.jl, " *
        "which holds only the currently-selected lattice -- runs on any other " *
        "would be given the wrong shells with no warning.")
    b = KNOWN_LATTICES[lat]
    n = maximum(max(p, q) for bl in (b.x, b.y, b.z) for (p, q) in bl)
    return graph_distances(origin, n, b.x, b.y, b.z)
end


# BUTTERFLY VELOCITY AND THE Eq (14) FRONT ====================================
# Step 3 of the lost-weight note: record the first time t_j at which C_{S,j}
# crosses a small threshold, then fit |j - j0| = v_B t_j -- a straight line
# through the origin, since the front leaves the source at t = 0. Minimising
# sum_j (d_j - v_B t_j)^2 gives v_B = sum(d t)/sum(t^2).
#
# `thr` is in Csq units (the note's C_S threshold doubled, since Csq = 2 C_note).
# Only sites that actually crossed, that are not the source, and whose crossing
# happened while l was still small ("obtain v_B at early times, while l ~ 0")
# inform the fit -- a crossing taken after truncation has eaten into C is a
# measurement of the truncation, not of the front.
function fit_front_velocity(time, Csq, norm2, dist; thr::Real = 0.2,
                            ell_max::Real = 0.05)
    δ = 1 .- norm2
    pts = NamedTuple[]
    for j in 1:size(Csq, 2)
        (j <= length(dist) && dist[j] != 0) || continue
        k = findfirst(>(thr), Csq[:, j])
        k === nothing && continue
        δ[k] <= ell_max || continue
        push!(pts, (site = j, d = Float64(dist[j]), t = time[k], ell = δ[k]))
    end
    isempty(pts) && return (vB = NaN, used = 0, resid = NaN, pts = pts)
    den = sum(p.t^2 for p in pts)
    vB  = den > 0 ? sum(p.d * p.t for p in pts) / den : NaN
    resid = sqrt(sum((p.d - vB * p.t)^2 for p in pts) / length(pts))
    return (vB = vB, used = length(pts), resid = resid, pts = pts)
end

# Step 4: Theta(v_B t - |j - j0|), a sharp straight-line front, then
#     Csq += 2 l Theta        (the note's C += l Theta, in Csq units)
#     F   += l (1 - Theta)    (same p_j model applied to F: F_exact = F_S +
#                              (1-2p_j) l, so +l ahead of the front, 0 behind)
# Differs from `front_correct` ONLY in how Theta is built: there, each site's own
# measured crossing; here, one fitted velocity for all sites, which also gives a
# front time to sites whose C never crossed the threshold.
function vb_front_correct(time, F, Csq, norm2, dist, vB)
    δ = 1 .- norm2
    nT, nS = size(Csq)
    Θ = [ (j <= length(dist) && vB * time[k] > dist[j]) ? 1.0 : 0.0
          for k in 1:nT, j in 1:nS ]
    return F .+ δ .* (1 .- Θ), Csq .+ 2 .* δ .* Θ, Θ
end


# Every script saves here, matching the project's existing convention
# (Correlations/ writes to its own plots/ folder the same way). Created on
# demand, so a fresh checkout needs no setup.
plots_dir() = (d = joinpath(@__DIR__, "plots"); mkpath(d); d)
