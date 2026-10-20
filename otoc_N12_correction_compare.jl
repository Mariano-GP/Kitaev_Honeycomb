# Compare the truncation corrections on the OTOC-PP 2x3 Kitaev-Heisenberg series
# (kh_honeycomb_N12_J0.1_cut*.csv) against the exact reference run.
#
#     julia --project=. otoc_N12_correction_compare.jl
#     $env:N12_DATA_DIR = "D:\some\folder"; julia --project=. otoc_N12_correction_compare.jl
#
# Three estimates of the squared commutator are scored, per site and per time:
#
#   raw       otoc_j       the truncated run as it is
#   otoc_cor  otoc_cor_j   OTOC-PP's own correction: the anticommuting weight
#                          actually discarded for site j, summed over every
#                          truncation (OtocCorrection in otoc_observables.jl)
#   Vb        Eq (14)      raw + 2 l(t) Theta(v_B t - |j - j0|), l = 1 - w_norm2,
#                          v_B fitted to raw threshold crossings -- the same
#                          procedure as otoc_vb_front_correction.jl
#
# REFERENCE. The cut1e-12 run is exact up to Trotter error: it reaches all 4^11
# strings of the symmetry sector reachable from Z_1 (H commutes with the global
# pi-rotations prod X, prod Y, prod Z) and keeps w_norm2 = 1. Trotter error is
# identical in every run, so differences against it are pure truncation error.
#
# UNITS. otoc_j = 4 sum_anti |c|^2, i.e. the project's Csq. The note's threshold
# 0.1 is therefore 0.2 here and its correction l*Theta is 2l*Theta -- see the
# header of otoc_vb_front_correction.jl. Time is in units of the NORMALIZED
# Hamiltonian (lambda = 4.705), so v_B is not comparable to the K = 1 runs.
#
# OUTPUT (plots/):
#   N12compare_cut<c>.png   one figure per cutoff: v_B fit + one panel per site
#   N12compare_errors.png   RMS error over sites vs time, one panel per cutoff
# and a printed table of max |error| per site, plus Eq (12) bracket violations.

include(joinpath(@__DIR__, "otoc_plotting.jl"))

# --- configuration -----------------------------------------------------------
const DATA_DIR = get(ENV, "N12_DATA_DIR",
    raw"C:\Users\maria\Downloads\data-trunc_series_N12\data-trunc_series_N12")
const PREFIX   = "kh_honeycomb_N12_J0.1_cut"
ref_cut  = "1e-12"                                  # the exact reference
cuts     = ["1e-3", "5e-4", "2e-4", "1e-4", "5e-5", "1e-5", "1e-6"]

thr_note    = 0.1      # note's threshold on C_S; doubled below for Csq
ell_fit_max = 0.05     # use a crossing in the v_B fit only while l is this small
grid_shape  = (3, 4)   # v_B panel + 11 site panels
# -----------------------------------------------------------------------------

const THR_CSQ = 2 * thr_note
const OUTDIR  = joinpath(@__DIR__, "plots")
isdir(OUTDIR) || mkpath(OUTDIR)

# CSV ------------------------------------------------------------------------
# Plain comma-separated numbers with one header row; no CSV.jl dependency.
function read_run(cut)
    path = joinpath(DATA_DIR, PREFIX * cut * ".csv")
    isfile(path) || error("missing $path (set N12_DATA_DIR)")
    lines = filter(!isempty, readlines(path))
    hdr = split(lines[1], ',')
    rows = [parse.(Float64, split(l, ',')) for l in lines[2:end]]
    M = permutedims(hcat(rows...))
    col(name) = M[:, findfirst(==(name), hdr)]
    sites = [parse(Int, h[6:end]) for h in hdr if startswith(h, "otoc_") && !startswith(h, "otoc_cor_")]
    raw = hcat([col("otoc_$j") for j in sites]...)
    cor = hcat([col("otoc_cor_$j") for j in sites]...)
    return (cut = cut, time = col("time"), norm2 = col("w_norm2"),
            sites = sites, raw = raw, cor = cor, path = path)
end

# LATTICE --------------------------------------------------------------------
# Distances come from the run's own Hamiltonian file: every two-body string is
# a bond. Nothing about the 2x3 geometry is typed in by hand.
function distances_from_hamiltonian(cut, origin, N)
    path = joinpath(DATA_DIR, PREFIX * cut * "_hamiltonian.txt")
    adj = [Set{Int}() for _ in 1:N]
    for l in eachline(path)
        (startswith(l, "#") || isempty(strip(l))) && continue
        s = split(l)[2]
        q = findall(!=('I'), s)
        length(q) == 2 || continue
        push!(adj[q[1]], q[2]); push!(adj[q[2]], q[1])
    end
    d = fill(-1, N); d[origin] = 0; queue = [origin]
    while !isempty(queue)
        u = popfirst!(queue)
        for v in adj[u]
            d[v] < 0 && (d[v] = d[u] + 1; push!(queue, v))
        end
    end
    any(<(0), d) && error("lattice from $path is disconnected")
    return d
end

# VB CORRECTION (same procedure as otoc_vb_front_correction.jl) --------------
function fit_vB(time, Csq, norm2, dist)
    ℓ = 1 .- norm2
    pts = NamedTuple[]
    for j in axes(Csq, 2)
        k = findfirst(>(THR_CSQ), Csq[:, j])
        k === nothing && continue
        ℓ[k] <= ell_fit_max || continue
        push!(pts, (col = j, d = Float64(dist[j]), t = time[k]))
    end
    isempty(pts) && return (vB = NaN, pts = pts, resid = NaN)
    vB = sum(p.d * p.t for p in pts) / sum(p.t^2 for p in pts)
    resid = sqrt(sum((p.d - vB * p.t)^2 for p in pts) / length(pts))
    return (vB = vB, pts = pts, resid = resid)
end

vb_correct(time, Csq, norm2, dist, vB) =
    Csq .+ 2 .* (1 .- norm2) .* [vB * t > d ? 1.0 : 0.0 for t in time, d in dist]

# MAIN -----------------------------------------------------------------------
ref = read_run(ref_cut)
N = maximum(ref.sites)
origin = only(setdiff(1:N, ref.sites))           # W's site: the column that is absent
dist_all = distances_from_hamiltonian(ref_cut, origin, N)
dist = dist_all[ref.sites]                        # aligned with the otoc_j columns
@printf("reference %s: N=%d, W at site %d, w_norm2 at end = %.3e\n",
        ref_cut, N, origin, ref.norm2[end])
println("distances from site $origin: ",
        join(["$j=>$(dist_all[j])" for j in ref.sites], "  "))

const METHODS = ("raw", "otoc_cor", "Vb")
summary = Dict{String,Any}()

for cut in cuts
    r = read_run(cut)
    r.sites == ref.sites || error("$cut: site columns differ from the reference")
    nT = min(length(r.time), length(ref.time))
    maximum(abs.(r.time[1:nT] .- ref.time[1:nT])) < 1e-9 || error("$cut: time grids differ")
    t = r.time[1:nT]; exact = ref.raw[1:nT, :]
    ℓ = 1 .- r.norm2[1:nT]

    fit = fit_vB(t, r.raw[1:nT, :], r.norm2[1:nT], dist)
    est = Dict("raw" => r.raw[1:nT, :], "otoc_cor" => r.cor[1:nT, :],
               "Vb" => isfinite(fit.vB) ? vb_correct(t, r.raw[1:nT, :], r.norm2[1:nT], dist, fit.vB) :
                                          fill(NaN, nT, length(dist)))
    err = Dict(m => est[m] .- exact for m in METHODS)
    summary[cut] = (t = t, err = err, ℓ = ℓ)

    # --- printed table ------------------------------------------------------
    println("\n", "="^92)
    @printf("cut %s   l(t_end) = %.4e   v_B = %.3f bonds/time  (RMS residual %.2f, %d sites)\n",
            cut, ℓ[end], fit.vB, fit.resid, length(fit.pts))
    @printf("  %5s %5s | %12s %12s %12s | %s\n", "site", "dist",
            "max|raw|", "max|cor|", "max|Vb|", "bracket raw <= exact <= raw+4l violated at")
    for (c, j) in enumerate(r.sites)
        mx = [maximum(abs, err[m][:, c]) for m in METHODS]
        # how far exact sits OUTSIDE [raw, raw + 4l]; 0 when inside
        viol = [max(r.raw[k, c] - exact[k, c], exact[k, c] - r.raw[k, c] - 4ℓ[k], 0.0) for k in 1:nT]
        bad = findall(>(1e-12), viol)
        brk = isempty(bad) ? "never" :
              @sprintf("%3d/%d times, worst %.1e (first t=%.2f)", length(bad), nT, maximum(viol), t[bad[1]])
        best = METHODS[argmin(mx)]
        @printf("  %5d %5d | %12.4e %12.4e %12.4e | %s   best: %s\n",
                j, dist[c], mx..., brk, best)
    end

    # --- per-site figure ----------------------------------------------------
    tmax = t[end]
    p1 = plot(; base_attrs()..., title = "v_B fit = $(round(fit.vB, digits = 3))",
              xlabel = "crossing time", ylabel = "|j - j0|", legend = :topleft)
    if !isempty(fit.pts)
        scatter!(p1, [p.t for p in fit.pts], [p.d for p in fit.pts];
                 color = INK_PRIMARY, ms = 4, label = "Csq > $(THR_CSQ)")
        plot!(p1, [0, tmax], [0, fit.vB * tmax]; color = series_color(4), lw = 2, label = "v_B t")
    end
    ylims!(p1, (0, maximum(dist) + 0.5))
    panels = Any[p1]
    for (c, j) in enumerate(r.sites)
        p = plot(; base_attrs()..., title = "site $j  (d = $(dist[c]))",
                 xlabel = "t", ylabel = "Csq", legend = c == 1 ? :bottomright : false)
        plot!(p, t, exact[:, c]; color = INK_PRIMARY, lw = 2.5, label = "exact (cut $ref_cut)")
        plot!(p, t, est["raw"][:, c]; color = series_color(1), lw = 1.6, label = "raw")
        plot!(p, t, est["otoc_cor"][:, c]; color = series_color(2), lw = 1.6, label = "otoc_cor")
        plot!(p, t, est["Vb"][:, c]; color = series_color(3), lw = 1.6, ls = :dash, label = "Vb (Eq 14)")
        isfinite(fit.vB) && vline!(p, [dist[c] / fit.vB]; color = INK_MUTED, lw = 0.8, ls = :dot, label = "")
        push!(panels, p)
    end
    nr, nc = grid_shape
    length(panels) > nr * nc && @warn "only $(nr*nc) of $(length(panels)) panels shown"
    fig = plot(panels[1:min(end, nr * nc)]...; layout = (nr, nc), size = (420nc, 320nr),
               plot_title = "N=12 2x3 Kitaev-Heisenberg, J=0.1: corrections at cut $cut  (l_end = $(round(ℓ[end], sigdigits = 3)))",
               left_margin = 7Plots.mm, bottom_margin = 5Plots.mm)
    out = joinpath(OUTDIR, "N12compare_cut$(cut).png")
    savefig(fig, out); println("  wrote ", out)
end

# --- error summary figure: RMS over sites vs time, one panel per cutoff ------
epanels = Any[]
for cut in cuts
    s = summary[cut]
    p = plot(; base_attrs()..., title = "cut $cut", xlabel = "t", ylabel = "RMS error over sites",
             legend = cut == cuts[1] ? :topleft : false)
    for (i, m) in enumerate(METHODS)
        rms = vec(sqrt.(sum(abs2, s.err[m]; dims = 2) ./ size(s.err[m], 2)))
        plot!(p, s.t, rms; color = series_color(i), lw = 1.8, ls = m == "Vb" ? :dash : :solid, label = m)
    end
    plot!(p, s.t, 2 .* s.ℓ; color = INK_MUTED, lw = 1, ls = :dot, label = "2l (lost weight)")
    push!(epanels, p)
end
nc = 4; nr = cld(length(epanels), nc)
fig = plot(epanels...; layout = (nr, nc), size = (420nc, 320nr),
           plot_title = "RMS truncation error vs exact: raw, otoc_cor, Vb",
           left_margin = 7Plots.mm, bottom_margin = 5Plots.mm)
out = joinpath(OUTDIR, "N12compare_errors.png")
savefig(fig, out); println("\nwrote ", out)
