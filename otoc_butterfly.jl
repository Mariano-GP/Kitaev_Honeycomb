# Butterfly velocity from the OTOC front.
#
#     julia --project=. otoc_butterfly.jl
#
# WHY THIS REPLACES skw_spectra.jl RATHER THAN ADAPTING IT
# --------------------------------------------------------
# Correlations/skw_spectra.jl computes S(k,ω) by Fourier transforming C(t,j)
# over space and time. That works because a two-point correlator is a linear
# response function: its transform IS the dynamical structure factor, with poles
# on the magnon dispersion, and the green ω(k) curve overlaid there means
# something.
#
# None of that carries over. The squared commutator is a FOUR-point, non-linear
# object; it is not a response function, has no spectral representation, and no
# dispersion relation to overlay. Fourier transforming it would produce a
# picture that looks like S(k,ω) and means nothing. There is also no wavevector
# to transform over: the transform in skw_spectra.jl relies on a 1D periodic
# ring indexed by j (note its `circshift` recentering), whereas this is a
# 2D honeycomb where site index carries no geometry at all.
#
# The quantity an OTOC does define, and that S(k,ω) is the two-point analogue
# of, is the BUTTERFLY VELOCITY v_B -- the speed of the operator front. This
# script extracts it: for each graph-distance shell d, find the first time
# t_d at which the shell-averaged C_{ij}(t) crosses a threshold, then fit
# t_d = d / v_B + const. The slope of the front in the (d, t) plane is 1/v_B,
# so this is the quantitative reading of the light cone that otoc_lightcone.jl
# shows qualitatively.
#
# CAVEAT, stated because it changes the number: v_B extracted this way depends
# on the threshold, and genuinely so -- a contour of constant C is not unique.
# Several thresholds are therefore fitted and reported together. If they
# disagree strongly, the front is not sharp enough for v_B to be well defined
# over this window, and that is a result about the data, not a fitting problem
# to tune away.

include(joinpath(@__DIR__, "kitaev_model.jl"))
include(joinpath(@__DIR__, "otoc_plotting.jl"))

# --- Edit these ---------------------------------------------------------
npz_path = joinpath(@__DIR__, "otoc_outputs",
                    "PP_kitaev_otoc_infT_N16_L243535_K1.0-1.0-1.0_V0.3_J0.0_ZZ_site2_dt0.05_tmax3.0.npz")
thresholds = [0.02, 0.05, 0.10, 0.20]     # C_ij crossing levels to fit
# ------------------------------------------------------------------------

run = load_run(npz_path)
m = run.meta
m.N == N || error("file is N=$(m.N) but kitaev_model.jl defines N=$N -- distances would be wrong")
# Same N with different bonds gives different distance shells, so check the
# lattice fingerprint too (see lattice_tag() in kitaev_model.jl).
if isempty(m.lat)
    @warn "This run predates the lattice fingerprint; if the bond lists changed since it was " *
          "generated, every distance shell -- and so v_B -- is wrong."
elseif m.lat != lattice_tag()
    error("run is on lattice $(m.lat) but kitaev_model.jl is now $(lattice_tag()) -- same N, " *
          "different bonds, so the fitted v_B would be meaningless. Regenerate the run.")
end

dist = graph_distances(m.site, N, X_BONDS, Y_BONDS, Z_BONDS)
max_d = maximum(dist)
t = run.time

@printf("Run: %s\n  reference site=%d  pauli=%s  V=%s  J=%s\n",
        basename(npz_path), m.site, m.pauli, m.V, m.J)

# Shell-averaged squared commutator.
C = zeros(Float64, max_d + 1, length(t))
for d in 0:max_d
    cols = [run.col(s) for s in findall(dist .== d) if run.col(s) !== nothing]
    isempty(cols) || (C[d+1, :] = vec(sum(run.Csq[:, cols], dims = 2)) ./ length(cols))
end

# First crossing time per shell, linearly interpolated between samples so the
# answer is not quantized to the dt grid (which would bias every fitted slope).
function crossing_time(series, times, level)
    for k in 2:length(series)
        if series[k] >= level && series[k-1] < level
            f = (level - series[k-1]) / (series[k] - series[k-1])
            return times[k-1] + f * (times[k] - times[k-1])
        end
    end
    return nothing
end

# Least-squares fit of t_d = a*d + b; v_B = 1/a. Shell 0 is excluded: it is the
# reference site itself, where C rises instantly by construction and carries no
# propagation information.
function fit_front(ds, ts)
    length(ds) < 2 && return (NaN, NaN, NaN)
    n = length(ds); sd = sum(ds); st = sum(ts)
    sdd = sum(ds .^ 2); sdt = sum(ds .* ts)
    den = n * sdd - sd^2
    den == 0 && return (NaN, NaN, NaN)
    a = (n * sdt - sd * st) / den
    b = (st - a * sd) / n
    pred = a .* ds .+ b
    ss_res = sum((ts .- pred) .^ 2); ss_tot = sum((ts .- st/n) .^ 2)
    r2 = ss_tot == 0 ? NaN : 1 - ss_res / ss_tot
    return (a, b, r2)
end

fig = plot(; base_attrs()..., xlabel = "graph distance d from site $(m.site)",
             ylabel = "front arrival time  t(d)",
             title = "Butterfly front — $(m.pauli), $(m.solver), N=$N, V=$(m.V), J=$(m.J)",
             legend = :topleft, legendfontsize = 9,
             guidefontsize = 11, tickfontsize = 10, titlefontsize = 11,
             size = (760, 520), dpi = 200,
             left_margin = 8Plots.mm, bottom_margin = 8Plots.mm)

println("\n" * "-"^66)
@printf("%12s %10s %12s %10s %8s\n", "threshold", "shells", "v_B", "1/v_B", "R²")
println("-"^66)

results = []
for (i, lev) in enumerate(thresholds)
    ds = Float64[]; ts = Float64[]
    for d in 1:max_d
        tc = crossing_time(C[d+1, :], t, lev)
        tc === nothing && continue
        push!(ds, d); push!(ts, tc)
    end
    a, b, r2 = fit_front(ds, ts)
    vB = 1 / a
    push!(results, (lev, length(ds), vB, r2))
    @printf("%12.3f %10d %12.4f %10.4f %8.4f\n", lev, length(ds),
            vB, a, r2)

    c = series_color(i)
    scatter!(fig, ds, ts; label = "C = $lev", color = c, markersize = 6,
             markerstrokewidth = 0.8, markerstrokecolor = SURFACE)
    if isfinite(a)
        xs = [0.0, Float64(max_d)]
        plot!(fig, xs, a .* xs .+ b; label = "", color = c, lw = 2, linestyle = :dash)
    end
end
println("-"^66)

# One annotation carrying the headline number, rather than a value printed on
# every marker.
finite_vB = [r[3] for r in results if isfinite(r[3])]
if !isempty(finite_vB)
    @printf("v_B across thresholds: %.3f – %.3f  (spread %.1f%%)\n",
            minimum(finite_vB), maximum(finite_vB),
            100 * (maximum(finite_vB) - minimum(finite_vB)) / maximum(finite_vB))
    # Placed against the ARRIVAL-TIME range, not the run's full time axis: the
    # front reaches the far shell long before t_max, so anchoring to maximum(t)
    # puts the annotation far outside the visible y-range.
    all_ts = vcat([[crossing_time(C[d+1, :], t, lev) for d in 1:max_d
                    if crossing_time(C[d+1, :], t, lev) !== nothing]
                   for lev in thresholds]...)
    y_at = isempty(all_ts) ? 0.0 : maximum(all_ts)
    annotate!(fig, [(0.15, y_at,
        text("v_B ≈ $(round(sum(finite_vB)/length(finite_vB), digits=2))", 11, :left, INK_PRIMARY))])
    if (maximum(finite_vB) - minimum(finite_vB)) / maximum(finite_vB) > 0.25
        @warn "v_B varies by more than 25% across thresholds -- the front is not sharp over " *
              "this window, so a single butterfly velocity is not well defined here. Run to " *
              "larger t_max, or read the light cone qualitatively instead."
    end
end

if any(r[2] < 2 for r in results)
    @warn "Some thresholds were crossed in fewer than 2 shells -- extend t_max so the front " *
          "reaches farther before fitting."
end

display(fig)
out = joinpath(plots_dir(), "otoc_butterfly_" * replace(basename(npz_path), ".npz" => "") * ".png")
savefig(fig, out)
println("\nSaved: $out")
