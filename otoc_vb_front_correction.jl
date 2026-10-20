# Eq (14) of "Squared commutator from truncated Pauli propagation and a
# lost-weight correction" (Sept 2026), implemented literally.
#
#     julia --project=. otoc_vb_front_correction.jl
#
# This differs from otoc_front_correction.jl in exactly one respect: the FRONT.
# That script uses each site's own measured threshold-crossing time as its
# arrival. The note instead prescribes a two-step procedure --
#
#     3. Obtain v_B at early times, while l ~ 0. Record the first time t_j at
#        which C_{S,j} exceeds a small threshold (e.g. 0.1), and fit
#        |j - j0| = v_B t_j.
#     4. Add l(t) to every site with |j - j0| < v_B t.
#
# -- so the front is a straight line with ONE fitted parameter, and a site whose
# C never reaches the threshold still gets a front time by extrapolation. Both
# are computed here and scored against ED, so the choice can be made on evidence.
#
# UNITS -- THE FACTOR OF 2 IS REAL AND IS HANDLED HERE
# ----------------------------------------------------
# The note defines       C_note = 1/2 tr([W,V]^dag [W,V]) = 2 sum_anti c^2,
# and Eq (7) gives       C_note = norm2 - F.
# This project stores    Csq    = 2 (norm2 - F) = 4 sum_anti c^2.
# Hence
#       Csq = 2 * C_note ,
# verified on file to 1e-16. Every quantity from the note is therefore doubled
# before it touches a Csq array:
#       note threshold C_S > 0.1          ->  Csq > 0.2
#       note correction  C += l * Theta   ->  Csq += 2 l * Theta
#       note bracket  C_S <= C <= C_S+2l  ->  Csq_S <= Csq <= Csq_S + 4l
# Getting this wrong by a factor of 2 is the single easiest way to misapply the
# note, so the conversion is done once, here, and never repeated inline.
#
# ASSUMPTIONS CARRIED OVER FROM THE NOTE (Sec. 3, "When the assumptions fail")
#   * Assumption A: discarded weight does not flow back onto kept strings.
#   * p_j = 1/2 behind the front -- fails AT the front, where strings are not
#     yet scrambled.
#   * the sharp Theta overcounts within about one front width of |j-j0| = v_B t.
#   * "Do not renormalize the kept coefficients" -- so renormalized runs are
#     refused outright rather than silently corrected.

include(joinpath(@__DIR__, "otoc_plotting.jl"))

# --- configuration -----------------------------------------------------------
run_files = [
#  "PP_kitaev_otoc_infT_N32_L48439d_coeff1e3_w32_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
#  "PP_kitaev_otoc_infT_N32_L48439d_coeff5e4_w32_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
#  "PP_kitaev_otoc_infT_N32_L48439d_coeff1e4_w32_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
#  "PP_kitaev_otoc_infT_N32_L48439d_coeff8e5_w32_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz"
   "PP_kitaev_otoc_infT_N32_L48439d_coeff1e4_w32_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
  "PP_kitaev_otoc_infT_N32_L48439d_coeff8e5_w32_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
  "PP_kitaev_otoc_infT_N32_L48439d_coeff5e5_w32_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
]

thr_note = 0.1        # the note's threshold on C_{S,j}; doubled for Csq below
ell_fit_max = 0.05    # "obtain v_B at early times, while l ~ 0": only use a
                      # site's crossing in the fit if l has not yet grown past
                      # this. Truncation has then not yet distorted the front.
probe_sites = [2, 3, 4, 5, 6, 12, 23]
# Draw the model-free band of Eq (12), Csq_S <= Csq <= Csq_S + 4*l? It is the
# range the exact answer MUST lie in given only 0 <= p_j <= 1, with no modelling
# at all -- but it widens as l grows and at late times swamps the curves it is
# meant to bound. Off by default; the bracket is still TESTED numerically in the
# table above regardless of this setting.
show_bracket = false

# Draw the Eq (14) / fitted-v_B curve at all? With this false the figure shows
# only the raw run, the per-site front correction and ED -- useful when the
# question is what the per-site scheme does on its own, without the modelled
# p = 1/2 curve sitting on top of it. The v_B fit is still computed and still
# scored in the printed table; this controls the FIGURE only, and also drops
# the fit panel, which exists solely to justify the v_B curve.
show_vb = true

# Panel grid. Panels are the v_B fit (when show_vb) followed by one per probe
# site. A single row renders thousands of pixels wide and unreadably short once
# there are more than about four, so the shape is set explicitly here; extra
# panels beyond rows*cols are dropped with a warning rather than silently.
grid_shape = (2, 3)
# -----------------------------------------------------------------------------

const DIR = joinpath(@__DIR__, "otoc_outputs")
const THR_CSQ = 2 * thr_note     # the one and only note -> project conversion

# Bond distances come from the shared KNOWN_LATTICES table in otoc_plotting.jl,
# keyed by the run's OWN lattice fingerprint. Previously each of these scripts
# carried its own hardcoded copy of the L243535 bonds, which meant a run on any
# other lattice (torus16's current L24e521, or the 32-site L48439d) was silently
# given the wrong shells. An unknown tag is now a hard error instead.
dists_from(origin::Int, lat::AbstractString) = lattice_distances(lat, origin)

# STEP 3 OF THE NOTE: fit |j - j0| = v_B t_j ----------------------------------
# A line THROUGH THE ORIGIN, because the front leaves the source site at t = 0:
# minimising sum_j (d_j - v_B t_j)^2 gives v_B = sum(d t) / sum(t^2).
# Only sites that actually crossed, that are not the source itself, and whose
# crossing happened while l was still small, are used.
function fit_vB(time, Csq, norm2, dist; thr = THR_CSQ, ell_max = ell_fit_max)
    nT = length(time)
    δ = 1 .- norm2
    pts = NamedTuple[]
    for j in 1:size(Csq, 2)
        dist[j] == 0 && continue                       # source carries no information
        k = findfirst(>(thr), Csq[:, j])
        k === nothing && continue                      # never arrived: cannot inform the fit
        δ[k] <= ell_max || continue                    # crossing already polluted by truncation
        push!(pts, (site = j, d = Float64(dist[j]), t = time[k], ell = δ[k]))
    end
    isempty(pts) && return (vB = NaN, pts = pts, resid = NaN, used = 0)
    num = sum(p.d * p.t for p in pts)
    den = sum(p.t^2 for p in pts)
    vB = den > 0 ? num / den : NaN
    # RMS residual in distance units -- how far from a straight front the data is.
    resid = sqrt(sum((p.d - vB * p.t)^2 for p in pts) / length(pts))
    return (vB = vB, pts = pts, resid = resid, used = length(pts))
end

# STEP 4: Theta(v_B t - |j - j0|), then Csq += 2 l Theta.
function note_correct(time, Csq, norm2, dist, vB)
    δ = 1 .- norm2
    nT, nS = size(Csq)
    Θ = [ (vB * time[k] > dist[j]) ? 1.0 : 0.0 for k in 1:nT, j in 1:nS ]
    return Csq .+ 2 .* δ .* Θ, Θ            # 2*l*Theta == the note's l*Theta in C_note
end

function find_ed(m::RunMeta)
    isdir(DIR) || return nothing
    pat = "ED_kitaev_otoc_infT_N$(m.N)_$(m.lat)_exact_V$(m.V)_J$(m.J)_" *
          "$(m.pauli)_site$(m.site)_dt$(m.dt)_"
    hits = filter(f -> startswith(f, pat) && endswith(f, ".npz"), readdir(DIR))
    isempty(hits) ? nothing : load_run(joinpath(DIR, sort(hits)[end]))
end

maxdev(a, b) = (n = min(length(a), length(b)); maximum(abs.(a[1:n] .- b[1:n])))

function save_resilient(fig, path)
    for attempt in 1:3
        try
            savefig(fig, path); return path
        catch
            attempt < 3 && (sleep(1.5); continue)
            alt = replace(path, ".png" => "_new.png")
            try
                savefig(fig, alt); @warn "wrote $(basename(alt))"; return alt
            catch err
                @error "could not save $(basename(path))" exception = err; return nothing
            end
        end
    end
end

println("Eq (14) of the lost-weight note, implemented literally")
println("note C = Csq/2, so threshold $(thr_note) -> Csq > $(THR_CSQ), and  C += l  ->  Csq += 2l")

for f in run_files
    path = joinpath(DIR, f)
    isfile(path) || error("not in otoc_outputs/: $f")
    r = load_run(path)
    d = npzread(path)
    is_norm = (haskey(d, "normalized") && d["normalized"][1] == 1.0) || haskey(d, "cum_retention")

    println("\n", "="^100)
    println(basename(path))
    println("="^100)
    if is_norm
        println("  REFUSED: this run is renormalized. The note's step 1 says explicitly")
        println("  \"Do not renormalize the kept coefficients\" -- l is then 0 and the")
        println("  derivation of Eq (11) does not apply.")
        continue
    end

    δ = 1 .- r.norm2
    dist = dists_from(r.meta.site, r.meta.lat)
    @printf("  V=%s J=%s  W@%d   l: 0 -> %.5f\n", r.meta.V, r.meta.J, r.meta.site, δ[end])

    fit = fit_vB(r.time, r.Csq, r.norm2, dist)
    if !isfinite(fit.vB)
        println("  no usable crossings while l <= $ell_fit_max -- cannot fit v_B; skipping")
        continue
    end
    @printf("\n  STEP 3  fit |j-j0| = v_B t_j   over %d site(s) with l <= %.3f at crossing\n",
            fit.used, ell_fit_max)
    @printf("          v_B = %.4f bonds / unit time     RMS residual = %.4f bonds\n",
            fit.vB, fit.resid)
    println("  " * "-"^72)
    @printf("  %6s %6s %9s %9s %11s\n", "site", "dist", "t_j", "v_B t_j", "l at t_j")
    println("  " * "-"^72)
    for p in sort(fit.pts, by = x -> (x.d, x.t))
        @printf("  %6d %6.0f %9.2f %9.3f %11.2e\n", p.site, p.d, p.t, fit.vB * p.t, p.ell)
    end
    println("  " * "-"^72)

    Csq_note, Θ = note_correct(r.time, r.Csq, r.norm2, dist, fit.vB)
    # The per-site variant, for comparison (otoc_front_correction.jl's choice).
    _, Csq_site, _, tj = front_correct(r.F, r.Csq, r.norm2; thr = THR_CSQ)

    ed = find_ed(r.meta)
    if ed === nothing
        println("  (no ED at this physics point -- corrected but not scored)")
    else
        println("\n  ED reference: ", basename(ed.meta.path))
        println("  " * "-"^98)
        @printf("  %6s %6s | %10s %14s %14s | %s\n", "site", "dist",
                "raw", "Eq(14) v_B fit", "per-site front", "Eq(12) bracket vs ED")
        println("  " * "-"^98)
        for j in probe_sites
            cj = r.col(j); ej = ed.col(j)
            (cj === nothing || ej === nothing) && continue
            n = min(length(r.time), length(ed.time))
            # Eq (12): the model-free bracket, Csq_S <= Csq_exact <= Csq_S + 4l.
            # No p_j assumption enters it, so a REAL excursion means Assumption A
            # has failed. But ED here is a stochastic trace estimate carrying its
            # own 1-sigma (2*F_err on Csq, ~2e-3), which is the same order as the
            # excursions seen -- so the bound is tested AGAINST THAT ERROR BAR.
            # Without it the test reports a violation on essentially every run and
            # indicts Assumption A for what is only sampling noise.
            tol  = 2 .* ed.F_err[1:n, ej]
            lo_v = maximum(r.Csq[1:n, cj] .- ed.Csq[1:n, ej] .- tol)
            up_v = maximum(ed.Csq[1:n, ej] .- r.Csq[1:n, cj] .- 4 .* δ[1:n] .- tol)
            viol = max(lo_v, up_v, 0.0)
            @printf("  %6d %6d | %10.4f %14.4f %14.4f | %s\n", j, dist[j],
                    maxdev(r.Csq[:, cj], ed.Csq[:, ej]),
                    maxdev(Csq_note[:, cj], ed.Csq[:, ej]),
                    maxdev(Csq_site[:, cj], ed.Csq[:, ej]),
                    viol <= 0 ? "holds within ED 1-sigma" :
                                @sprintf("exceeds by %.1e", viol))
        end
        println("  " * "-"^98)
        # How much do the two front definitions actually differ? If the front
        # passes while l is still tiny, Theta differs only where there is almost
        # nothing to add, and the two schemes coincide.
        dfront = maximum(abs.(Csq_note .- Csq_site))
        @printf("  max|Eq(14) - per-site| over all sites and times: %.2e\n", dfront)
        agg(x) = maximum(maxdev(x[:, r.col(j)], ed.Csq[:, ed.col(j)]) for j in probe_sites
                         if r.col(j) !== nothing && ed.col(j) !== nothing)
        @printf("  WORST over probes:  raw %.4f   Eq(14) %.4f   per-site %.4f\n",
                agg(r.Csq), agg(Csq_note), agg(Csq_site))
    end

    # ---- figure -------------------------------------------------------------
    # Panel 1 is the v_B fit itself: the note's Eq (14) stands or falls on the
    # front being a straight line, so the evidence for that belongs in the figure
    # rather than in a single printed number.
    p1 = plot(; base_attrs()..., title = "front fit: v_B = $(round(fit.vB, digits=3))",
              titlefontsize = 11, xlabel = "crossing time  t_j", ylabel = "bond distance",
              guidefontsize = 10, tickfontsize = 9, legend = :bottomright,
              legendfontsize = 7, grid = true)
    tmaxf = maximum(p.t for p in fit.pts) * 1.15
    plot!(p1, [0, tmaxf], [0, fit.vB * tmaxf]; color = series_color(1), lw = 2,
          label = "|j-j0| = v_B t")
    scatter!(p1, [p.t for p in fit.pts], [p.d for p in fit.pts];
             color = series_color(2), ms = 6, msw = 0, label = "measured t_j")

    # The fit panel justifies the v_B curve; with that curve off it is noise.
    panels = show_vb ? Any[p1] : Any[]
    for j in probe_sites
        cj = r.col(j)
        cj === nothing && continue
        p = plot(; base_attrs()..., title = "site $j  (d=$(dist[j]))", titlefontsize = 11,
                 xlabel = "time  t", ylabel = j == first(probe_sites) ? L"C_{ij}(t)" : "",
                 guidefontsize = 10, tickfontsize = 9,
                 legend = j == first(probe_sites) ? :best : false, legendfontsize = 6,
                 xlims = (0, r.meta.tmax), grid = true)
        hline!(p, [2.0]; color = INK_MUTED, lw = 1, ls = :dot, label = "")
        # Eq (12) as a band: everything between the raw curve and raw + 4l is
        # allowed without any modelling of p_j at all.
        if show_bracket
            plot!(p, r.time, r.Csq[:, cj] .+ 4 .* δ;
                  fillrange = r.Csq[:, cj], fillalpha = 0.13, fillcolor = series_color(1),
                  linealpha = 0, label = "Eq(12) bracket")
        end
        if ed !== nothing && ed.col(j) !== nothing
            plot!(p, ed.time, ed.Csq[:, ed.col(j)]; color = INK_PRIMARY, lw = 2.4, label = "ED")
        end
        show_vb && plot!(p, r.time, Csq_note[:, cj]; color = series_color(1), lw = 3.0,
              label = "Eq(14), fitted v_B")
        plot!(p, r.time, Csq_site[:, cj]; color = series_color(3), lw = 1.8,
              ls = :dashdot, label = "per-site front")
        plot!(p, r.time, r.Csq[:, cj]; color = series_color(2), lw = 1.6,
              ls = :dot, label = "raw PP")
        # The note's front for this site, and the per-site one, so the single
        # difference between the two schemes is visible rather than inferred.
        show_vb && vline!(p, [dist[j] / fit.vB]; color = series_color(1), lw = 0.9, ls = :dash, label = "")
        k = tj[cj]
        k <= length(r.time) && vline!(p, [r.time[k]]; color = series_color(3),
                                      lw = 0.9, ls = :dot, label = "")
        push!(panels, p)
    end

    # Fit the panels to the requested grid. More panels than slots would make
    # Plots silently reflow them, so the excess is dropped and NAMED instead.
    nrow, ncol = grid_shape
    if length(panels) > nrow * ncol
        # Expected whenever the ladder is longer than the grid, so this is a
        # note rather than a warning -- but it still names what was dropped,
        # because a silently missing probe site is indistinguishable from one
        # that was never computed.
        dropped = [j for j in probe_sites][(nrow*ncol - (show_vb ? 1 : 0) + 1):end]
        @printf("  grid %dx%d shows %d of %d panels; omitting site(s) %s
",
                nrow, ncol, nrow*ncol, length(panels), join(dropped, ", "))
        panels = panels[1:nrow*ncol]
    end
    lay = length(panels) < nrow * ncol ?
          (ceil(Int, length(panels) / ncol), min(ncol, length(panels))) : (nrow, ncol)
    fig = plot(panels...; layout = lay, size = (430 * lay[2], 400 * lay[1] + 70), dpi = 200,
               plot_title = "Lost-weight correction, Eq (14) with fitted v_B - " *
                            "$(r.meta.trunc)   N=$(r.meta.N) $(r.meta.lat) J=$(r.meta.J) W@$(r.meta.site)",
               plot_titlefontsize = 11, top_margin = 10Plots.mm,
               left_margin = 10Plots.mm, bottom_margin = 9Plots.mm, right_margin = 4Plots.mm)
    out = joinpath(plots_dir(),
        "otoc_vbfront_$(r.meta.trunc)_N$(r.meta.N)_$(r.meta.lat)_V$(r.meta.V)_J$(r.meta.J)" *
        "_site$(r.meta.site).png")
    s = save_resilient(fig, out)
    s === nothing || println("\n  Saved: $(basename(s))")
end
