# Front-aware truncation correction: restore C(t) behind the front, F(t) ahead.
#
#     julia --project=. otoc_front_correction.jl
#
# THE PROBLEM
# -----------
# A truncated Pauli-propagation run loses operator weight, so
#     norm2(t) = sum_k |c_k|^2 = 1 - delta(t)        with delta growing in t.
# The two reported quantities are related exactly by
#     Csq = 2*(norm2 - F) = 2*(1 - delta - F),
# and the lost weight delta corrupts them in DIFFERENT PLACES:
#
#   * AHEAD of the front (W(t) has not reached site j): the true F is 1 and the
#     true C is 0. The run instead reports F ~ norm2 = 1 - delta, which sags
#     away from 1 purely because weight went missing -- a spurious decay that
#     looks like early scrambling but is pure truncation.
#   * BEHIND the front (W(t) has arrived and C has saturated): the true C is 2.
#     The run reports C ~ 2*norm2, too small by 2*delta -- the plateau sinks
#     instead of holding.
#
# So one correction cannot be applied everywhere: ahead of the front the damage
# is in F, behind it the damage is in C.
#
# THE CORRECTION
# --------------
# With Theta_kj = 1 once site j's front has arrived (and 0 before):
#     Csq_corr = Csq + 2*delta*Theta           (lift the plateau, behind only)
#     F_corr   = F   +   delta*(1 - Theta)     (lift F back to 1, ahead only)
#
# WHAT IT REALLY IS. Substituting Csq = 2*(1 - delta - F) into both lines gives,
# in BOTH regions,
#     Csq_corr = 2*(1 - F_corr).
# That is the identity Csq = 2*(norm2 - F) with norm2 FORCED TO 1. So the scheme
# is "assume no weight was lost", and Theta only decides which member of the
# pair is held fixed while the other is re-derived:
#     behind the front -> trust F, rebuild C from it
#     ahead of it      -> trust C (which is ~0 and correct), rebuild F
# The script asserts that identity numerically as a self-test; if it ever fails,
# the algebra above has been broken by an edit.
#
# WHY THIS IS NOT CIRCULAR. It does not invent information. Truncation removes
# Pauli strings, and both failure modes above come from the SAME missing weight.
# The correction says where that weight would have gone had it been kept: into
# the anticommuting sector behind the front (hence +2*delta on C) and into the
# identity-like commuting sector ahead of it (hence +delta on F). Both are
# statements about the geometry of operator spreading, not fits to the ED answer.
#
# WHAT IT CANNOT FIX. delta measures how much weight was dropped, never which
# strings they were. If truncation biases the surviving set -- keeps the
# anticommuting part preferentially, say -- the resulting error is invisible to
# norm2 and this correction leaves it untouched. It is therefore expected to
# work well where the error is dominated by MISSING WEIGHT (un-normalized runs)
# and to do nothing at all for renormalized runs, where norm2 is pinned to 1 by
# construction, delta = 0, and the error lives entirely in that bias.
#
# A MULTIPLICATIVE alternative is computed alongside for comparison:
#     F_mult = F / norm2 ,  Csq_mult = Csq / norm2
# which rescales the surviving weight back to unit norm proportionally. It needs
# no front and no threshold, which makes it the fair baseline: the front-aware
# scheme has to beat it to be worth its extra machinery.

include(joinpath(@__DIR__, "otoc_plotting.jl"))

# --- what to process ---------------------------------------------------------
# UN-NORMALIZED runs only: a renormalized run has norm2 == 1 identically, so
# delta == 0 and every correction here is the identity map.
run_files = [
 "PP_kitaev_otoc_infT_N16_L243535_coeff1e4_w16_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
 "PP_kitaev_otoc_infT_N16_L243535_coeff8e5_w16_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
 "PP_kitaev_otoc_infT_N16_L243535_coeff5e5_w16_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz"
 
]

# Front threshold, in Csq units (Csq = 2C in the note's normalization, so 0.2
# here is the note's C = 0.1).
thr = 0.2

# Draw the multiplicative baseline (raw / norm2) alongside raw and corrected?
# It stays in the printed table either way; this only controls the figure.
show_multiplicative = false

# Probe sites to score and plot. On L243535, from site 2, these sit at bond
# distances 0, 2, 3 and 4 -- near, middle and far, which is what makes a
# FRONT-aware correction worth testing at all.
probe_sites = [2, 12, 5, 16]
# -----------------------------------------------------------------------------

const DIR = joinpath(@__DIR__, "otoc_outputs")

# L243535 bond lists. The data here was generated on the PREVIOUS lattice, which
# kitaev_model.jl no longer holds -- reading distances from the live model would
# silently label every curve with the wrong shell.
const L243535 = (
    z = [(2,1),(4,15),(6,7),(14,13),(16,11),(8,9),(3,12),(5,10)],
    x = [(2,3),(4,5),(14,15),(16,7),(12,11),(10,9),(1,6),(8,13)],
    y = [(3,4),(5,6),(1,14),(15,16),(7,8),(13,12),(11,10),(9,2)])

function dists_from(origin::Int)
    adj = [Int[] for _ in 1:16]
    for bl in (L243535.x, L243535.y, L243535.z), (a, b) in bl
        push!(adj[a], b); push!(adj[b], a)
    end
    d = fill(-1, 16); d[origin] = 0; q = [origin]
    while !isempty(q)
        u = popfirst!(q)
        for v in adj[u]; d[v] == -1 && (d[v] = d[u] + 1; push!(q, v)); end
    end
    return d
end

# THE CORRECTION itself now lives in otoc_plotting.jl as `front_correct`, so
# this script and the overlay script share one definition and cannot drift apart.

# Find the ED run at the same physics point. Matched on lattice, V, J, pauli and
# W site -- never on t_max, since ED is usually run over a different window.
function find_ed(m::RunMeta)
    isdir(DIR) || return nothing
    pat = "ED_kitaev_otoc_infT_N$(m.N)_$(m.lat)_exact_V$(m.V)_J$(m.J)_" *
          "$(m.pauli)_site$(m.site)_dt$(m.dt)_"
    hits = filter(f -> startswith(f, pat) && endswith(f, ".npz"), readdir(DIR))
    isempty(hits) && return nothing
    # Prefer the longest window, so the overlap with the PP run is maximal.
    return load_run(joinpath(DIR, sort(hits)[end]))
end

maxdev(a, b) = (n = min(length(a), length(b)); maximum(abs.(a[1:n] .- b[1:n])))
rmsdev(a, b) = (n = min(length(a), length(b)); sqrt(sum(abs2, a[1:n] .- b[1:n]) / n))

results = NamedTuple[]
for f in run_files
    path = joinpath(DIR, f)
    isfile(path) || error("not in otoc_outputs/: $f")
    r = load_run(path)
    d = npzread(path)
    is_norm = (haskey(d, "normalized") && d["normalized"][1] == 1.0) || haskey(d, "cum_retention")

    println("\n", "="^104)
    println(basename(path))
    println("="^104)
    if is_norm
        println("  RENORMALIZED run: norm2 == 1 by construction, so delta == 0 and this")
        println("  correction is the identity. Skipped -- it would report a fake 0% change.")
        continue
    end

    δ = 1 .- r.norm2
    @printf("  V=%s J=%s  W@%d  t_max=%.1f   norm2: 1 -> %.5f   (delta_end = %.5f)\n",
            r.meta.V, r.meta.J, r.meta.site, r.meta.tmax, r.norm2[end], δ[end])

    F_c, C_c, Θ, tj = front_correct(r.F, r.Csq, r.norm2; thr = thr)
    F_m = r.F ./ r.norm2
    C_m = r.Csq ./ r.norm2

    # SELF-TEST: the two corrected arrays must satisfy Csq_corr = 2*(1 - F_corr)
    # in every cell. This is the algebraic content of the scheme; if it fails,
    # the correction is not doing what the header claims.
    resid = maximum(abs.(C_c .- 2 .* (1 .- F_c)))
    @printf("  self-test  max|Csq_corr - 2(1-F_corr)| = %.2e  %s\n", resid,
            resid < 1e-9 ? "OK" : "!! BROKEN")

    # Sites whose front never arrived: Theta is 0 for the whole column, so C was
    # left uncorrected there. Reported because it is a silent failure otherwise.
    never = [j for j in 1:size(r.Csq, 2) if tj[j] > size(r.Csq, 1)]
    isempty(never) || println("  !! front never crossed thr=$thr at site(s) " *
                              "$(join(never, ", ")) -- C left UNCORRECTED there")

    ed = find_ed(r.meta)
    if ed === nothing
        println("  (no ED at this physics point -- correction applied but not scored)")
        push!(results, (r = r, F_c = F_c, C_c = C_c, F_m = F_m, C_m = C_m,
                        Θ = Θ, tj = tj, ed = nothing))
        continue
    end
    println("  ED reference: ", basename(ed.meta.path))

    dist = dists_from(r.meta.site)
    println("  " * "-"^100)
    @printf("  %5s %5s | %-26s | %-26s\n", "site", "dist",
            "max|F - ED|  raw/front/mult", "max|C - ED|  raw/front/mult")
    println("  " * "-"^100)
    for j in probe_sites
        cj = r.col(j); ej = ed.col(j)
        (cj === nothing || ej === nothing) && continue
        @printf("  %5d %5d | %7.4f %8.4f %8.4f | %7.4f %8.4f %8.4f\n", j, dist[j],
                maxdev(r.F[:, cj], ed.F[:, ej]), maxdev(F_c[:, cj], ed.F[:, ej]),
                maxdev(F_m[:, cj], ed.F[:, ej]),
                maxdev(r.Csq[:, cj], ed.Csq[:, ej]), maxdev(C_c[:, cj], ed.Csq[:, ej]),
                maxdev(C_m[:, cj], ed.Csq[:, ej]))
    end
    println("  " * "-"^100)
    # Aggregate over the probe set, which is the number that decides whether the
    # scheme is worth using.
    agg(x, y) = maximum(maxdev(x[:, r.col(j)], y[:, ed.col(j)]) for j in probe_sites
                        if r.col(j) !== nothing && ed.col(j) !== nothing)
    @printf("  WORST over probes   F: raw %.4f -> front %.4f (mult %.4f)\n",
            agg(r.F, ed.F), agg(F_c, ed.F), agg(F_m, ed.F))
    @printf("                      C: raw %.4f -> front %.4f (mult %.4f)\n",
            agg(r.Csq, ed.Csq), agg(C_c, ed.Csq), agg(C_m, ed.Csq))

    push!(results, (r = r, F_c = F_c, C_c = C_c, F_m = F_m, C_m = C_m,
                    Θ = Θ, tj = tj, ed = ed, dist = dist))
end

# FIGURES =====================================================================
# One figure per run: rows = quantity (F, then C), columns = probe site. Raw and
# corrected are drawn against ED so the correction can be judged where it acts
# rather than through a single summary number.
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

for res in results
    r = res.r
    sites = [j for j in probe_sites if r.col(j) !== nothing]
    ncol = length(sites)
    panels = Plots.Plot[]
    for (row, (lbl, raw, cor, mul, edq, rule)) in enumerate((
            ("F", r.F, res.F_c, res.F_m, res.ed === nothing ? nothing : res.ed.F, 0.0),
            ("C", r.Csq, res.C_c, res.C_m, res.ed === nothing ? nothing : res.ed.Csq, 2.0)))
        for (ci, j) in enumerate(sites)
            cj = r.col(j)
            p = plot(; base_attrs()...,
                     title = row == 1 ? "site $j  (d=$(res.dist[j]))" : "",
                     titlefontsize = 11, tickfontsize = 9,
                     xlabel = row == 2 ? "time  t" : "",
                     ylabel = ci == 1 ? (lbl == "F" ? L"F_{ij}(t)" : L"C_{ij}(t)") : "",
                     guidefontsize = 10,
                     legend = (row == 1 && ci == 1) ? :best : false, legendfontsize = 6,
                     # ED usually runs to a longer t_max than the PP run. Showing
                     # its full window stretches the axis and shrinks the region
                     # where the two can actually be compared.
                     xlims = (0, r.meta.tmax),
                     grid = true)
            hline!(p, [rule]; color = INK_MUTED, lw = 1, ls = :dot, label = "")
            if edq !== nothing
                ej = res.ed.col(j)
                ej === nothing || plot!(p, res.ed.time, edq[:, ej];
                                        color = INK_PRIMARY, lw = 2.4, label = "ED")
            end
            # DRAW ORDER MATTERS HERE. On the F row the corrected curve is
            # almost exactly the raw one: the front arrives while delta is still
            # ~1e-4, and BEHIND the front F is untouched by construction, so
            # max|F_corr - F_raw| is ~2e-4 at worst. Drawing the correction last
            # therefore painted over the raw curve completely and made it look
            # like the data was missing. Raw now goes on TOP, thin, so the
            # overlap reads as "the correction left F alone" -- which is the
            # whole point of the scheme -- instead of as an absence.
            plot!(p, r.time, cor[:, cj]; color = series_color(1), lw = 3.4,
                  label = "front-corrected")
            # The raw/norm2 rescaling is still computed and still scored in the
            # table above -- it is the baseline the front-aware scheme has to
            # beat -- but it is kept OFF the figure so the panels carry only the
            # before/after pair. Set show_multiplicative = true to draw it.
            show_multiplicative && plot!(p, r.time, mul[:, cj];
                  color = series_color(3), lw = 1.8, ls = :dashdot,
                  label = "raw / norm2  (rescaled)")
            plot!(p, r.time, raw[:, cj]; color = series_color(2), lw = 1.6,
                  ls = :dot, label = "raw PP")
            # Where the front was declared: the correction is discontinuous
            # there, so mark it rather than let the kink look like physics.
            k = res.tj[cj]
            k <= length(r.time) && vline!(p, [r.time[k]];
                                          color = INK_MUTED, lw = 0.8, ls = :dot, label = "")
            push!(panels, p)
        end
    end
    # The title band needs real height: with two lines and per-column panel
    # titles on the top row, too little top margin draws the second line
    # straight through the column headings.
    fig = plot(panels...; layout = (2, ncol),
               size = (max(360 * ncol, 900), 820), dpi = 200,
               plot_title = "Front-aware truncation correction - $(r.meta.trunc)\n" *
                   "N=16 J=$(r.meta.J)  site $(r.meta.site)  " *
                   "thr=$thr ",
               plot_titlefontsize = 11, top_margin = 15Plots.mm,
               left_margin = 10Plots.mm, bottom_margin = 8Plots.mm, right_margin = 4Plots.mm)
    out = joinpath(plots_dir(),
        "otoc_frontcorr_$(r.meta.trunc)_N16_$(r.meta.lat)_V$(r.meta.V)_J$(r.meta.J)" *
        "_site$(r.meta.site)_thr$(thr).png")
    s = save_resilient(fig, out)
    s === nothing || println("\nSaved: $(basename(s))")
end
