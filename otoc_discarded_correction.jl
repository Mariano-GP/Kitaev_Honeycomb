# The DISCARDED-weight correction on its own -- no v_B, no front, no threshold.
#
#     julia --project=. otoc_discarded_correction.jl
#
# Same layout as otoc_vb_front_correction.jl, with the modelling removed. That
# script fits a butterfly velocity, assumes p_j = 1/2 behind a straight front,
# and adds 2*l. This one adds the discarded anticommuting weight that
# discarded_kitaev_OTOC_infinite_temp.jl actually recorded:
#
#     Csq_corrected = Csq + Csq_lost ,
#     Csq_lost[j]   = 4 * sum over strings discarded up to time t that
#                     anticommute with V_j.
#
# Nothing is fitted and nothing is assumed about where the front is -- so the
# first panel is not a v_B fit but the quantity this correction rests on
# instead: the measured p_j, drawn against the 1/2 the note models.
#
# WHAT IS PLOTTED PER SITE
#   raw                 the truncated run, uncorrected
#   discarded-corrected raw + Csq_lost, straight from the solver's payload
#   ED                  exact reference, when one exists at this physics point
#
# F IS DERIVED, NOT STORED. Either correction forces norm2 = 1, so
#     Csq_corrected = 2 (1 - F_corrected)   ->   F_corrected = 1 - Csq_corrected/2
# which is how the F row is built. The solver saves only the C side.
#
# REQUIRES runs from discarded_kitaev_OTOC_infinite_temp.jl (filenames carry
# "_disc"). A plain run has no Csq_lost and is refused rather than silently
# plotted uncorrected.

include(joinpath(@__DIR__, "otoc_plotting.jl"))

# --- configuration -----------------------------------------------------------
run_files = [
 "PP_kitaev_otoc_infT_N16_L24e521_coeff1e4_w32_disc_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
 "PP_kitaev_otoc_infT_N16_L24e521_coeff5e5_w32_disc_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
]

# nothing -> one site per distance shell, computed from the run's own lattice.
probe_sites = nothing
quantity    = :C       # :C (squared commutator) or :F (the OTOC)
flip_F      = true     # invert the F axis so it rises like C; axis only, not data
# -----------------------------------------------------------------------------

const DIR = joinpath(@__DIR__, "otoc_outputs")

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

yflip_on = (quantity === :F) && flip_F

for f in run_files
    path = joinpath(DIR, f)
    isfile(path) || error("not in otoc_outputs/: $f")
    r = load_run(path)
    d = npzread(path)

    haskey(d, "Csq_corrected") || error(
        "$(basename(path)) has no \"Csq_lost\"/\"Csq_corrected\". Only runs from " *
        "discarded_kitaev_OTOC_infinite_temp.jl carry the recorded discard tally; " *
        "a plain run cannot be corrected this way.")

    δ    = 1 .- r.norm2
    ell  = d["ell"]
    pm   = d["p_measured"]
    Ccor = d["Csq_corrected"]
    # Both schemes force norm2 = 1, so F follows from C and need not be stored.
    Fcor = 1 .- Ccor ./ 2
    dist = lattice_distances(r.meta.lat, r.meta.site)
    sites = probe_sites === nothing ?
        [findfirst(==(s), dist) for s in 0:maximum(dist) if findfirst(==(s), dist) !== nothing] :
        probe_sites

    raw_q = quantity === :F ? r.F   : r.Csq
    cor_q = quantity === :F ? Fcor  : Ccor
    rule  = quantity === :F ? 0.0   : 2.0
    ylab  = quantity === :F ? L"F_{ij}(t)" : L"C_{ij}(t)"

    println("\n", "="^96)
    println(basename(path))
    println("="^96)
    @printf("  N=%d  %s  V=%s  J=%s  W@%d   l: 0 -> %.5f   p_measured(end) = %.4f\n",
            r.meta.N, r.meta.lat, r.meta.V, r.meta.J, r.meta.site, ell[end], pm[end, r.col(r.meta.site)])
    # The tally must account for every discard; the gap is whatever the in-place
    # 1e-12 clip removed without reporting it.
    @printf("  tally check: measured l = %.6f vs 1 - norm2 = %.6f   gap = %.2e\n",
            ell[end], 1 - r.norm2[end], abs(ell[end] - (1 - r.norm2[end])))

    ed = find_ed(r.meta)
    if ed === nothing
        println("  (no ED at this physics point -- corrected but not scored)")
    else
        ed_q = quantity === :F ? ed.F : ed.Csq
        println("  ED reference: ", basename(ed.meta.path))
        println("  " * "-"^58)
        @printf("  %6s %6s | %12s %12s\n", "site", "dist", "raw", "discarded")
        println("  " * "-"^58)
        for j in sites
            cj = r.col(j); ej = ed.col(j)
            (cj === nothing || ej === nothing) && continue
            @printf("  %6d %6d | %12.4f %12.4f\n", j, dist[j],
                    maxdev(raw_q[:, cj], ed_q[:, ej]), maxdev(cor_q[:, cj], ed_q[:, ej]))
        end
        println("  " * "-"^58)
    end

    # ---- figure -------------------------------------------------------------
    # Panel 1 takes the place of the v_B fit in otoc_vb_front_correction.jl.
    # That correction stands or falls on the front being a straight line; this
    # one stands or falls on p_measured, so that is what belongs in the figure.
    p1 = plot(; base_attrs()..., title = "measured p_j (model assumes 1/2)",
              titlefontsize = 11, xlabel = "time  t", ylabel = L"p_j(t)",
              guidefontsize = 10, tickfontsize = 9, legend = :best, legendfontsize = 7,
              xlims = (0, r.meta.tmax), ylims = (0, 1), grid = true)
    hline!(p1, [0.5]; color = series_color(1), lw = 2, ls = :dot, label = "modelled 1/2")
    plot!(p1, r.time, pm[:, r.col(r.meta.site)]; color = series_color(3), lw = 2.4,
          label = "measured at discard")
    # l on the same axes: a p gap costs nothing while almost no weight is gone.
    plot!(p1, r.time, ell; color = INK_MUTED, lw = 1.5, ls = :dashdot, label = "lost weight l(t)")

    panels = Any[p1]
    for j in sites
        cj = r.col(j)
        cj === nothing && continue
        p = plot(; base_attrs()..., title = "site $j  (d=$(dist[j]))", titlefontsize = 11,
                 xlabel = "time  t", ylabel = j == first(sites) ? ylab : "",
                 guidefontsize = 10, tickfontsize = 9,
                 legend = j == first(sites) ? :best : false, legendfontsize = 6,
                 xlims = (0, r.meta.tmax), yflip = yflip_on, grid = true)
        hline!(p, [rule]; color = INK_MUTED, lw = 1, ls = :dot, label = "")
        if ed !== nothing && ed.col(j) !== nothing
            plot!(p, ed.time, (quantity === :F ? ed.F : ed.Csq)[:, ed.col(j)];
                  color = INK_PRIMARY, lw = 2.4, label = "ED")
        end
        plot!(p, r.time, cor_q[:, cj]; color = series_color(3), lw = 3.0,
              label = "discarded-corrected")
        plot!(p, r.time, raw_q[:, cj]; color = series_color(2), lw = 1.6,
              ls = :dot, label = "raw PP")
        push!(panels, p)
    end

    # Wrap into a grid rather than one row: with a full distance ladder this is
    # 7+ panels, and a single row renders 5000+ px wide and unreadably short.
    ncol = min(4, length(panels))
    nrow = ceil(Int, length(panels) / ncol)
    fig = plot(panels...; layout = (nrow, ncol),
               size = (400 * ncol, 400 * nrow + 60), dpi = 200,
               plot_title = "Discarded-weight correction (no v_B, no front) - " *
                   "$(r.meta.trunc)   N=$(r.meta.N) $(r.meta.lat) J=$(r.meta.J) W@$(r.meta.site)" *
                   (yflip_on ? "\ny axis INVERTED: F falls upward" : ""),
               plot_titlefontsize = 11, top_margin = 10Plots.mm,
               left_margin = 10Plots.mm, bottom_margin = 9Plots.mm, right_margin = 4Plots.mm)
    out = joinpath(plots_dir(),
        "otoc_disccorr_$(quantity)_$(r.meta.trunc)_N$(r.meta.N)_$(r.meta.lat)" *
        "_V$(r.meta.V)_J$(r.meta.J)_site$(r.meta.site).png")
    s = save_resilient(fig, out)
    s === nothing || println("\n  Saved: $(basename(s))")
end
