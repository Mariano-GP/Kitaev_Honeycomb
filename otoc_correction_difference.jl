# How far apart are the two lost-weight corrections, in C and in F?
#
#     julia --project=. otoc_correction_difference.jl
#
# Both corrections add the discarded anticommuting weight to C. They differ only
# in where that weight comes from:
#
#   Eq (14), v_B   modelled: p_j = 1/2 behind a fitted straight front, 0 ahead,
#                  so it adds 2*l*Theta.
#   discarded      measured: the anticommuting weight actually thrown away,
#                  recorded at each truncation, so it adds Csq_lost = 4*p_meas*l.
#
# Plotted here is the DIFFERENCE between them,
#     dC(t) = C_vb - C_discarded ,      dF(t) = F_vb - F_discarded
# for each truncation threshold.
#
# dF IS NOT INDEPENDENT OF dC. Both schemes force norm2 = 1, so each obeys
#     Csq_corrected = 2 (1 - F_corrected)   ->   F_corrected = 1 - Csq_corrected/2
# and therefore
#     dF = -dC/2
# EXACTLY, at every time and every site. The F panel is drawn because it was
# asked for and because it sets the scale in F's own units, but it carries no
# information the C panel does not. The script asserts the identity rather than
# asserting it in a comment.
#
# A single run file is enough per threshold: a _disc run stores the raw Csq as
# well as the correction, and its raw curve is identical to the plain solver's
# (verified elsewhere to 1e-14), so the v_B correction is computed from the same
# file it is being compared against.

include(joinpath(@__DIR__, "otoc_plotting.jl"))

# --- configuration -----------------------------------------------------------
# One _disc run per truncation threshold, all at the SAME physics point.
run_files = [
 "PP_kitaev_otoc_infT_N16_L24e521_coeff1e4_w32_disc_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
 "PP_kitaev_otoc_infT_N16_L24e521_coeff5e5_w32_disc_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
]

probe_site  = nothing   # nothing -> each run's own W site (the self-correlator)
thr_note    = 0.1       # the note's C_S threshold; doubled for Csq
ell_fit_max = 0.05      # "obtain v_B while l ~ 0"
# -----------------------------------------------------------------------------

const DIR = joinpath(@__DIR__, "otoc_outputs")
const THR_CSQ = 2 * thr_note

rows = NamedTuple[]
for f in run_files
    path = joinpath(DIR, f)
    isfile(path) || error("not in otoc_outputs/: $f")
    r = load_run(path)
    d = npzread(path)
    haskey(d, "Csq_corrected") || error("$(basename(path)) carries no Csq_lost; " *
        "it is not from discarded_kitaev_OTOC_infinite_temp.jl")

    site = probe_site === nothing ? r.meta.site : probe_site
    c = r.col(site); c === nothing && error("site $site absent from $(basename(path))")

    dist = lattice_distances(r.meta.lat, r.meta.site)
    fv = fit_front_velocity(r.time, r.Csq, r.norm2, dist; thr = THR_CSQ, ell_max = ell_fit_max)
    isfinite(fv.vB) || error("$(r.meta.trunc): could not fit v_B")
    _, C_vb, _ = vb_front_correct(r.time, r.F, r.Csq, r.norm2, dist, fv.vB)

    C_di = d["Csq_corrected"]
    # F from C, the same way for both schemes (see header).
    F_vb = 1 .- C_vb ./ 2
    F_di = 1 .- C_di ./ 2

    dC = C_vb[:, c] .- C_di[:, c]
    dF = F_vb[:, c] .- F_di[:, c]
    # The identity dF = -dC/2 holds by construction; check it instead of trusting it.
    resid = maximum(abs.(dF .+ dC ./ 2))
    resid < 1e-12 || error("dF != -dC/2 (residual $resid) -- the two corrections are " *
                           "no longer both forcing norm2 = 1")

    push!(rows, (meta = r.meta, time = r.time, site = site, dC = dC, dF = dF,
                 ell = d["ell"], pm = d["p_measured"][:, c], vB = fv.vB,
                 thresh = d["trunc"][1], resid = resid))
end

# ---- report ------------------------------------------------------------------
println("\n", "="^92)
println("DIFFERENCE BETWEEN THE TWO LOST-WEIGHT CORRECTIONS  (v_B minus discarded)")
println("="^92)
@printf("  physics point: N=%d  %s  V=%s  J=%s  W@%d  %s self-correlator\n",
        rows[1].meta.N, rows[1].meta.lat, rows[1].meta.V, rows[1].meta.J,
        rows[1].meta.site, rows[1].meta.pauli)
println("  identity dF = -dC/2 holds to ",
        @sprintf("%.1e", maximum(r.resid for r in rows)), " on every run")
println("-"^92)
@printf("%-16s %8s %8s %9s %10s %10s %10s\n",
        "truncation", "v_B", "l(end)", "p_meas(e)", "dC(end)", "dF(end)", "max|dC|")
println("-"^92)
for r in rows
    @printf("%-16s %8.3f %8.4f %9.4f %10.4f %10.4f %10.4f\n",
            r.meta.trunc, r.vB, r.ell[end], r.pm[end], r.dC[end], r.dF[end],
            maximum(abs.(r.dC)))
end
println("-"^92)
println("dC > 0 means the v_B correction adds MORE than the measured one, i.e.")
println("p_measured < 1/2. dC = 4*l*(1/2 - p_measured) wherever the front has passed.")

# ---- figure -------------------------------------------------------------------
tmax = maximum(r.meta.tmax for r in rows)
col(i) = series_color(i)

pC = plot(; base_attrs()..., xlabel = "time  t", ylabel = L"\Delta C_{ij}(t)",
          title = "C: v_B correction minus discarded", titlefontsize = 12,
          guidefontsize = 11, tickfontsize = 10, legend = :topleft, legendfontsize = 8,
          xlims = (0, tmax), grid = true)
hline!(pC, [0.0]; color = INK_MUTED, lw = 1, ls = :dot, label = "")
for (i, r) in enumerate(rows)
    plot!(pC, r.time, r.dC; color = col(i), lw = 2.4,
          label = "$(r.meta.trunc)  (l_end = $(round(r.ell[end], digits=2)))")
end

pF = plot(; base_attrs()..., xlabel = "time  t", ylabel = L"\Delta F_{ij}(t)",
          title = "F: same difference, = -dC/2 exactly", titlefontsize = 12,
          guidefontsize = 11, tickfontsize = 10, legend = false,
          xlims = (0, tmax), grid = true)
hline!(pF, [0.0]; color = INK_MUTED, lw = 1, ls = :dot, label = "")
for (i, r) in enumerate(rows)
    plot!(pF, r.time, r.dF; color = col(i), lw = 2.4, label = "")
end

# The driver of both panels: how far the measured p_j sits from the modelled 1/2.
pP = plot(; base_attrs()..., xlabel = "time  t", ylabel = L"p_j(t)",
          title = "why they differ: measured p vs 1/2", titlefontsize = 12,
          guidefontsize = 11, tickfontsize = 10, legend = :best, legendfontsize = 8,
          xlims = (0, tmax), ylims = (0, 1), grid = true)
hline!(pP, [0.5]; color = INK_PRIMARY, lw = 2, ls = :dot, label = "modelled 1/2")
for (i, r) in enumerate(rows)
    plot!(pP, r.time, r.pm; color = col(i), lw = 2.4, label = "$(r.meta.trunc) measured")
end

fig = plot(pC, pF, pP; layout = (1, 3), size = (1550, 520), dpi = 200,
           plot_title = "Lost-weight corrections compared: v_B (modelled p=1/2) minus " *
                        "discarded (measured p) | N=$(rows[1].meta.N) " *
                        "$(rows[1].meta.lat) J=$(rows[1].meta.J) $(rows[1].meta.pauli) self-correlator",
           plot_titlefontsize = 11, top_margin = 7Plots.mm,
           left_margin = 11Plots.mm, bottom_margin = 10Plots.mm, right_margin = 5Plots.mm)

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

out = joinpath(plots_dir(),
    "otoc_corrdiff_N$(rows[1].meta.N)_$(rows[1].meta.lat)_V$(rows[1].meta.V)" *
    "_J$(rows[1].meta.J)_site$(rows[1].site).png")
s = save_resilient(fig, out)
s === nothing || println("\nSaved: $(basename(s))")
