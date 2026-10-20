# MEASURED p_j versus MODELLED p_j = 1/2: the two lost-weight corrections, side by side.
#
#     julia --project=. otoc_measured_vs_modelled.jl
#
# Both corrections add the same THING -- the discarded anticommuting weight,
# C_lost,j = 2 p_j l (note units) = 4 p_j l in this project's Csq units. They
# differ only in where p_j comes from:
#
#   Eq (14), MODELLED   p_j = 1/2 behind a fitted straight front, 0 ahead.
#                       Needs v_B, a threshold, and a sharp-front assumption the
#                       note itself says fails at the front. Adds 2*l behind.
#
#   DISCARDED, MEASURED p_j read off the strings themselves at the moment they
#                       were thrown away (discarded_kitaev_OTOC_infinite_temp.jl).
#                       No front, no velocity, no threshold.
#
# The measured version is exact AT THE INSTANT OF DISCARD, but it then FREEZES
# that population: discarded weight would have gone on evolving and scrambling,
# and a scrambled population tends to p_j -> 1/2. So the measured curve is
# expected to sit BELOW the modelled one and to fall further below as t grows.
# The right answer should lie between them, which is why both are drawn.
#
# NOTE ON GROUND TRUTH: at N = 32 there is no ED (a single state vector is 69 GB),
# so neither curve can be scored. The dotted rule at C = 2 is the value the
# squared commutator SETTLES at once the operator has spread -- an expectation
# from the physics, not a measurement of this system. Treat it as orientation,
# not as the exact answer.

include(joinpath(@__DIR__, "otoc_plotting.jl"))

# --- configuration -----------------------------------------------------------
# The plain run supplies the raw curve and the v_B correction; the _disc run
# supplies the measured one. They must be the SAME physics point -- the script
# checks that the raw arrays agree before comparing anything.
plain_file = "PP_kitaev_otoc_infT_N32_L48439d_coeff1e3_w32_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz"
disc_file  = "PP_kitaev_otoc_infT_N32_L48439d_coeff1e3_w32_disc_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz"

probe_site = nothing   # nothing -> the run's own W site (self-correlator)
thr_note   = 0.1       # the note's C_S threshold; doubled for Csq below
ell_fit_max = 0.05     # "obtain v_B while l ~ 0"
# -----------------------------------------------------------------------------

const DIR = joinpath(@__DIR__, "otoc_outputs")
const THR_CSQ = 2 * thr_note

pr = load_run(joinpath(DIR, plain_file))
dr = load_run(joinpath(DIR, disc_file))
dd = npzread(joinpath(DIR, disc_file))

for k in ("Csq_corrected", "Csq_lost", "ell", "p_measured")
    haskey(dd, k) || error("$(basename(disc_file)) has no \"$k\" -- it was not produced by " *
                           "discarded_kitaev_OTOC_infinite_temp.jl")
end

site = probe_site === nothing ? pr.meta.site : probe_site
cp_, cd_ = pr.col(site), dr.col(site)
(cp_ === nothing || cd_ === nothing) && error("site $site is not a column of these runs")

# The two solvers differ ONLY in that one records what it discards. If their raw
# curves disagree, the recording has perturbed the dynamics and nothing below is
# a like-for-like comparison.
dev = maximum(abs.(pr.Csq .- dr.Csq))
dev < 1e-9 || error("raw Csq differs between the plain and _disc runs by $dev -- " *
                    "they are not the same trajectory")

δ = 1 .- pr.norm2

# --- Eq (14): fitted v_B, p_j = 1/2 behind a straight front -------------------
dist = lattice_distances(pr.meta.lat, pr.meta.site)
fv   = fit_front_velocity(pr.time, pr.Csq, pr.norm2, dist; thr = THR_CSQ, ell_max = ell_fit_max)
isfinite(fv.vB) || error("could not fit v_B")
_, C_vb, Θ = vb_front_correct(pr.time, pr.F, pr.Csq, pr.norm2, dist, fv.vB)

C_raw  = pr.Csq[:, cp_]
C_meas = dd["Csq_corrected"][:, cd_]
p_meas = dd["p_measured"][:, cd_]
ell    = dd["ell"]

# --- report -------------------------------------------------------------------
println("\n", "="^92)
println("MEASURED vs MODELLED p_j   --   ", basename(plain_file))
println("="^92)
@printf("  N=%d  lattice %s  V=%s  J=%s  W@%d  probe %d (d=%d)\n",
        pr.meta.N, pr.meta.lat, pr.meta.V, pr.meta.J, pr.meta.site, site, dist[site])
@printf("  raw Csq agrees between the two solvers to %.2e  (recording is non-invasive)\n", dev)
@printf("  lost-weight tally closes to %.2e  (measured l vs 1 - norm2)\n",
        abs(ell[end] - (1 - pr.norm2[end])))
@printf("  v_B = %.4f bonds/time from %d crossings (RMS %.3f bonds)\n", fv.vB, fv.used, fv.resid)
println("-"^92)
@printf("%6s %9s %11s %12s %12s %12s\n", "t", "l(t)", "p_measured", "raw", "Eq(14) p=1/2", "measured p")
println("-"^92)
for tt in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
    k = findmin(abs.(pr.time .- tt))[2]
    @printf("%6.2f %9.4f %11.4f %12.4f %12.4f %12.4f\n",
            pr.time[k], ell[k], p_meas[k], C_raw[k], C_vb[k, cp_], C_meas[k])
end
println("-"^92)
println("Eq(14) adds 2*l behind the front (p = 1/2); the measured version adds 4*p_meas*l.")
println("Their ratio is therefore p_measured/(1/2) wherever both are active.")

# --- figure -------------------------------------------------------------------
# Left: the three C(t) curves. Right: the quantity that explains the gap between
# them -- the measured p_j against the 1/2 the note assumes.
p1 = plot(; base_attrs()..., xlabel = "time  t", ylabel = L"C_{ij}(t)",
          title = "squared commutator", titlefontsize = 12,
          guidefontsize = 11, tickfontsize = 10, legend = :best, legendfontsize = 8,
          xlims = (0, pr.meta.tmax), grid = true)
hline!(p1, [2.0]; color = INK_MUTED, lw = 1, ls = :dot, label = "C = 2 (saturation)")
plot!(p1, pr.time, C_vb[:, cp_]; color = series_color(1), lw = 3.0,
      label = "Eq(14): modelled p = 1/2")
plot!(p1, pr.time, C_meas; color = series_color(3), lw = 2.4, ls = :dash,
      label = "measured p (discarded)")
plot!(p1, pr.time, C_raw; color = series_color(2), lw = 1.8, ls = :dot,
      label = "raw (no correction)")

p2 = plot(; base_attrs()..., xlabel = "time  t", ylabel = L"p_j(t)",
          title = "the quantity the two disagree on", titlefontsize = 12,
          guidefontsize = 11, tickfontsize = 10, legend = :best, legendfontsize = 8,
          xlims = (0, pr.meta.tmax), ylims = (0, 0.75), grid = true)
hline!(p2, [0.5]; color = series_color(1), lw = 2, ls = :dot,
       label = "modelled: p = 1/2")
plot!(p2, pr.time, p_meas; color = series_color(3), lw = 2.4,
      label = "measured at discard")
# l(t) on the same axes, rescaled, to show WHEN the correction has any size: a
# large p gap costs nothing while l is still ~0.
plot!(p2, pr.time, ell .* 0.75; color = INK_MUTED, lw = 1.5, ls = :dashdot,
      label = "lost weight l(t)  (x0.75)")

fig = plot(p1, p2; layout = (1, 2), size = (1300, 560), dpi = 200,
           plot_title = "Lost-weight correction: measured vs modelled p | " *
                        "N=$(pr.meta.N) $(pr.meta.lat) J=$(pr.meta.J) " *
                        "$(pr.meta.trunc) $(pr.meta.pauli) self-correlator",
           plot_titlefontsize = 12, top_margin = 7Plots.mm,
           left_margin = 11Plots.mm, bottom_margin = 9Plots.mm, right_margin = 5Plots.mm)

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
    "otoc_pmeasured_vs_modelled_N$(pr.meta.N)_$(pr.meta.lat)_$(pr.meta.trunc)" *
    "_V$(pr.meta.V)_J$(pr.meta.J)_site$(site).png")
s = save_resilient(fig, out)
s === nothing || println("\nSaved: $(basename(s))")
