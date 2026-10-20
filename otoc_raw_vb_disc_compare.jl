# Raw vs v_B-corrected vs discarded-corrected, for several truncation thresholds.
#
#     julia --project=. otoc_raw_vb_disc_compare.jl
#
# Three treatments of the SAME runs, on one axis:
#
#   raw         the truncated run, uncorrected -- C sags as weight is discarded
#   v_B         Eq (14): p_j modelled as 1/2 behind a fitted straight front,
#               so it adds 2*l*Theta
#   discarded   the anticommuting weight actually recorded at each truncation
#               by discarded_kitaev_OTOC_infinite_temp.jl, so it adds
#               Csq_lost = 4*p_measured*l. No fit, no front, no threshold.
#
# ENCODING: colour is the THRESHOLD, linestyle is the TREATMENT. A reader can
# then follow one threshold across all three treatments, or compare treatments
# within one threshold, without a twelve-entry legend.
#
# F IS DERIVED FOR THE CORRECTED CURVES. Both corrections force norm2 = 1, so
#     Csq_corrected = 2 (1 - F_corrected)  ->  F_corrected = 1 - Csq_corrected/2.
# The RAW F is read from the file instead: with norm2 < 1 the raw pair obeys
# Csq = 2(norm2 - F), NOT 2(1 - F), so deriving it the same way would be wrong.
#
# One file per threshold is enough: a _disc run stores the raw arrays too, and
# its raw curve is identical to the plain solver's (checked to 1e-14), so the
# v_B correction is built from the same file it is compared against.

include(joinpath(@__DIR__, "otoc_plotting.jl"))

# --- configuration -----------------------------------------------------------
run_files = [
 "PP_kitaev_otoc_infT_N16_L24e521_coeff1e4_w32_disc_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
 "PP_kitaev_otoc_infT_N16_L24e521_coeff5e5_w32_disc_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
]

quantity    = :C        # :C or :F  -- run twice to get both figures
probe_site  = nothing   # nothing -> each run's own W site (self-correlator)
flip_F      = true      # invert the F axis so it rises like C (axis, not data)
thr_note    = 0.1
ell_fit_max = 0.05
show_ed     = true
# -----------------------------------------------------------------------------

const DIR = joinpath(@__DIR__, "otoc_outputs")
const THR_CSQ = 2 * thr_note
yflip_on = (quantity === :F) && flip_F

function find_ed(m::RunMeta)
    isdir(DIR) || return nothing
    pat = "ED_kitaev_otoc_infT_N$(m.N)_$(m.lat)_exact_V$(m.V)_J$(m.J)_" *
          "$(m.pauli)_site$(m.site)_dt$(m.dt)_"
    hits = filter(f -> startswith(f, pat) && endswith(f, ".npz"), readdir(DIR))
    isempty(hits) ? nothing : load_run(joinpath(DIR, sort(hits)[end]))
end
maxdev(a, b) = (n = min(length(a), length(b)); maximum(abs.(a[1:n] .- b[1:n])))

rows = NamedTuple[]
for f in run_files
    path = joinpath(DIR, f)
    isfile(path) || error("not in otoc_outputs/: $f")
    r = load_run(path); d = npzread(path)
    haskey(d, "Csq_corrected") || error("$(basename(path)) carries no Csq_lost; " *
        "it is not from discarded_kitaev_OTOC_infinite_temp.jl")
    site = probe_site === nothing ? r.meta.site : probe_site
    c = r.col(site); c === nothing && error("site $site absent from $(basename(path))")

    dist = lattice_distances(r.meta.lat, r.meta.site)
    fv = fit_front_velocity(r.time, r.Csq, r.norm2, dist; thr = THR_CSQ, ell_max = ell_fit_max)
    isfinite(fv.vB) || error("$(r.meta.trunc): could not fit v_B")
    _, C_vb, _ = vb_front_correct(r.time, r.F, r.Csq, r.norm2, dist, fv.vB)
    C_di = d["Csq_corrected"]

    push!(rows, (meta = r.meta, time = r.time, site = site, col = c,
                 raw = quantity === :F ? r.F[:, c]      : r.Csq[:, c],
                 vb  = quantity === :F ? 1 .- C_vb[:, c] ./ 2 : C_vb[:, c],
                 di  = quantity === :F ? 1 .- C_di[:, c] ./ 2 : C_di[:, c],
                 ell = d["ell"], pm = d["p_measured"][:, c], vB = fv.vB))
end

tmax_runs = maximum(r.meta.tmax for r in rows)
ed    = show_ed ? find_ed(rows[1].meta) : nothing
ed_q  = ed === nothing ? nothing : (quantity === :F ? ed.F : ed.Csq)
ed_c  = ed === nothing ? nothing : ed.col(rows[1].site)

# ---- report ------------------------------------------------------------------
m0 = rows[1].meta
println("\n", "="^88)
println("RAW vs v_B vs DISCARDED   --   $(quantity)")
println("="^88)
@printf("  N=%d  %s  V=%s  J=%s  W@%d  probe %d   v_B = %.3f\n",
        m0.N, m0.lat, m0.V, m0.J, m0.site, rows[1].site, rows[1].vB)
if ed === nothing
    println("  no ED at this physics point -- curves are shown but not scored")
else
    println("  ED: ", basename(ed.meta.path),
            "  (window t <= ", ed.time[end], ")")
end
println("-"^88)
@printf("%-16s %8s %9s %10s %10s %10s\n",
        "truncation", "l(end)", "p_meas(e)", "raw(end)", "v_B(end)", "disc(end)")
println("-"^88)
for r in rows
    @printf("%-16s %8.4f %9.4f %10.4f %10.4f %10.4f\n",
            r.meta.trunc, r.ell[end], r.pm[end], r.raw[end], r.vb[end], r.di[end])
end
println("-"^88)
if ed !== nothing
    @printf("%-16s %10s %10s %10s   (max |%s - ED| over the ED window)\n",
            "truncation", "raw", "v_B", "disc", string(quantity))
    for r in rows
        @printf("%-16s %10.4f %10.4f %10.4f\n", r.meta.trunc,
                maxdev(r.raw, ed_q[:, ed_c]), maxdev(r.vb, ed_q[:, ed_c]),
                maxdev(r.di, ed_q[:, ed_c]))
    end
    println("-"^88)
    # Only warn when the reference is SHORTER than the runs being scored -- with
    # a full-window ED the scores are complete and the caveat would be false.
    if ed.time[end] < tmax_runs - 1e-9
        @printf("NOTE: the ED window stops at t = %.2f but the runs reach %.2f, so these
", ed.time[end], tmax_runs)
        println("scores cover only the early window and miss where the corrections diverge.")
    else
        println("ED covers the full run window, so these scores are complete.")
    end
end

# ---- figure -------------------------------------------------------------------
rule  = quantity === :F ? 0.0 : 2.0
ylab  = quantity === :F ? L"F_{ij}(t)" : L"C_{ij}(t)"
qname = quantity === :F ? "Infinite temperature OTOC F(t)" : "Squared commutator C(t)"
tmax  = maximum(r.meta.tmax for r in rows)

p = plot(; base_attrs()..., size = (1150, 680), dpi = 200,
         xlabel = "time  t", ylabel = ylab, guidefontsize = 12, tickfontsize = 11,
         legend = :best, legendfontsize = 8, grid = true,
         xlims = (0, tmax), yflip = yflip_on,
         title = "$qname | N=$(m0.N) Kitaev-Heisenberg PBC Lattice at J=$(m0.J), " *
                 "$(m0.pauli) self-correlator\nraw vs v_B-corrected vs discarded-corrected" *
                 (yflip_on ? "   |   y axis INVERTED: F falls upward" : ""),
         titlefontsize = 12,
         left_margin = 11Plots.mm, bottom_margin = 9Plots.mm,
         top_margin = 6Plots.mm, right_margin = 5Plots.mm)
hline!(p, [rule]; color = INK_MUTED, lw = 1, ls = :dot, label = "")

# ED first so it lies under the approximations rather than painting over them.
ed === nothing || plot!(p, ed.time, ed_q[:, ed_c]; color = INK_PRIMARY, lw = 3.0,
                        label = "ED (exact, t <= $(ed.time[end]))")

for (i, r) in enumerate(rows)
    cc = series_color(i)
    plot!(p, r.time, r.vb;  color = cc, lw = 2.8, ls = :solid,
          label = "$(r.meta.trunc)  v_B corrected")
    plot!(p, r.time, r.di;  color = cc, lw = 2.2, ls = :dash,
          label = "$(r.meta.trunc)  discarded corrected")
    plot!(p, r.time, r.raw; color = cc, lw = 1.5, ls = :dot,
          label = "$(r.meta.trunc)  raw")
end

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
    "otoc_raw_vb_disc_$(quantity)_N$(m0.N)_$(m0.lat)_V$(m0.V)_J$(m0.J)_site$(rows[1].site).png")
s = save_resilient(p, out)
s === nothing || println("\nSaved: $(basename(s))")
