# One truncated run, three ways: raw, front-corrected, and Eq (14) with a fitted
# butterfly velocity -- overlaid against ED.
#
#     julia --project=. otoc_correction_overlay.jl
#
# The three curves differ ONLY in how the front Theta is located:
#
#   raw                 no correction at all; C sags as weight is discarded.
#   per-site front      Theta from each site's own threshold crossing
#                       (what otoc_front_correction.jl does).
#   Eq (14), fitted v_B the note's prescription: record every site's crossing,
#                       fit |j-j0| = v_B t_j through the origin, then use the
#                       straight-line front Theta(v_B t - |j-j0|).
#
# Both corrections add the same amount -- 2*l*Theta in this project's Csq units,
# which is the note's C += l*Theta since Csq = 2*C_note. They can therefore only
# differ where their two Thetas disagree, i.e. in the window between the two
# declared front times. That window is narrow here and sits where l is still
# ~1e-4, so expect the two to nearly coincide; the printed gap says by how much.
#
# Self-contained on purpose: the lattice table and the v_B fit are repeated from
# otoc_vb_front_correction.jl rather than shared, so editing this figure cannot
# disturb that script. `front_correct` is the one piece taken from the shared
# helpers, so the per-site scheme here is literally the same code as everywhere
# else.

include(joinpath(@__DIR__, "otoc_plotting.jl"))

# --- configuration -----------------------------------------------------------
run_file = "PP_kitaev_otoc_infT_N16_L243535_coeff1e4_w16_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz"

quantity    = :C      # :C (squared commutator) or :F (the OTOC itself)
probe_site  = nothing # nothing -> the run's own W site (self-correlator); or e.g. 16
thr_note    = 0.1     # the note's threshold on C_S; doubled for Csq below
ell_fit_max = 0.05    # only use crossings taken while l <= this, per the note's
                      # "obtain v_B at early times, while l ~ 0"
# -----------------------------------------------------------------------------

const DIR = joinpath(@__DIR__, "otoc_outputs")
const THR_CSQ = 2 * thr_note          # the single note -> project conversion

# Bond distances come from the shared KNOWN_LATTICES table in otoc_plotting.jl,
# keyed by the run's OWN lattice fingerprint. Previously each of these scripts
# carried its own hardcoded copy of the L243535 bonds, which meant a run on any
# other lattice (torus16's current L24e521, or the 32-site L48439d) was silently
# given the wrong shells. An unknown tag is now a hard error instead.
dists_from(origin::Int, lat::AbstractString) = lattice_distances(lat, origin)

# Note Step 3: least squares through the origin, v_B = sum(d t)/sum(t^2).
function fit_vB(time, Csq, norm2, dist; thr = THR_CSQ, ell_max = ell_fit_max)
    δ = 1 .- norm2
    pts = NamedTuple[]
    for j in 1:size(Csq, 2)
        dist[j] == 0 && continue
        k = findfirst(>(thr), Csq[:, j])
        k === nothing && continue
        δ[k] <= ell_max || continue
        push!(pts, (site = j, d = Float64(dist[j]), t = time[k]))
    end
    isempty(pts) && return (vB = NaN, n = 0, resid = NaN)
    vB = sum(p.d * p.t for p in pts) / sum(p.t^2 for p in pts)
    resid = sqrt(sum((p.d - vB * p.t)^2 for p in pts) / length(pts))
    return (vB = vB, n = length(pts), resid = resid)
end

# ---- load --------------------------------------------------------------------
path = joinpath(DIR, run_file)
isfile(path) || error("not in otoc_outputs/: $run_file")
r = load_run(path)
d = npzread(path)
((haskey(d, "normalized") && d["normalized"][1] == 1.0) || haskey(d, "cum_retention")) &&
    error("$(basename(path)) is renormalized: l == 0 and the note's step 1 says " *
          "\"Do not renormalize the kept coefficients\". Pick an un-normalized run.")

δ    = 1 .- r.norm2
dist = dists_from(r.meta.site, r.meta.lat)
site = probe_site === nothing ? r.meta.site : probe_site
cj   = r.col(site)
cj === nothing && error("site $site is not a column of this run")

# ---- the two corrections -----------------------------------------------------
F_site, C_site, _, tj = front_correct(r.F, r.Csq, r.norm2; thr = THR_CSQ)

fit = fit_vB(r.time, r.Csq, r.norm2, dist)
isfinite(fit.vB) || error("no usable crossings while l <= $ell_fit_max; cannot fit v_B")
Θ_vb  = [ (fit.vB * r.time[k] > dist[j]) ? 1.0 : 0.0
          for k in 1:length(r.time), j in 1:size(r.Csq, 2) ]
C_vb  = r.Csq .+ 2 .* δ .* Θ_vb          # == note's C += l*Theta, in Csq units
# F under the same p_j model: F_exact = F_S + (1-2p_j) l, so +l ahead of the
# front and nothing behind it. Same construction as front_correct, different
# Theta -- stated here rather than reused so the two Thetas stay independent.
F_vb  = r.F .+ δ .* (1 .- Θ_vb)

dat_raw  = quantity === :F ? r.F     : r.Csq
dat_site = quantity === :F ? F_site  : C_site
dat_vb   = quantity === :F ? F_vb    : C_vb

# ---- ED ----------------------------------------------------------------------
pat = "ED_kitaev_otoc_infT_N$(r.meta.N)_$(r.meta.lat)_exact_V$(r.meta.V)_J$(r.meta.J)_" *
      "$(r.meta.pauli)_site$(r.meta.site)_dt$(r.meta.dt)_"
hits = filter(f -> startswith(f, pat) && endswith(f, ".npz"), readdir(DIR))
ed = isempty(hits) ? nothing : load_run(joinpath(DIR, sort(hits)[end]))
ed_dat = ed === nothing ? nothing : (quantity === :F ? ed.F : ed.Csq)

# ---- report ------------------------------------------------------------------
maxdev(a, b) = (n = min(length(a), length(b)); maximum(abs.(a[1:n] .- b[1:n])))

println("\n", "="^88)
println(basename(path))
println("="^88)
@printf("  V=%s  J=%s  W@%d   probe site %d (d=%d)   l: 0 -> %.5f\n",
        r.meta.V, r.meta.J, r.meta.site, site, dist[site], δ[end])
@printf("  v_B = %.4f bonds/time   from %d crossing(s)   RMS residual %.4f bonds\n",
        fit.vB, fit.n, fit.resid)
@printf("  front at this site:  per-site t_j = %.2f    fitted d/v_B = %.2f\n",
        r.time[min(tj[cj], length(r.time))], dist[site] / fit.vB)

if ed !== nothing
    ej = ed.col(site)
    println("  ED reference: ", basename(ed.meta.path))
    println("  " * "-"^60)
    @printf("  %-28s %14s\n", "curve", "max |$(quantity) - ED|")
    println("  " * "-"^60)
    @printf("  %-28s %14.4f\n", "raw (no correction)",    maxdev(dat_raw[:, cj],  ed_dat[:, ej]))
    @printf("  %-28s %14.4f\n", "front-corrected",        maxdev(dat_site[:, cj], ed_dat[:, ej]))
    @printf("  %-28s %14.4f\n", "Eq(14), fitted v_B",     maxdev(dat_vb[:, cj],   ed_dat[:, ej]))
    println("  " * "-"^60)
end
@printf("  max|per-site - Eq(14)| at this site: %.3e   (over all sites: %.3e)\n",
        maxdev(dat_site[:, cj], dat_vb[:, cj]), maximum(abs.(dat_site .- dat_vb)))

# ---- figure ------------------------------------------------------------------
rule = quantity === :F ? 0.0 : 2.0
ylab = quantity === :F ? L"F_{ij}(t)" : L"C_{ij}(t)"

# F and C run in OPPOSITE directions: C starts at 0 and climbs to 2, F starts
# at 1 and falls to 0, though they carry the same information. Flipping F's
# axis puts both "more scrambled" directions upward so the two figures can be
# read against each other. The AXIS is reversed, not the data -- ticks still
# read 1.0 at the start and 0.0 at saturation. Set false for a conventional
# F axis. Same switch as in otoc_overlay_select.jl.
flip_F   = true
yflip_on = (quantity === :F) && flip_F
qlabel = quantity === :F ? "Infinite temperature OTOC F(t)" : "Squared commutator C(t)"
probe_txt = site == r.meta.site ? "$(r.meta.pauli) self-correlator" :
            "$(r.meta.pauli) probe j=$site"

p = plot(; base_attrs()..., size = (1000, 640), dpi = 200,
         xlabel = "time  t", ylabel = ylab, guidefontsize = 12, tickfontsize = 11,
         legend = :best, legendfontsize = 9, grid = true,
         xlims = (0, r.meta.tmax), yflip = yflip_on,
         title = "$qlabel | N=$(r.meta.N) Kitaev-Heisenberg PBC Lattice " *
                 "at J=$(r.meta.J), $probe_txt" *
                 (yflip_on ? "\ny axis INVERTED: F falls upward, so F and C read the same way" : ""),
         titlefontsize = 12,
         left_margin = 10Plots.mm, bottom_margin = 8Plots.mm,
         top_margin = 6Plots.mm, right_margin = 5Plots.mm)
hline!(p, [rule]; color = INK_MUTED, lw = 1, ls = :dot, label = "")

ed === nothing || plot!(p, ed.time, ed_dat[:, ed.col(site)];
                        color = INK_PRIMARY, lw = 2.8, label = "ED (exact)")
# The two corrections sit almost on top of each other, so they get clearly
# different weights and styles: a thick solid underneath and a dashed line over
# it, which reads as agreement rather than as one curve being absent.
plot!(p, r.time, dat_vb[:, cj];   color = series_color(1), lw = 3.4,
      label = "Eq(14), fitted v_B = $(round(fit.vB, digits=2))")
plot!(p, r.time, dat_site[:, cj]; color = series_color(3), lw = 2.0, ls = :dash,
      label = "front-corrected (per-site)")
plot!(p, r.time, dat_raw[:, cj];  color = series_color(2), lw = 2.0, ls = :dot,
      label = "raw $(r.meta.trunc)")

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
    "otoc_corrections_$(quantity)_$(r.meta.trunc)_N$(r.meta.N)_$(r.meta.lat)" *
    "_V$(r.meta.V)_J$(r.meta.J)_site$(site).png")
s = save_resilient(p, out)
s === nothing || println("\nSaved: $(basename(s))")
