# C(t): v_B correction minus discarded correction, resolved by bond distance.
#
#     julia --project=. otoc_corrdiff_by_distance.jl
#
# One panel per distance shell from the W site, each holding EVERY site at that
# distance, for every truncation threshold. Colour is the threshold; the several
# curves sharing a colour in one panel are the distinct sites in that shell.
#
# WHY THE SHELLS ARE DRAWN SEPARATELY. Both corrections add the discarded
# anticommuting weight, and they differ by
#
#     dC_j(t) = C_vb - C_disc = 4 l(t) * ( 1/2 - p_measured,j(t) )    (behind the front)
#
# so the gap is set by how far the measured p_j sits from the modelled 1/2 --
# and p_j is a PER-SITE quantity. Collapsing the lattice onto the self-
# correlator, as the other difference script does, hides whether that gap is a
# property of the whole operator or only of the site W started on. Ahead of the
# front the v_B scheme adds nothing while the discarded one still adds whatever
# it recorded, so the two also differ there, for a quite different reason.
#
# Sites within a shell are equivalent under the lattice symmetry only if the
# lattice is vertex-transitive AND the symmetry fixes the W site; neither is
# assumed here. Spread within a colour is therefore real structure, not noise,
# and is worth looking at.

include(joinpath(@__DIR__, "otoc_plotting.jl"))

# --- configuration -----------------------------------------------------------
run_files = [
 "PP_kitaev_otoc_infT_N16_L24e521_coeff1e4_w32_disc_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
 "PP_kitaev_otoc_infT_N16_L24e521_coeff5e5_w32_disc_V0.0_J0.05_ZZ_site2_dt0.05_tmax5.0.npz",
]

thr_note    = 0.1
ell_fit_max = 0.05
grid_cols   = 3        # panels wrap into this many columns
show_ed     = true     # score both corrections against ED where one exists
# -----------------------------------------------------------------------------

const DIR = joinpath(@__DIR__, "otoc_outputs")
const THR_CSQ = 2 * thr_note

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
    dist = lattice_distances(r.meta.lat, r.meta.site)
    fv = fit_front_velocity(r.time, r.Csq, r.norm2, dist; thr = THR_CSQ, ell_max = ell_fit_max)
    isfinite(fv.vB) || error("$(r.meta.trunc): could not fit v_B")
    _, C_vb, _ = vb_front_correct(r.time, r.F, r.Csq, r.norm2, dist, fv.vB)
    push!(rows, (meta = r.meta, time = r.time, dist = dist, raw = r.Csq,
                 vb = C_vb, di = d["Csq_corrected"], ell = d["ell"],
                 pm = d["p_measured"], vB = fv.vB, col = r.col))
end

m0    = rows[1].meta
dist0 = rows[1].dist
shells = sort(unique(dist0))
ed   = show_ed ? find_ed(m0) : nothing

# ---- report ------------------------------------------------------------------
println("\n", "="^100)
println("C(t): v_B correction MINUS discarded correction, by bond distance")
println("="^100)
@printf("  N=%d  %s  V=%s  J=%s  W@%d   v_B = %.3f\n",
        m0.N, m0.lat, m0.V, m0.J, m0.site, rows[1].vB)
println("  dC = C_vb - C_disc = 4*l*(1/2 - p_measured) behind the front.")
println("  dC > 0: v_B adds more (p_measured < 1/2).  dC < 0: the measured p is larger.")
ed === nothing || println("  ED: ", basename(ed.meta.path), "  (t <= ", ed.time[end], ")")
println("-"^100)
@printf("%5s %6s %6s | %-26s | %s\n", "dist", "sites", "in ED", "mean dC(end)",
        "max|C-ED|  v_B / discarded")
println("-"^100)
# ED probed only the sites in its own `probe_sites` (here the distance ladder),
# so most sites have no column in it. Scoring those would index `nothing`;
# they are skipped and the count of covered sites is reported instead, so a
# shell scored from one site is never mistaken for one scored from four.
for s in shells
    js  = [j for j in 1:length(dist0) if dist0[j] == s]
    jse = ed === nothing ? Int[] : [j for j in js if ed.col(j) !== nothing]
    dtxt = join([@sprintf("%-12s %+.3f", r.meta.trunc,
                          sum(r.vb[end, r.col(j)] - r.di[end, r.col(j)] for j in js) / length(js))
                 for r in rows], "  ")
    etxt = isempty(jse) ? "(no ED site in this shell)" :
        join([@sprintf("%.3f/%.3f",
              maximum(maxdev(r.vb[:, r.col(j)], ed.Csq[:, ed.col(j)]) for j in jse),
              maximum(maxdev(r.di[:, r.col(j)], ed.Csq[:, ed.col(j)]) for j in jse))
              for r in rows], "  ")
    @printf("%5d %6d %6d | %-26s | %s\n", s, length(js), length(jse), dtxt, etxt)
end
println("-"^100)
println("mean dC(end) is averaged over every site in the shell; the figure draws them")
println("individually, so spread within one colour is real per-site structure.")

# ---- figure -------------------------------------------------------------------
tmax = maximum(r.meta.tmax for r in rows)

# ONE SITE PER SHELL -- the one ED actually probed. Drawing all four sites of a
# shell times two thresholds times two corrections would be sixteen curves in a
# panel, and only one of them would have a reference to be judged against. The
# chosen site is named in each panel title so it is never ambiguous which one is
# shown; the per-site spread within a shell is the other figure's job.
function shell_site(s)
    js = [j for j in 1:length(dist0) if dist0[j] == s]
    ed === nothing && return first(js)
    k = findfirst(j -> ed.col(j) !== nothing, js)
    k === nothing ? first(js) : js[k]
end

panels = map(enumerate(shells)) do (idx, s)
    js = [j for j in 1:length(dist0) if dist0[j] == s]
    j  = shell_site(s)
    has_ed = ed !== nothing && ed.col(j) !== nothing
    p = plot(; base_attrs()...,
             title = "d = $s   site $j" * (length(js) > 1 ? "  (of $(length(js)) in shell)" : ""),
             titlefontsize = 11, tickfontsize = 9,
             xlabel = idx > length(shells) - grid_cols ? "time  t" : "",
             ylabel = mod1(idx, grid_cols) == 1 ? L"C_{ij}(t)" : "",
             guidefontsize = 10,
             legend = idx == 1 ? :bottomright : false, legendfontsize = 6,
             xlims = (0, tmax), ylims = (0, 2.75), grid = true)
    hline!(p, [2.0]; color = INK_MUTED, lw = 1, ls = :dot, label = "")
    # ED first so it lies beneath the approximations rather than hiding them.
    has_ed && plot!(p, ed.time, ed.Csq[:, ed.col(j)];
                    color = INK_PRIMARY, lw = 3.0, label = "ED (exact)")
    for (i, r) in enumerate(rows)
        plot!(p, r.time, r.vb[:, r.col(j)]; color = series_color(i), lw = 2.4, ls = :solid,
              label = "$(r.meta.trunc)  v_B")
        plot!(p, r.time, r.di[:, r.col(j)]; color = series_color(i), lw = 2.0, ls = :dash,
              label = "$(r.meta.trunc)  discarded")
    end
    p
end

nrow = ceil(Int, length(panels) / grid_cols)
fig = plot(panels...; layout = (nrow, grid_cols),
           size = (430 * grid_cols, 380 * nrow + 80), dpi = 200,
           plot_title = "C(t) corrected two ways, against ED, by bond distance | " *
                        "N=$(m0.N) $(m0.lat) J=$(m0.J) W@$(m0.site)
" *
                        "solid = v_B (modelled p=1/2)   dashed = discarded (measured p)   black = ED",
           plot_titlefontsize = 11, top_margin = 7Plots.mm,
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
    "otoc_C_bydist_vb_vs_disc_N$(m0.N)_$(m0.lat)_V$(m0.V)_J$(m0.J)_site$(m0.site).png")
s = save_resilient(fig, out)
s === nothing || println("\nSaved: $(basename(s))")
