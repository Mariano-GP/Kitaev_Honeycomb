#=
Overlay comparison: dynamical correlators C_{ref,j}(t) from BOTH the exact ED
solver (ED_Validation/ED_Kitaev_Correlation.jl -- ground truth on the small
13-site cluster) and the CAMPS + Pauli-propagation pipeline
(Correlations/kitaev_correlation.jl), plotted together as a grid of subplots
(one per target site j) so the Trotter/truncation approximation can be judged
directly against the exact result.

Both files must be for the SAME reference site and same Pauli type for this to
be a meaningful comparison -- this is checked automatically below (from each
file's own "_site<N>_"/"_XX_|_YY_|_ZZ_" filename tags, the same convention used
throughout this project) and a warning is printed (not fatal) if they don't
match, since overlaying mismatched reference sites/operators compares
physically different quantities, not the same correlator computed two ways.
=#

using NPZ
using Printf
using Plots

V = 0.3       # effective-field three-body term strength (H_V), matches Kitaev_honeycomb_CAMPS.py's --V default
h_field = 0.1 # boundary field strength (H_V), matches Kitaev_honeycomb_CAMPS.py's --h_field default
ref_site = 12
Pauli = :Y
# --- Edit these to match the runs you want to compare ---
pp_npz_path = joinpath(@__DIR__, "correlation_outputs", "PP_kitaev_corr_camps_camps_N22_D100_V$(V)_h$(h_field)_$(Pauli)$(Pauli)_site$(ref_site)_V$(V)_h$(h_field)_dt0.05_tmax5.0_N=22.npz")
#ed_npz_path = joinpath(@__DIR__, "..", "ED_Validation", "correlation_outputs", "ED_kitaev_corr_N13_$(Pauli)$(Pauli)_site$(ref_site)_V$(V)_h$(h_field)_dt0.05_tmax5.0.npz")

# --- Optional: overlay a handful of hand-picked correlator series on ONE
# shared axis (instead of the full NxN grid below) -- e.g. compare a site's
# autocorrelation against its bond-neighbor cross-correlators seen from BOTH
# reference sites (C_89 from the i_site=8 run vs C_98 from the i_site=9 run)
# to check the dynamics at a single edge. Each entry is
# (source, i_site, pauli, j_target, label); source is :ED or :PP, i_site
# selects which correlator-run file to load (its own npz has a column for
# every target site j), and j_target picks the column out of that file.
PLOT_EDGE_OVERLAY = false
overlay_configs = [
    #(:PP, 13, :Z, 13, "C_13_13"),
    (:PP, 11, :Z, 11, "C_11_11"),
    (:PP, 11, :Z, 13, "C_11_13"),
    (:PP, 11, :Z, 9, "C_11_9"),
    # (:PP, 7, :Y, 8, "C_7_8"),
    # (:PP, 7, :X, 6, "C_7_6"),


]

plots_dir = joinpath(@__DIR__, "plots")
mkpath(plots_dir)

function detect_pauli(path)
    if occursin("_XX_", path)
        "XX"
    elseif occursin("_YY_", path)
        "YY"
    elseif occursin("_ZZ_", path)
        "ZZ"
    else
        println("Note: no _XX_/_YY_/_ZZ_ tag found in $(basename(path)) -- assuming XX.")
        "XX"
    end
end

function detect_site(path, fallback)
    m = match(r"_site(\d+)_", basename(path))
    if m !== nothing
        parse(Int, m.captures[1])
    else
        println("Note: no _site<N>_ tag found in $(basename(path)) -- using fallback ref_site=$fallback.")
        fallback
    end
end

pp_pauli = detect_pauli(pp_npz_path)
#ed_pauli = detect_pauli(ed_npz_path)
pp_site  = detect_site(pp_npz_path, 1)
#ed_site  = detect_site(ed_npz_path, 1)

#if pp_pauli != ed_pauli
 #   @warn "PP file is $(pp_pauli) but ED file is $(ed_pauli) -- these are different operators; overlaying them does not validate the same quantity."
#end
#if pp_site != ed_site
 #   @warn "PP file's reference site is $(pp_site) but ED file's is $(ed_site) -- overlaying different reference sites compares physically different correlators. Regenerate one of the two runs so both use the same i_site."
#end

pp_data = npzread(pp_npz_path)
#ed_data = npzread(ed_npz_path)

t_pp = pp_data["time"]
#t_ed = ed_data["time"]
N_pp = size(pp_data["C_re"], 2)
#N_ed = size(ed_data["C_re"], 2)

#N_pp == N_ed || error("PP file has N=$N_pp sites but ED file has N=$N_ed sites -- these are different " *
                       "clusters and cannot be overlaid. (Are you comparing the old 22-site PP run " *
                       "against the new 13-site ED run by mistake?)"
#N = N_pp

@printf("PP: %s (%s, reference site = %d)\n", basename(pp_npz_path), pp_pauli, pp_site)
#@printf("ED: %s (%s, reference site = %d)\n", basename(ed_npz_path), ed_pauli, ed_site)

# One small subplot per target site j, ED (exact, solid) overlaid with PP
# (CAMPS + Trotter, dashed), all combined into a single grid figure.
cols = ceil(Int, sqrt(N))
rows = ceil(Int, N / cols)

subplots = map(1:N) do j
    #C_ed = ed_data["C_re"][:, j] .+ im .* ed_data["C_im"][:, j]
    C_pp = pp_data["C_re"][:, j] .+ im .* pp_data["C_im"][:, j]
    p = plot(t_ed, real.(C_pp);
              label          = j == 1 ? "PP Re" : "",
              color          = :black,
              lw             = 2.0,
              title          = "j = $j",
              titlefontsize  = 8,
              tickfontsize   = 6,
              legend         = j == 1 ? :topright : false,
              legendfontsize = 6,
              grid           = true,
              ylims          = (-1.1, 1.1))
    #plot!(p, t_ed, imag.(C_ed); label = j == 1 ? "ED Im" : "", color = :gray, lw = 2.0)
    #plot!(p, t_pp, real.(C_pp); label = j == 1 ? "PP Re" : "", color = :red, lw = 1.2, linestyle = :dot)
    plot!(p, t_pp, imag.(C_pp); label = j == 1 ? "PP Im" : "", color = :blue, lw = 1.2)
    p
end

fig = plot(subplots...;
           layout     = (rows, cols),
           size       = (250 * cols, 200 * rows),
           plot_title = "Kitaev Honeycomb <$(Pauli)$(Pauli)> Correlators, PP vs ED -- C_$(ref_site),j(t), N=$N, h=$(h_field)",
           dpi        = 150)

display(fig)
out_path = joinpath(plots_dir, "ED_vs_PP_corr_grid_N$(N)_" * replace(basename(pp_npz_path), ".npz" => "") * ".png")
savefig(fig, out_path)
println("\nSaved: $(out_path)")

# --- Optional single-axis overlay of hand-picked (source, i_site, j) series ---
function build_corr_path(source::Symbol, i_site::Int, pauli::Symbol, V, h_field;
                          N::Int = 13, dt::Float64 = 0.05, t_max::Float64 = 5.0)
    if source == :PP
        joinpath(@__DIR__, "correlation_outputs",
                 "PP_kitaev_corr_camps_camps_N$(N)_D100_V$(V)_h$(h_field)_$(pauli)$(pauli)_site$(i_site)_V$(V)_h$(h_field)_dt$(dt)_tmax$(t_max)_N=$(N).npz")
    elseif source == :ED
        joinpath(@__DIR__, "..", "ED_Validation", "correlation_outputs",
                 "ED_kitaev_corr_N$(N)_$(pauli)$(pauli)_site$(i_site)_V$(V)_h$(h_field)_dt$(dt)_tmax$(t_max).npz")
    else
        error("Unknown source $source -- expected :PP or :ED")
    end
end

if PLOT_EDGE_OVERLAY
    palette = [:black, :red, :blue, :green, :purple, :orange, :brown, :cyan]
    p_overlay = plot(; xlabel = "t", ylabel = "Re C(t)",
                      title  = " <ZZ> Correlators overlay for a single reference site, N=$N, site=$(ref_site)",
                      titlefontsize = 10,
                      legend = :topright,
                      grid   = true,
                      size   = (900, 600),
                      margin = 8Plots.mm,
                      top_margin = 12Plots.mm)
    for (idx, (source, i_site, pauli, j, label)) in enumerate(overlay_configs)
        path = build_corr_path(source, i_site, pauli, V, h_field; N = N)
        if !isfile(path)
            @warn "Skipping overlay series \"$(label)\": file not found -- $(path)"
            continue
        end
        data = npzread(path)
        t = data["time"]
        C = data["C_re"][:, j] .+ im .* data["C_im"][:, j]
        color = palette[mod1(idx, length(palette))]
        style = source == :ED ? :solid : :dot
        plot!(p_overlay, t, real.(C); label = "$(source) $(label)", color = color, lw = 1.8, linestyle = style,
              marker = :circle, markersize = 3, markerstrokewidth = 0)
    end
    display(p_overlay)
    overlay_out = joinpath(plots_dir, "edge_overlay_N$(N)_$(Pauli)$(Pauli).png")
    savefig(p_overlay, overlay_out)
    println("\nSaved overlay: $(overlay_out)")
end


