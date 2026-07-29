
# Computes S(k,ω) magnon spectra from the C(t, j) data that correlation.jl's
# run_corr_configs already computed and saved to PP_corr_*.npz files (run
# `julia correlation.jl` to (re)generate them). No PP propagation happens in
# this script — we just load each configuration's saved correlator and FFT it.

using LinearAlgebra
using Plots, LaTeXStrings
using FFTW
using NPZ

# Given M of shape (n_t × (1+N)) where column 1 is time and columns 2:end are
# C(t, j), we compute S(k, ω) via:
#  1. Spatial FFT over sites j  =  C(t, k)
#  2. Temporal FFT over time t  =  S(k, ω)

function Compute_Skw(M_corr::Matrix{Float64}; pad_factor::Int = 6, subtract_dc::Bool = false, damping::Float64 = 1.0)
    times = M_corr[:, 1]
    C = M_corr[:, 2:end]
    n_t, N_sites = size(C)

    # Exponential damping: drives signal to zero at t_max, equivalent to Lorentzian broadening in ω
    if damping > 0.0
        C = C .* exp.(-damping .* times)
    end

    # Re-center: its important to shift source site (col 1) to center col so PBC wraps symmetrically
    C = circshift(C, (0, N_sites ÷ 2))

    # Zero-pad for finer interpolation in both k and ω this makes a finer grid we can adjust
    C_pad = zeros(Float64, n_t * pad_factor, N_sites * pad_factor)
    C_pad[1:n_t, 1:N_sites] = C

    #Actual ffts 
    C_k  = fft(C_pad, 2) #space
    S_kw = fft(C_k,   1) #time

    return fftshift(S_kw)
end

out_dir   = @__DIR__
plots_dir = joinpath(out_dir, "plots")
mkpath(plots_dir)   # figures go in plots/, data (.npy) stays alongside the scripts

N = 22  # derived from current J.npy

tick_positions = [-pi, -pi/2, 0, pi/2, pi]

tick_labels = [L"-\pi", L"\frac{-\pi}{2}", 0, L"\frac{\pi}{2}", L"\pi"]

# We can modify to test any configuration we want — each entry must have a
# corresponding PP_corr_eps<ε>_w<w>.npz file saved by correlation.jl.
configs = [
    #  label            alpha   thresh (ε)
    ("α=0.05, ε=1e-4",    0.05,   1e-4), 
    ("α=0.1, ε=1e-4",    0.1,   1e-4), 
    ("α=0.2, ε=1e-4",    0.2,   1e-4),
    ("α=0.3, ε=1e-4",    0.3,   1e-4),
    ("α=0.4, ε=1e-4",    0.4,   1e-4),  
    ("α=0.5, ε=1e-4",    0.5,   1e-4), 
    ("α=1.0, ε=1e-4",    1.0,   1e-4), 
	("α=2.0, ε=1e-4",    2.0,   1e-4), 
	("α=3.0, ε=1e-4",    3.0,   1e-4), 
	("α=4.0, ε=1e-4",    4.0,   1e-4), 
	#("α=5.0, ε=1e-4",    5.0,   1e-4), 
	# ("α=7.0, ε=1e-4",    7.0,   1e-4), 
	# ("α=9.0, ε=1e-4",    9.0,   1e-4),
    # ("α=2.0, ε=1e-2",    2.0,   1e-2), 
	# ("α=2.0, ε=1e-3",    2.0,   1e-3), 
	# ("α=2.0, ε=1e-4",    2.0,   1e-4), 
	# ("α=2.0, ε=1e-5",    2.0,   1e-5), 
	# ("α=2.0, ε=1e-6",    2.0,   1e-6), 
	# ("α=2.0, ε=1e-7",    2.0,   1e-7), 
	# ("α=2.0, ε=1e-8",    2.0,   1e-8), 
	# ("α=2.0, ε=1e-9",    2.0,   1e-9),

]
# Loads PP_corr_eps<ε>_w<w>.npz (saved by correlation.jl's run_corr_configs)
# and reassembles it into the (n_t × (1+N)) [time | C(t,j)] matrix Compute_Skw
# expects — using the real part only, same as the old load_and_run_corr path.
function load_PP_corr_matrix(alpha, thresh)
    safe_label = "N=$(N)_alpha$(alpha)_thresh$(thresh)"
    path       = joinpath(out_dir, "PP_corr_$(safe_label).npz")
    data       = npzread(path)
    return hcat(data["time"], data["C_re"])
end

pp_panels = Plots.Plot[]

for (label, alpha, thresh) in configs
    println("\n" * "="^55)
    println("Loading PP config: $label")
    println("="^55)

    M_i = load_PP_corr_matrix(alpha, thresh)
    println("  M_corr size: ", size(M_i))

    S_kw_i = Compute_Skw(M_i; subtract_dc = true, damping = 0.3)
    Mag2_i = abs2.(S_kw_i)

    sorted_i = sort(vec(Mag2_i))
    p99_i    = sorted_i[floor(Int, 0.994 * length(sorted_i))] #gets rid of outlier pixels and keeps 99.5% of everything
    norm_i   = clamp.(Mag2_i ./ p99_i, 0.0, 1.0)
    nf       = 0.05 #Noise floor gets rid of small fluctuations and cleans up the graph
    norm_i   = clamp.(norm_i .- nf, 0.0, 1.0) ./ (1.0 - nf)
    norm_i   = norm_i .^ 1.1  # gamma<1: compresses bright spots, lifts dim parts of the curve

    safe = replace(label, "=" => "", "," => "", " " => "_")
    npzwrite(joinpath(out_dir, "Mag2_Sqw_$(safe)_N=$(N).npy"), Mag2_i)
    println("  Saved: Mag2_Sqw_$(safe)_N=$(N).npy")

    n_t_pad_i, N_pad_i = size(S_kw_i)
    dt_i     = M_i[2, 1] - M_i[1, 1]
    ω_ax_i   = fftshift(fftfreq(n_t_pad_i, 1/dt_i)) .* 2π
    k_ax_i   = fftshift(fftfreq(N_pad_i, 1.0)) .* 2π
    ω_disp_i = 4 .* (1 .- cos.(k_ax_i))

    plt_i = heatmap(k_ax_i, ω_ax_i, norm_i,
        xlabel = "k",            ylims = [0, 12],
        ylabel = "ω", xlims = [-π, π],
        xticks = (tick_positions, tick_labels),
        tickfontsize = 11,
        title  = "|S(k,ω)|², PP $label",
        color  = :inferno,       clims = (0, 0.7))
    plot!(plt_i, k_ax_i, ω_disp_i, label = "ω(k)=4(1−cos k)", color = :green, lw = 2)

    savefig(plt_i, joinpath(plots_dir, "Skw_PP_$(safe)_N=$(N).png"))
    println("  Saved: Skw_PP_$(safe)_N=$(N).png")
    push!(pp_panels, plt_i)
end

n_t_pad_ed, N_pad_ed = size(S_kw_ED)
dt_ed      = ed_times[2] - ed_times[1]
ω_ax_ed    = fftshift(fftfreq(n_t_pad_ed, 1/dt_ed)) .* 2π
k_ax_ed    = fftshift(fftfreq(N_pad_ed, 1.0)) .* 2π
ω_disp_ed  = 4 .* (1 .- cos.(k_ax_ed))

plt_ed = heatmap(k_ax_ed, ω_ax_ed, Mag2_ED_norm,
    xlabel = "k",  ylims = [0, 12],
    ylabel = "ω", xlims = [-pi, pi],
    xticks = (tick_positions, tick_labels),
    tickfontsize = 11,
    title  = "|S(k,ω)|² ED 1-magnon ",
    color  = :inferno,
    clims  = (0, 0.7))
plot!(plt_ed, k_ax_ed, ω_disp_ed, label = "ω(k)=4(1−cos k)", color = :green, lw = 2)
savefig(plt_ed, joinpath(plots_dir, "Skw_ED_N=$(N).png"))
println("Saved: Skw_ED_N=$(N).png")

# ── Combined grid: ED first, then PP configs ──────────────────────────────────
all_panels  = vcat([plt_ed], pp_panels)
n_panels    = length(all_panels)
n_cols      = 3
n_rows      = cld(n_panels, n_cols)
empty_panel = plot(legend=false, grid=false, showaxis=false, ticks=nothing,
                   background_color=:white)
grid_panels = vcat(all_panels, fill(empty_panel, n_rows * n_cols - n_panels))

fig_grid = plot(grid_panels...;
    layout             = (n_rows, n_cols),
    size               = (1080 * n_cols, 760 * n_rows),
    dpi                = 150,
    left_margin        = 15Plots.mm,
    right_margin       = 12Plots.mm,
    top_margin         = 15Plots.mm,
    bottom_margin      = 15Plots.mm,
    plot_titlefontsize = 20,
    plot_title         = "|S(k,ω)|²  —  ED reference + PP configs")

savefig(fig_grid, joinpath(plots_dir, "Fixed_ε_Skw_all_combined_N=$(N).png"))
println("Saved: Skw_all_combined_N=$(N).png")
display(fig_grid)


#EXTRA PLOTTING MEASURES THAT SOMETIMES IMPROVE VISUAL RESULTS:

# Spatial Hann window — tapers the circshifted array to zero at both edges.
    # For large w with long T the magnon reaches all N sites, making C(t,j) non-zero
    # at the array boundary.
    #hann_s = 0.5 .* (1 .- cos.(2π .* (0:N_sites-1) ./ (N_sites - 1)))
    #C      = C .* hann_s'

    # Temporal Hann window — tapers signal to zero at t_max to suppress Gibbs ringing
    # which are vertical black fringes that form on the graph due to the way the foruier 
    # transfrom is calculated
    #hann_t = 0.5 .* (1 .- cos.(2π .* (0:n_t-1) ./ (n_t - 1)))
    #C      = C .* hann_t

    # DC subtraction AFTER Hann: subtracting the mean of the already-windowed signal
    # guarantees exactly zero mean → zero ω=0 power in the temporal FFT.
    #if subtract_dc
     #   C = C .- sum(C, dims=1) ./ n_t
    #end