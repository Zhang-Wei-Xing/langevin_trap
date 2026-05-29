from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

if "--ui" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


@dataclass(frozen=True)
class TrapParameters:
    kappa: np.ndarray
    gamma: float
    diffusion: float
    dt: float
    steps: int
    initial_position: np.ndarray
    seed: int | None


@dataclass(frozen=True)
class AnalysisResults:
    time: np.ndarray
    positions: np.ndarray
    sampled: np.ndarray
    sigmas: np.ndarray
    acf_lags: np.ndarray
    acf: np.ndarray
    msd_lags: np.ndarray
    msd: np.ndarray


def parse_vector(text: str, *, name: str) -> np.ndarray:
    values = np.fromstring(text, sep=",", dtype=float)
    if values.shape != (3,):
        raise argparse.ArgumentTypeError(f"{name} must contain exactly 3 comma-separated values")
    return values


def simulate(params: TrapParameters) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(params.seed)
    positions = np.empty((params.steps + 1, 3), dtype=float)
    positions[0] = params.initial_position

    drift = params.kappa / params.gamma
    noise_scale = np.sqrt(2.0 * params.diffusion * params.dt)

    for i in range(1, params.steps + 1):
        random_kick = noise_scale * rng.normal(size=3)
        restoring_step = drift * positions[i - 1] * params.dt
        positions[i] = positions[i - 1] - restoring_step + random_kick

    time = np.arange(params.steps + 1) * params.dt
    return time, positions


def equilibrium_sigmas(params: TrapParameters) -> np.ndarray:
    return np.sqrt(params.diffusion * params.gamma / params.kappa)


def potential_energy(positions: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(kappa * positions**2, axis=1)


def smooth2d(values: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0:
        return values
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / max(radius / 2.0, 1.0)) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 0, padded)
    padded = np.pad(smoothed, ((0, 0), (radius, radius)), mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)


def density_grid(
    sampled: np.ndarray,
    x_index: int,
    y_index: int,
    bins: int,
    sigmas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = sampled[:, x_index]
    y = sampled[:, y_index]
    x_span = max(float(np.max(np.abs(x))), float(3.0 * sigmas[x_index]), 1e-9)
    y_span = max(float(np.max(np.abs(y))), float(3.0 * sigmas[y_index]), 1e-9)
    hist, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[-x_span, x_span], [-y_span, y_span]],
        density=True,
    )
    return smooth2d(hist.T, radius=2), x_edges, y_edges


def fokker_planck_equilibrium_density(
    x_index: int,
    y_index: int,
    bins: int,
    sigmas: np.ndarray,
    *,
    span_scale: float = 3.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_x = float(sigmas[x_index])
    sigma_y = float(sigmas[y_index])
    x_span = max(span_scale * sigma_x, 1e-9)
    y_span = max(span_scale * sigma_y, 1e-9)
    x_edges = np.linspace(-x_span, x_span, bins + 1)
    y_edges = np.linspace(-y_span, y_span, bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    density = np.exp(-0.5 * ((xx / sigma_x) ** 2 + (yy / sigma_y) ** 2))
    density /= 2.0 * np.pi * sigma_x * sigma_y
    return density, x_edges, y_edges


def normalized_position_acf(sampled: np.ndarray, dt: float, max_lags: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    centered = sampled - sampled.mean(axis=0)
    n = len(centered)
    max_lags = min(max_lags, n - 1)
    if max_lags < 1:
        return np.array([0.0]), np.array([1.0])

    fft_size = 1 << (2 * n - 1).bit_length()
    acfs = []
    for axis in range(centered.shape[1]):
        spectrum = np.fft.rfft(centered[:, axis], n=fft_size)
        corr = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[: max_lags + 1]
        corr /= np.arange(n, n - max_lags - 1, -1)
        if corr[0] > 0.0:
            corr /= corr[0]
        acfs.append(corr)
    one_sided = np.mean(acfs, axis=0)
    lags = np.arange(max_lags + 1) * dt
    symmetric_lags = np.concatenate((-lags[:0:-1], lags))
    symmetric_acf = np.concatenate((one_sided[:0:-1], one_sided))
    return symmetric_lags, symmetric_acf


def mean_squared_displacement(sampled: np.ndarray, dt: float, points: int = 90) -> tuple[np.ndarray, np.ndarray]:
    n = len(sampled)
    if n < 3:
        return np.array([dt]), np.array([0.0])
    max_lag = max(1, min(n // 2, 5000))
    lags = np.unique(np.geomspace(1, max_lag, points).astype(int))
    msd = np.empty_like(lags, dtype=float)
    for i, lag in enumerate(lags):
        delta = sampled[lag:] - sampled[:-lag]
        msd[i] = np.mean(np.sum(delta * delta, axis=1))
    return lags * dt, msd


def analyze(
    time: np.ndarray,
    positions: np.ndarray,
    params: TrapParameters,
    burn_in_fraction: float,
) -> AnalysisResults:
    burn_in = int(len(positions) * burn_in_fraction)
    sampled = positions[burn_in:]
    sigmas = equilibrium_sigmas(params)
    acf_lags, acf = normalized_position_acf(sampled, params.dt)
    msd_lags, msd = mean_squared_displacement(sampled, params.dt)
    return AnalysisResults(time, positions, sampled, sigmas, acf_lags, acf, msd_lags, msd)


def make_ellipsoid(sigmas: np.ndarray, scale: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, 80)
    v = np.linspace(0.0, np.pi, 40)
    x = scale * sigmas[0] * np.outer(np.cos(u), np.sin(v))
    y = scale * sigmas[1] * np.outer(np.sin(u), np.sin(v))
    z = scale * sigmas[2] * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def set_equal_3d_limits(ax: plt.Axes, positions: np.ndarray, sigmas: np.ndarray) -> None:
    span = max(np.max(np.abs(positions)), float(np.max(2.5 * sigmas)), 1.0)
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(-span, span)


def plot_results(
    time: np.ndarray,
    positions: np.ndarray,
    params: TrapParameters,
    output: Path,
    bins: int,
    burn_in_fraction: float,
) -> None:
    analysis = analyze(time, positions, params, burn_in_fraction)
    sigmas = analysis.sigmas
    kappa_label = ", ".join(f"{value:g}" for value in params.kappa)
    sampled = analysis.sampled

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    stride = max(1, len(positions) // 2500)
    ax3d.plot(
        positions[::stride, 0],
        positions[::stride, 1],
        positions[::stride, 2],
        color="#3f3f46",
        linewidth=0.7,
        alpha=0.85,
    )
    ex, ey, ez = make_ellipsoid(sigmas)
    ax3d.plot_surface(ex, ey, ez, color="#86a7b8", alpha=0.18, linewidth=0)
    ax3d.scatter(*positions[0], color="#2f6fdd", s=28, label="start")
    ax3d.scatter(*positions[-1], color="#d33f49", s=28, label="end")
    set_equal_3d_limits(ax3d, sampled, sigmas)
    ax3d.set_title("3D trajectory in harmonic trap")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.legend(loc="upper left")

    ax_xz = fig.add_subplot(gs[0, 1])
    plot_density(ax_xz, sampled, 0, 2, bins, sigmas, cmap="magma")
    ax_xz.set_title("Smoothed probability density: x-z")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")

    ax_xy = fig.add_subplot(gs[1, 1])
    plot_fokker_planck_density(ax_xy, 0, 1, bins, sigmas, cmap="viridis")
    ax_xy.set_title("Fokker-Planck theory density: x-y")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")

    ax_acf = fig.add_subplot(gs[0, 2])
    plot_acf(ax_acf, analysis.acf_lags, analysis.acf)

    ax_msd = fig.add_subplot(gs[1, 2])
    plot_msd(ax_msd, analysis.msd_lags, analysis.msd, sigmas)

    fig.suptitle(
        "Overdamped Langevin particle: "
        f"kappa=({kappa_label}), gamma={params.gamma:g}, D={params.diffusion:g}, dt={params.dt:g}",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_density(
    ax: plt.Axes,
    sampled: np.ndarray,
    x_index: int,
    y_index: int,
    bins: int,
    sigmas: np.ndarray,
    *,
    cmap: str,
    colorbar: bool = True,
) -> None:
    density, x_edges, y_edges = density_grid(sampled, x_index, y_index, bins, sigmas)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    mesh = ax.imshow(density, origin="lower", extent=extent, aspect="auto", cmap=cmap)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    positive = density[density > 0.0]
    if positive.size:
        levels = np.quantile(positive, [0.55, 0.72, 0.86, 0.94])
        ax.contour(x_centers, y_centers, density, levels=np.unique(levels), colors="white", linewidths=0.55, alpha=0.75)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label="probability density")
    ax.axhline(0.0, color="white", linewidth=0.7, alpha=0.7)
    ax.axvline(0.0, color="white", linewidth=0.7, alpha=0.7)


def plot_fokker_planck_density(
    ax: plt.Axes,
    x_index: int,
    y_index: int,
    bins: int,
    sigmas: np.ndarray,
    *,
    cmap: str,
    colorbar: bool = True,
) -> None:
    density, x_edges, y_edges = fokker_planck_equilibrium_density(x_index, y_index, bins, sigmas)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    mesh = ax.imshow(density, origin="lower", extent=extent, aspect="auto", cmap=cmap)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    levels = np.quantile(density, [0.55, 0.72, 0.86, 0.94])
    ax.contour(x_centers, y_centers, density, levels=np.unique(levels), colors="white", linewidths=0.55, alpha=0.75)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label="theory probability density")
    ax.axhline(0.0, color="white", linewidth=0.7, alpha=0.7)
    ax.axvline(0.0, color="white", linewidth=0.7, alpha=0.7)


def plot_acf(ax: plt.Axes, lags: np.ndarray, acf: np.ndarray) -> None:
    ax.plot(lags, acf, color="#202124", linewidth=1.4)
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.8)
    ax.set_title("Position ACF")
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF [n.u.]")
    ax.set_ylim(min(-0.15, float(np.nanmin(acf)) * 1.05), 1.08)


def plot_msd(ax: plt.Axes, lags: np.ndarray, msd: np.ndarray, sigmas: np.ndarray) -> None:
    ax.plot(lags, msd, color="#4b5563", linewidth=1.4, marker=".", markersize=3)
    plateau = 2.0 * np.sum(sigmas * sigmas)
    ax.axhline(plateau, color="#d97706", linewidth=1.0, linestyle="--", label="theory plateau")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Mean squared displacement")
    ax.set_xlabel("lag")
    ax.set_ylabel("MSD")
    ax.legend(fontsize=8)


def render_analysis_axes(
    fig: plt.Figure,
    axes: dict[str, plt.Axes],
    params: TrapParameters,
    bins: int,
    burn_in_fraction: float,
    *,
    include_colorbars: bool = False,
) -> None:
    time, positions = simulate(params)
    analysis = analyze(time, positions, params, burn_in_fraction)

    for ax in axes.values():
        ax.clear()

    ax3d = axes["trajectory"]
    stride = max(1, len(positions) // 2200)
    ax3d.plot(
        positions[::stride, 0],
        positions[::stride, 1],
        positions[::stride, 2],
        color="#3f3f46",
        linewidth=0.7,
        alpha=0.85,
    )
    ex, ey, ez = make_ellipsoid(analysis.sigmas)
    ax3d.plot_surface(ex, ey, ez, color="#86a7b8", alpha=0.18, linewidth=0)
    ax3d.scatter(*positions[0], color="#2f6fdd", s=22)
    ax3d.scatter(*positions[-1], color="#d33f49", s=22)
    set_equal_3d_limits(ax3d, analysis.sampled, analysis.sigmas)
    ax3d.set_title("3D trajectory")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")

    plot_density(
        axes["density_xz"],
        analysis.sampled,
        0,
        2,
        bins,
        analysis.sigmas,
        cmap="magma",
        colorbar=include_colorbars,
    )
    axes["density_xz"].set_title("Density x-z")
    axes["density_xz"].set_xlabel("x")
    axes["density_xz"].set_ylabel("z")

    plot_fokker_planck_density(
        axes["density_xy"],
        0,
        1,
        bins,
        analysis.sigmas,
        cmap="viridis",
        colorbar=include_colorbars,
    )
    axes["density_xy"].set_title("Fokker-Planck theory x-y")
    axes["density_xy"].set_xlabel("x")
    axes["density_xy"].set_ylabel("y")

    plot_acf(axes["acf"], analysis.acf_lags, analysis.acf)
    plot_msd(axes["msd"], analysis.msd_lags, analysis.msd, analysis.sigmas)
    fig.canvas.draw_idle()


def run_interactive_ui(params: TrapParameters, bins: int, burn_in_fraction: float) -> None:
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.78, 1.35, 1.05, 1.05], wspace=0.45, hspace=0.34)

    controls_ax = fig.add_subplot(gs[:, 0])
    controls_ax.set_axis_off()
    controls_ax.set_title("Parameters", loc="left", pad=18)

    axes = {
        "trajectory": fig.add_subplot(gs[:, 1], projection="3d"),
        "density_xz": fig.add_subplot(gs[0, 2]),
        "density_xy": fig.add_subplot(gs[1, 2]),
        "acf": fig.add_subplot(gs[0, 3]),
        "msd": fig.add_subplot(gs[1, 3]),
    }

    slider_specs = [
        ("kx", params.kappa[0], 0.05, 8.0, 0.05),
        ("ky", params.kappa[1], 0.05, 8.0, 0.05),
        ("kz", params.kappa[2], 0.05, 8.0, 0.05),
        ("gamma", params.gamma, 0.1, 5.0, 0.05),
        ("D", params.diffusion, 0.001, 0.5, 0.001),
        ("dt", params.dt, 0.001, 0.05, 0.001),
        ("steps", params.steps, 1000, 100000000, 1000),
        ("bins", bins, 40, 180, 5),
        ("burn-in", burn_in_fraction, 0.0, 0.9, 0.05),
    ]

    sliders: dict[str, Slider] = {}
    y = 0.84
    for name, value, min_value, max_value, step in slider_specs:
        slider_ax = fig.add_axes([0.055, y, 0.18, 0.026])
        sliders[name] = Slider(slider_ax, name, min_value, max_value, valinit=value, valstep=step)
        y -= 0.075

    button_ax = fig.add_axes([0.075, 0.085, 0.14, 0.045])
    button = Button(button_ax, "Run simulation")
    status = fig.text(0.055, 0.045, "", fontsize=9, color="#4b5563")

    def current_params() -> tuple[TrapParameters, int, float]:
        return (
            TrapParameters(
                kappa=np.array([sliders["kx"].val, sliders["ky"].val, sliders["kz"].val], dtype=float),
                gamma=float(sliders["gamma"].val),
                diffusion=float(sliders["D"].val),
                dt=float(sliders["dt"].val),
                steps=int(sliders["steps"].val),
                initial_position=params.initial_position,
                seed=params.seed,
            ),
            int(sliders["bins"].val),
            float(sliders["burn-in"].val),
        )

    def update(_event: object | None = None) -> None:
        next_params, next_bins, next_burn_in = current_params()
        status.set_text("running...")
        fig.canvas.draw_idle()
        render_analysis_axes(fig, axes, next_params, next_bins, next_burn_in)
        kappa_text = ", ".join(f"{v:g}" for v in next_params.kappa)
        status.set_text(f"kappa=({kappa_text}), gamma={next_params.gamma:g}, D={next_params.diffusion:g}")
        fig.canvas.draw_idle()

    button.on_clicked(update)
    update()
    plt.show()


def print_summary(time: np.ndarray, positions: np.ndarray, params: TrapParameters, burn_in_fraction: float) -> None:
    burn_in = int(len(positions) * burn_in_fraction)
    sampled = positions[burn_in:]
    measured_mean = sampled.mean(axis=0)
    measured_std = sampled.std(axis=0)
    expected_std = equilibrium_sigmas(params)
    final_position = positions[-1]
    final_energy = potential_energy(positions[-1:], params.kappa)[0]

    print("Simulation complete")
    print(f"duration: {time[-1]:.6g}")
    print(f"samples after burn-in: {len(sampled)}")
    print(f"final position: x={final_position[0]:.6g}, y={final_position[1]:.6g}, z={final_position[2]:.6g}")
    print(f"final potential energy: {final_energy:.6g}")
    print(
        "measured std after burn-in: "
        f"x={measured_std[0]:.6g}, y={measured_std[1]:.6g}, z={measured_std[2]:.6g}"
    )
    print(
        "theory std sqrt(D*gamma/kappa): "
        f"x={expected_std[0]:.6g}, y={expected_std[1]:.6g}, z={expected_std[2]:.6g}"
    )
    print(f"measured mean after burn-in: x={measured_mean[0]:.6g}, y={measured_mean[1]:.6g}, z={measured_mean[2]:.6g}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate a Brownian particle in a damped 3D harmonic potential well."
    )
    parser.add_argument(
        "--kappa",
        type=lambda text: parse_vector(text, name="kappa"),
        default=parse_vector("1.0,1.0,0.25", name="kappa"),
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="friction coefficient")
    parser.add_argument("--diffusion", "--D", dest="diffusion", type=float, default=0.05, help="diffusion coefficient")
    parser.add_argument("--dt", type=float, default=0.005, help="time step")
    parser.add_argument("--steps", type=int, default=500000, help="number of finite-difference steps")
    parser.add_argument(
        "--initial-position",
        type=lambda text: parse_vector(text, name="initial-position"),
        default=parse_vector("0.0,0.0,0.0", name="initial-position"),
    )
    parser.add_argument("--seed", type=int, default=7, help="random seed; use a negative value for no fixed seed")
    parser.add_argument("--bins", type=int, default=90, help="histogram bins for probability density plots")
    parser.add_argument("--burn-in-fraction", type=float, default=0.2, help="fraction of initial samples to discard")
    parser.add_argument("--output", type=Path, default=Path("outputs/langevin_trap.png"), help="output image path")
    parser.add_argument("--ui", action="store_true", help="open an interactive parameter-tuning interface")
    return parser


def validate_args(args: argparse.Namespace) -> TrapParameters:
    if np.any(args.kappa <= 0.0):
        raise SystemExit("All kappa values must be positive.")
    if args.gamma <= 0.0:
        raise SystemExit("gamma must be positive.")
    if args.diffusion <= 0.0:
        raise SystemExit("diffusion must be positive.")
    if args.dt <= 0.0:
        raise SystemExit("dt must be positive.")
    if args.steps < 2:
        raise SystemExit("steps must be at least 2.")
    if args.bins < 8:
        raise SystemExit("bins must be at least 8.")
    if not 0.0 <= args.burn_in_fraction < 0.95:
        raise SystemExit("burn-in-fraction must be in [0, 0.95).")
    seed = None if args.seed < 0 else args.seed
    return TrapParameters(
        kappa=args.kappa,
        gamma=args.gamma,
        diffusion=args.diffusion,
        dt=args.dt,
        steps=args.steps,
        initial_position=args.initial_position,
        seed=seed,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    params = validate_args(args)
    if args.ui:
        if "agg" in matplotlib.get_backend().lower():
            print("Matplotlib is using a non-interactive Agg backend; the UI window may not open in this session.")
        run_interactive_ui(params, args.bins, args.burn_in_fraction)
        return
    time, positions = simulate(params)
    plot_results(time, positions, params, args.output, args.bins, args.burn_in_fraction)
    print_summary(time, positions, params, args.burn_in_fraction)
    print(f"plot saved to: {args.output}")


if __name__ == "__main__":
    main()
