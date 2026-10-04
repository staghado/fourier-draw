#!/usr/bin/env python3
# coding: utf-8
"""Draw an image's contours as Fourier epicycles.

Extracts contours from an image, computes the Fourier series of the (complex)
contour with an FFT, and renders the rotating-vector animation to mp4 or gif.

Dependencies: numpy + matplotlib only (no scipy / opencv).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from math import tau

import numpy as np


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Draw an image's contours with Fourier epicycles (fast).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True, help="input image path")
    p.add_argument("-o", "--output", required=True, help="output animation path (.mp4 or .gif)")
    p.add_argument("-f", "--frames", type=int, default=300, help="number of frames")
    p.add_argument("-N", "--N", type=int, default=300,
                   help="number of Fourier coefficients (epicycles) to keep")
    p.add_argument("--mode", choices=("magnitude", "range"), default="magnitude",
                   help="'magnitude': keep the N strongest harmonics; "
                        "'range': keep harmonics in [-N, N] (original behaviour)")
    p.add_argument("--samples", type=int, default=2048,
                   help="points used to arc-length-resample the path (0 = keep raw)")
    p.add_argument("--all-contours", action="store_true",
                   help="use every contour, joined with a nearest-neighbour path")
    p.add_argument("--min-contour-len", type=int, default=25,
                   help="ignore contours shorter than this many points")
    p.add_argument("--blur", type=int, default=3, help="box-blur radius for de-noising (0 = off)")
    p.add_argument("--max-dim", type=int, default=2048,
                   help="downscale the image so its longest side is at most this (0 = off)")
    p.add_argument("--invert", action="store_true", help="force inverting threshold polarity")
    p.add_argument("--no-auto-invert", action="store_true",
                   help="disable automatic threshold polarity detection")
    p.add_argument("--fps", type=int, default=30, help="output frames per second")
    p.add_argument("--size", type=float, default=10.0, help="figure size in inches")
    p.add_argument("--dpi", type=int, default=100, help="output dpi (resolution = size*dpi)")
    p.add_argument("--codec", choices=("auto", "libx264", "videotoolbox"), default="libx264",
                   help="video codec; libx264 is usually fastest")
    p.add_argument("--preset", choices=("ultrafast", "superfast", "veryfast", "faster", "medium"),
                   default="veryfast", help="libx264 speed/quality preset")
    p.add_argument("--no-circles", action="store_true", help="hide epicycle rings/vectors")
    p.add_argument("--no-gradient", action="store_true", help="solid trail instead of a gradient")
    p.add_argument("--no-blit", action="store_true",
                   help="disable blitting during save (slightly slower, safest)")
    p.add_argument("--theme", choices=("midnight", "blueprint", "paper"), default="midnight",
                   help="colour theme")
    p.add_argument("--show", action="store_true", help="open an interactive window instead of saving")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# 1. Image loading / filtering / thresholding  (pure numpy + matplotlib)
# --------------------------------------------------------------------------- #
def load_gray(path):
    """Read any image Pillow/matplotlib understands -> float64 grayscale in [0, 1]."""
    from matplotlib.image import imread
    img = np.asarray(imread(path))
    if img.ndim == 3:
        rgb = img[..., :3]
        if not np.issubdtype(rgb.dtype, np.floating):
            rgb = rgb.astype(np.float64) / 255.0
        gray = rgb @ np.array([0.299, 0.587, 0.114])
    else:
        gray = img.astype(np.float64)
    gray = np.clip(gray, 0.0, 1.0)
    if gray.max() > 1.0 + 1e-9:
        gray /= 255.0
    return gray


def _downscale(img, max_dim):
    if not max_dim or max(img.shape) <= max_dim:
        return img
    f = int(np.ceil(max(img.shape) / max_dim))
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def _box_blur_axis(a, r, axis):
    a = np.moveaxis(a, axis, 0)
    pad = [(r, r)] + [(0, 0)] * (a.ndim - 1)
    ap = np.pad(a, pad, mode="edge")
    c = np.cumsum(ap, axis=0)
    c = np.concatenate([np.zeros((1,) + ap.shape[1:]), c], axis=0)
    out = (c[2 * r + 1:] - c[:-(2 * r + 1)]) / (2 * r + 1)
    return np.moveaxis(out, 0, axis)


def box_blur(img, radius, passes=3):
    """Separable box blur; 3 passes approximate a Gaussian. O(N), pure numpy."""
    if radius < 1:
        return img
    for _ in range(passes):
        img = _box_blur_axis(img, radius, 0)
        img = _box_blur_axis(img, radius, 1)
    return img


def otsu_threshold(gray):
    """Otsu's method, vectorised over the 256-bin histogram."""
    q = np.clip(np.rint(gray * 255), 0, 255).astype(np.uint8)
    hist = np.bincount(q.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    denom = omega * (total - omega)
    denom[denom == 0] = 1.0
    sigma_b = (mu[-1] * omega - mu * total) ** 2 / denom
    return float(np.argmax(sigma_b)) / 255.0


def extract_contours(path, blur=3, min_len=25, max_dim=2048,
                     invert=False, auto_invert=True):
    """Return a list of (P, 2) float arrays of (x, y) contour points.

    Uses matplotlib's marching-squares tracer, so OpenCV is not required.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    gray = _downscale(load_gray(path), max_dim)
    gray = box_blur(gray, blur) if blur else gray

    thr = otsu_threshold(gray)
    mask = gray <= thr  # assume dark subject on light background by default
    if auto_invert and mask.mean() > 0.5:
        mask = ~mask
    if invert:
        mask = ~mask

    padded = np.pad(mask.astype(float), 1, constant_values=0.0)
    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    contours = ax.contour(padded, levels=[0.5]).allsegs[0]

    out = [c.astype(float) - 1.0 for c in contours if len(c) >= min_len]  # undo pad
    if not out:
        raise SystemExit("no contours found -- try adjusting --blur/--min-contour-len")
    return out


def connect_contours(contours):
    """Join contours into one open path with a greedy nearest-endpoint heuristic.

    Cheap O(K^2) cousin of the travelling-salesman problem: start from the
    largest contour and repeatedly append whichever contour has the nearest
    endpoint, oriented so its near end comes first.  Short connectors => few
    crossings.  This is the improvement the original README asks for.
    """
    contours = sorted(contours, key=len, reverse=True)
    remaining = [np.asarray(c, dtype=float) for c in contours[1:]]
    path = [np.asarray(contours[0], dtype=float)]
    end = path[0][-1]

    while remaining:
        best_i, best_rev, best_d = 0, False, np.inf
        for i, c in enumerate(remaining):
            d_start = np.hypot(*(c[0] - end))
            d_end = np.hypot(*(c[-1] - end))
            if d_start <= d_end and d_start < best_d:
                best_i, best_rev, best_d = i, False, d_start
            elif d_end < d_start and d_end < best_d:
                best_i, best_rev, best_d = i, True, d_end
        c = remaining.pop(best_i)
        if best_rev:
            c = c[::-1]
        path.append(c)
        end = c[-1]
    return np.concatenate(path, axis=0)


# --------------------------------------------------------------------------- #
# 2. Geometry helpers
# --------------------------------------------------------------------------- #
def contour_to_complex(pts, center=True, flip_y=True):
    z = pts[:, 0] + 1j * pts[:, 1]
    if flip_y:
        z = pts[:, 0] - 1j * pts[:, 1]
    if center:
        z = z - z.mean()
    return z


def resample_uniform(z, n):
    """Resample a closed complex path to n points uniformly by arc length."""
    if n <= 0 or n == len(z):
        return z
    zc = np.concatenate([z, z[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(zc)))])
    if s[-1] == 0:
        return z
    targets = np.linspace(0, s[-1], n, endpoint=False)
    return np.interp(targets, s, zc.real) + 1j * np.interp(targets, s, zc.imag)


# --------------------------------------------------------------------------- #
# 3. Fourier coefficients (FFT, not numerical integration)
# --------------------------------------------------------------------------- #
def fourier_coeffs(z, n_keep, mode="magnitude"):
    """Return (coeffs, harmonics) with the strongest circles first."""
    m = len(z)
    cn = np.fft.fft(z) / m
    freqs = np.rint(np.fft.fftfreq(m, d=1.0 / m)).astype(int)

    if mode == "range":
        n_keep = min(n_keep, (m - 1) // 2)
        idx = np.nonzero(np.abs(freqs) <= n_keep)[0]
    else:
        n_keep = min(n_keep, m)
        idx = np.argpartition(np.abs(cn), -n_keep)[-n_keep:]

    c, f = cn[idx], freqs[idx]
    order = np.argsort(-np.abs(c))  # big epicycles first: nicer reveal
    return c[order], f[order]


def precompute_centers(coeffs, harmonics, frames, turns=1.0):
    """prefix[:, k] = position after the first k epicycles, per frame.

    Shape (frames, K+1) complex; prefix[:, -1] is the pen tip.  One broadcast
    + cumsum computes every frame at once.
    """
    t = np.linspace(0.0, turns, frames, endpoint=False)
    rot = coeffs[None, :] * np.exp(1j * (harmonics[None, :] * tau * t[:, None]))
    zeros = np.zeros((frames, 1), dtype=complex)
    return np.concatenate([zeros, np.cumsum(rot, axis=1)], axis=1)


# --------------------------------------------------------------------------- #
# 4. Themes
# --------------------------------------------------------------------------- #
THEMES = {
    "midnight": dict(bg="#0d1117", fg="#f0f6fc", ring="#2d6cdf", vector="#3fb950",
                     ghost="#30363d", cmap="winter"),
    "blueprint": dict(bg="#f7f3e8", fg="#1b3a6b", ring="#7aa2c8", vector="#c0392b",
                      ghost="#d8cfba", cmap="autumn"),
    "paper": dict(bg="#ffffff", fg="#111111", ring="#bbbbbb", vector="#e07b39",
                  ghost="#e8e8e8", cmap="plasma"),
}


# --------------------------------------------------------------------------- #
# 5. Figure + per-frame update
# --------------------------------------------------------------------------- #
def build_figure(path, coeffs, harms, args, theme):
    """Return (fig, ax, update) where update(i) draws frame i."""
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import EllipseCollection, LineCollection
    from matplotlib.patches import Circle

    frames = max(2, args.frames)
    prefix = precompute_centers(coeffs, harms, frames)

    k = len(coeffs)
    radii = np.abs(coeffs)
    cx, cy = prefix[:, :k].real, prefix[:, :k].imag
    tip_x, tip_y = prefix[:, -1].real, prefix[:, -1].imag

    fig, ax = plt.subplots(figsize=(args.size, args.size), dpi=args.dpi)
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["bg"])

    ax.plot(path.real, path.imag, color=theme["ghost"], lw=1.0, alpha=0.6, zorder=1)

    max_r = float(radii.max()) if k else 1.0
    vectors = LineCollection([], colors=theme["vector"], linewidths=0.7, alpha=0.55, zorder=2)
    ax.add_collection(vectors)
    circles = None
    if not args.no_circles:
        circles = EllipseCollection(
            widths=2 * radii, heights=2 * radii, angles=np.zeros(k), units="xy",
            offsets=np.stack([cx[0], cy[0]], axis=-1), transOffset=ax.transData,
            facecolors="none", edgecolors=theme["ring"],
            linewidths=min(max(max_r / 60.0, 0.3), 1.2), alpha=0.5, zorder=2,
        )
        ax.add_collection(circles)

    if args.no_gradient:
        trail = ax.plot([], [], color=theme["fg"], lw=2.0, zorder=4)[0]
        seg_colors = None
    else:
        seg_colors = plt.get_cmap(theme["cmap"])(np.linspace(0.15, 1.0, max(frames - 1, 2)))
        trail = LineCollection([], linewidths=2.6, capstyle="round", zorder=4)
        ax.add_collection(trail)

    pen = Circle((tip_x[0], tip_y[0]), radius=max_r / 45.0, color=theme["fg"], zorder=5)
    ax.add_patch(pen)

    pad = 0.08 * max(float(np.ptp(tip_x)), float(np.ptp(tip_y)), 1.0)
    ax.set_xlim(tip_x.min() - pad, tip_x.max() + pad)
    ax.set_ylim(tip_y.min() - pad, tip_y.max() + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    tip_xy = np.stack([tip_x, tip_y], axis=-1)      # (frames, 2)
    circle_xy = np.stack([cx, cy], axis=-1)         # (frames, k, 2)

    def update(i):
        if circles is not None:
            circles.set_offsets(circle_xy[i])
        starts = circle_xy[i][:, None, :]
        ends = np.concatenate([circle_xy[i][1:], tip_xy[i][None, :]], axis=0)[:, None, :]
        vectors.set_segments(np.concatenate([starts, ends], axis=1))
        if args.no_gradient:
            trail.set_data(tip_x[: i + 1], tip_y[: i + 1])
        else:
            pts = tip_xy[: i + 1]
            if len(pts) > 1:
                trail.set_segments(np.stack([pts[:-1], pts[1:]], axis=1))
                trail.set_color(seg_colors[: len(pts) - 1])
        pen.center = (tip_x[i], tip_y[i])
        artists = [vectors, pen] + ([circles] if circles is not None else []) + [trail]
        return tuple(artists)

    return fig, ax, update


# --------------------------------------------------------------------------- #
# 6. Saving (direct-buffer writer)
# --------------------------------------------------------------------------- #
def _ffmpeg_has_encoder(name):
    if shutil.which("ffmpeg") is None:
        return False
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return False
    return name in out


def _resolved_codec(codec):
    if codec != "auto":
        return codec
    return "videotoolbox" if (sys.platform == "darwin"
                               and _ffmpeg_has_encoder("h264_videotoolbox")) else "libx264"


def _ffmpeg_input_args(codec, preset):
    if codec == "videotoolbox":
        return dict(codec="h264_videotoolbox",
                    extra_args=["-b:v", "12M", "-pix_fmt", "yuv420p"])
    return dict(codec="libx264",
                extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", preset])


def save_animation(fig, ax, update, args):
    """Save mp4/gif. Renders each frame once and streams the Agg buffer."""
    frames = max(2, args.frames)
    use_blit = not args.no_blit

    if args.output.lower().endswith(".gif"):
        from PIL import Image
        imgs = []
        for i in range(frames):
            update(i)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            imgs.append(Image.frombytes("RGBA", (buf.shape[1], buf.shape[0]),
                                        buf.tobytes()).convert("RGB"))
        imgs[0].save(args.output, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / max(args.fps, 1)), loop=0, optimize=False)
        return

    from matplotlib.animation import FFMpegWriter

    class _DirectWriter(FFMpegWriter):
        """Writes the canvas buffer straight to ffmpeg instead of savefig()."""

        def write_buffer(self):
            self._proc.stdin.write(self.fig.canvas.buffer_rgba())

        def grab_frame(self, **kwargs):
            self.fig.canvas.draw()
            self.write_buffer()

    writer = _DirectWriter(fps=args.fps, **_ffmpeg_input_args(_resolved_codec(args.codec),
                                                              args.preset))
    writer.setup(fig, args.output, dpi=args.dpi)
    try:
        if use_blit:
            # capture a background with the moving artists hidden, then only
            # redraw those artists each frame
            moving = list(update(0))
            for a in moving:
                a.set_visible(False)
            fig.canvas.draw()
            background = fig.canvas.copy_from_bbox(fig.bbox)
            for a in moving:
                a.set_visible(True)
            for i in range(frames):
                artists = update(i)
                fig.canvas.restore_region(background)
                for a in artists:
                    ax.draw_artist(a)
                fig.canvas.blit(fig.bbox)
                writer.write_buffer()
        else:
            for i in range(frames):
                update(i)
                writer.grab_frame()
    finally:
        writer.finish()


# --------------------------------------------------------------------------- #
# 7. Entry point
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    theme = THEMES[args.theme]

    contours = extract_contours(
        args.input, blur=args.blur, min_len=args.min_contour_len,
        max_dim=args.max_dim, invert=args.invert,
        auto_invert=not args.no_auto_invert,
    )
    pts = connect_contours(contours) if args.all_contours else max(contours, key=len)
    print(f"contours: {len(contours)}  |  path points: {len(pts)}")

    z = resample_uniform(contour_to_complex(pts), args.samples)
    coeffs, harms = fourier_coeffs(z, args.N, mode=args.mode)
    fig, ax, update = build_figure(z, coeffs, harms, args, theme)

    if args.show:
        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt
        FuncAnimation(fig, update, frames=max(2, args.frames),
                      interval=1000 / max(args.fps, 1), blit=not args.no_blit)
        plt.show()
        return

    ext = args.output.lower().rsplit(".", 1)[-1]
    if ext not in ("mp4", "gif", "mov", "webm", "mkv", "avi", "m4v"):
        raise SystemExit(f"unsupported output extension '.{ext}'. Use .mp4 or .gif")
    print(f"saving {args.output}  ({args.frames} frames @ {args.fps} fps, "
          f"{args.size:g}in @ {args.dpi}dpi, codec={args.codec})...")
    save_animation(fig, ax, update, args)
    print("done.")


if __name__ == "__main__":
    main()
