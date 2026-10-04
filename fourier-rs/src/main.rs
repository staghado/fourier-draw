//! Draw an image's contours with Fourier epicycles.
//!
//! Contours are rasterized with tiny-skia, converted to I420, and encoded with
//! libx264 in-process; only the compressed H.264 stream leaves the program.

use std::io::Write;
use std::mem::MaybeUninit;
use std::os::raw::{c_char, c_int};
use std::process::{Command, Stdio};
use std::ptr;

use clap::Parser;
use image::{GrayImage, Luma};
use imageproc::contours::find_contours;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use tiny_skia::{
    Color, FillRule, GradientStop, LineCap, LineJoin, LinearGradient, Paint, PathBuilder,
    Pixmap, Point, SpreadMode, Stroke, Transform,
};
use x264_sys::*;

const TAU: f32 = std::f32::consts::TAU;

#[derive(Parser, Debug)]
#[command(about = "Fast Fourier epicycle drawing (fused raster -> x264)")]
struct Args {
    #[arg(short = 'i', long)]
    input: String,
    #[arg(short = 'o', long)]
    output: String,
    #[arg(short = 'f', long, default_value_t = 300)]
    frames: u32,
    #[arg(short = 'N', long = "coeffs", default_value_t = 300)]
    n: usize,
    #[arg(long, default_value = "magnitude")]
    mode: String,
    #[arg(long)]
    all_contours: bool,
    #[arg(long, default_value_t = 25)]
    min_contour_len: usize,
    #[arg(long, default_value_t = 3.0)]
    blur: f32,
    #[arg(long, default_value_t = 2048)]
    max_dim: u32,
    #[arg(long, default_value_t = 2048)]
    samples: usize,
    #[arg(long, default_value_t = 30)]
    fps: u32,
    #[arg(long, default_value_t = 10.0)]
    size: f32,
    #[arg(long, default_value_t = 100)]
    dpi: u32,
    #[arg(long, default_value = "ultrafast")]
    preset: String,
    #[arg(long, default_value_t = 18.0)]
    crf: f32,
    #[arg(long)]
    no_circles: bool,
    #[arg(long)]
    no_gradient: bool,
    #[arg(long, default_value = "midnight")]
    theme: String,
}

#[derive(Clone, Copy)]
struct Theme {
    bg: [u8; 3],
    fg: [u8; 3],
    ring: [u8; 3],
    vector: [u8; 3],
    ghost: [u8; 3],
    grad0: [u8; 3],
    grad1: [u8; 3],
}

fn theme(name: &str) -> Theme {
    match name {
        "paper" => Theme {
            bg: [255, 255, 255], fg: [17, 17, 17], ring: [187, 187, 187],
            vector: [224, 123, 57], ghost: [232, 232, 232],
            grad0: [122, 4, 154], grad1: [242, 156, 32],
        },
        "blueprint" => Theme {
            bg: [247, 243, 232], fg: [27, 58, 107], ring: [122, 162, 200],
            vector: [192, 57, 43], ghost: [216, 207, 186],
            grad0: [180, 40, 20], grad1: [250, 200, 40],
        },
        _ => Theme {
            bg: [13, 17, 23], fg: [240, 246, 252], ring: [45, 108, 223],
            vector: [63, 185, 80], ghost: [48, 54, 61],
            grad0: [0, 38, 236], grad1: [0, 255, 128],
        },
    }
}

/// Real RGB -> tiny-skia color (straightforward now that we convert to YUV
/// ourselves rather than handing BGRA to x264).
fn col(c: [u8; 3]) -> Color {
    Color::from_rgba8(c[0], c[1], c[2], 255)
}

// ---------------------------------------------------------------------------
// image -> contour
// ---------------------------------------------------------------------------
fn otsu(hist: &[u64; 256], total: u64) -> u8 {
    let sum: f64 = (0..256).map(|i| i as f64 * hist[i] as f64).sum();
    let (mut wb, mut sb, mut best, mut thr) = (0.0f64, 0.0f64, -1.0f64, 0u8);
    for t in 0..256 {
        wb += hist[t] as f64;
        if wb == 0.0 {
            continue;
        }
        let wf = total as f64 - wb;
        if wf == 0.0 {
            break;
        }
        sb += t as f64 * hist[t] as f64;
        let mb = sb / wb;
        let mf = (sum - sb) / wf;
        let between = wb * wf * (mb - mf) * (mb - mf);
        if between > best {
            best = between;
            thr = t as u8;
        }
    }
    thr
}

fn load_mask(path: &str, blur: f32, max_dim: u32) -> GrayImage {
    let img = image::open(path).expect("could not open image").to_luma8();
    let img = if max_dim > 0 && img.width().max(img.height()) > max_dim {
        let s = max_dim as f32 / img.width().max(img.height()) as f32;
        let nw = ((img.width() as f32 * s) as u32).max(1);
        let nh = ((img.height() as f32 * s) as u32).max(1);
        image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let img = if blur > 0.0 { image::imageops::blur(&img, blur) } else { img };

    let mut hist = [0u64; 256];
    for p in img.pixels() {
        hist[p.0[0] as usize] += 1;
    }
    let total = (img.width() as u64) * (img.height() as u64);
    let thr = otsu(&hist, total);

    // default: dark pixels are the subject
    let fg = |v: u8| v <= thr;
    let count = img.pixels().filter(|p| fg(p.0[0])).count() as u64;
    let invert = count * 2 > total; // subject is the minority

    let mut out = GrayImage::new(img.width(), img.height());
    for (x, y, p) in img.enumerate_pixels() {
        let is_fg = if invert { !fg(p.0[0]) } else { fg(p.0[0]) };
        out.put_pixel(x, y, Luma([if is_fg { 255 } else { 0 }]));
    }
    out
}

fn contours_from(mask: &GrayImage, min_len: usize, all: bool) -> Vec<Vec<(f32, f32)>> {
    let found = find_contours::<i32>(mask);
    let mut cs: Vec<Vec<(f32, f32)>> = found
        .into_iter()
        .filter(|c| c.points.len() >= min_len)
        .map(|c| c.points.into_iter().map(|p| (p.x as f32, p.y as f32)).collect())
        .collect();
    if cs.is_empty() {
        panic!("no contours found");
    }
    if all {
        cs
    } else {
        vec![cs.swap_remove(
            (0..cs.len()).max_by_key(|&i| cs[i].len()).unwrap(),
        )]
    }
}

fn connect(mut cs: Vec<Vec<(f32, f32)>>) -> Vec<(f32, f32)> {
    cs.sort_by(|a, b| b.len().cmp(&a.len()));
    let mut path = cs[0].clone();
    let mut end = *path.last().unwrap();
    let mut rest = cs.split_off(1);
    while !rest.is_empty() {
        let mut best = (0usize, false, f32::INFINITY);
        for (i, c) in rest.iter().enumerate() {
            let ds = ((c[0].0 - end.0).powi(2) + (c[0].1 - end.1).powi(2)).sqrt();
            let de = ((c[c.len() - 1].0 - end.0).powi(2) + (c[c.len() - 1].1 - end.1).powi(2)).sqrt();
            let (d, rev) = if ds <= de { (ds, false) } else { (de, true) };
            if d < best.2 {
                best = (i, rev, d);
            }
        }
        let mut c = rest.remove(best.0);
        if best.1 {
            c.reverse();
        }
        end = *c.last().unwrap();
        path.extend(c);
    }
    path
}

// ---------------------------------------------------------------------------
// geometry + FFT
// ---------------------------------------------------------------------------
fn to_complex(pts: &[(f32, f32)]) -> Vec<Complex32> {
    let n = pts.len() as f32;
    let mx: f32 = pts.iter().map(|p| p.0).sum::<f32>() / n;
    let my: f32 = pts.iter().map(|p| p.1).sum::<f32>() / n;
    // flip y so the drawing is upright, then center on the origin
    pts.iter().map(|p| Complex32::new(p.0 - mx, my - p.1)).collect()
}

fn resample(z: &[Complex32], n: usize) -> Vec<Complex32> {
    let m = z.len();
    if n == 0 || n == m {
        return z.to_vec();
    }
    let mut cum = vec![0f32; m + 1];
    for i in 0..m {
        cum[i + 1] = cum[i] + (z[(i + 1) % m] - z[i]).norm();
    }
    let total = cum[m];
    if total == 0.0 {
        return z.to_vec();
    }
    let mut out = Vec::with_capacity(n);
    let mut j = 0usize;
    for k in 0..n {
        let target = total * k as f32 / n as f32;
        while j + 1 < cum.len() && cum[j + 1] < target {
            j += 1;
        }
        let a = z[j];
        let b = z[(j + 1) % m];
        let seg = cum[j + 1] - cum[j];
        let f = if seg > 0.0 { (target - cum[j]) / seg } else { 0.0 };
        out.push(a + (b - a) * f);
    }
    out
}

fn fourier(z: &[Complex32], nkeep: usize, mode: &str) -> Vec<(Complex32, i32)> {
    let m = z.len();
    let mut buf = z.to_vec();
    FftPlanner::<f32>::new().plan_fft_forward(m).process(&mut buf);
    let inv = 1.0 / m as f32;
    let items: Vec<(Complex32, i32)> = (0..m)
        .map(|i| {
            let n = if i <= m / 2 { i as i32 } else { i as i32 - m as i32 };
            (buf[i] * inv, n)
        })
        .collect();

    let mut sel: Vec<(Complex32, i32)> = if mode == "range" {
        items.iter().copied().filter(|(_, n)| n.abs() <= nkeep as i32).collect()
    } else {
        let k = nkeep.min(m);
        let mut mags: Vec<(f32, usize)> = items.iter().enumerate().map(|(i, c)| (c.0.norm(), i)).collect();
        mags.select_nth_unstable_by(k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
        mags[..k].iter().map(|x| items[x.1]).collect()
    };
    sel.sort_by(|a, b| b.0.norm().partial_cmp(&a.0.norm()).unwrap());
    sel
}

// ---------------------------------------------------------------------------
// tiny-skia raster state
// ---------------------------------------------------------------------------
struct Raster {
    s: f32,
    ox: f32,
    oy: f32,
    coeffs: Vec<(Complex32, i32)>,
    radii: Vec<f32>,
    static_bg: Pixmap,
    theme: Theme,
    no_circles: bool,
    gradient: bool,
    frames: u32,
    gx0: f32,
    gy0: f32,
    gx1: f32,
    gy1: f32,
    bg_y: u8,
    bg_u: u8,
    bg_v: u8,
    bg_px: u32,
}

/// A planar 4:2:0 frame.
struct I420 {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    w: usize,
    h: usize,
}

impl I420 {
    fn new(w: usize, h: usize) -> Self {
        I420 { y: vec![16; w * h], u: vec![128; w * h / 4], v: vec![128; w * h / 4], w, h }
    }
}

#[inline]
fn yuv(r: i32, g: i32, b: i32) -> (u8, u8, u8) {
    // BT.709 limited range (HD), matching the VUI flags we write
    let y = (((47 * r + 157 * g + 16 * b + 128) >> 8) + 16).clamp(0, 255) as u8;
    let u = (((-26 * r - 87 * g + 112 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
    let v = (((112 * r - 102 * g - 10 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
    (y, u, v)
}

impl Raster {
    fn sx(&self, x: f32) -> f32 {
        self.ox + x * self.s
    }
    fn sy(&self, y: f32) -> f32 {
        self.oy - y * self.s
    }

    /// Pixel-space epicycle centres and pen tip for every frame.
    fn precompute(&self) -> (Vec<Vec<(f32, f32)>>, Vec<(f32, f32)>) {
        let k = self.coeffs.len();
        let n = self.frames as usize;
        let mut centers = Vec::with_capacity(n);
        let mut tips = Vec::with_capacity(n);
        for i in 0..self.frames {
            let a0 = i as f32 / self.frames as f32 * TAU;
            let (mut x, mut y) = (0f32, 0f32);
            let mut cs = Vec::with_capacity(k);
            for j in 0..k {
                cs.push((self.sx(x), self.sy(y)));
                let (c, m) = self.coeffs[j];
                let (sn, co) = (m as f32 * a0).sin_cos();
                x += c.re * co - c.im * sn;
                y += c.re * sn + c.im * co;
            }
            tips.push((self.sx(x), self.sy(y)));
            centers.push(cs);
        }
        (centers, tips)
    }

    fn render(&self, pm: &mut Pixmap, centers: &[(f32, f32)], tip: (f32, f32), trail: &[(f32, f32)]) {
        // background + ghost are static: a contiguous memcpy beats re-stroking
        // the 2000-segment ghost path every frame
        pm.data_mut().copy_from_slice(self.static_bg.data());
        let identity = Transform::identity();

        if !self.no_circles {
            let mut circles = PathBuilder::new();
            let mut vecs = PathBuilder::new();
            for j in 0..centers.len() {
                let (px, py) = centers[j];
                circles.push_circle(px, py, self.radii[j] * self.s);
                let (ex, ey) = if j + 1 < centers.len() { centers[j + 1] } else { tip };
                vecs.move_to(px, py);
                vecs.line_to(ex, ey);
            }
            let mut pr = Paint::default();
            pr.set_color(col(self.theme.ring));
            pr.anti_alias = true;
            if let Some(p) = circles.finish() {
                pm.stroke_path(&p, &pr, &Stroke { width: 0.8, ..Default::default() }, identity, None);
            }
            let mut pv = Paint::default();
            pv.set_color(col(self.theme.vector));
            pv.anti_alias = true;
            if let Some(p) = vecs.finish() {
                pm.stroke_path(&p, &pv, &Stroke { width: 0.7, ..Default::default() }, identity, None);
            }
        }

        // glowing gradient trail
        if trail.len() > 1 {
            let mut tb = PathBuilder::new();
            tb.move_to(trail[0].0, trail[0].1);
            for p in &trail[1..] {
                tb.line_to(p.0, p.1);
            }
            if let Some(path) = tb.finish() {
                let mut paint = Paint::default();
                paint.anti_alias = true;
                if self.gradient {
                    let grad = LinearGradient::new(
                        Point::from_xy(self.gx0, self.gy0),
                        Point::from_xy(self.gx1, self.gy1),
                        vec![
                            GradientStop::new(0.0, col(self.theme.grad0)),
                            GradientStop::new(1.0, col(self.theme.grad1)),
                        ],
                        SpreadMode::Pad,
                        identity,
                    );
                    match grad {
                        Some(g) => paint.shader = g,
                        None => paint.set_color(col(self.theme.fg)),
                    }
                } else {
                    paint.set_color(col(self.theme.fg));
                }
                let stroke = Stroke {
                    width: 2.6,
                    line_cap: LineCap::Round,
                    line_join: LineJoin::Round,
                    ..Default::default()
                };
                pm.stroke_path(&path, &paint, &stroke, identity, None);
            }
        }

        // pen
        let mut pen = PathBuilder::new();
        pen.push_circle(tip.0, tip.1, 2.6);
        if let Some(path) = pen.finish() {
            let mut paint = Paint::default();
            paint.set_color(col(self.theme.fg));
            paint.anti_alias = true;
            pm.fill_path(&path, &paint, FillRule::Winding, identity, None);
        }
    }

    /// Convert the rendered RGBA buffer to I420. Pixels equal to the flat
    /// background are skipped, so only the (few) drawn pixels are converted.
    fn to_i420(&self, rgba: &[u8], out: &mut I420) {
        let (w, h) = (out.w, out.h);
        out.y.fill(self.bg_y);
        out.u.fill(self.bg_u);
        out.v.fill(self.bg_v);
        let bg = self.bg_px;

        for (i, px) in rgba.chunks_exact(4).enumerate() {
            if u32::from_ne_bytes([px[0], px[1], px[2], px[3]]) != bg {
                out.y[i] = yuv(px[0] as i32, px[1] as i32, px[2] as i32).0;
            }
        }

        let cw = w / 2;
        for cy in 0..h / 2 {
            for cx in 0..cw {
                let a = ((cy * 2) * w + cx * 2) * 4;
                let b = a + 4;
                let c = ((cy * 2 + 1) * w + cx * 2) * 4;
                let d = c + 4;
                let v0 = u32::from_ne_bytes([rgba[a], rgba[a + 1], rgba[a + 2], rgba[a + 3]]);
                let v1 = u32::from_ne_bytes([rgba[b], rgba[b + 1], rgba[b + 2], rgba[b + 3]]);
                let v2 = u32::from_ne_bytes([rgba[c], rgba[c + 1], rgba[c + 2], rgba[c + 3]]);
                let v3 = u32::from_ne_bytes([rgba[d], rgba[d + 1], rgba[d + 2], rgba[d + 3]]);
                if v0 == bg && v1 == bg && v2 == bg && v3 == bg {
                    continue;
                }
                let r = (rgba[a] as i32 + rgba[b] as i32 + rgba[c] as i32 + rgba[d] as i32 + 2) >> 2;
                let g = (rgba[a + 1] as i32 + rgba[b + 1] as i32 + rgba[c + 1] as i32 + rgba[d + 1] as i32 + 2) >> 2;
                let bl = (rgba[a + 2] as i32 + rgba[b + 2] as i32 + rgba[c + 2] as i32 + rgba[d + 2] as i32 + 2) >> 2;
                let (_, u, v) = yuv(r, g, bl);
                let j = cy * cw + cx;
                out.u[j] = u;
                out.v[j] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// x264
// ---------------------------------------------------------------------------
fn append_nals(out: &mut Vec<u8>, nal: *mut x264_nal_t, count: c_int) {
    unsafe {
        for i in 0..count as isize {
            let n = &*nal.offset(i);
            out.extend_from_slice(std::slice::from_raw_parts(n.p_payload, n.i_payload as usize));
        }
    }
}

struct Encoder {
    raw: *mut x264_t,
}

// The raw x264 context is only ever used from a single thread at a time.
unsafe impl Send for Encoder {}

impl Encoder {
    fn new(w: u32, h: u32, fps: u32, preset: &str, crf: f32) -> Self {
        unsafe {
            let mut p = MaybeUninit::<x264_param_t>::uninit();
            let preset = std::ffi::CString::new(preset).unwrap();
            x264_param_default_preset(
                p.as_mut_ptr(),
                preset.as_ptr(),
                b"\0".as_ptr() as *const c_char,
            );
            let mut p = p.assume_init();
            // Encoder target is I420 (p.i_csp stays at its default); the input
            // picture below declares BGRA, so x264 converts BGRA -> yuv420p.
            p.i_width = w as c_int;
            p.i_height = h as c_int;
            p.i_fps_num = fps;
            p.i_fps_den = 1;
            p.rc.i_rc_method = X264_RC_CRF as c_int;
            p.rc.f_rf_constant = crf;
            // tag the stream BT.709 limited-range to match our conversion
            p.vui.i_colorprim = 1;
            p.vui.i_transfer = 1;
            p.vui.i_colmatrix = 1;
            p.vui.b_fullrange = 0;
            let raw = x264_encoder_open(&mut p);
            assert!(!raw.is_null(), "x264_encoder_open failed");
            Encoder { raw }
        }
    }

    fn headers(&mut self) -> Vec<u8> {
        unsafe {
            let (mut nal, mut count) = (ptr::null_mut(), 0);
            x264_encoder_headers(self.raw, &mut nal, &mut count);
            let mut out = Vec::new();
            append_nals(&mut out, nal, count);
            out
        }
    }

    fn encode(&mut self, pts: i64, img: &I420, w: u32) -> Vec<u8> {
        unsafe {
            let mut pic = MaybeUninit::<x264_picture_t>::zeroed().assume_init();
            x264_picture_init(&mut pic);
            pic.i_pts = pts;
            pic.img.i_csp = X264_CSP_I420 as c_int;
            pic.img.i_plane = 3;
            pic.img.i_stride[0] = w as c_int;
            pic.img.i_stride[1] = (w / 2) as c_int;
            pic.img.i_stride[2] = (w / 2) as c_int;
            pic.img.plane[0] = img.y.as_ptr() as *mut u8;
            pic.img.plane[1] = img.u.as_ptr() as *mut u8;
            pic.img.plane[2] = img.v.as_ptr() as *mut u8;

            let (mut nal, mut count) = (ptr::null_mut(), 0);
            let mut out_pic = MaybeUninit::<x264_picture_t>::zeroed().assume_init();
            let ret = x264_encoder_encode(self.raw, &mut nal, &mut count, &mut pic, &mut out_pic);
            assert!(ret >= 0, "x264_encoder_encode failed");
            let mut out = Vec::new();
            append_nals(&mut out, nal, count);
            out
        }
    }

    fn flush(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        unsafe {
            while x264_encoder_delayed_frames(self.raw) > 0 {
                let (mut nal, mut count) = (ptr::null_mut(), 0);
                let mut out_pic = MaybeUninit::<x264_picture_t>::zeroed().assume_init();
                x264_encoder_encode(self.raw, &mut nal, &mut count, ptr::null_mut(), &mut out_pic);
                append_nals(&mut out, nal, count);
            }
        }
        out
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        unsafe { x264_encoder_close(self.raw) };
    }
}

/// Streams encoded NALs either to a raw `.h264` file or straight into ffmpeg,
/// which only remuxes (`-c copy`) — never re-encodes.
enum Sink {
    File(std::fs::File),
    Pipe { child: std::process::Child, stdin: std::process::ChildStdin },
}

impl Sink {
    fn new(output: &str, fps: u32) -> Sink {
        if output.ends_with(".h264") {
            Sink::File(std::fs::File::create(output).unwrap())
        } else {
            let mut child = Command::new("ffmpeg")
                .args([
                    "-y", "-loglevel", "error", "-f", "h264", "-i", "pipe:0",
                    "-c:v", "copy", "-r", &fps.to_string(), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", output,
                ])
                .stdin(Stdio::piped())
                .spawn()
                .expect("failed to spawn ffmpeg (needed only for muxing)");
            let stdin = child.stdin.take().unwrap();
            Sink::Pipe { child, stdin }
        }
    }

    fn write(&mut self, data: &[u8]) {
        match self {
            Sink::File(f) => f.write_all(data).unwrap(),
            Sink::Pipe { stdin, .. } => stdin.write_all(data).unwrap(),
        }
    }

    fn finish(self) {
        match self {
            Sink::File(mut f) => f.flush().unwrap(),
            Sink::Pipe { mut child, stdin } => {
                drop(stdin);
                child.wait().unwrap();
            }
        }
    }
}

// ---------------------------------------------------------------------------
fn main() {
    let args = Args::parse();
    let mut w = (args.size * args.dpi as f32) as u32;
    let mut h = w;
    w += w % 2; // x264's 4:2:0 conversion wants even dimensions
    h += h % 2;

    let mask = load_mask(&args.input, args.blur, args.max_dim);
    let raw = contours_from(&mask, args.min_contour_len, args.all_contours);
    let pts = if args.all_contours { connect(raw) } else { raw[0].clone() };
    let z = resample(&to_complex(&pts), args.samples);
    let coeffs = fourier(&z, args.n, &args.mode);
    println!("path points: {}  |  epicycles: {}", z.len(), coeffs.len());

    let theme = theme(&args.theme);
    let radii: Vec<f32> = coeffs.iter().map(|(c, _)| c.norm()).collect();

    // world bounds -> pixel transform
    let (mut minx, mut miny, mut maxx, mut maxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in &z {
        minx = minx.min(p.re);
        maxx = maxx.max(p.re);
        miny = miny.min(p.im);
        maxy = maxy.max(p.im);
    }
    let dw = (maxx - minx).max(1e-6);
    let dh = (maxy - miny).max(1e-6);
    let s = (w as f32 / dw).min(h as f32 / dh) * 0.9;
    let ox = w as f32 / 2.0 - (minx + maxx) / 2.0 * s;
    let oy = h as f32 / 2.0 + (miny + maxy) / 2.0 * s;

    // precompute the ghost path
    let mut gb = PathBuilder::new();
    gb.move_to(ox + z[0].re * s, oy - z[0].im * s);
    for p in &z[1..] {
        gb.line_to(ox + p.re * s, oy - p.im * s);
    }
    gb.close();
    let ghost = gb.finish().unwrap();

    // static layer: background + ghost, drawn once
    let mut static_bg = Pixmap::new(w, h).unwrap();
    static_bg.fill(col(theme.bg));
    {
        let mut paint = Paint::default();
        paint.set_color(col(theme.ghost));
        paint.anti_alias = true;
        static_bg.stroke_path(
            &ghost, &paint, &Stroke { width: 1.0, ..Default::default() },
            Transform::identity(), None,
        );
    }

    let (bg_y, bg_u, bg_v) = yuv(theme.bg[0] as i32, theme.bg[1] as i32, theme.bg[2] as i32);
    let bg_px = u32::from_ne_bytes([theme.bg[0], theme.bg[1], theme.bg[2], 255]);
    let raster = Raster {
        s, ox, oy,
        coeffs,
        radii,
        static_bg,
        theme,
        no_circles: args.no_circles,
        gradient: !args.no_gradient,
        frames: args.frames,
        gx0: ox + minx * s,
        gy0: oy - maxy * s,
        gx1: ox + maxx * s,
        gy1: oy - miny * s,
        bg_y,
        bg_u,
        bg_v,
        bg_px,
    };

    let (centers, tips) = raster.precompute();

    let mut enc = Encoder::new(w, h, args.fps, &args.preset, args.crf);
    let mut sink = Sink::new(&args.output, args.fps);
    sink.write(&enc.headers());

    println!("rendering {} frames @ {}x{} ...", args.frames, w, h);
    let t0 = std::time::Instant::now();
    let n = args.frames as usize;
    let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4).min(n).max(1);
    let mut pool: Vec<I420> = (0..threads).map(|_| I420::new(w as usize, h as usize)).collect();
    let mut scratch: Vec<Pixmap> = (0..threads).map(|_| Pixmap::new(w, h).unwrap()).collect();
    let mut i = 0usize;
    while i < n {
        let end = (i + threads).min(n);
        std::thread::scope(|sc| {
            for (j, (pm, buf)) in scratch[..end - i].iter_mut().zip(pool[..end - i].iter_mut()).enumerate() {
                let idx = i + j;
                let (centers, tips, raster) = (&centers, &tips, &raster);
                sc.spawn(move || {
                    raster.render(pm, &centers[idx], tips[idx], &tips[..=idx]);
                    raster.to_i420(pm.data(), buf);
                });
            }
        });
        for j in 0..(end - i) {
            sink.write(&enc.encode((i + j) as i64, &pool[j], w));
        }
        i = end;
    }
    sink.write(&enc.flush());
    sink.finish();
    println!("done in {:.2}s -> {}", t0.elapsed().as_secs_f32(), args.output);
}
