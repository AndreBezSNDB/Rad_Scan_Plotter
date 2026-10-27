"""
beamcore.py - analysis core for Beam Lab (no UI code).

Loads 2-D hydrophone plane scans (.mat with PDx/PDy/PDz + VPP/VMAX/VMIN datastores) and
computes everything the app shows: display field, focal point, dB widths, beam-axis fit,
focal depth along the axis, width / on-axis signal vs depth, and a PNG export.

Analysis notes
  * The horizontal/depth ranges CROP the analysis region: normalization, focus search,
    dB boundary, widths and focal depth all use only the selected data.
  * Pressure = V / (sensitivity x gain); sensitivity in mV/MPa, gain in dB. No hydrophone
    frequency-response or spatial-averaging correction is applied.
  * Widths are measured at the dB level (default -3 dB = 0.708 x pressure) of each depth's
    own peak, linearly interpolated between grid points.
  * The beam axis is a straight line through (va, xa) and (vb, xb). Auto-fit fits it through
    the per-depth beam centroids. Deskew shears each depth row so the axis sits at 0
    (a shear, not a rotation).
"""
from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom

AXIS_NAMES = ("X", "Y", "Z")
UPSCALE_STEPS = (1, 2, 4, 8, 16)
DB_FLOOR = -60.0


class Unit:
    NORM = "Normalized (linear)"
    DB = "Normalized (dB)"
    VPP = "Vpp (mV)"
    PPP = "Peak-to-peak pressure (MPa)"
    PPOS = "Peak positive pressure p+ (MPa)"
    PNEG = "Peak negative pressure p− (MPa)"


# ============================================================================ loading
def _scalar(d: dict, key: str, default=None):
    if key not in d:
        return default
    a = np.asarray(d[key]).ravel()
    if a.size == 0:
        return default
    x = a[0]
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return default
        x = x.ravel()[0]
    return x.item() if hasattr(x, "item") else x


def _source(src):
    return io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src


def is_scan_file(src) -> bool:
    try:
        return any(n == "VPPDatastore" for n, _, _ in sio.whosmat(_source(src)))
    except Exception:  # noqa: BLE001
        return False


def content_id(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:12]


@dataclass
class Scan:
    """One 2-D plane of a hydrophone scan, sorted ascending on both axes."""
    name: str
    h: np.ndarray
    v: np.ndarray
    h_ax: int
    v_ax: int
    fixed_ax: int
    fixed_val: float
    vpp: np.ndarray
    vmax: Optional[np.ndarray]
    vmin: Optional[np.ndarray]
    device: str = ""
    freq: Optional[float] = None
    sens: float = 1.0
    gain: float = 0.0
    navg: Optional[int] = None

    @classmethod
    def load(cls, src, name: str = "") -> "Scan":
        d = sio.loadmat(_source(src))
        if "VPPDatastore" not in d:
            raise ValueError("This .mat file has no VPPDatastore variable — is it a beam scan?")
        vpp3 = np.asarray(d["VPPDatastore"], float)
        if vpp3.ndim != 3:
            raise ValueError(f"VPPDatastore should be 3-D (Nx, Ny, Nz); got shape {vpp3.shape}.")
        shape = vpp3.shape
        pos = []
        for i, key in enumerate(("PDx", "PDy", "PDz")):
            p = np.asarray(d[key], float).ravel() if key in d else np.array([])
            pos.append(p if p.size == shape[i] else np.arange(shape[i], dtype=float))
        varying = [i for i in range(3) if shape[i] > 1]
        if len(varying) != 2:
            raise ValueError(f"Expected a 2-D plane scan; grid shape is {shape}.")
        fixed = ({0, 1, 2} - set(varying)).pop()
        h_ax, v_ax = varying
        oh, ov = np.argsort(pos[h_ax]), np.argsort(pos[v_ax])

        def plane(key):
            if key not in d:
                return None
            a = np.asarray(d[key], float)
            if a.shape != shape:
                return None
            return np.take(a, 0, axis=fixed).T[np.ix_(ov, oh)]

        return cls(name=name or (isinstance(src, str) and os.path.basename(src)) or "scan.mat",
                   h=pos[h_ax][oh], v=pos[v_ax][ov], h_ax=h_ax, v_ax=v_ax, fixed_ax=fixed,
                   fixed_val=float(pos[fixed][0]), vpp=plane("VPPDatastore"), vmax=plane("VMAXDatastore"),
                   vmin=plane("VMINDatastore"), device=str(_scalar(d, "DeviceID", "") or ""),
                   freq=_scalar(d, "USfreq"), sens=float(_scalar(d, "HydrophoneSens", 1.0) or 1.0),
                   gain=float(_scalar(d, "HydrophoneGain", 0.0) or 0.0), navg=_scalar(d, "ScopeNAverages"))

    @property
    def hname(self):
        return AXIS_NAMES[self.h_ax]

    @property
    def vname(self):
        return AXIS_NAMES[self.v_ax]

    @property
    def plane(self):
        return self.hname + self.vname

    @property
    def depth_is_z(self):
        return self.v_ax == 2

    @property
    def is_yz(self):
        return self.h_ax == 1 and self.v_ax == 2

    @property
    def title(self):
        return self.device or os.path.splitext(self.name)[0]

    @property
    def dh(self):
        return float(np.median(np.diff(self.h))) if self.h.size > 1 else 0.0

    @property
    def dv(self):
        return float(np.median(np.diff(self.v))) if self.v.size > 1 else 0.0

    def units(self) -> list:
        u = [Unit.NORM, Unit.DB, Unit.VPP, Unit.PPP]
        if self.vmax is not None:
            u.append(Unit.PPOS)
        if self.vmin is not None:
            u.append(Unit.PNEG)
        return u


# ============================================================================ settings
@dataclass
class Settings:
    unit: str = Unit.PPP
    sens: float = 1.0                 # mV/MPa
    gain: float = 0.0                 # dB
    upscale: int = 4
    h_lo: float = -np.inf             # horizontal crop (mm, snapped to grid)
    h_hi: float = np.inf
    v_lo: float = -np.inf             # depth crop (mm)
    v_hi: float = np.inf
    level_db: float = -3.0
    depthnorm: bool = False
    deskew: bool = False
    axis_va: float = 0.0              # beam axis: horizontal position xa at depth va, xb at vb
    axis_vb: float = 1.0
    axis_xa: float = 0.0
    axis_xb: float = 0.0

    @property
    def volts_per_mpa(self) -> float:
        return self.sens * 1e-3 * 10 ** (self.gain / 20.0)

    def axis_h(self, v):
        if self.axis_vb == self.axis_va:
            return np.full_like(np.asarray(v, float), self.axis_xa)
        return self.axis_xa + (self.axis_xb - self.axis_xa) * (np.asarray(v, float) - self.axis_va) / (
            self.axis_vb - self.axis_va)

    @property
    def tilt_deg(self) -> float:
        if self.axis_vb == self.axis_va:
            return 0.0
        return float(np.degrees(np.arctan((self.axis_xb - self.axis_xa) / (self.axis_vb - self.axis_va))))

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        s = cls()
        for k, v in d.items():
            if hasattr(s, k) and v is not None:
                setattr(s, k, type(getattr(s, k))(v))
        return s


# ============================================================================ analysis
def cell_edges(c: np.ndarray) -> np.ndarray:
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    m = (c[1:] + c[:-1]) / 2
    return np.r_[c[0] - (m[0] - c[0]), m, c[-1] + (c[-1] - m[-1])]


def crossings(x, y, i_pk: int, frac: float):
    """Interpolated x either side of y[i_pk] where y first drops below frac*peak (None = never)."""
    thr = y[i_pk] * frac

    def walk(step):
        j = i_pk
        while 0 <= j + step < len(y):
            k = j + step
            if np.isnan(y[k]):
                return None
            if y[k] < thr:
                return float(x[j] + (thr - y[j]) * (x[k] - x[j]) / (y[k] - y[j]))
            j = k
        return None

    return walk(-1), walk(+1)


def fit_beam_axis(h, v, lin, thresh: float = 0.5):
    """Weighted straight-line fit through per-depth beam centroids -> (x at v[0], x at v[-1])."""
    rows = []
    gmax = np.nanmax(lin)
    for i in range(lin.shape[0]):
        r = np.nan_to_num(lin[i])
        m = r.max()
        if m <= 0.1 * gmax:
            continue
        w = np.where(r >= thresh * m, r, 0.0)
        rows.append((v[i], float((w * h).sum() / w.sum()), m))
    if len(rows) < 2:
        c = float(h[np.nanargmax(np.nanmax(lin, axis=0))])
        return c, c
    a = np.array(rows)
    if np.ptp(a[:, 0]) == 0:
        return float(a[0, 1]), float(a[0, 1])
    slope, icpt = np.polyfit(a[:, 0], a[:, 1], 1, w=np.sqrt(a[:, 2]))
    return float(icpt + slope * v[0]), float(icpt + slope * v[-1])


def axial_profile(h, v, lin, x0, x1, n=400):
    vs = np.linspace(v[0], v[-1], n)
    hs = x0 + (x1 - x0) * (vs - v[0]) / (v[-1] - v[0]) if v[-1] != v[0] else np.full(n, x0)
    f = RegularGridInterpolator((v, h), lin, bounds_error=False, fill_value=np.nan)
    return vs, hs, f(np.c_[vs, hs])


def depth_widths(h, lin, frac):
    out = np.full(lin.shape[0], np.nan)
    for i, row in enumerate(lin):
        if np.all(np.isnan(row)):
            continue
        j = int(np.nanargmax(row))
        a, b = crossings(h, row, j, frac)
        if a is not None and b is not None:
            out[i] = b - a
    return out


def full_field(scan: Scan, st: Settings, corrected: Optional[bool] = None):
    """Whole-scan field in the chosen scale (absolute: mV or MPa), optionally deskewed."""
    u = st.unit
    if u in (Unit.NORM, Unit.DB, Unit.VPP):
        a, unit = scan.vpp * 1e3, "mV"
    else:
        src = {Unit.PPP: scan.vpp, Unit.PPOS: scan.vmax, Unit.PNEG: scan.vmin}[u]
        a, unit = (-src if u == Unit.PNEG else src) / st.volts_per_mpa, "MPa"
    if st.deskew if corrected is None else corrected:
        out = np.full_like(a, np.nan)
        for i, vi in enumerate(scan.v):
            out[i] = np.interp(scan.h + float(st.axis_h(vi)), scan.h, a[i], left=np.nan, right=np.nan)
        a = out
    return a, unit


def crop_slices(scan: Scan, st: Settings):
    hi = np.where((scan.h >= st.h_lo - 1e-9) & (scan.h <= st.h_hi + 1e-9))[0]
    vi = np.where((scan.v >= st.v_lo - 1e-9) & (scan.v <= st.v_hi + 1e-9))[0]
    if hi.size == 0:
        hi = np.arange(scan.h.size)
    if vi.size == 0:
        vi = np.arange(scan.v.size)
    return slice(hi[0], hi[-1] + 1), slice(vi[0], vi[-1] + 1)


def view(scan: Scan, st: Settings, corrected: Optional[bool] = None):
    a, unit = full_field(scan, st, corrected)
    hs, vs = crop_slices(scan, st)
    return scan.h[hs], scan.v[vs], a[vs, hs], unit


def display(lin, st: Settings):
    """-> (norm, disp, label)."""
    if st.depthnorm:
        ref = np.nanmax(lin, axis=1, keepdims=True)
        ref = np.where(ref > 0, ref, np.nan)
    else:
        ref = np.nanmax(lin)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = lin / ref
    if st.unit == Unit.DB:
        return norm, 20 * np.log10(np.clip(norm, 10 ** (DB_FLOOR / 20), None)), \
            "dB re lateral max" if st.depthnorm else "dB re max"
    if st.unit == Unit.NORM or st.depthnorm:
        return norm, norm, "Laterally normalized" if st.depthnorm else "Normalized (linear)"
    return norm, lin, st.unit


def upscaled(h, v, lin, f: int):
    if f == 1 or lin.shape[0] < 2 or lin.shape[1] < 2:
        return h, v, lin
    up = np.clip(zoom(np.nan_to_num(lin), f, order=3, mode="nearest"), 0, None)
    mask = np.isnan(lin)
    if mask.any():
        up[zoom(mask.astype(float), f, order=0, mode="nearest") > 0.5] = np.nan
    return np.linspace(h[0], h[-1], up.shape[1]), np.linspace(v[0], v[-1], up.shape[0]), up


def colour_bounds(scan: Scan, st: Settings):
    """Colour-scale range from the whole scan (so cropping doesn't move the slider bounds)."""
    a, _ = full_field(scan, st)
    _, disp, _ = display(a, st)
    with np.errstate(all="ignore"):
        lo, hi = float(np.nanmin(disp)), float(np.nanmax(disp))
    if st.unit != Unit.DB:
        lo = min(lo, 0.0)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = 0.0, 1.0
    return lo, hi


def autofit(scan: Scan, st: Settings):
    """Fit the beam axis on the current (uncorrected) crop -> (va, vb, xa, xb)."""
    h, v, lin, _ = view(scan, st, corrected=False)
    if lin.shape[0] < 2:
        return float(v[0]), float(v[0]) + 1.0, float(h[lin.shape[1] // 2]), float(h[lin.shape[1] // 2])
    xa, xb = fit_beam_axis(h, v, lin)
    return float(v[0]), float(v[-1]), xa, xb


@dataclass
class Result:
    h: np.ndarray
    v: np.ndarray
    lin: np.ndarray
    unit: str
    hu: np.ndarray
    vu: np.ndarray
    linu: np.ndarray
    normu: np.ndarray
    dispu: np.ndarray
    label: str
    frac: float
    lvl: str
    hn: str
    fh: float
    fv: float
    fval: float
    fval_mpa: Optional[float]
    wa: Optional[float]
    wb: Optional[float]
    ax_x: tuple
    depth: Optional[dict]
    widths: np.ndarray
    amp: np.ndarray
    alab: str
    pv: Optional[np.ndarray] = None          # depth coordinates of widths/amp (upsampled grid if upscaled)
    meas_up: int = 1                         # upscale factor the measurements were made on
    lines: list = field(default_factory=list)

    @property
    def width(self):
        return None if self.wa is None or self.wb is None else self.wb - self.wa

    @property
    def dof(self):
        """(value, is_lower_bound) of the dB depth of field along the axis."""
        d = self.depth
        if not d:
            return None, False
        if d["za"] is not None and d["zb"] is not None:
            return d["zb"] - d["za"], False
        return d["zb_"] - d["za_"], True


def compute(scan: Scan, st: Settings, display_up: Optional[int] = None) -> Optional[Result]:
    """Measurements use the grid upsampled by st.upscale (bicubic); the returned display arrays
    (hu/vu/linu/normu/dispu) use display_up instead when given (to keep the on-screen map light).
    Peak pressure and focal position always come from the measured (raw) grid."""
    h, v, lin, unit = view(scan, st)
    if lin.size == 0 or np.all(np.isnan(lin)):
        return None
    mh, mv, mlin = upscaled(h, v, lin, st.upscale)                 # measurement grid
    meas_up = st.upscale if mlin.shape != lin.shape else 1
    if display_up is None or display_up == st.upscale:
        hu, vu, linu = mh, mv, mlin
    else:
        hu, vu, linu = upscaled(h, v, lin, display_up)
    normu, dispu, label = display(linu, st)
    level = st.level_db if st.level_db < 0 else -3.0
    frac, lvl = 10 ** (level / 20), f"{level:g} dB"
    hn = f"{scan.hname}′" if st.deskew else scan.hname
    iv, ih = np.unravel_index(np.nanargmax(lin), lin.shape)
    fh, fv, fval = float(h[ih]), float(v[iv]), float(lin[iv, ih])
    fval_mpa = fval * 1e-3 / st.volts_per_mpa if unit == "mV" else (fval if st.unit == Unit.PPP else None)
    # focal width on the measurement grid: the row nearest the focal depth, around its own peak
    iu = int(np.abs(mv - fv).argmin())
    row = mlin[iu, :]
    ju = int(np.nanargmax(row)) if not np.all(np.isnan(row)) else ih
    wa, wb = crossings(mh, row, ju, frac)
    ax_x = (0.0, 0.0) if st.deskew else (float(st.axis_h(v[0])), float(st.axis_h(v[-1])))

    depth, widths = None, np.full(len(mv), np.nan)
    amp, alab = np.full(len(mv), np.nan), "On-axis signal"
    if lin.shape[0] >= 2 and lin.shape[1] >= 2:
        vs, hs, prof = axial_profile(mh, mv, mlin, *ax_x)
        if not np.all(np.isnan(prof)):
            ip = int(np.nanargmax(prof))
            za, zb = crossings(vs, prof, ip, frac)
            za_, zb_ = (v[0] if za is None else za), (v[-1] if zb is None else zb)
            depth = dict(peak=float(vs[ip]), za=za, zb=zb, za_=float(za_), zb_=float(zb_),
                         xa=float(np.interp(za_, vs, hs)), xb=float(np.interp(zb_, vs, hs)))
        widths = depth_widths(mh, mlin, frac)
        _, _, onaxis = axial_profile(mh, mv, mlin, *ax_x, n=len(mv))
        ref = np.nanmax(lin)
        if st.unit == Unit.DB:
            amp, alab = 20 * np.log10(np.clip(onaxis / ref, 10 ** (DB_FLOOR / 20), None)), "On-axis (dB re max)"
        elif st.unit == Unit.NORM:
            amp, alab = onaxis / ref, "On-axis (normalized)"
        elif st.unit == Unit.VPP:
            amp, alab = onaxis, "On-axis Vpp (mV)"
        else:
            amp = onaxis
            alab = {Unit.PPP: "On-axis p–p (MPa)", Unit.PPOS: "On-axis p+ (MPa)", Unit.PNEG: "On-axis p− (MPa)"}[st.unit]

    r = Result(h=h, v=v, lin=lin, unit=unit, hu=hu, vu=vu, linu=linu, normu=normu, dispu=dispu, label=label,
               frac=frac, lvl=lvl, hn=hn, fh=fh, fv=fv, fval=fval, fval_mpa=fval_mpa, wa=wa, wb=wb, ax_x=ax_x,
               depth=depth, widths=widths, amp=amp, alab=alab, pv=mv, meas_up=meas_up)

    s = scan
    # short lines (≤ ~55 chars) so the details box doesn't wrap
    fmax = f"{fval_mpa:.4g} MPa" if fval_mpa is not None else f"{fval:.4g} {unit}"
    at = f"{hn} = {fh:.2f}" + (f" ({s.hname} {fh + float(st.axis_h(fv)):.2f})" if st.deskew else "")
    r.lines.append(f"Focal max: {fmax} [at {at}]")
    r.lines.append(f"Beamwidth ({lvl}): {wb - wa:.3f} mm at {s.vname} = {fv:.2f}" if r.width is not None
                   else f"Beamwidth ({lvl}): beam doesn't fall to {lvl} in range")
    ok = ~np.isnan(widths)
    if ok.any():
        k = int(np.nanargmin(widths))
        r.lines.append(f"Focal beamwidth: {widths[k]:.3f} mm at {s.vname} = {mv[k]:.2f}")
    if depth:
        if depth["za"] is not None and depth["zb"] is not None:
            r.lines.append(f"Focal depth ({lvl}): {depth['zb'] - depth['za']:.2f} mm "
                           f"({depth['za']:.2f} → {depth['zb']:.2f})")
        else:
            r.lines.append(f"Focal depth ({lvl}): > {depth['zb_'] - depth['za_']:.2f} mm "
                           f"({depth['za_']:.2f} → {depth['zb_']:.2f})")
        r.lines.append(f"Axial peak: {s.vname} = {depth['peak']:.2f} mm")
    r.lines.append(f"Axis tilt: {st.tilt_deg:+.2f}° ({np.tan(np.radians(st.tilt_deg)) * 1e3:+.1f} µm/mm)"
                   + (" · deskewed" if st.deskew else ""))
    r.lines.append(f"Grid: {s.plane} at {AXIS_NAMES[s.fixed_ax]} = {s.fixed_val:.2f} mm"
                   + (f" · widths on {meas_up}× grid" if meas_up > 1 else ""))
    return r


# ============================================================================ exports
def profile_csv(scan: Scan, r: Result) -> str:
    lines = [f"# {scan.title} - {scan.plane} scan, width at {r.lvl} of each depth's peak; on-axis in {r.alab}",
             f"{scan.vname}_mm,width_mm,on_axis"]
    for z, w, a in zip(r.pv if r.pv is not None else r.v, r.widths, r.amp):
        lines.append(f"{z:.4f},{'' if np.isnan(w) else f'{w:.5f}'},{'' if np.isnan(a) else f'{a:.6g}'}")
    return "\n".join(lines) + "\n"


def render_png(scan: Scan, st: Settings, r: Result, overlays: dict, cmap: str, clim: tuple,
               dark: bool = False, profile: bool = True) -> bytes:
    """Static PNG (map + optional profile) via matplotlib, for reports."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    T = dict(bg="#191919", fg="#d4d4d4", mute="#9b9b9b", line="#444444", grid="#2f2f2f", acc="#529cca",
             org="#ffa344", red="#ff7369", axis="#4cd7ff") if dark else \
        dict(bg="#ffffff", fg="#37352f", mute="#787774", line="#d9d9d6", grid="#ededeb", acc="#2383e2",
             org="#d9730d", red="#e03e3e", axis="#00a3bf")
    W, H = (13.0, 7.2) if profile else (6.0, 7.2)
    fig = Figure(figsize=(W, H), dpi=200, facecolor=T["bg"])
    he, ve = cell_edges(r.hu), cell_edges(r.vu)
    # map axes with exact aspect, top-left anchored
    L, R_, Tm, B = 0.8, 1.05, 1.15, 0.3
    aw, ah = 5.6 - L - R_, H - Tm - B
    ratio = (he[-1] - he[0]) / abs(ve[-1] - ve[0])
    if aw / ah > ratio:
        aw = ah * ratio
    else:
        ah = aw / ratio
    y0 = H - Tm - ah
    ax = fig.add_axes([L / W, y0 / H, aw / W, ah / H])
    cax = fig.add_axes([(L + aw + 0.14) / W, y0 / H, 0.12 / W, ah / H])
    pc = ax.pcolormesh(he, ve, r.dispu, cmap=cmap, vmin=clim[0], vmax=max(clim[1], clim[0] + 1e-12),
                       shading="flat", rasterized=True)
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label(r.label, color=T["fg"])
    cb.ax.tick_params(colors=T["mute"])
    cb.outline.set_edgecolor(T["line"])
    v = r.v
    if overlays.get("contour") and r.normu.shape[0] > 1 and r.normu.shape[1] > 1:
        ax.contour(r.hu, r.vu, r.normu, levels=[r.frac], colors="w", linewidths=1.1, linestyles="--")
    if overlays.get("depth") and r.depth:
        d = r.depth
        ax.plot(r.ax_x, [v[0], v[-1]], color=T["axis"], lw=0.9)
        ax.plot([d["xa"], d["xb"]], [d["za_"], d["zb_"]], color=T["red"], lw=2.4)
    if overlays.get("width"):
        ax.axhline(r.fv, color="w", lw=0.7, alpha=0.7)
        if r.width is not None:
            ax.plot([r.wa, r.wb], [r.fv, r.fv], color=T["red"], lw=2.4)
    if overlays.get("target"):
        ax.plot(r.fh, r.fv, "+", color=T["org"], ms=15, mew=2)
        ax.plot(r.fh, r.fv, "o", mfc="none", color=T["org"], ms=10, mew=1.5)
    ax.set_xlim(he[0], he[-1])
    ax.set_ylim(ve[0], ve[-1])
    if scan.depth_is_z != overlays.get("flip", False):
        ax.invert_yaxis()
    ax.set_xlabel(f"{r.hn} re beam axis (mm)" if st.deskew else f"{scan.hname} position (mm)")
    ax.set_ylabel(f"{scan.vname} (mm from probe)" if scan.depth_is_z else f"{scan.vname} position (mm)")
    if overlays.get("top", True):
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
    axes = [ax]
    if profile:
        axp = fig.add_axes([6.75 / W, 0.75 / H, 5.2 / W, (H - 1.9) / H])
        wl = "slice thickness" if scan.is_yz else "beam width"
        pv = r.pv if r.pv is not None else r.v
        axp.plot(pv, r.widths, "o-" if len(pv) <= 60 else "-", color=T["acc"], ms=3.5, lw=1.8, label=f"{r.lvl} {wl}")
        axp.set_ylabel(f"{r.lvl} {wl} (mm)", color=T["acc"])
        axp.set_xlabel(f"{scan.vname} (mm from probe)" if scan.depth_is_z else f"{scan.vname} position (mm)")
        axp.grid(True, color=T["grid"])
        if np.any(~np.isnan(r.widths)):
            axp.set_ylim(0, np.nanmax(r.widths) * 1.15)
        ax2 = axp.twinx()
        ax2.plot(pv, r.amp, "s-" if len(pv) <= 60 else "-", color=T["org"], ms=3.5, lw=1.8, label=r.alab)
        ax2.set_ylabel(r.alab, color=T["org"])
        if st.unit != Unit.DB and np.any(~np.isnan(r.amp)) and np.nanmax(r.amp) > 0:
            ax2.set_ylim(0, np.nanmax(r.amp) * 1.12)
        axp.axvline(r.fv, color=T["mute"], ls=":")
        h1, l1 = axp.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        lg = axp.legend(h1 + h2, l1 + l2, loc="best", fontsize=9, frameon=False)
        for tx in lg.get_texts():
            tx.set_color(T["fg"])
        axes += [axp, ax2]
    for a in axes:
        a.set_facecolor(T["bg"])
        for sp in a.spines.values():
            sp.set_color(T["line"])
        a.tick_params(colors=T["mute"])
        a.xaxis.label.set_color(T["fg"])
        if a is not axes[-1] or not profile:
            a.yaxis.label.set_color(a.yaxis.label.get_color() if profile and a is not ax else T["fg"])
    title = f"{scan.title} — {scan.plane} hydrophone scan" + (f", {scan.freq:g} MHz" if scan.freq else "")
    fig.text(0.25 / W, 1 - 0.35 / H, title, color=T["fg"], fontsize=13, fontweight="bold")
    buf = io.BytesIO()
    FigureCanvasAgg(fig).print_png(buf)
    return buf.getvalue()
