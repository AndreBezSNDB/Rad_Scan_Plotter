"""
merge.py - combine a coarse (base) scan with a fine (detail) scan of the same plane.

Typical use: a 0.2 mm survey of the whole beam plus a 0.05 mm scan of a small box around
the focus. The result is a single scan on the fine grid that the rest of the app treats
like any other scan.

How it avoids throwing the data off
  1. Position: the detail scan is placed in the base scan's coordinate frame using the
     absolute stage positions saved in each file ("coords"), not the per-scan PDx/PDy/PDz
     values (those are relative to each scan's own centre / entered start distance).
     Optional auto-align then refines the offset by matching the two scans where they
     overlap (useful if the stage was re-homed or a coordinate was entered differently).
  2. Amplitude: the detail scan is compared with the base scan over the overlap and the
     best-fit ratio is reported (it should be ~1.00). Optionally the detail scan is scaled
     by that ratio so the base scan's absolute level is kept.
  3. Seams: inside the box edge the two are blended with a cosine taper a couple of coarse
     steps wide; deep inside the box only fine data is used; outside it only coarse data.
  4. Provenance: the merged file records what was merged, the offsets, the scale factor,
     the overlap residual and the detail box (drawn on the map).

The base scan outside the box is resampled onto the fine grid with cubic interpolation, so
measurements there are no better than the coarse data - only the boxed region is "real"
fine data.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from beamcore import Scan

ALIGN_STAGE = "Stage coordinates"
ALIGN_SCAN = "Scan coordinates (PDx/PDy/PDz)"
ALIGN_AUTO = "Auto-align (refine by matching overlap)"


@dataclass
class MergeReport:
    align: str
    offset_h: float = 0.0           # extra shift applied to the detail scan (mm), after the alignment frame
    offset_v: float = 0.0
    scale: float = 1.0              # base / detail amplitude ratio in the overlap
    scale_applied: bool = False
    residual_pct: float = float("nan")   # RMS(base - scaled detail) / peak, over the overlap
    overlap_pts: int = 0
    step_h: float = 0.0
    step_v: float = 0.0
    box: tuple = (0, 0, 0, 0)
    shape: tuple = (0, 0)
    warnings: list = field(default_factory=list)

    def summary(self, hname="H", vname="V") -> str:
        return (f"{self.align}; offset {hname} {self.offset_h:+.3f} mm, {vname} {self.offset_v:+.3f} mm; "
                f"amplitude ratio {self.scale:.3f}{' (applied)' if self.scale_applied else ''}; "
                f"overlap residual {self.residual_pct:.1f}% of peak over {self.overlap_pts} pts")


def _stage_fit(stage: Optional[np.ndarray], pd: np.ndarray):
    """Linear map stage -> scan coordinate (slope is +1 or -1 in practice)."""
    if stage is None or stage.size < 2 or np.ptp(stage) == 0:
        return None
    a, b = np.polyfit(stage, pd, 1)
    return float(a), float(b)


def _detail_positions(base: Scan, det: Scan, use_stage: bool):
    """Detail-scan coordinates expressed in the base scan's frame."""
    if use_stage:
        fh, fv = _stage_fit(base.h_stage, base.h), _stage_fit(base.v_stage, base.v)
        if fh is None or fv is None or det.h_stage is None or det.v_stage is None:
            raise ValueError("Stage coordinates aren't available in both files — use scan coordinates "
                             "or auto-align instead.")
        return fh[0] * det.h_stage + fh[1], fv[0] * det.v_stage + fv[1]
    return det.h.copy(), det.v.copy()


def _sorted(h, v, *arrays):
    oh, ov = np.argsort(h), np.argsort(v)
    return (h[oh], v[ov]) + tuple(None if a is None else a[np.ix_(ov, oh)] for a in arrays)


def _overlap(base: Scan, hd, vd, d_vpp, dh=0.0, dv=0.0):
    """Base (cubic-interpolated) and detail values at the detail points inside the base extent."""
    f = RegularGridInterpolator((base.v, base.h), base.vpp, method="cubic", bounds_error=False, fill_value=np.nan)
    V, H = np.meshgrid(vd + dv, hd + dh, indexing="ij")
    b = f(np.c_[V.ravel(), H.ravel()])
    d = d_vpp.ravel()
    ok = np.isfinite(b) & np.isfinite(d)
    return b[ok], d[ok]


def _fit(b, d):
    if b.size < 4 or np.dot(d, d) == 0:
        return 1.0, float("nan")
    k = float(np.dot(b, d) / np.dot(d, d))
    res = float(np.sqrt(np.mean((b - k * d) ** 2)) / max(np.max(b), 1e-30) * 100)
    return k, res


def _snap(pts, v, h, tol=1e-6):
    """Pull points within tol of a grid's edge onto it (floating-point safety for edge rows/columns)."""
    p = pts.copy()
    for col, ax in ((0, v), (1, h)):
        lo, hi = ax[0], ax[-1]
        p[:, col] = np.where((p[:, col] < lo) & (p[:, col] > lo - tol), lo, p[:, col])
        p[:, col] = np.where((p[:, col] > hi) & (p[:, col] < hi + tol), hi, p[:, col])
    return p


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def merge_scans(base: Scan, det: Scan, align: str = ALIGN_STAGE, extra_h: float = 0.0, extra_v: float = 0.0,
                match_amplitude: bool = True, feather_steps: float = 2.0) -> tuple[Scan, MergeReport]:
    warn = []
    if base.plane != det.plane:
        raise ValueError(f"Different planes: base is {base.plane}, detail is {det.plane}.")
    if base.freq is not None and det.freq is not None and base.freq != det.freq:
        warn.append(f"Different drive frequency ({base.freq:g} vs {det.freq:g} MHz).")
    if abs(base.sens - det.sens) > 1e-6 or abs(base.gain - det.gain) > 1e-6:
        warn.append("Hydrophone sensitivity/gain differ between files; the base scan's values are kept.")
    if base.fixed_stage is not None and det.fixed_stage is not None:
        dfx = det.fixed_stage - base.fixed_stage
        if abs(dfx) > 0.02:
            warn.append(f"The planes are {dfx:+.3f} mm apart in the fixed axis — they may not be the same slice.")
    if det.dh <= 0 or det.dv <= 0:
        raise ValueError("The detail scan needs at least 2 points in each direction.")
    if det.dh > base.dh or det.dv > base.dv:
        warn.append("The detail scan isn't finer than the base scan in both directions.")

    # 1) position the detail scan in the base frame
    use_stage = align in (ALIGN_STAGE, ALIGN_AUTO) and base.h_stage is not None and det.h_stage is not None
    if align in (ALIGN_STAGE, ALIGN_AUTO) and not use_stage:
        warn.append("No stage coordinates in one of the files — using scan coordinates instead.")
    hd, vd = _detail_positions(base, det, use_stage)
    hd, vd, d_vpp, d_vmax, d_vmin = _sorted(hd, vd, det.vpp, det.vmax, det.vmin)
    hd, vd = hd + extra_h, vd + extra_v

    off_h, off_v = 0.0, 0.0
    if align == ALIGN_AUTO:
        def cost(x):
            b, d = _overlap(base, hd, vd, d_vpp, x[0], x[1])
            if b.size < 10:
                return 1e9
            k, res = _fit(b, d)
            return res
        lim_h, lim_v = 2 * base.dh, 2 * base.dv
        best = min((minimize(cost, x0, method="Nelder-Mead",
                             options=dict(xatol=1e-4, fatol=1e-6, maxiter=400)) for x0 in
                    ([0, 0], [lim_h / 2, 0], [-lim_h / 2, 0], [0, lim_v / 2], [0, -lim_v / 2])),
                   key=lambda r_: r_.fun)
        off_h, off_v = float(np.clip(best.x[0], -lim_h, lim_h)), float(np.clip(best.x[1], -lim_v, lim_v))
        hd, vd = hd + off_h, vd + off_v
        if abs(off_h) >= lim_h * 0.99 or abs(off_v) >= lim_v * 0.99:
            warn.append("Auto-align hit its search limit (±2 coarse steps) — check the scans really overlap.")

    b, d = _overlap(base, hd, vd, d_vpp)
    if b.size < 10:
        raise ValueError("The scans barely overlap in this alignment — check the alignment mode/offsets.")
    k, res = _fit(b, d)
    if abs(k - 1) > 0.05:
        warn.append(f"Amplitude differs by {100 * (k - 1):+.1f}% between scans (drift, warm-up or settings?).")
    if not match_amplitude:                      # residual of the unscaled detail data
        res = float(np.sqrt(np.mean((b - d) ** 2)) / max(np.max(b), 1e-30) * 100)
    scale = k if match_amplitude else 1.0

    # 2) output grid on the fine spacing, aligned to the detail samples, covering both scans
    sh, sv = det.dh, det.dv
    lo_h, hi_h = min(base.h[0], hd[0]), max(base.h[-1], hd[-1])
    lo_v, hi_v = min(base.v[0], vd[0]), max(base.v[-1], vd[-1])
    # stay inside the measured extent (ceil/floor) so no edge row/column falls outside both scans
    H = hd[0] + sh * np.arange(int(np.ceil((lo_h - hd[0]) / sh - 1e-6)), int(np.floor((hi_h - hd[0]) / sh + 1e-6)) + 1)
    V = vd[0] + sv * np.arange(int(np.ceil((lo_v - vd[0]) / sv - 1e-6)), int(np.floor((hi_v - vd[0]) / sv + 1e-6)) + 1)
    if H.size * V.size > 2_000_000:
        raise ValueError(f"Merged grid would be {V.size} × {H.size} points — too large.")
    VV, HH = np.meshgrid(V, H, indexing="ij")
    pts = np.c_[VV.ravel(), HH.ravel()]

    # 3) blend weight: 1 deep inside the detail box, cosine taper to 0 at its edges (only on edges that
    #    lie inside the base scan — where the box reaches the edge of the base scan there is nothing to blend)
    h0, h1, v0, v1 = hd[0], hd[-1], vd[0], vd[-1]
    th, tv = max(feather_steps * base.dh, 1e-9), max(feather_steps * base.dv, 1e-9)
    inf = np.inf
    dist_h = np.minimum(HH - h0 if h0 > base.h[0] + 1e-9 else inf, h1 - HH if h1 < base.h[-1] - 1e-9 else inf)
    dist_v = np.minimum(VV - v0 if v0 > base.v[0] + 1e-9 else inf, v1 - VV if v1 < base.v[-1] - 1e-9 else inf)
    inside = (HH >= h0 - 1e-9) & (HH <= h1 + 1e-9) & (VV >= v0 - 1e-9) & (VV <= v1 + 1e-9)
    w = np.where(inside, _smooth(dist_h / th) * _smooth(dist_v / tv), 0.0)
    if feather_steps <= 0:
        w = inside.astype(float)

    def combine(base_arr, det_arr, sgn=1.0):
        if base_arr is None or det_arr is None:
            return None
        fb = RegularGridInterpolator((base.v, base.h), base_arr, method="cubic", bounds_error=False,
                                     fill_value=np.nan)
        fd = RegularGridInterpolator((vd, hd), det_arr * scale, method="linear", bounds_error=False,
                                     fill_value=np.nan)
        bv = fb(_snap(pts, base.v, base.h)).reshape(VV.shape)
        dv_ = fd(_snap(pts, vd, hd)).reshape(VV.shape)
        out = np.where(np.isnan(dv_), bv, np.where(np.isnan(bv), dv_, w * dv_ + (1 - w) * bv))
        return out

    m_vpp = combine(base.vpp, d_vpp)
    m_vmax = combine(base.vmax, d_vmax)
    m_vmin = combine(base.vmin, d_vmin)

    rep = MergeReport(align=align, offset_h=extra_h + off_h, offset_v=extra_v + off_v, scale=k,
                      scale_applied=match_amplitude, residual_pct=res, overlap_pts=int(b.size),
                      step_h=sh, step_v=sv, box=(float(h0), float(h1), float(v0), float(v1)),
                      shape=(V.size, H.size), warnings=warn)
    if res > 5:
        warn.append(f"Overlap residual is {res:.1f}% of peak — the scans may not match well.")

    # stage coordinates for the merged grid (so it can be merged again)
    fh, fv = _stage_fit(base.h_stage, base.h), _stage_fit(base.v_stage, base.v)
    info = (f"Merged {base.name} (coarse {base.dh:g}×{base.dv:g} mm) + {det.name} "
            f"(fine {sh:g}×{sv:g} mm in {base.hname} {h0:.2f}…{h1:.2f}, {base.vname} {v0:.2f}…{v1:.2f}); "
            + rep.summary(base.hname, base.vname))
    merged = Scan(name=f"{base.title}_merged.mat", h=H, v=V, h_ax=base.h_ax, v_ax=base.v_ax,
                  fixed_ax=base.fixed_ax, fixed_val=base.fixed_val, vpp=m_vpp, vmax=m_vmax, vmin=m_vmin,
                  device=f"{base.title} (merged)", freq=base.freq, sens=base.sens, gain=base.gain, navg=base.navg,
                  h_stage=None if fh is None else (H - fh[1]) / fh[0],
                  v_stage=None if fv is None else (V - fv[1]) / fv[0],
                  fixed_stage=base.fixed_stage, merge_info=info, detail_box=rep.box)
    return merged, rep


def scan_to_mat_bytes(sc: Scan) -> bytes:
    """Write a Scan back to a .mat file the app (and MATLAB) can read."""
    n = [1, 1, 1]
    n[sc.h_ax], n[sc.v_ax] = sc.h.size, sc.v.size

    def cube(plane):
        if plane is None:
            return None
        return np.expand_dims(plane.T, axis=sc.fixed_ax)        # (Nh, Nv) -> 3-D with the fixed axis length 1

    pd = [None, None, None]
    pd[sc.h_ax], pd[sc.v_ax], pd[sc.fixed_ax] = sc.h, sc.v, np.array([sc.fixed_val])
    stage = [None, None, None]
    stage[sc.h_ax], stage[sc.v_ax] = sc.h_stage, sc.v_stage
    stage[sc.fixed_ax] = None if sc.fixed_stage is None else np.array([sc.fixed_stage])
    out = {"PDx": pd[0].reshape(1, -1), "PDy": pd[1].reshape(1, -1), "PDz": pd[2].reshape(1, -1),
           "VPPDatastore": cube(sc.vpp), "DeviceID": sc.device, "HydrophoneSens": sc.sens,
           "HydrophoneGain": sc.gain, "MergeInfo": sc.merge_info}
    if sc.freq is not None:
        out["USfreq"] = sc.freq
    if sc.navg is not None:
        out["ScopeNAverages"] = sc.navg
    if sc.vmax is not None:
        out["VMAXDatastore"] = cube(sc.vmax)
    if sc.vmin is not None:
        out["VMINDatastore"] = cube(sc.vmin)
    if sc.detail_box is not None:
        out["DetailBox"] = np.array(sc.detail_box, float)
    if all(x is not None for x in stage):
        c = np.empty((3, 1), dtype=object)
        for i in range(3):
            c[i, 0] = np.asarray(stage[i], float).reshape(1, -1)
        out["coords"] = c
    buf = io.BytesIO()
    sio.savemat(buf, out, do_compression=True)
    return buf.getvalue()
