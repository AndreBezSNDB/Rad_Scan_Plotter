"""
Beam Lab (web) - hydrophone beam-scan viewer as a Streamlit app.

Run locally:   streamlit run app.py
Then open:     http://localhost:8501   (others on your network: http://<this-pc>:8501)

Users upload .mat scans in the browser; nothing to install on their side. Each signed-in
user gets their own scan library and saved views (stored under BEAMLAB_DATA on the server).
Sign-in is optional: configure [auth] in .streamlit/secrets.toml to require it (see README).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from beamcore import (AXIS_NAMES, Scan, Settings, Unit, UPSCALE_STEPS, autofit, colour_bounds, compute,
                      profile_csv, render_png)
from store import UserStore

st.set_page_config(page_title="Beam Lab", page_icon="〰️", layout="wide", initial_sidebar_state="expanded")

CMAPS = {"viridis": "Viridis", "inferno": "Inferno", "magma": "Magma", "plasma": "Plasma",
         "cividis": "Cividis", "turbo": "Turbo", "jet": "Jet", "gray": "Greys_r"}
TOGGLES = dict(target=True, contour=True, width=True, depth=False, depthnorm=False, flip=False, top=True,
               equal=True, profile=True, deskew=False)
ss = st.session_state

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 1.5rem; max-width: 100%;}
section[data-testid="stSidebar"] .stButton button, section[data-testid="stSidebar"] .stButton button > div
    {justify-content: flex-start; text-align: left; width: 100%;}
section[data-testid="stSidebar"] .stButton button p {white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.bl-crumb {opacity: .6; font-size: .9rem; margin-bottom: .2rem;}
.bl-props {border-collapse: collapse; margin: .2rem 0 .4rem 0; font-size: .93rem;}
.bl-props td {padding: 3px 28px 3px 0; border: none;}
.bl-props td.k {opacity: .6;}
.bl-props tr, .bl-props table {border: none; background: transparent !important;}
.bl-chip {background: rgba(128,128,128,.16); border-radius: 4px; padding: 1px 7px; margin-right: 6px; font-size: .86rem;}
.bl-sec {opacity: .55; font-size: .78rem; font-weight: 600; margin: .9rem 0 .2rem 0; letter-spacing: .02em;}
.bl-empty {text-align: center; padding: 14vh 0 2rem 0; opacity: .8;}
div[data-testid="stMetricValue"] {font-size: 1.55rem;}
</style>
""", unsafe_allow_html=True)


# ============================================================================ auth
def current_user() -> str:
    """'local' when no sign-in is configured; otherwise the signed-in email (enforced)."""
    try:
        auth = st.secrets.get("auth")
    except Exception:  # noqa: BLE001  (no secrets.toml)
        auth = None
    if not auth:
        return "local"
    providers = [k for k, v in dict(auth).items() if isinstance(v, dict) or hasattr(v, "keys")]
    if not st.user.get("is_logged_in", False):
        st.markdown("<div class='bl-empty'><h2>Beam Lab</h2><p>Sign in to view and analyse hydrophone scans."
                    "</p></div>", unsafe_allow_html=True)
        c = st.columns([2, 1, 2])[1]
        if c.button("Sign in", type="primary", use_container_width=True):
            st.login(providers[0]) if providers else st.login()
        st.stop()
    email = (st.user.get("email") or st.user.get("name") or "user").lower()
    try:
        allowed = [d.lower().lstrip("@") for d in st.secrets.get("allowed_domains", [])]
    except Exception:  # noqa: BLE001
        allowed = []
    if allowed and email.split("@")[-1] not in allowed:
        st.error(f"{email} isn't allowed to use this app.")
        st.button("Sign out", on_click=st.logout)
        st.stop()
    return email


USER = current_user()
store = UserStore(USER)


@st.cache_data(show_spinner=False, max_entries=24)
def load_scan(path: str, mtime: float, filename: str) -> Scan:
    return Scan.load(path, filename)


def get_scan(sid: str) -> Scan | None:
    e = store.scan(sid)
    p = store.scan_path(sid) if e else None
    if not e or not p.exists():
        return None
    return load_scan(str(p), p.stat().st_mtime, e.get("filename", ""))


# ============================================================================ per-scan state
def K(sid: str, name: str) -> str:
    return f"{sid}.{name}"


def opts(arr) -> list:
    return [round(float(x), 4) for x in arr]


def ensure_defaults(sid: str, scan: Scan) -> None:
    if ss.get(K(sid, "init")):
        return
    ss[K(sid, "unit")] = Unit.PPP
    ss[K(sid, "sens")] = float(scan.sens)
    ss[K(sid, "gain")] = float(scan.gain)
    ss[K(sid, "cmap")] = "viridis"
    ss[K(sid, "up")] = 4
    oh, ov = opts(scan.h), opts(scan.v)
    ss[K(sid, "hr")] = (oh[0], oh[-1])
    ss[K(sid, "vr")] = (ov[0], ov[-1])
    ss[K(sid, "level")] = -3.0
    for k, v in TOGGLES.items():
        ss[K(sid, k)] = v
    va, vb, xa, xb = autofit(scan, settings_from_state(sid, scan, axis=False))
    ss[K(sid, "va")], ss[K(sid, "vb")] = va, vb
    ss[K(sid, "xa")], ss[K(sid, "xb")] = round(xa, 4), round(xb, 4)
    ss[K(sid, "init")] = True


def settings_from_state(sid: str, scan: Scan, axis: bool = True) -> Settings:
    g = lambda n, d=None: ss.get(K(sid, n), d)  # noqa: E731
    hr, vr = g("hr", (-np.inf, np.inf)), g("vr", (-np.inf, np.inf))
    s = Settings(unit=g("unit", Unit.PPP), sens=float(g("sens", scan.sens) or scan.sens),
                 gain=float(g("gain", scan.gain) or 0.0), upscale=int(g("up", 4) or 1),
                 h_lo=hr[0], h_hi=hr[1], v_lo=vr[0], v_hi=vr[1], level_db=float(g("level", -3.0)),
                 depthnorm=bool(g("depthnorm", False)), deskew=bool(g("deskew", False) and axis))
    if axis:
        s.axis_va, s.axis_vb = float(g("va", 0.0)), float(g("vb", 1.0))
        s.axis_xa, s.axis_xb = float(g("xa", 0.0)), float(g("xb", 0.0))
    return s


def range_widget(widget, stored_key: str, label: str, **kw):
    """Two-handle slider whose value lives in ss[stored_key]; the widget has its own key so that
    callbacks can reset it (pop the widget key) and range mode survives reruns."""
    wk = stored_key + "#w"

    def sync():
        ss[stored_key] = tuple(ss[wk])

    return widget(label, value=tuple(ss[stored_key]), key=wk, on_change=sync, **kw)


def reset_widget(stored_key: str, value) -> None:
    ss[stored_key] = tuple(value)
    ss.pop(stored_key + "#w", None)


def clim_key(sid: str, scan: Scan, s: Settings):
    lo, hi = colour_bounds(scan, s)
    return K(sid, f"clim:{lo:.6g}:{hi:.6g}"), lo, hi


# ============================================================================ callbacks
def cb_select(sid: str) -> None:
    ss.scan_id = sid
    ss.active_view = None


def cb_autofit(sid: str) -> None:
    scan = get_scan(sid)
    if scan:
        va, vb, xa, xb = autofit(scan, settings_from_state(sid, scan))
        ss[K(sid, "va")], ss[K(sid, "vb")] = va, vb
        ss[K(sid, "xa")], ss[K(sid, "xb")] = round(xa, 4), round(xb, 4)


def cb_reset_ranges(sid: str) -> None:
    scan = get_scan(sid)
    if scan:
        oh, ov = opts(scan.h), opts(scan.v)
        reset_widget(K(sid, "hr"), (oh[0], oh[-1]))
        reset_widget(K(sid, "vr"), (ov[0], ov[-1]))


def cb_auto_colour(key: str) -> None:
    ss.pop(key, None)
    ss.pop(key + "#w", None)


def snapshot(sid: str, scan: Scan) -> tuple[dict, dict]:
    s = settings_from_state(sid, scan)
    ck, lo, hi = clim_key(sid, scan, s)
    disp = dict(cmap=ss.get(K(sid, "cmap"), "viridis"), clim=list(ss.get(ck, (lo, hi))),
                toggles={k: bool(ss.get(K(sid, k), v)) for k, v in TOGGLES.items()})
    return s.as_dict(), disp


def cb_save_view(sid: str) -> None:
    name = (ss.get("new_view_name") or "").strip()
    scan = get_scan(sid)
    if not name or not scan:
        return
    settings, disp = snapshot(sid, scan)
    v = store.add_view(name, sid, scan.title, settings, disp)
    ss.active_view = v["id"]
    ss.new_view_name = ""


def cb_update_view(vid: str, sid: str) -> None:
    scan = get_scan(sid)
    if scan:
        settings, disp = snapshot(sid, scan)
        store.update_view(vid, settings=settings, display=disp, scan_id=sid, scan_name=scan.title)
        ss.active_view = vid


def cb_rename_view(vid: str) -> None:
    name = (ss.get(f"rn_{vid}") or "").strip()
    if name:
        store.update_view(vid, name=name)


def cb_delete_view(vid: str) -> None:
    store.delete_view(vid)
    if ss.get("active_view") == vid:
        ss.active_view = None


def cb_remove_scan(sid: str) -> None:
    store.remove_scan(sid)
    if ss.get("scan_id") == sid:
        ss.scan_id = store.scans[0]["id"] if store.scans else None


def cb_apply_view(vid: str) -> None:
    v = store.view(vid)
    if not v:
        return
    sid = v["scan_id"]
    scan = get_scan(sid)
    if not scan:
        ss.flash = f"The scan for “{v['name']}” is no longer in your library."
        return
    ss.scan_id, ss.active_view = sid, vid
    ensure_defaults(sid, scan)
    st_ = Settings.from_dict(v.get("settings", {}))
    if st_.unit in scan.units():
        ss[K(sid, "unit")] = st_.unit
    ss[K(sid, "sens")], ss[K(sid, "gain")] = st_.sens, st_.gain
    ss[K(sid, "up")] = st_.upscale if st_.upscale in UPSCALE_STEPS else 4
    ss[K(sid, "level")] = st_.level_db
    oh, ov = np.array(opts(scan.h)), np.array(opts(scan.v))
    snap = lambda arr, x, d: float(arr[np.abs(arr - x).argmin()]) if np.isfinite(x) else d  # noqa: E731
    reset_widget(K(sid, "hr"), (snap(oh, st_.h_lo, oh[0]), snap(oh, st_.h_hi, oh[-1])))
    reset_widget(K(sid, "vr"), (snap(ov, st_.v_lo, ov[0]), snap(ov, st_.v_hi, ov[-1])))
    ss[K(sid, "va")], ss[K(sid, "vb")] = st_.axis_va, st_.axis_vb
    ss[K(sid, "xa")], ss[K(sid, "xb")] = st_.axis_xa, st_.axis_xb
    d = v.get("display", {})
    ss[K(sid, "cmap")] = d.get("cmap", "viridis") if d.get("cmap") in CMAPS else "viridis"
    for k, val in d.get("toggles", {}).items():
        if k in TOGGLES:
            ss[K(sid, k)] = bool(val)
    ss[K(sid, "depthnorm")], ss[K(sid, "deskew")] = st_.depthnorm, st_.deskew
    ck, lo, hi = clim_key(sid, scan, settings_from_state(sid, scan))
    if d.get("clim"):
        a, b = d["clim"]
        a, b = min(max(a, lo), hi), min(max(b, lo), hi)
        reset_widget(ck, (a, b) if b > a else (lo, hi))


# ============================================================================ sidebar
ss.setdefault("scan_id", store.scans[0]["id"] if store.scans else None)
ss.setdefault("active_view", None)
ss.setdefault("up_n", 0)
if ss.scan_id and not store.scan(ss.scan_id):
    ss.scan_id = store.scans[0]["id"] if store.scans else None

with st.sidebar:
    st.markdown("### 〰 Beam Lab")
    ups = st.file_uploader("Import .mat scans", type=["mat"], accept_multiple_files=True,
                           key=f"uploader_{ss.up_n}", label_visibility="collapsed")
    if ups:
        first, errors = None, []
        for f in ups:
            try:
                e = store.add_scan(f.getvalue(), f.name)
                first = first or e["id"]
            except Exception as ex:  # noqa: BLE001
                errors.append(str(ex))
        ss.up_n += 1
        if first:
            cb_select(first)
        if errors:
            ss.flash = " · ".join(errors)
        st.rerun()

    q = st.text_input("Search", placeholder="Search scans & views", label_visibility="collapsed").strip().lower()

    st.markdown("<div class='bl-sec'>SCANS</div>", unsafe_allow_html=True)
    lib = [e for e in store.scans if q in f"{e['name']} {e.get('filename', '')}".lower()]
    if not lib:
        st.caption("No matches" if q else "None yet — drop .mat files above")
    for e in lib:
        c1, c2 = st.columns([6, 1], vertical_alignment="center")
        active = e["id"] == ss.scan_id
        c1.button(f"{'▣' if active else '▤'}  {e['name']}", key=f"s_{e['id']}", on_click=cb_select,
                  args=(e["id"],), type="secondary" if active else "tertiary", use_container_width=True,
                  help=f"{e.get('filename', '')} · {e.get('plane', '')} · added {e.get('added', '')}")
        with c2.popover("⋯", type="tertiary"):
            st.caption(e.get("filename", ""))
            p = store.scan_path(e["id"])
            if p.exists():
                st.download_button("Download .mat", data=p.read_bytes, file_name=e.get("filename") or f"{e['id']}.mat",
                                   key=f"dl_{e['id']}", use_container_width=True)
            st.button("Remove from library", key=f"rm_{e['id']}", on_click=cb_remove_scan, args=(e["id"],),
                      use_container_width=True)

    st.markdown("<div class='bl-sec'>SAVED VIEWS</div>", unsafe_allow_html=True)
    views = [v for v in store.views if q in f"{v['name']} {v.get('scan_name', '')}".lower()]
    if not views:
        st.caption("No matches" if q else "None yet — use “Save view”")
    for v in views:
        c1, c2 = st.columns([6, 1], vertical_alignment="center")
        active = v["id"] == ss.active_view
        c1.button(f"◧  {v['name']}", key=f"v_{v['id']}", on_click=cb_apply_view, args=(v["id"],),
                  type="secondary" if active else "tertiary", use_container_width=True,
                  help=f"{v.get('scan_name', '')} · saved {v.get('created', '')}")
        with c2.popover("⋯", type="tertiary"):
            if ss.scan_id:
                st.button("Update with current settings", key=f"uv_{v['id']}", on_click=cb_update_view,
                          args=(v["id"], ss.scan_id), use_container_width=True)
            st.text_input("Rename", value=v["name"], key=f"rn_{v['id']}")
            st.button("Rename", key=f"rnb_{v['id']}", on_click=cb_rename_view, args=(v["id"],),
                      use_container_width=True)
            st.button("Delete view", key=f"dv_{v['id']}", on_click=cb_delete_view, args=(v["id"],),
                      use_container_width=True)

    st.divider()
    if USER != "local":
        st.caption(f"Signed in as {USER}")
        st.button("Sign out", on_click=st.logout, type="tertiary")
    st.caption("Light / dark mode: ⋮ menu (top right) → Settings")

if ss.get("flash"):
    st.toast(ss.pop("flash"), icon="⚠️")

# ============================================================================ empty state
sid = ss.scan_id
scan = get_scan(sid) if sid else None
if scan is None:
    st.markdown("<div class='bl-empty'><h2>No scan loaded</h2>"
                "<p>Drop one or more hydrophone scans (.mat) into the box in the sidebar.<br>"
                "They're kept in your library so you can come back to them.</p></div>", unsafe_allow_html=True)
    st.stop()

ensure_defaults(sid, scan)
dark = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
C = dict(red="#ff7369" if dark else "#e03e3e", org="#ffa344" if dark else "#d9730d",
         acc="#529cca" if dark else "#2383e2", axis="#4cd7ff" if dark else "#00a3bf",
         mute="#9b9b9b" if dark else "#787774")

# ============================================================================ header
top_l, top_r = st.columns([3, 2], vertical_alignment="center")
top_l.markdown(f"<div class='bl-crumb'>Scans / {scan.title}</div>", unsafe_allow_html=True)
with top_r.container(horizontal=True, horizontal_alignment="right"):
    with st.popover("Save view", type="tertiary"):
        st.text_input("View name", key="new_view_name", placeholder=f"{scan.title} – elevation focus")
        st.button("Save", type="primary", on_click=cb_save_view, args=(sid,), use_container_width=True)
    export_slot = st.empty()
    csv_slot = st.empty()
    show_panel = st.toggle("View settings", value=True, key="show_panel")

st.markdown(f"# {scan.title}")
fixed = AXIS_NAMES[scan.fixed_ax]
props = [("◎ Plane", f"<span class='bl-chip'>{scan.plane}</span>at {fixed} = {scan.fixed_val:.2f} mm"),
         ("∿ Frequency", f"{scan.freq:g} MHz" if scan.freq is not None else "—"),
         ("▦ Grid", f"Δ{scan.hname} {scan.dh:.3g} mm · Δ{scan.vname} {scan.dv:.3g} mm · {len(scan.h)} × {len(scan.v)}"),
         ("◉ Hydrophone", f"{scan.sens:g} mV/MPa · {scan.gain:g} dB gain"
                          + (f" · {scan.navg} averages" if scan.navg else "")),
         ("▣ Source", f"<span style='opacity:.6'>{store.scan(sid).get('filename', '')}</span>")]
st.markdown("<table class='bl-props'>" + "".join(f"<tr><td class='k'>{k}</td><td>{v}</td></tr>" for k, v in props)
            + "</table>", unsafe_allow_html=True)
st.divider()

if show_panel:
    col_map, col_info, col_set = st.columns([1.25, 1.1, 0.8], gap="large")
else:
    col_map, col_info = st.columns([1.0, 1.4], gap="large")
    col_set = None

# ============================================================================ settings panel
s0 = settings_from_state(sid, scan)
ck, c_lo, c_hi = clim_key(sid, scan, s0)


def settings_panel(box):
    with box:
        st.markdown("**View settings**")
        st.markdown("<div class='bl-sec'>SCALE</div>", unsafe_allow_html=True)
        st.selectbox("Scale", scan.units(), key=K(sid, "unit"), label_visibility="collapsed")
        a, b = st.columns(2)
        a.number_input("Sensitivity (mV/MPa)", min_value=0.001, format="%.2f", key=K(sid, "sens"))
        b.number_input("Gain (dB)", format="%.1f", key=K(sid, "gain"))
        st.selectbox("Colormap", list(CMAPS), key=K(sid, "cmap"))
        st.segmented_control("Upscale", UPSCALE_STEPS, format_func=lambda x: f"{x}×", key=K(sid, "up"),
                             required=True, width="stretch")

        st.markdown("<div class='bl-sec'>RANGES</div>", unsafe_allow_html=True)
        range_widget(st.select_slider, K(sid, "hr"), f"{scan.hname} range (mm)", options=opts(scan.h),
                     format_func=lambda x: f"{x:.2f}")
        range_widget(st.select_slider, K(sid, "vr"), f"{scan.vname} range (mm)", options=opts(scan.v),
                     format_func=lambda x: f"{x:.2f}")
        if ck not in ss:
            ss[ck] = (c_lo, c_hi)
        range_widget(st.slider, ck, "Colour range", min_value=c_lo, max_value=c_hi, step=(c_hi - c_lo) / 200,
                     format="%.3g")
        a, b = st.columns(2)
        a.button("Auto colour", on_click=cb_auto_colour, args=(ck,), type="tertiary")
        b.button("Reset ranges", on_click=cb_reset_ranges, args=(sid,), type="tertiary")

        st.markdown("<div class='bl-sec'>OVERLAYS</div>", unsafe_allow_html=True)
        for k, label in (("target", "Focal target"), ("contour", "dB boundary"), ("width", "Focal width"),
                         ("depth", "Focal depth (along axis)"), ("depthnorm", "Normalize by depth"),
                         ("flip", "Flip vertical axis"), ("top", "Axis labels on top"),
                         ("equal", "Equal axis scaling"), ("profile", "Depth profile")):
            st.toggle(label, key=K(sid, k))
        st.number_input("Level (dB) — boundary, width & depth", step=0.5, format="%.1f", key=K(sid, "level"),
                        help="Negative dB relative to the peak (−3 dB = 0.708 × pressure). Positive values fall back to −3.")

        st.markdown("<div class='bl-sec'>BEAM AXIS</div>", unsafe_allow_html=True)
        a, b = st.columns(2)
        a.number_input(f"{scan.hname} at {scan.vname} {ss[K(sid, 'va')]:.2f}", step=0.01, format="%.3f",
                       key=K(sid, "xa"))
        b.number_input(f"{scan.hname} at {scan.vname} {ss[K(sid, 'vb')]:.2f}", step=0.01, format="%.3f",
                       key=K(sid, "xb"))
        st.toggle(f"Deskew to {scan.hname} = 0", key=K(sid, "deskew"), help=f"Tilt {s0.tilt_deg:+.2f}°. "
                  "Shifts each depth row so the beam axis sits at 0 (a shear, not a rotation).")
        st.button("Auto-fit axis", on_click=cb_autofit, args=(sid,), use_container_width=True)


if col_set is not None:
    settings_panel(col_set)

# ============================================================================ compute
s = settings_from_state(sid, scan)
tg = {k: bool(ss.get(K(sid, k), v)) for k, v in TOGGLES.items()}
ck, c_lo, c_hi = clim_key(sid, scan, s)
clim = ss.get(ck, (c_lo, c_hi))
r = compute(scan, s)
if r is None:
    st.warning("No data in the selected range.")
    st.stop()

# ============================================================================ beam map
def map_figure():
    hu, vu = r.hu, r.vu
    lab = r.label
    dunit = "dB" if s.unit == Unit.DB else ("" if lab.startswith("Normalized") else r.unit)
    extra = "" if dunit == r.unit else f"  (%{{customdata:.4g}} {r.unit})"
    fig = go.Figure(go.Heatmap(
        x=hu, y=vu, z=r.dispu, customdata=r.linu, colorscale=CMAPS[ss[K(sid, "cmap")]],
        zmin=clim[0], zmax=clim[1], zsmooth=False,
        colorbar=dict(title=dict(text=lab, side="right"), thickness=12, outlinewidth=0),
        hovertemplate=f"{r.hn} %{{x:.3f}} mm · {scan.vname} %{{y:.3f}} mm<br><b>%{{z:.4g}} {dunit}</b>{extra}<extra></extra>"))
    v = r.v
    if tg["contour"] and r.normu.shape[0] > 1 and r.normu.shape[1] > 1:
        fig.add_trace(go.Contour(x=hu, y=vu, z=r.normu, showscale=False, hoverinfo="skip",
                                 contours=dict(start=r.frac, end=r.frac, size=1, coloring="none"),
                                 line=dict(color="white", dash="dash", width=1.3), name=f"{r.lvl} boundary"))
    if tg["depth"] and r.depth:
        d = r.depth
        fig.add_trace(go.Scatter(x=list(r.ax_x), y=[v[0], v[-1]], mode="lines", hoverinfo="skip",
                                 line=dict(color=C["axis"], width=1), name="beam axis"))
        fig.add_trace(go.Scatter(x=[d["xa"], d["xb"]], y=[d["za_"], d["zb_"]], mode="lines",
                                 line=dict(color=C["red"], width=4), name=f"{r.lvl} depth of field",
                                 hovertemplate=f"{r.lvl} depth of field<extra></extra>"))
    if tg["width"]:
        fig.add_hline(y=r.fv, line=dict(color="white", width=1), opacity=0.7)
        if r.width is not None:
            fig.add_trace(go.Scatter(x=[r.wa, r.wb], y=[r.fv, r.fv], mode="lines+markers",
                                     marker=dict(symbol="line-ns", size=12, line=dict(width=2, color=C["red"])),
                                     line=dict(color=C["red"], width=4), name=f"{r.lvl} width",
                                     hovertemplate=f"{r.lvl} width {r.width:.3f} mm<extra></extra>"))
    if tg["target"]:
        fig.add_trace(go.Scatter(x=[r.fh], y=[r.fv], mode="markers", name="focal max",
                                 marker=dict(symbol="circle-cross-open", size=16, line=dict(width=2, color=C["org"])),
                                 hovertemplate=f"Focal max<br>{r.hn} {r.fh:.2f} · {scan.vname} {r.fv:.2f} mm<br>"
                                               f"{r.fval:.4g} {r.unit}<extra></extra>"))
    he = (hu[0] - (hu[1] - hu[0]) / 2, hu[-1] + (hu[1] - hu[0]) / 2) if hu.size > 1 else (hu[0] - .5, hu[0] + .5)
    ve = (vu[0] - (vu[1] - vu[0]) / 2, vu[-1] + (vu[1] - vu[0]) / 2) if vu.size > 1 else (vu[0] - .5, vu[0] + .5)
    down = scan.depth_is_z != tg["flip"]
    xaxis = dict(range=list(he), title=f"{r.hn} re beam axis (mm)" if s.deskew else f"{scan.hname} position (mm)",
                 side="top" if tg["top"] else "bottom", showgrid=False, zeroline=False,
                 constrain="domain", constraintoward="left")
    yaxis = dict(range=[ve[1], ve[0]] if down else list(ve), showgrid=False, zeroline=False,
                 title=f"{scan.vname} (mm from probe)" if scan.depth_is_z else f"{scan.vname} position (mm)",
                 constrain="domain", constraintoward="top")
    ml, mr = 62, 92                                   # left labels; right = colourbar + its label
    mt, mb = (72, 16) if tg["top"] else (16, 62)
    if tg["equal"]:
        # size the figure so the plot area has exactly the data's aspect ratio
        yaxis.update(scaleanchor="x", scaleratio=1)
        ratio = (he[1] - he[0]) / (ve[1] - ve[0])
        pw = 245 if show_panel else 380
        ph = pw / ratio
        if ph > 620:
            ph, pw = 620, 620 * ratio
        if ph < 240:
            ph, pw = 240, 240 * ratio
        W, H = int(pw + ml + mr), int(ph + mt + mb)
    else:
        W, H = None, 640
    fig.update_layout(height=H, width=W, margin=dict(l=ml, r=mr, t=mt, b=mb),
                      xaxis=xaxis, yaxis=yaxis, showlegend=False, hoverlabel=dict(font_size=12))
    return fig, W


with col_map:
    st.markdown("**Beam map**")
    fig, W = map_figure()
    st.plotly_chart(fig, width=W or "stretch", config={"displaylogo": False, "toImageButtonOptions":
                    {"filename": f"{scan.title}_map", "scale": 3}})

# ============================================================================ summary + profile + details
with col_info:
    st.markdown("**Summary**")
    wl = "slice thickness" if scan.is_yz else "beam width"
    with st.container(border=True):
        a, b = st.columns(2)
        if s.unit in (Unit.PPOS, Unit.PNEG):
            a.metric("Focal p+" if s.unit == Unit.PPOS else "Focal p−", f"{r.fval:.3g} MPa")
        else:
            a.metric("Focal pressure (p–p)", f"{r.fval_mpa:.3g} MPa" if r.fval_mpa is not None else "—")
        b.metric(f"Focal depth · {r.hn} {r.fh:.2f}", f"{r.fv:.2f} mm")
        a, b = st.columns(2)
        a.metric(f"{r.lvl} {wl}", f"{r.width:.3f} mm" if r.width is not None else "—")
        dof, lower = r.dof
        b.metric(f"{r.lvl} depth of field", "—" if dof is None else f"{'> ' if lower else ''}{dof:.2f} mm",
                 help="Still above the level at the edge of the selected depth range — scan deeper to close it."
                 if lower else None)

    if tg["profile"]:
        st.markdown(f"**{'Thickness' if scan.is_yz else 'Width'} & on-axis signal vs depth**")
        pf = make_subplots(specs=[[{"secondary_y": True}]])
        pf.add_trace(go.Scatter(x=r.v, y=r.widths, mode="lines+markers", name=f"{r.lvl} {wl}",
                                line=dict(color=C["acc"], width=2), marker=dict(size=5),
                                hovertemplate="%{y:.3f} mm"), secondary_y=False)
        pf.add_trace(go.Scatter(x=r.v, y=r.amp, mode="lines+markers", name=r.alab,
                                line=dict(color=C["org"], width=2), marker=dict(size=5, symbol="square"),
                                hovertemplate="%{y:.4g}"), secondary_y=True)
        pf.add_vline(x=r.fv, line=dict(color=C["mute"], dash="dot", width=1))
        pf.update_xaxes(title=f"{scan.vname} (mm from probe)" if scan.depth_is_z else f"{scan.vname} position (mm)")
        wmax = np.nanmax(r.widths) if np.any(~np.isnan(r.widths)) else 1
        pf.update_yaxes(title=dict(text=f"{r.lvl} {wl} (mm)", font=dict(color=C["acc"])), range=[0, wmax * 1.15],
                        tickfont=dict(color=C["acc"]), secondary_y=False)
        a2 = dict(title=dict(text=r.alab, font=dict(color=C["org"])), tickfont=dict(color=C["org"]), showgrid=False,
                  tickmode="auto", nticks=6)
        if s.unit != Unit.DB and np.any(~np.isnan(r.amp)) and np.nanmax(r.amp) > 0:
            a2["range"] = [0, float(np.nanmax(r.amp)) * 1.12]
        pf.update_yaxes(**a2, secondary_y=True)
        pf.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                         legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(pf, width="stretch", config={"displaylogo": False})

    st.markdown("**Details**")
    st.code("\n".join(r.lines), language=None, wrap_lines=True)

# ============================================================================ exports (top bar)
export_slot.download_button(
    "Export PNG", type="tertiary", file_name=f"{scan.title}_beam.png", mime="image/png",
    data=lambda: render_png(scan, s, r, tg, ss[K(sid, "cmap")], clim, dark=dark, profile=tg["profile"]))
csv_slot.download_button("Profile CSV", type="tertiary", file_name=f"{scan.title}_profile.csv", mime="text/csv",
                         data=profile_csv(scan, r))
