"""
Beam Lab (web) - hydrophone beam-scan viewer as a Streamlit app.

Run locally:   streamlit run app.py
Then open:     http://localhost:8501   (others on your network: http://<this-pc>:8501)

Users upload .mat scans in the browser; nothing to install on their side. Each signed-in
user gets their own scan library and saved views (stored under BEAMLAB_DATA on the server).
Sign-in is optional: configure [auth] in .streamlit/secrets.toml to require it (see README).
"""
from __future__ import annotations

import json
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from beamcore import (AXIS_NAMES, Scan, Settings, Unit, UPSCALE_STEPS, autofit, colour_bounds, compute,
                      profile_csv, render_png)
from store import UserStore
import merge as mg

APP_VERSION = "2026-10-28 build 11"     # shown in the sidebar so you can confirm what is deployed

st.set_page_config(page_title="Beam Lab", page_icon="〰️", layout="wide", initial_sidebar_state="expanded")

CMAPS = {"viridis": "Viridis", "inferno": "Inferno", "magma": "Magma", "plasma": "Plasma",
         "cividis": "Cividis", "turbo": "Turbo", "jet": "Jet", "gray": "Greys_r"}
TOGGLES = dict(target=True, contour=True, width=True, depth=False, depthnorm=False, flip=True,
               profile=True, deskew=False)
MAP_CELLS_MAX = 60_000            # cap on heat-map cells sent to the browser (speed); export uses full upscale
ss = st.session_state

st.markdown("""
<style>
.block-container {padding-top: 3.2rem; padding-bottom: 1.5rem; max-width: 100%;}
section[data-testid="stSidebar"] .stButton button, section[data-testid="stSidebar"] .stButton button > div
    {justify-content: flex-start; text-align: left; width: 100%;}
section[data-testid="stSidebar"] .stButton button p {white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.bl-crumb {opacity: .6; font-size: .9rem; margin-bottom: .2rem;}
.bl-props {border-collapse: collapse; margin: .2rem 0 .4rem 0; font-size: .93rem;}
.bl-props td {padding: 3px 28px 3px 0; border: none;}
.bl-props td.k {opacity: .6; white-space: nowrap;}
.bl-props td {word-break: break-word;}
.bl-props tr, .bl-props table {border: none; background: transparent !important;}
.bl-chip {background: rgba(128,128,128,.16); border-radius: 4px; padding: 1px 7px; margin-right: 6px; font-size: .86rem;}
.bl-sec {opacity: .55; font-size: .78rem; font-weight: 600; margin: .9rem 0 .2rem 0; letter-spacing: .02em;}
.bl-empty {text-align: center; padding: 14vh 0 2rem 0; opacity: .8;}
div[data-testid="stMetricValue"] {font-size: 1.5rem;}
.bl-sum-title {font-weight: 600; margin: .1rem 0 .4rem 0;}
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


# Settings live in plain session-state keys (they survive when the settings panel is hidden);
# each widget gets its own "#w" key and copies its value back on change.
def _sync(stored_key: str):
    def f():
        ss[stored_key] = ss[stored_key + "#w"]
    return f


def w_toggle(sid, name, label, **kw):
    k = K(sid, name)
    return st.toggle(label, value=bool(ss[k]), key=k + "#w", on_change=_sync(k), **kw)


def w_number(container, sid, name, label, **kw):
    k = K(sid, name)
    return container.number_input(label, value=float(ss[k]), key=k + "#w", on_change=_sync(k), **kw)


def w_select(sid, name, label, options, **kw):
    k = K(sid, name)
    opts_ = list(options)
    idx = opts_.index(ss[k]) if ss[k] in opts_ else 0
    return st.selectbox(label, opts_, index=idx, key=k + "#w", on_change=_sync(k), **kw)


def w_segment(sid, name, label, options, **kw):
    k = K(sid, name)
    return st.segmented_control(label, options, default=ss[k], key=k + "#w", on_change=_sync(k), **kw)


def drop_widgets(sid: str) -> None:
    """Forget widget state for a scan so widgets re-read the stored settings (after programmatic changes)."""
    for key in [x for x in ss.keys() if str(x).startswith(sid + ".") and str(x).endswith("#w")]:
        del ss[key]


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
        drop_widgets(sid)


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
    drop_widgets(sid)


# ============================================================================ sidebar
ss.setdefault("scan_id", store.scans[0]["id"] if store.scans else None)
ss.setdefault("active_view", None)
ss.setdefault("up_n", 0)
if ss.scan_id and not store.scan(ss.scan_id):
    ss.scan_id = store.scans[0]["id"] if store.scans else None

with st.sidebar:
    st.markdown("### Beam Lab")
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
    names = [e["name"] for e in lib]
    for e in lib:
        c1, c2 = st.columns([6, 1], vertical_alignment="center")
        active = e["id"] == ss.scan_id
        shown = e["name"]
        if names.count(e["name"]) > 1:                    # same device name twice: add the file name
            shown += f" · {os.path.splitext(e.get('filename', ''))[0][:22]}"
        c1.button(f"{'▣' if active else '▤'}  {shown}", key=f"s_{e['id']}", on_click=cb_select,
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
    # Dark-mode switch. Streamlit has no Python API to change the theme, so the switch operates
    # Streamlit's own Light/Dark item in the ⋮ menu with a tiny script (same effect, remembered per browser).
    is_dark_now = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
    if "dark" not in ss or (not ss.get("theme_pending") and ss.dark != is_dark_now):
        ss.dark = is_dark_now                      # follow changes made from the ⋮ menu
        ss.pop("dark#w", None)

    def _toggle_dark():
        ss.dark = ss["dark#w"]
        ss.theme_pending = "Dark" if ss.dark else "Light"

    st.toggle("Dark mode", value=ss.dark, key="dark#w", on_change=_toggle_dark)
    st.caption(f"Beam Lab {APP_VERSION}")
    if ss.get("theme_pending"):
        want = ss.pop("theme_pending")
        ss.theme_nonce = ss.get("theme_nonce", 0) + 1
        st.html(f"""<span data-nonce="{ss.theme_nonce}"></span><script>
        (function () {{
          function pick(n) {{
            var item = document.querySelector('[data-testid="stMainMenuItem-theme-{want}"]');
            if (item) {{
              item.click();
              setTimeout(function () {{           // close Streamlit's menu again if it stayed open
                if (document.querySelector('[data-testid="stMainMenuList"]')) {{
                  var m = document.querySelector('[data-testid="stMainMenu"] button') ||
                          document.querySelector('[data-testid="stMainMenu"]');
                  if (m) m.click();
                }}
              }}, 150);
              return;
            }}
            var menu = document.querySelector('[data-testid="stMainMenu"] button') ||
                       document.querySelector('[data-testid="stMainMenu"]');
            if (menu && n === 0) menu.click();
            if (n < 30) setTimeout(function () {{ pick(n + 1); }}, 100);
          }}
          pick(0);
        }})();
        </script>""", unsafe_allow_javascript=True, width="content")

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
entry = store.scan(sid)
spath = store.scan_path(sid)
SKEY = (str(spath), spath.stat().st_mtime, entry.get("filename", ""))   # cache key for this scan


@st.cache_data(show_spinner=False, max_entries=64)
def cached_compute(path: str, mtime: float, filename: str, settings_json: str, display_up: int | None = None):
    return compute(load_scan(path, mtime, filename), Settings.from_dict(json.loads(settings_json)), display_up)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_bounds(path: str, mtime: float, filename: str, settings_json: str):
    return colour_bounds(load_scan(path, mtime, filename), Settings.from_dict(json.loads(settings_json)))


def sjson(st_: Settings) -> str:
    return json.dumps(st_.as_dict(), sort_keys=True)


def clim_key_fast(st_: Settings):
    lo, hi = cached_bounds(*SKEY, sjson(st_))
    return K(sid, f"clim:{lo:.6g}:{hi:.6g}"), lo, hi


ss.setdefault("show_panel", True)
lvl_txt = f"{ss[K(sid, 'level')] if ss[K(sid, 'level')] < 0 else -3.0:g} dB"
if ss.show_panel:
    main, side = st.columns([3.3, 1], gap="large")
else:
    main, side = st.container(), None

# ============================================================================ settings panel (full height, right)
s0 = settings_from_state(sid, scan)
ck, c_lo, c_hi = clim_key_fast(s0)

if side is not None:
    with side:
        st.markdown("**View settings**")
        st.markdown("<div class='bl-sec'>SCALE</div>", unsafe_allow_html=True)
        w_select(sid, "unit", "Scale", scan.units(), label_visibility="collapsed")
        a, b = st.columns(2)
        w_number(a, sid, "sens", "Sensitivity (mV/MPa)", min_value=0.001, format="%.2f")
        w_number(b, sid, "gain", "Gain (dB)", format="%.1f")
        w_select(sid, "cmap", "Colormap", CMAPS)
        w_segment(sid, "up", "Upscale", UPSCALE_STEPS, format_func=lambda x: f"{x}×", required=True, width="stretch")
        ss.setdefault("map_h", 700)

        def _sync_maph():
            ss.map_h = ss["map_h#w"]

        st.segmented_control("Map size", [500, 700, 900], format_func=lambda x: {500: "S", 700: "M", 900: "L"}[x],
                             default=ss.map_h, key="map_h#w", on_change=_sync_maph, required=True, width="stretch")

        st.markdown("<div class='bl-sec'>BEAM AXIS CORRECTION</div>", unsafe_allow_html=True)
        w_toggle(sid, "deskew", f"Deskew (beam axis → {scan.hname} = 0)",
                  help=f"Fitted tilt {s0.tilt_deg:+.2f}°. Shifts each depth row so the beam axis sits at "
                       f"{scan.hname} = 0 (a shear, not a rotation). Turn off to see the raw data.")
        a, b = st.columns(2)
        w_number(a, sid, "xa", f"{scan.hname} at {scan.vname} {ss[K(sid, 'va')]:.2f}", step=0.01, format="%.3f")
        w_number(b, sid, "xb", f"{scan.hname} at {scan.vname} {ss[K(sid, 'vb')]:.2f}", step=0.01, format="%.3f")
        st.button(f"Auto-fit axis  (tilt {s0.tilt_deg:+.2f}°)", on_click=cb_autofit, args=(sid,),
                  use_container_width=True)

        st.markdown("<div class='bl-sec'>RANGES</div>", unsafe_allow_html=True)
        rng = st.container(gap=None)
        rng.__enter__()
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
        rng.__exit__(None, None, None)

        st.markdown("<div class='bl-sec'>OVERLAYS</div>", unsafe_allow_html=True)
        for k, label in (("target", "Focal target"), ("contour", f"dB boundary ({lvl_txt})"),
                         ("width", f"Focal width ({lvl_txt})"), ("depth", f"Plot focal depth axis ({lvl_txt})"),
                         ("depthnorm", "Lateral normalization"), ("flip", "Flip vertical axis (low at bottom)"),
                         ("profile", "Plot parameters")):
            w_toggle(sid, k, label)
        w_number(st, sid, "level", "Level (dB) — boundary, width & depth", step=0.5, format="%.1f",
                        help="Negative dB relative to the peak (−3 dB = 0.708 × pressure). Positive values fall back to −3.")

# ============================================================================ compute (cached)
s = settings_from_state(sid, scan)
tg = {k: bool(ss.get(K(sid, k), v)) for k, v in TOGGLES.items()}
ck, c_lo, c_hi = clim_key_fast(s)
clim = ss.get(ck, (c_lo, c_hi))
user_up = s.upscale
cells = len(scan.h) * len(scan.v)
cap = max([f for f in UPSCALE_STEPS if cells * f * f <= MAP_CELLS_MAX] or [1])
meas_cap = max([f for f in UPSCALE_STEPS if cells * f * f <= 1_500_000] or [1])   # memory guard for big grids
if s.upscale > meas_cap:
    s = Settings.from_dict({**s.as_dict(), "upscale": meas_cap})
r = cached_compute(*SKEY, sjson(s), min(user_up, cap))     # measure on full upscale, display capped
if r is None:
    with main:
        st.warning("No data in the selected range.")
    st.stop()
wl = "beamwidth"


# ============================================================================ figures
def map_figure(plot_h: int):
    hu, vu = r.hu, r.vu
    lab = r.label
    if s.unit == Unit.DB:
        dunit = "dB"
    elif s.unit == Unit.NORM or s.depthnorm:
        dunit = ""
    else:
        dunit = r.unit
    extra = "" if dunit == r.unit else f"  (%{{customdata:.4g}} {r.unit})"
    fig = go.Figure(go.Heatmap(
        x=hu, y=vu, z=r.dispu, customdata=r.linu, colorscale=CMAPS[ss[K(sid, "cmap")]],
        zmin=clim[0], zmax=clim[1], zsmooth="best" if user_up > 1 else False,
        showscale=False,
        hovertemplate=f"{r.hn} %{{x:.3f}} mm · {scan.vname} %{{y:.3f}} mm<br><b>%{{z:.4g}} {dunit}</b>{extra}<extra></extra>"))
    v = r.v
    if tg["contour"] and r.normu.shape[0] > 1 and r.normu.shape[1] > 1:
        fig.add_trace(go.Contour(x=hu, y=vu, z=r.normu, showscale=False, hoverinfo="skip",
                                 contours=dict(start=r.frac, end=r.frac, size=1, coloring="none"),
                                 line=dict(color="white", dash="dash", width=1.3, smoothing=1.0)))
    if tg["depth"] and r.depth:
        d = r.depth
        fig.add_trace(go.Scatter(x=list(r.ax_x), y=[v[0], v[-1]], mode="lines", hoverinfo="skip",
                                 line=dict(color=C["axis"], width=1)))
        fig.add_trace(go.Scatter(x=[d["xa"], d["xb"]], y=[d["za_"], d["zb_"]], mode="lines",
                                 line=dict(color=C["red"], width=4),
                                 hovertemplate=f"{r.lvl} depth of field<extra></extra>"))
    if tg["width"]:
        fig.add_hline(y=r.fv, line=dict(color="white", width=1), opacity=0.7)
        if r.width is not None:
            fig.add_trace(go.Scatter(x=[r.wa, r.wb], y=[r.fv, r.fv], mode="lines+markers",
                                     marker=dict(symbol="line-ns", size=12, line=dict(width=2, color=C["red"])),
                                     line=dict(color=C["red"], width=4),
                                     hovertemplate=f"{r.lvl} width {r.width:.3f} mm<extra></extra>"))
    if scan.detail_box:                                   # merged scan: outline the fine-data region
        bx = scan.detail_box
        fig.add_shape(type="rect", x0=bx[0], x1=bx[1], y0=bx[2], y1=bx[3], xref="x", yref="y",
                      line=dict(color="rgba(255,255,255,.75)", width=1, dash="dot"))
    if tg["target"]:
        fig.add_trace(go.Scatter(x=[r.fh], y=[r.fv], mode="markers",
                                 marker=dict(symbol="circle-cross-open", size=16, line=dict(width=2, color=C["org"])),
                                 hovertemplate=f"Focal max<br>{r.hn} {r.fh:.2f} · {scan.vname} {r.fv:.2f} mm<br>"
                                               f"{r.fval:.4g} {r.unit}<extra></extra>"))
    dh = (hu[1] - hu[0]) / 2 if hu.size > 1 else 0.5
    dv = (vu[1] - vu[0]) / 2 if vu.size > 1 else 0.5
    he, ve = (hu[0] - dh, hu[-1] + dh), (vu[0] - dv, vu[-1] + dv)
    down = scan.depth_is_z != tg["flip"]
    xt = f"{scan.hname} position (mm)"
    # x labels on top and bottom: a second x axis that mirrors the first
    fig.add_trace(go.Scatter(x=[he[0]], y=[ve[0]], xaxis="x2", mode="markers", marker=dict(opacity=0),
                             hoverinfo="skip", showlegend=False))

    # Colour scale as a horizontal bar under the map. It lives on its own x-axis that *matches* the map's
    # x-axis, so Plotly keeps it exactly as wide as the map (even when equal scaling narrows the map), and
    # the map is anchored to the bottom of its area so the bar always sits directly beneath it.
    lo, hi = clim
    g = np.linspace(lo, hi, 256)
    gx = np.linspace(he[0], he[1], 256)
    fig.add_trace(go.Heatmap(x=gx, y=[0, 1], z=[g, g], colorscale=CMAPS[ss[K(sid, "cmap")]], zmin=lo, zmax=hi,
                             showscale=False, xaxis="x3", yaxis="y3",
                             hovertemplate=f"%{{z:.4g}} {dunit}<extra></extra>"))
    from matplotlib.ticker import MaxNLocator
    tv = [t for t in MaxNLocator(nbins=6).tick_values(lo, hi) if lo - 1e-12 <= t <= hi + 1e-12]
    tpos = [he[0] + (t - lo) / (hi - lo) * (he[1] - he[0]) for t in tv]

    ml, mr, mt, mb = 62, 16, 64, 48                      # mt: top x labels; mb: bar ticks + title
    bar_px, gap_px = 14, 58                              # gap: bottom x tick labels + title above the bar
    ratio = (he[1] - he[0]) / (ve[1] - ve[0])
    ph = plot_h                                          # map height is fixed; the plots panel slides over
    H = int(round(mt + ph + gap_px + bar_px + mb))
    P = H - mt - mb
    fig.update_layout(
        height=H, margin=dict(l=ml, r=mr, t=mt, b=mb), showlegend=False, hoverlabel=dict(font_size=12),
        xaxis=dict(range=list(he), title=xt, side="bottom", showgrid=False, zeroline=False, ticks="outside",
                   constrain="domain", constraintoward="left"),        # map sits top-left in its zone
        xaxis2=dict(range=list(he), title=xt, side="top", overlaying="x", matches="x", showgrid=False,
                    zeroline=False, ticks="outside"),
        yaxis=dict(range=[ve[1], ve[0]] if down else list(ve), showgrid=False, zeroline=False, ticks="outside",
                   title=f"{scan.vname} (mm from probe)" if scan.depth_is_z else f"{scan.vname} position (mm)",
                   domain=[(bar_px + gap_px) / P, 1], scaleanchor="x", scaleratio=1,
                   constrain="domain", constraintoward="top"),
        xaxis3=dict(matches="x", anchor="y3", constrain="domain", constraintoward="left", showgrid=False,
                    zeroline=False, ticks="outside", tickvals=tpos, ticktext=[f"{t:g}" for t in tv],
                    title=dict(text=lab, standoff=4)),
        yaxis3=dict(domain=[0, bar_px / P], anchor="x3", showticklabels=False, ticks="", showgrid=False,
                    zeroline=False, fixedrange=True, range=[0, 1]))
    return fig, H


CUR_COL = (("#9a6dd7", "#9065b0"), ("#4dab9a", "#448361"))      # (dark, light) colours of markers 1 and 2


def profile_figure(H: int, ncur: int = 0):
    pf = make_subplots(specs=[[{"secondary_y": True}]])
    pv = r.pv if r.pv is not None else r.v
    mode = "lines+markers" if len(pv) <= 60 else "lines"
    pf.add_trace(go.Scatter(x=pv, y=r.widths, mode=mode, name=f"{r.lvl} Beamwidth",
                            line=dict(color=C["acc"], width=2), marker=dict(size=5),
                            hovertemplate="%{y:.3f} mm"), secondary_y=False)
    pf.add_trace(go.Scatter(x=pv, y=r.amp, mode=mode, name=r.alab,
                            line=dict(color=C["org"], width=2), marker=dict(size=5, symbol="square"),
                            hovertemplate="%{y:.4g}"), secondary_y=True)
    wmax = np.nanmax(r.widths) if np.any(~np.isnan(r.widths)) else 1
    # focal-depth guide as a trace (not a shape) so that only the markers are draggable
    pf.add_trace(go.Scatter(x=[r.fv, r.fv], y=[0, wmax * 1.15], mode="lines", showlegend=False, hoverinfo="skip",
                            line=dict(color=C["mute"], dash="dot", width=1)), secondary_y=False)
    pf.update_xaxes(title=f"{scan.vname} (mm from probe)" if scan.depth_is_z else f"{scan.vname} position (mm)")
    pf.update_yaxes(title=dict(text=f"{r.lvl} Beamwidth (mm)", font=dict(color=C["acc"])), range=[0, wmax * 1.15],
                    tickfont=dict(color=C["acc"]), secondary_y=False)
    a2 = dict(title=dict(text=r.alab, font=dict(color=C["org"])), tickfont=dict(color=C["org"]), showgrid=False,
              tickmode="auto", nticks=6)
    if s.unit != Unit.DB and np.any(~np.isnan(r.amp)) and np.nanmax(r.amp) > 0:
        a2["range"] = [0, float(np.nanmax(r.amp)) * 1.12]
    pf.update_yaxes(**a2, secondary_y=True)
    bottom = 58
    if ncur:
        # draggable markers: vertical line shapes moved by the app's own JS drag (see CURSOR_JS), which also
        # writes the readout under the plot
        lo, hi = float(np.nanmin(pv)), float(np.nanmax(pv))
        span = hi - lo
        x2 = r.fv + span / 4 if r.fv + span / 4 <= hi else r.fv - span / 4
        for i, x in enumerate((r.fv, x2)[:ncur]):
            col = CUR_COL[i][0 if dark else 1]
            pf.add_shape(type="line", name=f"cur{i + 1}", xref="x", yref="paper", x0=x, x1=x, y0=0, y1=1,
                         line=dict(color=col, width=2, dash="dash"), layer="above")
            pf.add_annotation(name=f"curlab{i + 1}", text=f"<b>{i + 1}</b>", x=x, xref="x", y=1, yref="paper",
                              yanchor="bottom", showarrow=False, font=dict(color="#ffffff", size=11),
                              bgcolor=col, borderpad=2)
        pf.add_annotation(name="readout", text=" ", x=0, xref="paper", xanchor="left", y=0, yref="paper",
                          yanchor="top", yshift=-50, align="left", showarrow=False,
                          font=dict(family="JetBrains Mono, SFMono-Regular, Menlo, Consolas, monospace", size=12))
        bottom += 20 * (ncur + (ncur == 2)) + 6
        sig_unit = {Unit.DB: "dB", Unit.NORM: "", Unit.VPP: "mV"}.get(s.unit, "MPa")
        pf.update_layout(meta=dict(bl=1, key=f"{sid}|{scan.vname}", z=scan.vname, unit=sig_unit,
                                   db=s.unit == Unit.DB, norm=s.unit == Unit.NORM,
                                   cols=[CUR_COL[i][0 if dark else 1] for i in range(ncur)]))
    pf.update_layout(height=H + bottom - 58, margin=dict(l=10, r=10, t=64, b=bottom), hovermode="x unified",
                     legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"))
    return pf


# JavaScript for the depth-plot markers. Our own drag handler (not Plotly shape editing) moves a marker
# horizontally only and redraws the readout on every animation frame while dragging. Positions are kept per
# scan across reruns.
# Note: Streamlit sanitizes st.html with DOMPurify, which drops a <script> whose text contains "<tag"-like
# sequences, so HTML inside the JS strings is written with \x3c instead of a literal "<".
CURSOR_JS = r"""<script>
(function () {
  if (window.__blCurInit) return;
  window.__blCurInit = true;
  window.__blCur = window.__blCur || {};
  var CIRC = ['①', '②'], GRAB = 8;          // grab distance in px either side of a marker
  var st = document.createElement('style');
  st.textContent = '.bl-cur-near .nsewdrag, .bl-cur-near .draglayer * {cursor: ew-resize !important}' +
                   '.bl-cur-drag, .bl-cur-drag * {cursor: ew-resize !important; user-select: none !important}' +
                   '.bl-cur-drag .hoverlayer, .bl-cur-drag .spikelines {display: none !important}';   // no stale hover box
  document.head.appendChild(st);
  function fin(v) { return typeof v === 'number' && isFinite(v); }
  function interp(xs, ys, x) {
    for (var i = 0; i < xs.length - 1; i++) {
      var a = xs[i], b = xs[i + 1];
      if (x >= Math.min(a, b) && x <= Math.max(a, b)) {
        var ya = ys[i], yb = ys[i + 1];
        if (a === b) return fin(ya) ? ya : NaN;
        var f = (x - a) / (b - a);
        if (!fin(ya) || !fin(yb)) return f < 0.5 ? (fin(ya) ? ya : NaN) : (fin(yb) ? yb : NaN);
        return ya + (yb - ya) * f;
      }
    }
    return xs.length === 1 && x === xs[0] ? ys[0] : NaN;
  }
  function sgn(v, d) { var q = Number(v.toFixed(d)); return (q >= 0 ? '+' : '−') + Math.abs(q).toFixed(d); }
  function fsig(v, m) {
    if (!fin(v)) return '—';
    if (m.db) return v.toFixed(2) + ' dB';
    if (m.norm) return v.toFixed(3);
    return Number(v.toPrecision(4)) + ' ' + m.unit;
  }
  function ok(gd) { var m = gd.layout && gd.layout.meta; return m && m.bl && gd._fullData && gd._fullData.length >= 2 && gd._fullLayout && gd._fullLayout.xaxis; }
  function data(gd) {
    var xs = Array.from(gd._fullData[0].x || []).map(Number), fx = xs.filter(fin);
    return {xs: xs, w: Array.from(gd._fullData[0].y || []).map(Number), sg: Array.from(gd._fullData[1].y || []).map(Number),
            lo: Math.min.apply(null, fx), hi: Math.max.apply(null, fx), n: fx.length};
  }
  function cursorShapes(gd) {
    var out = [];
    (gd.layout.shapes || []).forEach(function (s, i) { if (s.name && /^cur\d$/.test(s.name)) out.push([i, s]); });
    return out;
  }
  function annIndex(gd, name) {
    var a = gd.layout.annotations || [];
    for (var i = 0; i < a.length; i++) if (a[i].name === name) return i;
    return -1;
  }
  function memOf(gd) { var k = gd.layout.meta.key; return window.__blCur[k] = window.__blCur[k] || {}; }
  // bring the plot in line with the stored marker positions (only touches what differs)
  function sync(gd) {
    if (!ok(gd)) return;
    var m = gd.layout.meta, D = data(gd);
    if (!D.n) return;
    var mem = memOf(gd), upd = {}, rows = [], vals = [];
    function get(k) { return k.split('.').reduce(function (o, p) {
        var mm = p.match(/^(\w+)\[(\d+)\]$/); return o == null ? o : (mm ? (o[mm[1]] || [])[+mm[2]] : o[p]); }, gd.layout); }
    function set(k, v) { var cur = get(k);
      if (typeof v === 'number' ? !(fin(+cur) && Math.abs(+cur - v) < 1e-9) : cur !== v) upd[k] = v; }
    cursorShapes(gd).forEach(function (c) {
      var i = c[0], s = c[1], n = +s.name.slice(3);
      var x = mem[s.name] !== undefined ? mem[s.name] : +s.x0;
      if (!fin(x)) x = (D.lo + D.hi) / 2;
      x = Math.min(D.hi, Math.max(D.lo, x));
      mem[s.name] = x;
      set('shapes[' + i + '].x0', x); set('shapes[' + i + '].x1', x);
      var li = annIndex(gd, 'curlab' + n);
      if (li >= 0) set('annotations[' + li + '].x', x);
      var bw = interp(D.xs, D.w, x), sv = interp(D.xs, D.sg, x);
      vals.push([x, bw, sv]);
      var col = (m.cols || [])[n - 1] || 'inherit';
      rows.push('\x3cspan style="color:' + col + '">\x3cb>' + CIRC[n - 1] + '\x3c/b>\x3c/span> ' + m.z + ' ' + x.toFixed(2) +
                ' mm · BW ' + (fin(bw) ? bw.toFixed(3) + ' mm' : '—') + ' · ' + fsig(sv, m));
    });
    if (vals.length === 2) {
      var a = vals[0], b = vals[1], d = 'Δ' + m.z + ' ' + Math.abs(b[0] - a[0]).toFixed(2) + ' mm · ΔBW ' +
          (fin(a[1]) && fin(b[1]) ? sgn(b[1] - a[1], 3) + ' mm' : '—') +
          (gd._fullLayout.width < 620 ? '\x3cbr>   ' : ' · ') + 'Δsig ';   // wrap on narrow panels
      if (!fin(a[2]) || !fin(b[2])) d += '—';
      else if (m.db) d += sgn(b[2] - a[2], 2) + ' dB';
      else if (a[2] > 0 && b[2] > 0) d += sgn(100 * (b[2] / a[2] - 1), 1) + ' % (' + sgn(20 * Math.log10(b[2] / a[2]), 2) + ' dB)';
      else d += sgn(b[2] - a[2], 4) + (m.unit ? ' ' + m.unit : '');
      rows.push(d + '  \x3cspan style="opacity:.55">(②−①)\x3c/span>');
    }
    var ri = annIndex(gd, 'readout');
    if (ri >= 0) {
      var txt = rows.join('\x3cbr>') || ' ';
      set('annotations[' + ri + '].text', txt);
      set('margin.b', 58 + 20 * (txt.split('\x3cbr>').length) + 6);
    }
    if (Object.keys(upd).length && !gd.__blBusy) {
      gd.__blBusy = true;
      Promise.resolve(window.Plotly.relayout(gd, upd)).finally(function () {
        gd.__blBusy = false;
        if (gd.__blAgain) { gd.__blAgain = false; sync(gd); }     // a newer position arrived meanwhile
      });
    } else if (Object.keys(upd).length) gd.__blAgain = true;
  }
  // pixel helpers
  function geom(gd, ev) {
    var r = gd.getBoundingClientRect(), fl = gd._fullLayout, xa = fl.xaxis;
    return {px: ev.clientX - r.left - xa._offset, py: ev.clientY - r.top, xa: xa,
            top: fl._size.t - 22, bot: fl._size.t + fl._size.h, len: xa._length};
  }
  function nearest(gd, ev) {
    if (!ok(gd)) return null;
    var g = geom(gd, ev);
    if (g.py < g.top || g.py > g.bot || g.px < -GRAB || g.px > g.len + GRAB) return null;
    var mem = memOf(gd), best = null, bd = GRAB + 1;
    cursorShapes(gd).forEach(function (c) {
      var x = mem[c[1].name] !== undefined ? mem[c[1].name] : +c[1].x0, d = Math.abs(g.xa.l2p(x) - g.px);
      if (d < bd) { bd = d; best = c[1].name; }
    });
    return best;
  }
  var drag = null;
  function moveTo(gd, name, ev) {
    var g = geom(gd, ev), D = data(gd);
    var x = g.xa.p2l(Math.min(g.len, Math.max(0, g.px)));
    memOf(gd)[name] = Math.min(D.hi, Math.max(D.lo, x));
    if (!gd.__blRaf) gd.__blRaf = requestAnimationFrame(function () { gd.__blRaf = 0; sync(gd); });
  }
  function attach(gd) {
    var host = gd._ev || gd;            // Plotly drops all listeners (and _ev) when it re-creates a plot
    if (!host.__blCursors && gd.on) {
      host.__blCursors = true;
      gd.on('plotly_afterplot', function () { if (!drag) sync(gd); });
      gd.on('plotly_click', function (e) {   // click on the plot: move the nearest marker there
        if (!e || !e.points || !e.points.length || !fin(+e.points[0].x) || !ok(gd)) return;
        var x = +e.points[0].x, mem = memOf(gd), best = null, bd = Infinity;
        cursorShapes(gd).forEach(function (c) { var v = mem[c[1].name] !== undefined ? mem[c[1].name] : +c[1].x0;
                                                if (Math.abs(v - x) < bd) { bd = Math.abs(v - x); best = c[1].name; } });
        if (best) { mem[best] = x; sync(gd); }
      });
    }
    if (!gd.__blDom) {                  // DOM listeners survive a Plotly re-plot of the same div
      gd.__blDom = true;
      gd.addEventListener('mousedown', function (ev) {
        if (ev.button !== 0) return;
        var name = nearest(gd, ev);
        if (!name) return;
        ev.stopPropagation(); ev.preventDefault();      // keep Plotly from starting a zoom box
        drag = {gd: gd, name: name};
        document.body.classList.add('bl-cur-drag');
        if (window.Plotly.Fx) window.Plotly.Fx.unhover(gd);
        moveTo(gd, name, ev);
      }, true);
      gd.addEventListener('mousemove', function (ev) {
        if (drag) return;
        gd.classList.toggle('bl-cur-near', !!nearest(gd, ev));
      });
      gd.addEventListener('mouseleave', function () { gd.classList.remove('bl-cur-near'); });
    }
  }
  window.addEventListener('mousemove', function (ev) {
    if (!drag) return;
    ev.preventDefault(); ev.stopPropagation();
    moveTo(drag.gd, drag.name, ev);
  }, true);
  window.addEventListener('mouseup', function (ev) {
    if (!drag) return;
    ev.stopPropagation();
    var d = drag; drag = null;
    document.body.classList.remove('bl-cur-drag');
    moveTo(d.gd, d.name, ev);
  }, true);
  setInterval(function () {
    if (!window.Plotly) return;
    document.querySelectorAll('.st-key-profchart .js-plotly-plot').forEach(function (gd) {
      attach(gd);
      if (!drag && !gd.__blBusy) sync(gd);
    });
  }, 400);
})();
</script>"""


def export_png() -> bytes:
    rf = cached_compute(*SKEY, sjson(s), None)               # full requested upscale for the export
    return render_png(scan, s, rf, {**tg, "top": True}, ss[K(sid, "cmap")], clim, dark=dark, profile=tg["profile"])


# ============================================================================ merge dialog
@st.dialog("Merge scans", width="large")
def merge_dialog():
    lib = [e for e in store.scans if store.scan_path(e["id"]).exists()]
    if len(lib) < 2:
        st.info("Import at least two scans of the same plane first (a coarse survey and a fine detail scan).")
        return
    ids = [e["id"] for e in lib]
    label = {e["id"]: f"{e['name']} · {e.get('plane', '')} · {e.get('filename', '')}" for e in lib}
    st.caption("Combines a coarse survey with a fine scan of a small region (e.g. 0.2 mm overall + 0.05 mm "
               "around the focus). The result is saved to your library as a new scan; the originals are untouched.")
    a, b = st.columns(2)
    base_id = a.selectbox("Base scan (coarse, whole beam)", ids, index=ids.index(sid) if sid in ids else 0,
                          format_func=lambda i: label[i])
    det_opts = [i for i in ids if i != base_id]
    det_id = b.selectbox("Detail scan (fine, small region)", det_opts, format_func=lambda i: label[i])
    align = st.radio("Position the detail scan using", [mg.ALIGN_STAGE, mg.ALIGN_AUTO, mg.ALIGN_SCAN],
                     help="Stage coordinates (recommended): absolute stage positions saved in each file — correct "
                          "even if the fine scan was re-centred or a different start distance was typed. "
                          "Auto-align: starts from stage coordinates and refines the offset by matching the overlap "
                          "(use if the stage was re-homed; accurate to ~0.02 mm). Scan coordinates: PDx/PDy/PDz as "
                          "written — only correct if both scans share the same centre and start distance.")
    with st.expander("Advanced"):
        c1, c2, c3 = st.columns(3)
        base = get_scan(base_id)
        xh = c1.number_input(f"Extra {base.hname} offset (mm)", value=0.0, step=0.01, format="%.3f")
        xv = c2.number_input(f"Extra {base.vname} offset (mm)", value=0.0, step=0.01, format="%.3f")
        feather = c3.slider("Blend width (coarse steps)", 0.0, 4.0, 2.0, 0.5,
                            help="Width of the cosine blend just inside the fine box, to avoid a seam.")
        match = st.toggle("Scale the detail scan to match the base amplitude in the overlap", value=True)
    base, det = get_scan(base_id), get_scan(det_id)
    try:
        merged, rep = mg.merge_scans(base, det, align=align, extra_h=xh, extra_v=xv,
                                     match_amplitude=match, feather_steps=feather)
    except ValueError as e:
        st.error(str(e))
        return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Offset applied", f"{rep.offset_h:+.3f} / {rep.offset_v:+.3f}", help=f"{base.hname} / {base.vname} (mm)")
    m2.metric("Amplitude ratio", f"{rep.scale:.3f}", help="Base ÷ detail over the overlap; ~1.00 expected.")
    m3.metric("Overlap residual", f"{rep.residual_pct:.1f} %",
              help="RMS difference between the scans where they overlap, as % of peak. Includes noise and the "
                   "coarse scan's own interpolation error; under ~2–3 % is a good match.")
    m4.metric("Merged grid", f"{rep.shape[1]} × {rep.shape[0]}", help=f"{rep.step_h:g} × {rep.step_v:g} mm spacing")
    for w in rep.warnings:
        st.warning(w)
    if not rep.warnings:
        st.success("The scans agree well in the overlap.")
    if st.button("Create merged scan", type="primary", use_container_width=True):
        e = store.add_scan(mg.scan_to_mat_bytes(merged), f"{base.title}_merged_{det.dh:g}mm.mat")
        cb_select(e["id"])
        st.rerun()


# ============================================================================ main column
with main:
    # top bar
    top_l, top_r = st.columns([2, 3], vertical_alignment="center")
    top_l.markdown(f"<div class='bl-crumb'>Scans / {scan.title}</div>", unsafe_allow_html=True)
    with top_r.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center"):
        if st.button("Merge scans…", type="tertiary"):
            merge_dialog()
        with st.popover("Save view", type="tertiary"):
            st.text_input("View name", key="new_view_name", placeholder=f"{scan.title} – elevation focus")
            st.button("Save", type="primary", on_click=cb_save_view, args=(sid,), use_container_width=True)
        st.download_button("Export PNG", type="tertiary", file_name=f"{scan.title}_beam.png", mime="image/png",
                           data=export_png)
        st.download_button("Profile CSV", type="tertiary", file_name=f"{scan.title}_profile.csv",
                           mime="text/csv", data=profile_csv(scan, r))
        st.toggle("View settings", key="show_panel")

    # header: scan info (left) | summary (right)
    h_left, h_right = st.columns([1.1, 1], gap="large", vertical_alignment="bottom")
    with h_left:
        st.markdown(f"# {scan.title}")
        fixed = AXIS_NAMES[scan.fixed_ax]
        props = [("◎ Plane", f"<span class='bl-chip'>{scan.plane}</span>at {fixed} = {scan.fixed_val:.2f} mm"),
                 ("∿ Frequency", f"{scan.freq:g} MHz" if scan.freq is not None else "—"),
                 ("▦ Grid", f"Δ{scan.hname} {scan.dh:.3g} mm · Δ{scan.vname} {scan.dv:.3g} mm · "
                            f"{len(scan.h)} × {len(scan.v)}"),
                 ("◉ Hydrophone", f"{scan.sens:g} mV/MPa · {scan.gain:g} dB gain"
                                  + (f" · {scan.navg} averages" if scan.navg else "")),
                 ("▣ Source", f"<span style='opacity:.6'>{entry.get('filename', '')}</span>")]
        st.markdown("<table class='bl-props'>" + "".join(f"<tr><td class='k'>{k}</td><td>{v}</td></tr>"
                                                          for k, v in props) + "</table>", unsafe_allow_html=True)
    with h_right:
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
    st.divider()

    # body: the beam map sits on a bottom layer across the full width; the plots panel is a top layer that
    # slides over it from the right (like a page over a page). The slider sets where that panel starts.
    ss.setdefault("split", 55)
    ss.setdefault("map_h", 700)

    def _sync_split():
        ss.split = ss["split#w"]

    st.slider("Plots panel position (%)", 20, 85, value=ss.split, step=1, key="split#w", on_change=_sync_split,
              format="Plots start at %d%%", label_visibility="collapsed",
              help="Slide the plots panel over the beam map")
    st.html("""<script>
      if (!window.__blBg) {           // mirror the theme background so the sliding panel always matches it
        window.__blBg = setInterval(function () {
          document.documentElement.style.setProperty('--bl-bg', getComputedStyle(document.body).backgroundColor);
        }, 300);
      }
    </script>""", unsafe_allow_javascript=True, width="content")
    st.markdown(f"""<style>
      .st-key-stack {{display: grid !important; grid-template-columns: minmax(0, 1fr);}}
      .st-key-stack > * {{grid-row: 1; grid-column: 1; min-width: 0; pointer-events: none;}}
      .st-key-mapzone, .st-key-overlay {{pointer-events: auto;}}   /* inherited, so Plotly's own rules still apply */
      .st-key-overlay {{margin-left: {ss.split}%; width: calc(100% - {ss.split}%) !important; box-sizing: border-box;
                        position: relative; z-index: 5; align-self: stretch;
                        background: var(--bl-bg, #ffffff); padding-left: 1.4rem;
                        border-left: 1px solid rgba(128,128,128,.28);
                        box-shadow: -18px 0 26px -22px rgba(0,0,0,.55);}}
    </style>""", unsafe_allow_html=True)

    with st.container(key="stack"):
        with st.container(key="mapzone"):
            st.markdown("**Beam map**" + (f"  <span style='opacity:.5;font-size:.85rem'>display {min(user_up, cap)}× "
                                           f"+ smoothing</span>" if user_up > cap else ""), unsafe_allow_html=True)
            fig, H = map_figure(ss.map_h)
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "toImageButtonOptions":
                            {"filename": f"{scan.title}_map", "scale": 3}})
        with st.container(key="overlay"):
            if tg["profile"]:
                ss.setdefault("ncur", "2")

                def _sync_ncur():
                    ss.ncur = ss["ncur#w"] or "Off"

                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown("**Beamwidth and Signal Strength vs depth**", width="stretch")
                    st.segmented_control("Markers", ["Off", "1", "2"], default=ss.ncur, key="ncur#w",
                                         on_change=_sync_ncur, label_visibility="collapsed",
                                         help="Depth markers: drag a dashed line (or click the plot to move the "
                                              "nearest one). Readout appears under the plot.")
                ncur = 0 if ss.ncur == "Off" else int(ss.ncur)
                with st.container(key="profchart"):
                    st.plotly_chart(profile_figure(min(max(H - 320, 360), 560), ncur), width="stretch",
                                    config={"displaylogo": False,
                                            "toImageButtonOptions": {"filename": f"{scan.title}_profile",
                                                                     "scale": 3}})
                if ncur:
                    st.html(CURSOR_JS, unsafe_allow_javascript=True, width="content")
            st.markdown("**Details**")
            st.code("\n".join(r.lines), language=None, wrap_lines=True)
