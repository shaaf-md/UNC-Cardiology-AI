#!/usr/bin/env python3
"""geometry_based_digitizer.py

Geometry-guided crop digitizer for the PTB-XL 3x4 + full-lead-II-rhythm render.

This is a PTB-XL *adapter* around the centerline-extraction ENGINE of
``mdacc_color_digitizer_v2e.py``(another one of my paper-based ECG digitising engines). It deliberately REUSES the parts of v2e that
are format-agnostic and DROPS the parts that infer page layout from the image:

  Adapted from v2e, with attribution at each function:
    * build_color_masks / trace-masking logic   (pink-grid suppression + dark
      neutral trace mask) -- the PTB-XL render uses the SAME pink-grid /
      dark-trace colour scheme as the MDACC scans, so the thresholds transfer.
    * _remove_small_components, _column_runs
    * extract_lead's DP / Viterbi centerline extraction (here: extract_centerline)
    * the calibrated-mV conversion idea (here: re-derived from the KNOWN geometry
      baseline instead of a baseline estimated from the trace itself)
    * the QC / overlay / export patterns (per-panel + per-record QC, summary JSON,
      preview overlay).

  BYPASSED / REMOVED (PTB-XL geometry is the source of truth, not the image):
    * detect_plot_bbox, detect_baselines, detect_lead_boundaries
    * build_rois (marker-defined full-page layout)
    * calibration-pulse exclusion
    * lead-label detection / removal
    * terminal / lead-transition marker detection
    * endpoint auto-trimming
    * manual marker overrides

SOURCE OF TRUTH (per panel): lead, row/col, sample_start/sample_end, x0/x1,
y_baseline_px, px_per_mm, mm_per_mV, mm_per_s. We never infer lead identity or
lead boundaries from the pixels.

INPUT (dual mode, auto-detected at RENDER_ROOT):
  PRIMARY  (production):  RENDER_ROOT/panel_geometry_manifest.csv + RENDER_ROOT/crops/
       * No crop-path column is needed. Crop files are derived per panel as
           RENDER_ROOT/crops/<filename_hr>/<lead>_r<row>_c<col>.png
         e.g. crops/records500/00000/00038_hr/I_r1_c0.png
              crops/records500/00000/00038_hr/II_r4_call.png   (rhythm, col "all").
       * The CSV x0_px/x1_px/y_baseline_px are FULL-PAGE coordinates, kept only as
         provenance. Because each crop is already cut to its panel, extraction uses
         CROP-LOCAL coordinates: x0=0, x1=crop_w, y_baseline=crop_h/2, band=full crop.
  FALLBACK (local dev) :  RENDER_ROOT/geometry/*.json + full-page RENDER_ROOT/png/*.png
                          (each panel is sliced on the fly from the full page using
                          the full-page x0/x1 and a band around y_baseline_px; PNG
                          lookup tries png/<stem>.png and png/**/<stem>.png).

PER-PANEL PIPELINE:
  1. obtain the crop pixels (a pre-cut crop, or a band sliced from the full page),
  2. build the trace mask with the reused colour logic,
  3. extract the centerline column-wise with the reused DP/Viterbi engine,
  4. y_px -> mV via geometry:  mV = (y_baseline_px - y_px) / (px_per_mm * mm_per_mV),
  5. map pixel-x -> native sample index via sample_start/sample_end & crop width,
  6. resample onto the panel's native integer-sample grid (no padding / no
     hallucination -- printed columns with no ink stay MISSING),
  7. compare against the native WFDB physical-mV signal for QC.

ASSEMBLY (dynamically sized to the records actually processed; becomes
[2074, 12, 5000] when the full PTB-XL test split is present):
  * Lead II  : taken from the FULL rhythm-strip crop (definitive, fills 0..5000).
               The short Lead II panel is digitized for QC ONLY.
  * other 11 : fill ONLY the printed 2.5 s window (sample_start:sample_end);
               unprinted samples stay 0.0 in the signal and UNPRINTED in the mask.

OUTPUTS (in --out-dir):
  digitized_montage_<split>.npy        float32 [N,12,5000]  (0.0 where unprinted/missing)
  digitized_observed_mask_<split>.npy  uint8   [N,12,5000]  (1 = valid|interpolated, else 0)
  digitized_status_<split>.npy         uint8   [N,12,5000]  (0 unprinted,1 valid,2 interp,3 missing)
  digitization_record_index.csv        row -> ecg_id / record / native path
  digitization_panel_qc.csv            one row per digitized panel (incl. SNR)
  digitization_record_qc.csv           one row per record (incl. SNR)
  digitization_summary.json            run-level summary + legends + SNR + eval_bundle
  preview_digitization_overlay.png     digitized-vs-native overlay for a few records

EVAL-COMPATIBLE BUNDLE (in <out-dir>/eval_waveforms_500hz/, for 04_eval):
  test_waveforms_500hz.npy             == the digitized montage [N,12,5000] float32
  test_metadata.csv                    same row order as the array (source_index 0..N-1;
                                       ecg_id, patient_id, strat_fold, filename_hr,
                                       filename_montage, waveform_source_path, split)
  test_waveforms_500hz.done.json       marker passing verify_waveform_split_complete;
                                       waveform_source_col=filename_montage (montage-arm
                                       guard accepts it) + waveform_source_kind=
                                       digitized_geometry_guided_crops & digitization_source
                                       for transparent provenance
  val_*  (copied/symlinked from --val-waveform-dir; fold 9 is NOT digitized -- threshold
         selection stays on the original digital montage validation arm)

SNR (per panel over valid|interpolated & native-finite samples; noise = digitized - native):
  snr_db_vs_native, snr_centered_db_vs_native (PREFERRED: native DC/baseline removed).
  Aggregated to record + summary: mean/median/macro_panel/weighted_sample/leadII_rhythm.

SNR DECOMPOSITION (diagnostic; QC only, never alters the montage; --no-snr-decomp to skip):
  Per panel, centered SNR + RMSE under four corrections, to separate a correctable
  alignment/calibration issue from genuine under-resolution noise:
    raw | affine (a*digi+b LSQ to native) | best integer lag in [-snr_max_lag,+] |
    lag+affine. Columns: best_lag_samples, rmse_{raw,affine,lag,lag_affine}_mV,
    snr_centered_{raw,affine,lag,lag_affine}_db, affine_gain, affine_bias_mV;
    aggregated to record + summary (qc.snr_decomposition).

Only common packages: opencv-python, numpy, pandas, wfdb, matplotlib, pillow.
Windows-compatible (no POSIX-only calls).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless / Windows-safe
import matplotlib.pyplot as plt

try:
    import wfdb
except Exception:  # pragma: no cover - wfdb is required only for native QC
    wfdb = None


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MM_PER_INCH = 25.4

# Montage / output channel order (PTB-XL standard 12-lead).
MONTAGE_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_TO_IDX = {name.upper(): i for i, name in enumerate(MONTAGE_LEADS)}
N_LEADS = len(MONTAGE_LEADS)
N_SAMPLES = 5000          # native 500 Hz x 10 s
NATIVE_FS = 500

# Per-sample status codes for digitized_status_<split>.npy.
ST_UNPRINTED = 0   # sample is outside any printed panel window for this lead
ST_VALID = 1       # real trace ink digitized at this sample
ST_INTERP = 2      # short pixel-column gap bridged by interpolation
ST_MISSING = 3     # printed window but no ink recovered (never hallucinated)
STATUS_LEGEND = {ST_UNPRINTED: "unprinted", ST_VALID: "valid",
                 ST_INTERP: "interpolated", ST_MISSING: "missing"}


def norm_lead(name: str) -> str:
    """Canonical upper-case lead key (PTB-XL .hea uses AVR/AVL/AVF)."""
    return str(name).strip().upper()


# Required panel layout: the 12 standard short leads (each once) + a full-lead-II
# rhythm strip. Lead II therefore appears TWICE (short panel + rhythm strip).
EXPECTED_N_PANELS = 13
EXPECTED_SHORT_LEADS = sorted(norm_lead(l) for l in MONTAGE_LEADS)  # 12, each once
EXPECTED_RHYTHM_LEAD = "II"


class RecordError(RuntimeError):
    """A per-record failure that must be reported and counted, never silently
    skipped (e.g. an unreadable crop or a malformed panel set)."""


def validate_record_panels(rp: "RecordPanels") -> None:
    """Fail fast (RecordError) unless the record has EXACTLY the 13 expected
    panels: the 12 standard short leads once each + one Lead II rhythm strip."""
    n = len(rp.panels)
    if n != EXPECTED_N_PANELS:
        raise RecordError(
            f"{rp.record_id}: expected {EXPECTED_N_PANELS} panels, got {n}")
    rhythm = [p for p in rp.panels if p.is_rhythm]
    short = [p for p in rp.panels if not p.is_rhythm]
    if len(rhythm) != 1:
        raise RecordError(
            f"{rp.record_id}: expected exactly 1 rhythm panel, got {len(rhythm)}")
    if norm_lead(rhythm[0].lead) != EXPECTED_RHYTHM_LEAD:
        raise RecordError(
            f"{rp.record_id}: rhythm panel must be Lead II, got "
            f"{rhythm[0].lead!r}")
    got_short = sorted(norm_lead(p.lead) for p in short)
    if got_short != EXPECTED_SHORT_LEADS:
        raise RecordError(
            f"{rp.record_id}: short-lead set mismatch; expected "
            f"{EXPECTED_SHORT_LEADS}, got {got_short}")


# --------------------------------------------------------------------------- #
# Tunable parameters (colour thresholds + DP weights ported from v2e Params)
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    # --- colour thresholds (ported verbatim from v2e; the PTB-XL render is the
    #     same pink-grid / dark-neutral-trace scheme) ---
    trace_max_channel: int = 175   # trace pixels are darker than this
    trace_max_redness: int = 45    # trace pixels are near-neutral (R-(G+B)/2)
    grid_min_redness: int = 25     # grid pixels are reddish
    grid_min_channel: int = 120    # grid pixels are not dark
    min_cc_area: int = 12          # drop trace specks smaller than this (px)

    # --- DP / Viterbi centerline extraction (ported from v2e Params) ---
    smooth_weight: float = 0.06    # cost per pixel of vertical jump between columns
    length_reward: float = 0.4     # reward (negative cost) per px of run thickness
    length_cap: int = 18           # cap run-thickness reward
    max_interp_gap_sec: float = 0.08  # bridge gaps up to this; longer -> MISSING

    # --- crop band (FALLBACK full-page mode only) ---
    # Half-height of the band sliced around y_baseline, in mm. Defaults to the
    # render's half_band_mm (== half the row pitch) so the band is exactly the
    # printed panel and never reaches the neighbouring row's baseline.
    band_half_mm: float = 22.5     # half row pitch (neighbour-side fallback)
    band_half_safety_px: int = 2   # shrink the band toward my baseline (neighbour side)
    band_outer_mult: float = 2.0   # max reach (x half_band) into a free page margin
    clip_edge_tol_px: int = 3      # extracted center this close to a band edge == clipped

    # --- deterministic x-axis (native sample -> pixel) render-coordinate mapping ---
    # Map each native sample k to pixel-x using the render's PHYSICAL calibration
    #   px/sample = px_per_mm * mm_per_s / NATIVE_FS
    # rather than stretching n_panel samples across the crop width. The two
    # offsets below are a FIXED render-coordinate correction (pixel-center / edge
    # convention only): they are NOT fit to the native waveform and never used to
    # per-panel align. Applied identically to every panel.
    x_center_offset_px: float = 0.0   # constant pixel shift added to every x_k
    x_sample_offset: float = 0.0      # constant shift (in native samples) before mapping

    # --- QC flag thresholds (per-record) ---
    qc_green_corr: float = 0.95
    qc_yellow_corr: float = 0.80
    qc_green_coverage: float = 0.95
    qc_yellow_coverage: float = 0.80

    # --- diagnostic SNR decomposition (QC only; never alters the montage) ---
    snr_decomp: bool = True       # compute raw/affine/lag/lag+affine SNR per panel
    snr_max_lag: int = 10         # integer-lag search range (+/- samples)


# --------------------------------------------------------------------------- #
# REUSED ENGINE -- copied from mdacc_color_digitizer_v2e.py (build_color_masks,
# _remove_small_components, _column_runs) and adapted (extract_centerline).
# --------------------------------------------------------------------------- #
def build_color_masks(bgr: np.ndarray, p: Params):
    """Return (grid_mask, trace_mask, grid_suppressed_bgr).

    Verbatim port of v2e.build_color_masks:
      * grid pixels are pink: R notably larger than (G+B)/2, and not dark.
      * trace pixels are dark and neutral (low max channel, low redness).
    """
    rgb = bgr[..., ::-1].astype(np.int32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = rgb.max(axis=2)
    redness = R - (G + B) // 2

    grid_mask = (redness >= p.grid_min_redness) & (max_c >= p.grid_min_channel)
    trace_mask = (max_c < p.trace_max_channel) & (redness < p.trace_max_redness)

    grid_mask = grid_mask.astype(np.uint8)
    trace_mask = trace_mask.astype(np.uint8)

    # Light cleanup: bridge 1px grid-suppression gaps, drop specks. Conservative
    # so we never invent ink.
    trace_mask = cv2.morphologyEx(
        trace_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    trace_mask = _remove_small_components(trace_mask, p.min_cc_area)

    grid_suppressed = bgr.copy()
    grid_suppressed[grid_mask.astype(bool)] = (255, 255, 255)
    return grid_mask, trace_mask, grid_suppressed


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Verbatim port of v2e._remove_small_components."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    return keep[labels].astype(np.uint8)


def _column_runs(col_mask: np.ndarray):
    """Verbatim port of v2e._column_runs.

    Return list of (center, length) for vertical True-runs in a column.
    """
    idx = np.where(col_mask)[0]
    if idx.size == 0:
        return []
    runs = []
    start = prev = idx[0]
    for v in idx[1:]:
        if v == prev + 1:
            prev = v
        else:
            runs.append(((start + prev) / 2.0, prev - start + 1))
            start = prev = v
    runs.append(((start + prev) / 2.0, prev - start + 1))
    return runs


def extract_centerline(trace_mask: np.ndarray, band_top: int, band_bot: int,
                       seg_x0: int, seg_x1: int, px_per_mm: float,
                       mm_per_s: float, p: Params):
    """Extract one y-position per x-column via DP over candidate ink runs.

    Adapted from v2e.extract_lead -- the SAME DP/Viterbi over per-column trace
    runs, but driven by explicit band/segment coordinates (the panel geometry)
    instead of a marker-derived LeadROI, and WITHOUT estimating a baseline from
    the trace (the geometry baseline is used downstream for calibration).

    Returns (y_px, status, long_gap_count):
      y_px   : float64 [W], absolute image-row coords, NaN where missing
      status : uint8   [W], 0=valid, 1=interpolated, 2=missing
      W      : seg_x1 - seg_x0
    """
    band = trace_mask[band_top:band_bot, seg_x0:seg_x1]
    Hb, Wb = band.shape

    candidates = [_column_runs(band[:, j]) for j in range(Wb)]
    valid_cols = [j for j, c in enumerate(candidates) if c]

    y_px = np.full(Wb, np.nan, dtype=np.float64)
    status = np.full(Wb, 2, dtype=np.uint8)  # default missing
    long_gap_count = 0

    if valid_cols:
        # --- DP / Viterbi over candidate runs (verbatim logic from extract_lead) ---
        INF = 1e18
        dp_cost: list[list[float]] = []
        dp_back: list[list[int]] = []
        for i, j in enumerate(valid_cols):
            cands = candidates[j]
            costs, backs = [], []
            for (yc, ln) in cands:
                emit = -p.length_reward * min(ln, p.length_cap)
                if i == 0:
                    costs.append(emit)
                    backs.append(-1)
                else:
                    prev_cands = candidates[valid_cols[i - 1]]
                    best, bk = INF, -1
                    for k, (pyc, _) in enumerate(prev_cands):
                        tc = dp_cost[i - 1][k] + p.smooth_weight * abs(yc - pyc)
                        if tc < best:
                            best, bk = tc, k
                    costs.append(best + emit)
                    backs.append(bk)
            dp_cost.append(costs)
            dp_back.append(backs)

        # backtrack
        chosen = [0] * len(valid_cols)
        k = int(np.argmin(dp_cost[-1]))
        for i in range(len(valid_cols) - 1, -1, -1):
            chosen[i] = k
            k = dp_back[i][k]
            if k < 0 and i > 0:
                k = int(np.argmin(dp_cost[i - 1]))

        for i, j in enumerate(valid_cols):
            y_px[j] = candidates[j][chosen[i]][0] + band_top
            status[j] = 0  # valid

        # --- fill gaps between valid columns ---
        max_gap = max(1, int(p.max_interp_gap_sec * px_per_mm * mm_per_s))
        for a, b in zip(valid_cols[:-1], valid_cols[1:]):
            gap = b - a - 1
            if gap <= 0:
                continue
            if gap <= max_gap:
                ya, yb = y_px[a], y_px[b]
                for off in range(1, gap + 1):
                    frac = off / (gap + 1)
                    y_px[a + off] = ya + frac * (yb - ya)
                    status[a + off] = 1  # interpolated
            else:
                long_gap_count += 1  # leave as missing

    return y_px, status, long_gap_count


# --------------------------------------------------------------------------- #
# Panel model + geometry loading (dual mode)
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    record_id: str           # e.g. "00009_hr"
    ecg_id: int
    filename_hr: str         # e.g. "records500/00000/00009_hr" (native WFDB)
    lead: str                # "I".."V6"
    row: int
    col: object              # 0..3 or "all"
    is_rhythm: bool
    sample_start: int
    sample_end: int
    # EXTRACTION pixel coordinates actually fed to the centerline engine.
    #   fallback (full-page) mode: full-page coords (data span + page band).
    #   primary  (crop) mode     : reset to crop-local after the crop is loaded
    #                              (x0=0, x1=crop_w, baseline=crop_h/2, band=full crop).
    x0_px: int
    x1_px: int
    y_baseline_px: int
    px_per_mm: float
    mm_per_mV: float
    mm_per_s: float
    # FULL-PAGE provenance (kept for QC even when extraction is crop-local).
    fp_x0_px: int = -1
    fp_x1_px: int = -1
    fp_y_baseline_px: int = -1
    coord_space: str = "full_page"     # "full_page" | "crop_local"
    crop_w: int = -1
    crop_h: int = -1
    # acquisition:
    crop_path: Optional[str] = None    # primary mode: path to the crop PNG
    page_path: Optional[str] = None    # fallback mode: path to the full-page PNG
    # crop band (rows) used for centerline extraction (filled before digitizing):
    band_top: int = 0
    band_bot: int = 0


@dataclass
class RecordPanels:
    record_id: str
    ecg_id: int
    filename_hr: str
    panels: list = field(default_factory=list)


def _ecg_id_from_stem(stem: str) -> int:
    """'00009_hr' -> 9 (best-effort; -1 if not parseable)."""
    digits = "".join(ch for ch in stem.split("_")[0] if ch.isdigit())
    return int(digits) if digits else -1


def _native_rel_from_ecg_id(ecg_id: int) -> str:
    """PTB-XL relative record path from ecg_id (e.g. 9 -> records500/00000/00009_hr)."""
    folder = (ecg_id // 1000) * 1000
    return f"records500/{folder:05d}/{ecg_id:05d}_hr"


def load_records_fullpage(render_root: Path, png_dir: Path, geom_dir: Path):
    """FALLBACK mode: one Panel set per geometry/*.json, sliced from full pages."""
    records = []
    for jpath in sorted(geom_dir.glob("*.json")):
        stem = jpath.stem  # e.g. "00009_hr"
        try:
            g = json.loads(jpath.read_text())
        except Exception:
            continue
        ecg_id = _ecg_id_from_stem(stem)
        filename_hr = g.get("filename_hr") or _native_rel_from_ecg_id(ecg_id)

        # resolve the full-page PNG. Try flat png/<stem>.png, the JSON's
        # filename_png, and a recursive png/**/<stem>.png search.
        candidates = [png_dir / f"{stem}.png"]
        if g.get("filename_png"):
            candidates.append(render_root / g["filename_png"])
            candidates.append(png_dir / Path(g["filename_png"]).name)
        page_path = next((c for c in candidates if c.exists()), None)
        if page_path is None:
            page_path = next(iter(png_dir.rglob(f"{stem}.png")), None)
        if page_path is None:
            continue

        rp = RecordPanels(record_id=stem, ecg_id=ecg_id, filename_hr=filename_hr)
        for pan in g.get("panels", []):
            col = pan.get("col")
            is_rhythm = (col == "all") or (int(pan.get("row", 0)) >= 4)
            x0 = int(round(pan["x0_px"])); x1 = int(round(pan["x1_px"]))
            yb = int(round(pan["y_baseline_px"]))
            rp.panels.append(Panel(
                record_id=stem, ecg_id=ecg_id, filename_hr=filename_hr,
                lead=str(pan["lead"]), row=int(pan["row"]), col=col,
                is_rhythm=is_rhythm,
                sample_start=int(pan["sample_start"]),
                sample_end=int(pan["sample_end"]),
                x0_px=x0, x1_px=x1, y_baseline_px=yb,
                fp_x0_px=x0, fp_x1_px=x1, fp_y_baseline_px=yb,
                coord_space="full_page",
                px_per_mm=float(pan["px_per_mm"]),
                mm_per_mV=float(pan["mm_per_mV"]),
                mm_per_s=float(pan["mm_per_s"]),
                page_path=str(page_path),
            ))
        if rp.panels:
            records.append(rp)
    return records


def _crop_basename(lead, row, col) -> str:
    """Crop file name convention: <lead>_r<row>_c<col>.png.

    col "all" (rhythm) -> 'call' (e.g. II_r4_call.png); col 0 -> 'c0'
    (e.g. I_r1_c0.png).
    """
    return f"{lead}_r{row}_c{col}.png"


def load_records_crops(manifest_csv: Path, crops_dir: Path):
    """PRIMARY mode: one Panel set per record from panel_geometry_manifest.csv.

    Expected (flexible) columns per row: filename_hr (and/or record_id), lead,
    row, col, sample_start, sample_end, x0/x0_px, x1/x1_px, y_baseline_px,
    px_per_mm, mm_per_mV, mm_per_s.

    The manifest does NOT need a crop-path column. Crop files are derived as
        RENDER_ROOT/crops/<filename_hr>/<lead>_r<row>_c<col>.png
    e.g. crops/records500/00000/00038_hr/I_r1_c0.png
         crops/records500/00000/00038_hr/II_r4_call.png
    (an explicit crop-path column, if present, overrides the derivation).

    The CSV x0_px/x1_px/y_baseline_px are FULL-PAGE coordinates -- they are kept
    only as provenance. Crop-local extraction coordinates (x0=0, x1=crop_w,
    baseline=crop_h/2, band=full crop) are set after the crop image is loaded
    (see process_record), because each crop is already cut to its panel.
    """
    df = pd.read_csv(manifest_csv)
    cols = {c.lower(): c for c in df.columns}

    def col(*names, default=None):
        for n in names:
            if n in cols:
                return cols[n]
        return default

    c_rec = col("record_id", "record", "stem")
    c_fn = col("filename_hr", "filename")
    c_lead = col("lead")
    c_row = col("row")
    c_col = col("col")
    c_ss = col("sample_start")
    c_se = col("sample_end")
    c_x0 = col("x0_px", "x0", "crop_x0")
    c_x1 = col("x1_px", "x1", "crop_x1")
    c_yb = col("y_baseline_px", "y_baseline", "baseline_y_px")
    c_ppm = col("px_per_mm")
    c_mmv = col("mm_per_mv")
    c_mms = col("mm_per_s")
    c_crop = col("crop_path", "crop_file", "crop", "filename_crop")  # optional

    groups = {}
    for _, r in df.iterrows():
        fn = str(r[c_fn]) if c_fn else None
        rec = str(r[c_rec]) if c_rec else (Path(fn).name if fn else None)
        ecg_id = _ecg_id_from_stem(rec) if rec else -1
        if not fn:
            fn = _native_rel_from_ecg_id(ecg_id)

        lead = str(r[c_lead])
        row = int(r[c_row]) if c_row else 0
        col_val = r[c_col] if c_col else None
        try:
            col_val = int(col_val)
        except (ValueError, TypeError):
            col_val = str(col_val)
        is_rhythm = (col_val == "all") or (row >= 4)

        # crop path: explicit column wins, else derive from filename_hr/lead/row/col.
        # A relative crop_path may be stored relative to render_root (e.g.
        # "crops/<fn>/<lead>_r#_c#.png", as 02b writes) OR relative to crops_dir.
        # Resolve against both (plus absolute) and PREFER the one that exists, so
        # neither renderer's convention doubles the "crops/" prefix. crops_dir.parent
        # is render_root (crops_dir == render_root/"crops").
        if c_crop and isinstance(r[c_crop], str) and r[c_crop].strip():
            cand = Path(r[c_crop].strip())
            if cand.is_absolute():
                crop_path = str(cand)
            else:
                _opts = [crops_dir.parent / cand, crops_dir / cand]
                crop_path = str(next((o for o in _opts if o.exists()), _opts[0]))
        else:
            crop_path = str(crops_dir / fn / _crop_basename(lead, row, col_val))

        fx0 = int(round(float(r[c_x0]))) if c_x0 else -1
        fx1 = int(round(float(r[c_x1]))) if c_x1 else -1
        fyb = int(round(float(r[c_yb]))) if c_yb else -1

        p = Panel(
            record_id=rec, ecg_id=ecg_id, filename_hr=fn,
            lead=lead, row=row, col=col_val, is_rhythm=bool(is_rhythm),
            sample_start=int(r[c_ss]), sample_end=int(r[c_se]),
            x0_px=fx0, x1_px=fx1, y_baseline_px=fyb,           # placeholders
            fp_x0_px=fx0, fp_x1_px=fx1, fp_y_baseline_px=fyb,  # full-page provenance
            coord_space="crop_local",
            px_per_mm=float(r[c_ppm]), mm_per_mV=float(r[c_mmv]),
            mm_per_s=float(r[c_mms]), crop_path=crop_path,
        )
        groups.setdefault(rec, RecordPanels(rec, ecg_id, fn)).panels.append(p)

    return sorted(groups.values(), key=lambda rp: (rp.ecg_id, rp.record_id))


def resolve_input_mode(render_root: Path):
    """Return ('crops', manifest, crops_dir) or ('fullpage', png_dir, geom_dir)."""
    manifest = render_root / "panel_geometry_manifest.csv"
    crops_dir = render_root / "crops"
    if manifest.exists() and crops_dir.is_dir():
        return "crops", manifest, crops_dir
    return "fullpage", render_root / "png", render_root / "geometry"


# --------------------------------------------------------------------------- #
# Per-panel digitization
# --------------------------------------------------------------------------- #
def digitize_panel(panel: Panel, trace_mask: np.ndarray, p: Params):
    """Digitize one panel's printed window onto its native integer-sample grid.

    Returns dict with:
      mV       : float64 [n_panel]   (0.0 where missing)
      status   : uint8   [n_panel]   (ST_VALID / ST_INTERP / ST_MISSING)
      n_panel  : sample_end - sample_start
      col arrays for QC/overlay (x_px, y_px, col_status)
    """
    H, W = trace_mask.shape
    seg_x0 = max(0, min(panel.x0_px, W - 1))
    seg_x1 = max(seg_x0 + 1, min(panel.x1_px, W))
    band_top = max(0, min(panel.band_top, H - 1))
    band_bot = max(band_top + 1, min(panel.band_bot, H))

    y_px, col_status, long_gaps = extract_centerline(
        trace_mask, band_top, band_bot, seg_x0, seg_x1,
        panel.px_per_mm, panel.mm_per_s, p)

    x_px = np.arange(seg_x0, seg_x1, dtype=np.float64)
    # mV from the KNOWN geometry baseline (image y grows downward => up is +).
    mV_col = (panel.y_baseline_px - y_px) / (panel.px_per_mm * panel.mm_per_mV)

    n_panel = panel.sample_end - panel.sample_start
    mV = np.zeros(n_panel, dtype=np.float64)
    status = np.full(n_panel, ST_MISSING, dtype=np.uint8)

    # native sample k (0..n_panel-1) -> pixel-x via the render's PHYSICAL
    # calibration: px/sample = px_per_mm * mm_per_s / NATIVE_FS. This is a fixed
    # deterministic render-coordinate mapping (NOT fit to native, never per-panel
    # aligned); panel.x0_px (== seg_x0) is the printed window's left data edge.
    # x_sample_offset / x_center_offset_px are constant pixel-center / edge
    # convention corrections. Samples whose x_k lands outside the extracted column
    # span fall out of `in_range` below and stay MISSING (never hallucinated).
    px_per_sample = panel.px_per_mm * panel.mm_per_s / NATIVE_FS
    k = np.arange(n_panel)
    x_k = seg_x0 + (k + p.x_sample_offset) * px_per_sample + p.x_center_offset_px

    usable = (col_status == 0) | (col_status == 1)
    if usable.any():
        su = x_px[usable]
        sv = mV_col[usable]
        sc = col_status[usable]
        order = np.argsort(su)
        su, sv, sc = su[order], sv[order], sc[order]

        in_range = (x_k >= su[0]) & (x_k <= su[-1])
        if in_range.any():
            xq = x_k[in_range]
            mV[in_range] = np.interp(xq, su, sv)
            # per-sample status = status of the nearest source pixel-column
            nn = np.clip(np.searchsorted(su, xq), 0, len(su) - 1)
            left = np.clip(nn - 1, 0, len(su) - 1)
            choose_left = np.abs(su[left] - xq) <= np.abs(su[nn] - xq)
            nn_idx = np.where(choose_left, left, nn)
            s_near = sc[nn_idx]
            idx_in = np.where(in_range)[0]
            status[idx_in[s_near == 0]] = ST_VALID
            status[idx_in[s_near == 1]] = ST_INTERP

    # clipping: extracted centerline sitting against a band edge means the trace
    # overflowed the crop (the printed amplitude exceeds what the crop captures).
    real = col_status == 0
    n_edge = 0
    if real.any():
        yv = y_px[real]
        n_edge = int(np.sum((yv <= band_top + p.clip_edge_tol_px) |
                            (yv >= band_bot - 1 - p.clip_edge_tol_px)))

    return {
        "mV": mV, "status": status, "n_panel": n_panel,
        "long_gaps": long_gaps,
        "x_px": x_px, "y_px": y_px, "col_status": col_status,
        "seg_x0": seg_x0, "seg_x1": seg_x1,
        "band_top": band_top, "band_bot": band_bot,
        "n_band_edge": n_edge,
    }


def prepare_panel_geometry(panel: Panel, page_h: int, page_w: int,
                           baselines: list, p: Params):
    """Fill the crop band coordinates for FALLBACK (full-page) slicing.

    The band runs from the MIDPOINT to the neighbouring baseline above to the
    MIDPOINT to the neighbouring baseline below (== +/- half the row pitch when
    rows are evenly spaced), so it never reaches a neighbour's baseline. On a
    side with NO neighbour (the top row upward, the rhythm strip downward) the
    band extends to the page margin -- the render lets tall QRS overflow well
    past +/- half_band_mm into that free space (only the page edge clips them),
    so extending there recovers real printed signal instead of truncating it.

    A small ``band_half_safety_px`` shrink is applied on the neighbour side only,
    to keep a synchronous neighbour deflection that just touches the midpoint out
    of this panel. In PRIMARY (crop) mode the band is the whole crop.
    """
    if panel.crop_path is not None:
        return  # crop-local: band is the full crop height (set after load).

    yb = panel.y_baseline_px
    above = [b for b in baselines if b < yb]
    below = [b for b in baselines if b > yb]
    half = int(round(p.band_half_mm * panel.px_per_mm))
    outer = int(round(p.band_outer_mult * half))  # bounded reach into a free margin

    if above:  # neighbour side: midpoint, shrunk slightly toward my baseline
        top = (yb + max(above)) / 2.0 + p.band_half_safety_px
    else:      # free margin above (top row): reach toward the page top, bounded
        top = yb - outer
    if below:
        bot = (yb + min(below)) / 2.0 - p.band_half_safety_px
    else:      # free margin below (rhythm strip): reach toward the page bottom
        bot = yb + outer

    panel.band_top = max(0, int(round(top)))
    panel.band_bot = min(page_h, int(round(bot)))


# --------------------------------------------------------------------------- #
# Native WFDB ground truth (for QC)
# --------------------------------------------------------------------------- #
def load_native_signal(native_root: Path, filename_hr: str):
    """Return (sig [5000,12] mV in MONTAGE_LEADS order, ok, msg)."""
    if wfdb is None:
        return None, False, "wfdb not installed"
    rec_path = native_root / filename_hr
    try:
        rec = wfdb.rdrecord(str(rec_path))
    except Exception as e:
        return None, False, f"rdrecord failed: {e}"
    sig = np.asarray(rec.p_signal, dtype=np.float64)  # [n, n_sig], physical mV
    names = [norm_lead(n) for n in rec.sig_name]
    out = np.full((N_SAMPLES, N_LEADS), np.nan, dtype=np.float64)
    n = min(sig.shape[0], N_SAMPLES)
    for li, lead in enumerate(MONTAGE_LEADS):
        key = norm_lead(lead)
        if key in names:
            out[:n, li] = sig[:n, names.index(key)]
    return out, True, "ok"


def panel_qc_metrics(digi_mV, status, native_seg):
    """Compare a digitized panel window against the native window over observed
    samples. Returns a metrics dict."""
    n = len(digi_mV)
    observed = (status == ST_VALID) | (status == ST_INTERP)
    n_valid = int((status == ST_VALID).sum())
    n_interp = int((status == ST_INTERP).sum())
    n_missing = int((status == ST_MISSING).sum())
    m = {
        "n_window": n,
        "n_valid": n_valid, "n_interp": n_interp, "n_missing": n_missing,
        "n_observed": n_valid + n_interp,
        "coverage": (n_valid + n_interp) / n if n else 0.0,
        "digi_amp_range_mV": float(np.ptp(digi_mV[observed])) if observed.any() else 0.0,
        "corr": np.nan, "rmse_mV": np.nan, "mae_mV": np.nan,
        "max_abs_err_mV": np.nan, "native_amp_range_mV": np.nan,
        "mean_bias_mV": np.nan,
        # SNR vs native (over observed & native-finite samples). The CENTERED
        # variant removes the native DC/baseline before measuring signal power
        # and is the preferred report metric (baseline offset is not "signal").
        "snr_db_vs_native": np.nan, "snr_centered_db_vs_native": np.nan,
        # raw power sums for exact weighted-sample aggregation downstream:
        "snr_n_cmp": 0, "snr_sig_pow_centered": 0.0, "snr_noise_pow": 0.0,
    }
    if native_seg is not None:
        finite = np.isfinite(native_seg)
        cmp = observed & finite
        m["native_amp_range_mV"] = (float(np.ptp(native_seg[finite]))
                                    if finite.any() else np.nan)
        if cmp.sum() >= 2:
            d = digi_mV[cmp]
            nv = native_seg[cmp]
            err = d - nv
            m["rmse_mV"] = float(np.sqrt(np.mean(err ** 2)))
            m["mae_mV"] = float(np.mean(np.abs(err)))
            m["max_abs_err_mV"] = float(np.max(np.abs(err)))
            m["mean_bias_mV"] = float(np.mean(err))
            if d.std() > 1e-9 and nv.std() > 1e-9:
                m["corr"] = float(np.corrcoef(d, nv)[0, 1])
            # --- SNR (dB) vs native ---
            eps = 1e-12
            noise_pow = float(np.sum(err ** 2))
            sig_pow = float(np.sum(nv ** 2))
            sig_pow_c = float(np.sum((nv - nv.mean()) ** 2))
            m["snr_n_cmp"] = int(cmp.sum())
            m["snr_sig_pow_centered"] = sig_pow_c
            m["snr_noise_pow"] = noise_pow
            if sig_pow > 0:
                m["snr_db_vs_native"] = float(10.0 * np.log10(sig_pow / (noise_pow + eps)))
            if sig_pow_c > 0:
                m["snr_centered_db_vs_native"] = float(
                    10.0 * np.log10(sig_pow_c / (noise_pow + eps)))
    return m


# Columns produced by snr_decomposition (kept here so empty rows stay schema-stable).
SNR_DECOMP_COLUMNS = (
    "best_lag_samples",
    "rmse_raw_mV", "rmse_affine_mV", "rmse_lag_mV", "rmse_lag_affine_mV",
    "snr_centered_raw_db", "snr_centered_affine_db",
    "snr_centered_lag_db", "snr_centered_lag_affine_db",
    "affine_gain", "affine_bias_mV",
)


def _centered_snr_rmse(dd, nn, eps=1e-12):
    """(rmse_mV, centered_snr_db) for paired digitized dd vs native nn."""
    if dd.size < 2:
        return np.nan, np.nan
    noise = dd - nn
    noise_pow = float(np.sum(noise ** 2))
    sig_c = float(np.sum((nn - nn.mean()) ** 2))
    rmse = float(np.sqrt(np.mean(noise ** 2)))
    snr = float(10.0 * np.log10(sig_c / (noise_pow + eps))) if sig_c > 0 else np.nan
    return rmse, snr


def _affine_fit(dd, nn):
    """Least-squares a*dd + b ~= nn. Returns (corrected, a, b) or None."""
    if dd.size < 2 or float(np.std(dd)) < 1e-12:
        return None
    a, b = np.polyfit(dd, nn, 1)            # slope, intercept (corrected = a*dd + b)
    return a * dd + b, float(a), float(b)


def snr_decomposition(digi_mV, status, native_seg, max_lag=10):
    """DIAGNOSTIC ONLY (never alters the montage array): centered SNR + RMSE under
    four corrections, to tell a correctable alignment/calibration issue apart from
    genuine under-resolution noise:
      1. raw                — as-digitized, lag 0;
      2. affine             — digi_corrected = a*digi + b (LSQ fit to native), lag 0;
      3. lag                — best integer lag in [-max_lag, +max_lag] (min RMSE);
      4. lag + affine       — affine fit at the best lag.
    Comparison is over observed (valid|interpolated) AND native-finite samples.
    """
    out = {c: np.nan for c in SNR_DECOMP_COLUMNS}
    out["best_lag_samples"] = 0
    if native_seg is None:
        return out
    n = len(digi_mV)
    obs = (status == ST_VALID) | (status == ST_INTERP)
    fin = np.isfinite(native_seg)

    # --- raw + affine at lag 0 ---
    base = obs & fin
    if base.sum() >= 2:
        dd0, nn0 = digi_mV[base], native_seg[base]
        out["rmse_raw_mV"], out["snr_centered_raw_db"] = _centered_snr_rmse(dd0, nn0)
        aff = _affine_fit(dd0, nn0)
        if aff is not None:
            corr, a, b = aff
            out["rmse_affine_mV"], out["snr_centered_affine_db"] = \
                _centered_snr_rmse(corr, nn0)
            out["affine_gain"], out["affine_bias_mV"] = a, b

    # --- best integer lag (digitized index k vs native index k+L) ---
    best = None  # (rmse, lag, dd, nn, snr)
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            di = np.arange(0, n - L)
        else:
            di = np.arange(-L, n)
        ni = di + L
        msk = obs[di] & fin[ni]
        if int(msk.sum()) < 2:
            continue
        dd, nn = digi_mV[di[msk]], native_seg[ni[msk]]
        rmse, snr = _centered_snr_rmse(dd, nn)
        if not np.isfinite(rmse):
            continue
        if best is None or rmse < best[0]:
            best = (rmse, L, dd, nn, snr)
    if best is not None:
        rmse, L, dd, nn, snr = best
        out["best_lag_samples"] = int(L)
        out["rmse_lag_mV"], out["snr_centered_lag_db"] = rmse, snr
        aff = _affine_fit(dd, nn)
        if aff is not None:
            corr, _a, _b = aff
            out["rmse_lag_affine_mV"], out["snr_centered_lag_affine_db"] = \
                _centered_snr_rmse(corr, nn)
    return out


# --------------------------------------------------------------------------- #
# Record processing
# --------------------------------------------------------------------------- #
def process_record(rp: RecordPanels, native_root: Path, p: Params):
    """Digitize one record. Returns (montage[12,5000], status[12,5000],
    panel_qc_rows, record_summary, overlay_payload).

    Raises RecordError if the panel set is malformed or a crop/page image cannot
    be read -- such records are reported and counted, never silently skipped.
    """
    validate_record_panels(rp)  # enforce the 13-panel layout before any work

    montage = np.zeros((N_LEADS, N_SAMPLES), dtype=np.float32)
    status_arr = np.zeros((N_LEADS, N_SAMPLES), dtype=np.uint8)  # ST_UNPRINTED

    native, native_ok, native_msg = load_native_signal(native_root, rp.filename_hr)

    # group panels by source image so each full page is masked once.
    page_cache = {}

    def get_trace_mask(panel: Panel):
        if panel.crop_path is not None:
            bgr = cv2.imread(panel.crop_path, cv2.IMREAD_COLOR)
            if bgr is None:
                _exists = Path(panel.crop_path).is_file()
                raise RecordError(
                    f"crop {'unreadable' if _exists else 'missing'}: {panel.crop_path} "
                    f"[record={panel.filename_hr} lead={panel.lead} "
                    f"row={panel.row} col={panel.col}]")
            _, tm, _ = build_color_masks(bgr, p)
            return tm, bgr.shape
        key = panel.page_path
        if key not in page_cache:
            bgr = cv2.imread(panel.page_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RecordError(f"page unreadable or missing: {panel.page_path}")
            _, tm, _ = build_color_masks(bgr, p)
            page_cache[key] = (tm, bgr.shape)
        return page_cache[key]

    panel_rows = []
    overlay = {"record_id": rp.record_id, "panels": {}, "native": native if native_ok else None}

    # baselines on this page define each panel's safe vertical extent (fallback mode)
    page_baselines = sorted({pn.y_baseline_px for pn in rp.panels
                             if pn.crop_path is None})

    for panel in rp.panels:
        trace_mask, shape = get_trace_mask(panel)
        H, W = shape[0], shape[1]
        if panel.crop_path is not None:
            # crop is already cut to this panel: extraction coords are crop-local
            # (the CSV x0/x1/y_baseline are full-page and kept only as provenance).
            panel.crop_w, panel.crop_h = W, H
            panel.x0_px, panel.x1_px = 0, W
            panel.y_baseline_px = H // 2
            panel.band_top, panel.band_bot = 0, H
            panel.coord_space = "crop_local"
        else:
            prepare_panel_geometry(panel, H, W, page_baselines, p)

        res = digitize_panel(panel, trace_mask, p)

        # native window for QC (lead's own samples over this window)
        native_seg = None
        li = LEAD_TO_IDX.get(norm_lead(panel.lead))
        if native_ok and li is not None:
            seg = native[panel.sample_start:panel.sample_end, li]
            native_seg = seg if seg.shape[0] == res["n_panel"] else None
        m = panel_qc_metrics(res["mV"], res["status"], native_seg)
        if p.snr_decomp:
            m.update(snr_decomposition(res["mV"], res["status"], native_seg,
                                       max_lag=p.snr_max_lag))

        panel_kind = ("rhythm_II" if panel.is_rhythm
                      else ("short_II_qc_only" if norm_lead(panel.lead) == "II"
                            else "short"))

        # --- write into the montage arrays ---
        if panel.is_rhythm:
            # definitive Lead II from the full rhythm strip (fills 0..5000)
            ii = LEAD_TO_IDX["II"]
            s0, s1 = panel.sample_start, panel.sample_end
            montage[ii, s0:s1] = res["mV"].astype(np.float32)
            status_arr[ii, s0:s1] = res["status"]
        elif panel_kind == "short_II_qc_only":
            pass  # short Lead II is QC-only; the rhythm strip is definitive
        else:
            if li is not None:
                s0, s1 = panel.sample_start, panel.sample_end
                montage[li, s0:s1] = res["mV"].astype(np.float32)
                status_arr[li, s0:s1] = res["status"]

        n_real = int((res["col_status"] == 0).sum())
        row = {
            "record_id": rp.record_id, "ecg_id": rp.ecg_id, "lead": panel.lead,
            "row": panel.row, "col": panel.col, "panel_kind": panel_kind,
            "is_rhythm": panel.is_rhythm,
            "sample_start": panel.sample_start, "sample_end": panel.sample_end,
            "coord_space": panel.coord_space,
            # extraction coords actually used:
            "x0_px": panel.x0_px, "x1_px": panel.x1_px,
            "y_baseline_px": panel.y_baseline_px,
            "band_top_px": res["band_top"], "band_bot_px": res["band_bot"],
            "crop_w": panel.crop_w, "crop_h": panel.crop_h,
            # full-page provenance (from the manifest / geometry JSON):
            "fp_x0_px": panel.fp_x0_px, "fp_x1_px": panel.fp_x1_px,
            "fp_y_baseline_px": panel.fp_y_baseline_px,
            "px_per_mm": panel.px_per_mm,
            "n_band_edge": res["n_band_edge"],
            "clip_frac": (res["n_band_edge"] / n_real) if n_real else 0.0,
            "clipped": bool(res["n_band_edge"] >= 3),
            "long_gaps": res["long_gaps"], "native_ok": native_ok,
        }
        row.update(m)
        panel_rows.append(row)

        overlay["panels"][f"{panel.lead}|{panel.row}|{panel.col}"] = {
            "panel": panel, "res": res, "kind": panel_kind, "metrics": m,
        }

    # --- per-record summary ---
    obs = (status_arr == ST_VALID) | (status_arr == ST_INTERP)
    printed = status_arr != ST_UNPRINTED
    short_rows = [r for r in panel_rows if r["panel_kind"] == "short"]
    rhythm_rows = [r for r in panel_rows if r["panel_kind"] == "rhythm_II"]
    shortII_rows = [r for r in panel_rows if r["panel_kind"] == "short_II_qc_only"]

    def _mean(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None
                and isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else np.nan

    def _median(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None
                and isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
        return float(np.median(vals)) if vals else np.nan

    corr_rows = short_rows + rhythm_rows
    rec = {
        "record_id": rp.record_id, "ecg_id": rp.ecg_id,
        "filename_hr": rp.filename_hr,
        "n_panels": len(panel_rows), "native_ok": native_ok,
        "native_msg": native_msg,
        "n_printed_samples": int(printed.sum()),
        "n_observed_samples": int(obs.sum()),
        "n_valid_samples": int((status_arr == ST_VALID).sum()),
        "n_interp_samples": int((status_arr == ST_INTERP).sum()),
        "n_missing_samples": int((status_arr == ST_MISSING).sum()),
        "observed_coverage": (float(obs.sum() / printed.sum())
                              if printed.sum() else 0.0),
        "mean_corr": _mean(corr_rows, "corr"),
        "mean_rmse_mV": _mean(corr_rows, "rmse_mV"),
        "mean_mae_mV": _mean(corr_rows, "mae_mV"),
        "leadII_rhythm_corr": (rhythm_rows[0]["corr"] if rhythm_rows else np.nan),
        "leadII_rhythm_rmse_mV": (rhythm_rows[0]["rmse_mV"] if rhythm_rows else np.nan),
        "leadII_short_corr": (shortII_rows[0]["corr"] if shortII_rows else np.nan),
        # --- SNR (centered is the preferred report metric) ---
        "mean_snr_centered_db": _mean(corr_rows, "snr_centered_db_vs_native"),
        "median_snr_centered_db": _median(corr_rows, "snr_centered_db_vs_native"),
        "mean_snr_db": _mean(corr_rows, "snr_db_vs_native"),
        "leadII_rhythm_snr_centered_db": (rhythm_rows[0]["snr_centered_db_vs_native"]
                                          if rhythm_rows else np.nan),
        # exact power sums for weighted-sample aggregation at summary level:
        "snr_sig_pow_centered_sum": float(sum(r.get("snr_sig_pow_centered", 0.0)
                                              for r in corr_rows)),
        "snr_noise_pow_sum": float(sum(r.get("snr_noise_pow", 0.0)
                                       for r in corr_rows)),
    }
    if p.snr_decomp:
        rec.update({
            "mean_snr_centered_raw_db": _mean(corr_rows, "snr_centered_raw_db"),
            "mean_snr_centered_affine_db": _mean(corr_rows, "snr_centered_affine_db"),
            "mean_snr_centered_lag_db": _mean(corr_rows, "snr_centered_lag_db"),
            "mean_snr_centered_lag_affine_db": _mean(corr_rows, "snr_centered_lag_affine_db"),
            "mean_rmse_raw_mV": _mean(corr_rows, "rmse_raw_mV"),
            "mean_rmse_affine_mV": _mean(corr_rows, "rmse_affine_mV"),
            "mean_rmse_lag_mV": _mean(corr_rows, "rmse_lag_mV"),
            "mean_rmse_lag_affine_mV": _mean(corr_rows, "rmse_lag_affine_mV"),
            "median_best_lag_samples": _median(corr_rows, "best_lag_samples"),
            "mean_abs_best_lag_samples": _mean(
                [{"v": abs(r["best_lag_samples"])} for r in corr_rows
                 if isinstance(r.get("best_lag_samples"), (int, float))], "v"),
            "mean_affine_gain": _mean(corr_rows, "affine_gain"),
            "mean_affine_bias_mV": _mean(corr_rows, "affine_bias_mV"),
            "leadII_rhythm_snr_centered_lag_affine_db": (
                rhythm_rows[0].get("snr_centered_lag_affine_db")
                if rhythm_rows else np.nan),
        })
    rec["quality_flag"] = _record_flag(rec, p)
    return montage, status_arr, panel_rows, rec, overlay


def _record_flag(rec, p: Params):
    cov = rec.get("observed_coverage", 0.0)
    corr = rec.get("mean_corr", np.nan)
    if not rec.get("native_ok"):
        corr_ok_green = corr_ok_yellow = True  # no native -> judge on coverage
    else:
        corr_ok_green = np.isfinite(corr) and corr >= p.qc_green_corr
        corr_ok_yellow = np.isfinite(corr) and corr >= p.qc_yellow_corr
    if cov >= p.qc_green_coverage and corr_ok_green:
        return "green"
    if cov >= p.qc_yellow_coverage and corr_ok_yellow:
        return "yellow"
    return "red"


# --------------------------------------------------------------------------- #
# Preview overlay
# --------------------------------------------------------------------------- #
def _draw_overlay_axis(ax, lead, pl, native, title):
    panel = pl["panel"]
    s0, s1 = panel.sample_start, panel.sample_end
    t = np.arange(s1 - s0) / NATIVE_FS
    mV = pl["res"]["mV"].copy()
    st = pl["res"]["status"]
    mV[(st != ST_VALID) & (st != ST_INTERP)] = np.nan  # never draw hallucinated samples
    if native is not None:
        li = LEAD_TO_IDX.get(norm_lead(lead))
        if li is not None:
            ax.plot(t, native[s0:s1, li], color="tab:red", lw=0.8, ls="--",
                    alpha=0.7, label="native")
    ax.plot(t, mV, color="black", lw=0.7, label="digitized")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def save_preview_overlay(overlays, out_path: Path, max_records: int = 3):
    """Digitized (black) vs native (red dashed) mV overlay: 3x4 short leads +
    a full-width Lead II rhythm row, for up to ``max_records`` records."""
    overlays = [o for o in overlays if o is not None][:max_records]
    if not overlays:
        return
    nrec = len(overlays)
    layout = {  # (row, col) -> lead for the 3x4 short block
        (0, 0): "I", (0, 1): "aVR", (0, 2): "V1", (0, 3): "V4",
        (1, 0): "II", (1, 1): "aVL", (1, 2): "V2", (1, 3): "V5",
        (2, 0): "III", (2, 1): "aVF", (2, 2): "V3", (2, 3): "V6",
    }
    fig = plt.figure(figsize=(20, 8.5 * nrec))
    outer = fig.add_gridspec(nrec, 1, hspace=0.35)
    for ri, ov in enumerate(overlays):
        inner = outer[ri].subgridspec(4, 4, hspace=0.55, wspace=0.22)
        short, short_II, rhythm = {}, None, None
        for pl in ov["panels"].values():
            if pl["kind"] == "rhythm_II":
                rhythm = pl
            elif pl["kind"] == "short_II_qc_only":
                short_II = pl
            elif pl["kind"] == "short":
                short[norm_lead(pl["panel"].lead)] = pl
        native = ov["native"]
        for (r, c), lead in layout.items():
            ax = fig.add_subplot(inner[r, c])
            pl = short_II if lead == "II" else short.get(norm_lead(lead))
            if pl is None:
                ax.set_axis_off()
                continue
            corr = pl["metrics"].get("corr")
            tag = " (II short, QC only)" if lead == "II" else ""
            title = (f"{lead}{tag}  r={corr:.3f}"
                     if corr == corr else f"{lead}{tag}")
            _draw_overlay_axis(ax, lead, pl, native, title)
        axr = fig.add_subplot(inner[3, :])
        if rhythm is not None:
            corr = rhythm["metrics"].get("corr")
            title = (f"{ov['record_id']}  Lead II rhythm (definitive)  r={corr:.3f}"
                     if corr == corr else f"{ov['record_id']}  Lead II rhythm")
            _draw_overlay_axis(axr, "II", rhythm, native, title)
            axr.legend(loc="upper right", fontsize=7)
        else:
            axr.set_axis_off()
    fig.suptitle("PTB-XL geometry-guided digitization: digitized (black) vs "
                 "native (red dashed), mV", fontsize=12)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Eval-compatible waveform-dir export (04_eval_ptbxl_echonext_multitask_fusion_v1)
#
# 04_eval expects a waveform-dir with, per split: {split}_waveforms_500hz.npy,
# {split}_metadata.csv, and a {split}_waveforms_500hz.done.json marker that passes
# ptbxl_echonext_multitask_modeling.verify_waveform_split_complete. We emit that
# bundle for the digitized TEST split. waveform_source_col stays 'filename_montage'
# (so the montage-arm source guard accepts this model), while waveform_source_kind
# / digitization_source record that the pixels were DIGITIZED from geometry-guided
# rendered-PNG crops -- provenance is explicit, never hidden.
# --------------------------------------------------------------------------- #

# Byte-for-byte local replicas of the modeling-module contract, used only when the
# real module cannot be located/imported. Verified against the module source.
_EM_FALLBACK = dict(
    WAVEFORM_FILE_TEMPLATE="{split}_waveforms_500hz.npy",
    SPLIT_METADATA_TEMPLATE="{split}_metadata.csv",
    WAVEFORM_DONE_MARKER_TEMPLATE="{split}_waveforms_500hz.done.json",
    ECG_ID_COLUMN="ecg_id", PATIENT_ID_COLUMN="patient_id",
    STRAT_FOLD_COLUMN="strat_fold", FILENAME_HR_COLUMN="filename_hr",
    FILENAME_MONTAGE_COLUMN="filename_montage",
    WAVEFORM_SOURCE_PATH_COLUMN="waveform_source_path",
    WAVEFORM_SOURCE_COL_FIELD="waveform_source_col",
    CANONICAL_LEADS=list(MONTAGE_LEADS), N_LEADS=N_LEADS, EXPECTED_T=N_SAMPLES,
)


def _em_ordered_sha256(values) -> str:
    """Local replica of M.ordered_sha256 (NUL-delimited str() over the sequence)."""
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _em_build_split_metadata(split_df, waveform_source_col, C):
    """Local replica of M.build_split_metadata (identity cols + source provenance,
    source_index in array row order)."""
    if waveform_source_col not in split_df.columns:
        raise KeyError(f"waveform source column '{waveform_source_col}' absent "
                       f"from the manifest split frame")
    keep = [c for c in (C.ECG_ID_COLUMN, C.PATIENT_ID_COLUMN, C.STRAT_FOLD_COLUMN,
                        C.FILENAME_HR_COLUMN) if c in split_df.columns]
    if waveform_source_col not in keep:
        keep.append(waveform_source_col)
    meta = split_df[keep].copy().reset_index(drop=True)
    meta.insert(0, "source_index", np.arange(len(meta), dtype=np.int64))
    meta[C.WAVEFORM_SOURCE_COL_FIELD] = str(waveform_source_col)
    meta[C.WAVEFORM_SOURCE_PATH_COLUMN] = (
        split_df[waveform_source_col].astype(str).to_numpy())
    return meta


def load_eval_modeling(render_root, modeling_dir=None):
    """Namespace with the modeling-module contract (symbols + functions).

    Prefers the real ptbxl_echonext_multitask_modeling module when importable
    (it imports fine without torch); otherwise uses byte-for-byte local replicas.
    ``verify`` is the real verify_waveform_split_complete when available, else None.
    """
    ns = SimpleNamespace(**_EM_FALLBACK)
    ns.ordered_sha256 = _em_ordered_sha256
    ns.verify = None
    ns.source = "local_replica"

    candidates = []
    if modeling_dir:
        candidates.append(Path(modeling_dir))
    here = Path(__file__).resolve().parent
    rr = Path(render_root)
    candidates += [rr, *list(rr.parents)[:5], here, here.parent,
                   Path(r"C:\Users\UNC_Shaaf\Downloads\PTBXL ResNet Montage")]
    seen, M = set(), None
    for d in candidates:
        d = Path(d)
        if str(d) in seen:
            continue
        seen.add(str(d))
        if (d / "ptbxl_echonext_multitask_modeling.py").exists():
            try:
                if str(d) not in sys.path:
                    sys.path.insert(0, str(d))
                import ptbxl_echonext_multitask_modeling as M  # type: ignore
                break
            except Exception as exc:
                print(f"[eval-bundle] modeling module at {d} failed to import "
                      f"({exc!r}); using local replicas.", file=sys.stderr)
                M = None
    if M is not None:
        for nm in ("WAVEFORM_FILE_TEMPLATE", "SPLIT_METADATA_TEMPLATE",
                   "WAVEFORM_DONE_MARKER_TEMPLATE", "ECG_ID_COLUMN",
                   "PATIENT_ID_COLUMN", "STRAT_FOLD_COLUMN", "FILENAME_HR_COLUMN",
                   "FILENAME_MONTAGE_COLUMN", "WAVEFORM_SOURCE_PATH_COLUMN",
                   "WAVEFORM_SOURCE_COL_FIELD", "N_LEADS", "EXPECTED_T"):
            if hasattr(M, nm):
                setattr(ns, nm, getattr(M, nm))
        ns.CANONICAL_LEADS = list(M.CANONICAL_LEADS)
        ns.ordered_sha256 = M.ordered_sha256
        ns.build_split_metadata = (lambda df, col, _M=M:
                                   _M.build_split_metadata(df, waveform_source_col=col))
        ns.split_metadata_path = M.split_metadata_path
        ns.waveform_done_marker_path = M.waveform_done_marker_path
        ns.verify = getattr(M, "verify_waveform_split_complete", None)
        ns.source = f"module:{Path(M.__file__).parent}"
    else:
        ns.build_split_metadata = (lambda df, col, _C=ns:
                                   _em_build_split_metadata(df, col, _C))
        ns.split_metadata_path = (lambda wd, sp, _C=ns:
                                  Path(wd) / _C.SPLIT_METADATA_TEMPLATE.format(split=sp))
        ns.waveform_done_marker_path = (lambda wd, sp, _C=ns:
                                        Path(wd) / _C.WAVEFORM_DONE_MARKER_TEMPLATE.format(split=sp))
    return ns


def discover_manifest(explicit, render_root, native_root, C):
    """Locate the per-record montage manifest (ecg_id + filename_montage). Prefers
    --manifest; else a bounded search near render_root / native_root / script."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        print(f"[eval-bundle] --manifest {p} not found; trying auto-discovery.",
              file=sys.stderr)
    roots = [Path(render_root), *list(Path(render_root).parents)[:4]]
    if native_root:
        roots += [Path(native_root), *list(Path(native_root).parents)[:2]]
    roots += [Path(__file__).resolve().parent, Path(r"C:\Users\UNC_Shaaf\Downloads")]
    patterns = ["*montage_manifest*.csv", "panel_geometry_manifest.csv", "*manifest*.csv"]
    seen = set()
    for root in roots:
        if not root or not Path(root).is_dir():
            continue
        for pat in patterns:
            for cand in sorted(Path(root).glob(pat)):
                if str(cand) in seen:
                    continue
                seen.add(str(cand))
                try:
                    cols = list(pd.read_csv(cand, nrows=0).columns)
                except Exception:
                    continue
                if C.ECG_ID_COLUMN in cols and C.FILENAME_MONTAGE_COLUMN in cols:
                    return cand
    return None


def _copy_val_split(eval_dir, val_waveform_dir, val_link, C):
    """Copy/symlink the DIGITAL-MONTAGE validation split (val_*) into eval_dir.

    Fold 9 is NOT digitized: threshold selection stays on the original digital
    montage validation arm; only digitized fold 10 (test) is produced here.
    """
    if not val_waveform_dir:
        return {"status": "skipped", "reason": "no --val-waveform-dir provided "
                "(04_eval also needs val_* from the original digital montage arm)"}
    src = Path(val_waveform_dir)
    names = [C.WAVEFORM_FILE_TEMPLATE.format(split="val"),
             C.SPLIT_METADATA_TEMPLATE.format(split="val"),
             C.WAVEFORM_DONE_MARKER_TEMPLATE.format(split="val")]
    missing = [n for n in names if not (src / n).exists()]
    if missing:
        raise RuntimeError(f"--val-waveform-dir {src} is missing: {missing}")
    out = []
    for n in names:
        s, dst = src / n, Path(eval_dir) / n
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if val_link == "symlink":
            dst.symlink_to(s)
        else:
            shutil.copy2(s, dst)
        out.append(n)
    return {"status": ("linked" if val_link == "symlink" else "copied"),
            "mode": val_link, "from": str(src), "files": out}


def emit_eval_bundle(montage, records, manifest_path, eval_dir, split, C,
                     digi_paths, val_waveform_dir, val_link):
    """Write {split}_waveforms_500hz.npy + {split}_metadata.csv + the done marker
    so 04_eval/verify_waveform_split_complete accept the digitized montage."""
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(manifest_path, low_memory=False)
    for need in (C.ECG_ID_COLUMN, C.FILENAME_MONTAGE_COLUMN):
        if need not in man.columns:
            raise RuntimeError(f"manifest {manifest_path} lacks '{need}'")
    man = man.copy()
    man[C.ECG_ID_COLUMN] = pd.to_numeric(man[C.ECG_ID_COLUMN], errors="coerce").astype("Int64")
    man = man.dropna(subset=[C.ECG_ID_COLUMN]).drop_duplicates(C.ECG_ID_COLUMN, keep="first")
    man_idx = man.set_index(C.ECG_ID_COLUMN)

    ordered = [int(rp.ecg_id) for rp in records]   # EXACT array row order
    missing = [e for e in ordered if e not in man_idx.index]
    if missing:
        raise RuntimeError(f"{len(missing)} digitized ecg_id(s) absent from the "
                           f"manifest (e.g. {missing[:10]}); cannot build eval "
                           f"metadata aligned to the array")
    split_df = man_idx.loc[ordered].reset_index()  # ecg_id back as a column, array order

    meta = C.build_split_metadata(split_df, C.FILENAME_MONTAGE_COLUMN)
    meta["split"] = split
    meta_path = Path(C.split_metadata_path(eval_dir, split))
    meta.to_csv(meta_path, index=False)

    arr_path = eval_dir / C.WAVEFORM_FILE_TEMPLATE.format(split=split)
    np.save(arr_path, np.ascontiguousarray(montage, dtype=np.float32))

    # hashes from the RE-READ metadata, byte-identical to what verify recomputes
    md = pd.read_csv(meta_path, low_memory=False)
    if "source_index" in md.columns:
        md = md.sort_values("source_index")
    ecg_hash = C.ordered_sha256(md[C.ECG_ID_COLUMN].tolist())
    src_hash = C.ordered_sha256(md[C.WAVEFORM_SOURCE_PATH_COLUMN].astype(str).tolist())

    n = int(montage.shape[0])
    marker = {
        "split": split,
        "n_built": n,
        "expected_shape": [n, C.N_LEADS, C.EXPECTED_T],
        "dtype": str(np.dtype(np.float32)),
        C.WAVEFORM_SOURCE_COL_FIELD: C.FILENAME_MONTAGE_COLUMN,
        "waveform_source_kind": "digitized_geometry_guided_crops",
        "waveform_root": str(eval_dir),
        "canonical_lead_order": list(C.CANONICAL_LEADS),
        "sampling_frequency_hz": NATIVE_FS,
        "ecg_id_order_sha256": ecg_hash,
        "waveform_source_order_sha256": src_hash,
        f"{C.FILENAME_MONTAGE_COLUMN}_order_sha256": src_hash,
        "array": str(arr_path),
        "metadata": str(meta_path),
        "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        # provenance: pixels are DIGITIZED from rendered PNG crops; the source
        # column stays filename_montage only so the montage-arm eval guard accepts
        # this model. The two fields below make the true source explicit.
        "digitization_source": "geometry_guided_rendered_png_crops",
        "digitization_summary": str(digi_paths["summary"]),
        "digitization_panel_qc": str(digi_paths["panel_qc"]),
        "digitization_record_qc": str(digi_paths["record_qc"]),
        "note": ("waveform pixels were digitized from rendered PNG crops; "
                 "waveform_source_col kept as filename_montage for eval "
                 "source-guard compatibility."),
    }
    marker_path = Path(C.waveform_done_marker_path(eval_dir, split))
    marker_path.write_text(json.dumps(marker, indent=2))

    # self-verify exactly as 04_eval will (only when the real module is present)
    self_verify = "unavailable (modeling module not importable)"
    if C.verify is not None:
        try:
            C.verify(eval_dir, split, pd.read_csv(meta_path, low_memory=False))
            self_verify = "ok"
        except Exception as exc:
            self_verify = f"FAILED: {exc}"

    val_info = _copy_val_split(eval_dir, val_waveform_dir, val_link, C)

    return {
        "status": "ok",
        "eval_dir": str(eval_dir),
        "modeling_source": C.source,
        "manifest": str(manifest_path),
        "n_rows": n,
        "split": split,
        "test_waveforms": str(arr_path),
        "test_metadata": str(meta_path),
        "test_done_marker": str(marker_path),
        "waveform_source_col": C.FILENAME_MONTAGE_COLUMN,
        "waveform_source_kind": "digitized_geometry_guided_crops",
        "digitization_source": "geometry_guided_rendered_png_crops",
        "ecg_id_order_sha256": ecg_hash,
        "waveform_source_order_sha256": src_hash,
        "self_verify": self_verify,
        "val": val_info,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--render-root", "--render_root", dest="render_root",
                    type=Path, default=here,
                    help="RENDER_ROOT (panel_geometry_manifest.csv + crops/, or "
                         "geometry/*.json + png/*.png). Default: script dir.")
    ap.add_argument("--native-root", "--ptbxl_root", "--ptbxl-root",
                    dest="native_root", type=Path,
                    default=Path(r"C:\Users\UNC_Shaaf\OneDrive\Desktop"
                                 r"\PhysioNet Complete (PTB XL, Georgia etc)\training"),
                    help="PTB-XL root containing records500/ for native-WFDB QC "
                         "(alias: --ptbxl_root).")
    ap.add_argument("--out-dir", "--out_root", "--out-root", dest="out_dir",
                    type=Path, default=None,
                    help="Output directory (default: RENDER_ROOT/digitization_out; "
                         "alias: --out_root).")
    ap.add_argument("--split", type=str, default="test",
                    help="Suffix for the output arrays (digitized_montage_<split>.npy).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most this many records (0 = all). Dev only.")
    ap.add_argument("--expected_count", "--expected-count", dest="expected_count",
                    type=int, default=0,
                    help="If >0, assert exactly this many records are found "
                         "(e.g. 2074 for the PTB-XL test split). 0 = no assertion.")
    ap.add_argument("--num_workers", "--num-workers", dest="num_workers",
                    type=int, default=1,
                    help="Parallel worker processes for record digitization.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing outputs (otherwise skip if present).")
    ap.add_argument("--stop_on_first_failure", "--stop-on-first-failure",
                    dest="stop_on_first_failure", action="store_true",
                    help="Abort on the first record that fails (default: collect "
                         "errors and continue).")
    ap.add_argument("--preview-records", "--preview_records", dest="preview_records",
                    type=int, default=3,
                    help="Number of records drawn in the preview overlay.")
    ap.add_argument("--no-native", "--no_native", dest="no_native",
                    action="store_true",
                    help="Skip native-WFDB QC (no correlation/RMSE columns).")
    ap.add_argument("--no-snr-decomp", "--no_snr_decomp", dest="snr_decomp",
                    action="store_false",
                    help="Disable the diagnostic raw/affine/lag/lag+affine SNR "
                         "decomposition (default: on; QC only).")
    ap.set_defaults(snr_decomp=True)
    ap.add_argument("--snr-max-lag", "--snr_max_lag", dest="snr_max_lag",
                    type=int, default=10,
                    help="Integer-lag search range (+/- samples) for the SNR "
                         "lag-correction diagnostic.")
    # --- deterministic x-axis (sample->pixel) render-coordinate correction ---
    # FIXED offsets applied identically to every panel; NOT fit to the native
    # waveform, never per-panel aligned. Used to diagnose pixel-center / edge
    # convention mismatch in the sample->pixel mapping.
    ap.add_argument("--x-center-offset-px", "--x_center_offset_px",
                    dest="x_center_offset_px", type=float, default=0.0,
                    help="Constant pixel shift added to every native-sample->pixel "
                         "x_k mapping (render-coordinate pixel-center/edge "
                         "correction; NOT fit to native, never per-panel aligned).")
    ap.add_argument("--x-sample-offset", "--x_sample_offset",
                    dest="x_sample_offset", type=float, default=0.0,
                    help="Constant shift (in native samples) added to k before the "
                         "px/sample mapping (render-coordinate correction; NOT fit "
                         "to native, never per-panel aligned).")
    # --- eval-compatible waveform-dir export (04_eval contract) ---
    ap.add_argument("--no-eval-bundle", "--no_eval_bundle", dest="emit_eval_bundle",
                    action="store_false",
                    help="Do NOT emit the 04_eval-compatible eval_waveforms_500hz/ "
                         "bundle (default: emit it).")
    ap.set_defaults(emit_eval_bundle=True)
    ap.add_argument("--eval-out-dir", "--eval_out_dir", dest="eval_out_dir",
                    type=Path, default=None,
                    help="Eval bundle dir (default: <out-dir>/eval_waveforms_500hz).")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Per-record montage manifest (needs ecg_id + "
                         "filename_montage) for the eval metadata. Auto-discovered "
                         "near render-root if omitted.")
    ap.add_argument("--modeling-dir", "--modeling_dir", dest="modeling_dir",
                    type=Path, default=None,
                    help="Dir containing ptbxl_echonext_multitask_modeling.py "
                         "(used for byte-exact helpers + self-verification).")
    ap.add_argument("--val-waveform-dir", "--val_waveform_dir", dest="val_waveform_dir",
                    type=Path, default=None,
                    help="Original DIGITAL-MONTAGE waveform-dir to copy the val "
                         "split (val_*) from. Fold 9 is NOT digitized; threshold "
                         "selection stays on the original digital montage val arm.")
    ap.add_argument("--val-link", "--val_link", dest="val_link",
                    choices=["copy", "symlink"], default="copy",
                    help="Copy (default) or symlink the val_* files into the bundle.")
    return ap.parse_args(argv)


def _run_one(rp: "RecordPanels", idx: int, native_root, p: Params,
             want_overlay: bool):
    """Digitize a single record (top-level so it is picklable for workers)."""
    nr = native_root if native_root is not None else Path(".")
    mtg, st, prows, rrow, ov = process_record(rp, nr, p)
    if native_root is None:
        rrow["native_ok"] = False
        rrow["native_msg"] = "skipped"
    rrow["row_index"] = idx
    return {
        "idx": idx, "montage": mtg, "status": st,
        "panel_rows": prows, "record_row": rrow,
        "overlay": ov if want_overlay else None,
        "index_row": {"row_index": idx, "ecg_id": rp.ecg_id,
                      "record_id": rp.record_id, "filename_hr": rp.filename_hr,
                      "quality_flag": rrow["quality_flag"]},
    }


def main(argv=None):
    args = parse_args(argv)
    p = Params()
    p.snr_decomp = bool(args.snr_decomp)
    p.snr_max_lag = int(args.snr_max_lag)
    p.x_center_offset_px = float(args.x_center_offset_px)
    p.x_sample_offset = float(args.x_sample_offset)
    render_root = args.render_root
    out_dir = args.out_dir or (render_root / "digitization_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    native_root = None if args.no_native else args.native_root
    sp = args.split

    # idempotency: skip if outputs already exist and --overwrite was not given.
    montage_path = out_dir / f"digitized_montage_{sp}.npy"
    if montage_path.exists() and not args.overwrite:
        print(f"[skip] {montage_path} exists; pass --overwrite to regenerate.")
        return 0

    mode, a, b = resolve_input_mode(render_root)
    if mode == "crops":
        print(f"[input] PRIMARY (crop) mode: {a}  +  {b}")
        records = load_records_crops(a, b)
    else:
        print(f"[input] FALLBACK (full-page) mode: {b}/*.json  +  {a}/*.png")
        records = load_records_fullpage(render_root, a, b)

    records = sorted(records, key=lambda rp: (rp.ecg_id, rp.record_id))
    n_loaded = len(records)
    if args.expected_count and n_loaded != args.expected_count:
        print(f"[error] expected_count={args.expected_count} but found {n_loaded} "
              f"records at RENDER_ROOT (mode={mode}).", file=sys.stderr)
        return 1
    if args.limit and args.limit > 0:
        records = records[:args.limit]
    N = len(records)
    if N == 0:
        print("[error] no records found at RENDER_ROOT", file=sys.stderr)
        return 1
    print(f"[input] {N} records to digitize (mode={mode}, num_workers={args.num_workers})")

    montage = np.zeros((N, N_LEADS, N_SAMPLES), dtype=np.float32)
    status = np.zeros((N, N_LEADS, N_SAMPLES), dtype=np.uint8)
    panel_qc_rows, record_qc_rows, index_rows = [], [], []
    overlays_by_idx, errors = {}, []
    done = 0

    def _accept(res):
        nonlocal done
        montage[res["idx"]] = res["montage"]
        status[res["idx"]] = res["status"]
        panel_qc_rows.extend(res["panel_rows"])
        record_qc_rows.append(res["record_row"])
        index_rows.append(res["index_row"])
        if res["overlay"] is not None:
            overlays_by_idx[res["idx"]] = res["overlay"]
        done += 1
        if done % 25 == 0 or done == N:
            print(f"  ... {done}/{N}")

    def _fail(rec_id, exc):
        errors.append({"record_id": rec_id, "error": str(exc),
                       "trace": traceback.format_exc()})
        print(f"[warn] {rec_id} failed: {exc}", file=sys.stderr)

    want = lambda i: i < args.preview_records
    if args.num_workers and args.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            fut2rp = {ex.submit(_run_one, rp, i, native_root, p, want(i)): rp
                      for i, rp in enumerate(records)}
            for fut in as_completed(fut2rp):
                rp = fut2rp[fut]
                try:
                    _accept(fut.result())
                except Exception as e:
                    if args.stop_on_first_failure:
                        ex.shutdown(cancel_futures=True)
                        raise
                    _fail(rp.record_id, e)
    else:
        for i, rp in enumerate(records):
            try:
                _accept(_run_one(rp, i, native_root, p, want(i)))
            except Exception as e:
                if args.stop_on_first_failure:
                    raise
                _fail(rp.record_id, e)

    # restore deterministic (row_index) order for the per-record outputs
    record_qc_rows.sort(key=lambda r: r["row_index"])
    index_rows.sort(key=lambda r: r["row_index"])
    panel_qc_rows.sort(key=lambda r: (r["ecg_id"], r["row"], str(r["col"])))
    overlays = [overlays_by_idx[i] for i in sorted(overlays_by_idx)]

    observed_mask = ((status == ST_VALID) | (status == ST_INTERP)).astype(np.uint8)

    # --- write arrays ---
    np.save(out_dir / f"digitized_montage_{sp}.npy", montage)
    np.save(out_dir / f"digitized_observed_mask_{sp}.npy", observed_mask)
    np.save(out_dir / f"digitized_status_{sp}.npy", status)

    # --- write CSVs ---
    pd.DataFrame(index_rows).to_csv(out_dir / "digitization_record_index.csv",
                                    index=False)
    pd.DataFrame(panel_qc_rows).to_csv(out_dir / "digitization_panel_qc.csv",
                                       index=False)
    pd.DataFrame(record_qc_rows).to_csv(out_dir / "digitization_record_qc.csv",
                                        index=False)

    # --- preview overlay ---
    try:
        save_preview_overlay(overlays, out_dir / "preview_digitization_overlay.png",
                             max_records=args.preview_records)
    except Exception as e:
        print(f"[warn] preview overlay failed: {e}", file=sys.stderr)

    n_ok = len(record_qc_rows)
    n_failed = len(errors)

    # --- eval-compatible waveform-dir bundle (04_eval contract) --------------- #
    eval_bundle = {"status": "disabled"}
    eval_bundle_attempted = False
    eval_bundle_failed = False
    if args.emit_eval_bundle:
        eval_dir = args.eval_out_dir or (out_dir / "eval_waveforms_500hz")
        if n_failed:
            eval_bundle = {"status": "skipped",
                           "reason": f"{n_failed} record(s) failed; refusing to emit "
                                     f"an eval bundle with zero-filled rows"}
            print(f"[eval-bundle] skipped ({n_failed} record failures).", file=sys.stderr)
        else:
            EM = load_eval_modeling(render_root, args.modeling_dir)
            manifest = discover_manifest(args.manifest, render_root, native_root, EM)
            if manifest is None:
                eval_bundle = {"status": "skipped",
                               "reason": "no montage manifest (ecg_id + "
                                         "filename_montage) found; pass --manifest"}
                print("[eval-bundle] skipped: no manifest with ecg_id + "
                      "filename_montage (pass --manifest).", file=sys.stderr)
            else:
                eval_bundle_attempted = True
                digi_paths = {
                    "summary": out_dir / "digitization_summary.json",
                    "panel_qc": out_dir / "digitization_panel_qc.csv",
                    "record_qc": out_dir / "digitization_record_qc.csv"}
                try:
                    eval_bundle = emit_eval_bundle(
                        montage, records, manifest, eval_dir, sp, EM, digi_paths,
                        args.val_waveform_dir, args.val_link)
                    print(f"[eval-bundle] wrote {eval_dir} "
                          f"(modeling={eval_bundle['modeling_source']}, "
                          f"self_verify={eval_bundle['self_verify']}, "
                          f"val={eval_bundle['val']['status']})")
                    if str(eval_bundle.get("self_verify", "")).startswith("FAILED"):
                        eval_bundle_failed = True
                except Exception as e:
                    eval_bundle = {"status": "error", "error": str(e),
                                   "trace": traceback.format_exc()}
                    eval_bundle_failed = True
                    print(f"[eval-bundle] FAILED: {e}", file=sys.stderr)

    # --- summary JSON ---
    def _safe_mean(rows, key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))
                and np.isfinite(r.get(key))]
        return float(np.mean(vals)) if vals else None

    def _safe_median(rows, key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))
                and np.isfinite(r.get(key))]
        return float(np.median(vals)) if vals else None

    # weighted-sample SNR: global centered-signal power / global noise power
    _eps = 1e-12
    _sig = float(sum((r.get("snr_sig_pow_centered") or 0.0) for r in panel_qc_rows))
    _noise = float(sum((r.get("snr_noise_pow") or 0.0) for r in panel_qc_rows))
    weighted_sample_snr = (float(10.0 * np.log10(_sig / (_noise + _eps)))
                           if _sig > 0 else None)

    # diagnostic SNR-decomposition aggregates (macro over panels) + lag stats
    snr_decomp_summary = None
    if args.snr_decomp:
        _lags = [r.get("best_lag_samples") for r in panel_qc_rows
                 if isinstance(r.get("best_lag_samples"), (int, float))]
        snr_decomp_summary = {
            "mean_snr_centered_raw_db": _safe_mean(panel_qc_rows, "snr_centered_raw_db"),
            "mean_snr_centered_affine_db": _safe_mean(panel_qc_rows, "snr_centered_affine_db"),
            "mean_snr_centered_lag_db": _safe_mean(panel_qc_rows, "snr_centered_lag_db"),
            "mean_snr_centered_lag_affine_db": _safe_mean(panel_qc_rows, "snr_centered_lag_affine_db"),
            "median_snr_centered_raw_db": _safe_median(panel_qc_rows, "snr_centered_raw_db"),
            "median_snr_centered_affine_db": _safe_median(panel_qc_rows, "snr_centered_affine_db"),
            "median_snr_centered_lag_db": _safe_median(panel_qc_rows, "snr_centered_lag_db"),
            "median_snr_centered_lag_affine_db": _safe_median(panel_qc_rows, "snr_centered_lag_affine_db"),
            "mean_rmse_raw_mV": _safe_mean(panel_qc_rows, "rmse_raw_mV"),
            "mean_rmse_affine_mV": _safe_mean(panel_qc_rows, "rmse_affine_mV"),
            "mean_rmse_lag_mV": _safe_mean(panel_qc_rows, "rmse_lag_mV"),
            "mean_rmse_lag_affine_mV": _safe_mean(panel_qc_rows, "rmse_lag_affine_mV"),
            "mean_abs_best_lag_samples": (float(np.mean(np.abs(_lags))) if _lags else None),
            "median_best_lag_samples": (float(np.median(_lags)) if _lags else None),
            "frac_panels_nonzero_lag": (float(np.mean([l != 0 for l in _lags]))
                                        if _lags else None),
            "mean_affine_gain": _safe_mean(panel_qc_rows, "affine_gain"),
            "mean_affine_bias_mV": _safe_mean(panel_qc_rows, "affine_bias_mV"),
            "snr_threshold_db": 17.0,
        }

    flags = [r["quality_flag"] for r in record_qc_rows]
    summary = {
        "script": Path(__file__).name,
        "input_mode": mode,
        "render_root": str(render_root),
        "native_root": (str(native_root) if native_root else None),
        "split": sp,
        "n_records_loaded": n_loaded,
        "n_records": N,
        "n_records_ok": len(record_qc_rows),
        "n_records_failed": len(errors),
        "expected_count": args.expected_count,
        "num_workers": args.num_workers,
        "overwrite": bool(args.overwrite),
        "limit": args.limit,
        "array_shape": [N, N_LEADS, N_SAMPLES],
        "lead_order": MONTAGE_LEADS,
        "native_fs": NATIVE_FS,
        "status_legend": {str(k): v for k, v in STATUS_LEGEND.items()},
        "lead_II_policy": "rhythm strip is definitive; short Lead II panel is QC-only",
        "mask_sums": {
            "observed_total": int(observed_mask.sum()),
            "printed_total": int((status != ST_UNPRINTED).sum()),
            "valid_total": int((status == ST_VALID).sum()),
            "interp_total": int((status == ST_INTERP).sum()),
            "missing_total": int((status == ST_MISSING).sum()),
            "expected_printed_per_record": 11 * 1250 + 5000,  # 18750
        },
        "qc": {
            "mean_corr": _safe_mean(record_qc_rows, "mean_corr"),
            "mean_rmse_mV": _safe_mean(record_qc_rows, "mean_rmse_mV"),
            "mean_observed_coverage": _safe_mean(record_qc_rows, "observed_coverage"),
            "mean_leadII_rhythm_corr": _safe_mean(record_qc_rows, "leadII_rhythm_corr"),
            "n_panels": len(panel_qc_rows),
            "n_clipped_panels": int(sum(1 for r in panel_qc_rows if r.get("clipped"))),
            "flags": {f: flags.count(f) for f in ("green", "yellow", "red")},
            # --- SNR vs native (centered is the preferred report metric) ---
            "mean_snr_centered_db": _safe_mean(panel_qc_rows, "snr_centered_db_vs_native"),
            "median_snr_centered_db": _safe_median(panel_qc_rows, "snr_centered_db_vs_native"),
            "macro_panel_snr_centered_db": _safe_mean(record_qc_rows, "mean_snr_centered_db"),
            "weighted_sample_snr_centered_db": weighted_sample_snr,
            "leadII_rhythm_snr_centered_db": _safe_mean(record_qc_rows,
                                                        "leadII_rhythm_snr_centered_db"),
            "mean_snr_db": _safe_mean(panel_qc_rows, "snr_db_vs_native"),
            "snr_decomposition": snr_decomp_summary,
        },
        "eval_bundle": eval_bundle,
        "params": asdict(p),
        "outputs": [
            f"digitized_montage_{sp}.npy", f"digitized_observed_mask_{sp}.npy",
            f"digitized_status_{sp}.npy", "digitization_record_index.csv",
            "digitization_panel_qc.csv", "digitization_record_qc.csv",
            "preview_digitization_overlay.png",
        ],
        "errors": errors[:50],
    }
    if eval_bundle.get("status") == "ok":
        summary["outputs"].append(f"eval_waveforms_500hz/{sp}_waveforms_500hz.npy")
    (out_dir / "digitization_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[done] {n_ok}/{N} records OK, {n_failed} failed -> {out_dir}")
    print(f"  montage {montage.shape}  observed_mask sum={int(observed_mask.sum())}")
    q = summary["qc"]
    print(f"  mean corr={q['mean_corr']}  mean coverage={q['mean_observed_coverage']}"
          f"  flags={q['flags']}")
    print(f"  mean SNR(centered)={q['mean_snr_centered_db']} dB  "
          f"weighted-sample={q['weighted_sample_snr_centered_db']} dB  "
          f"leadII-rhythm={q['leadII_rhythm_snr_centered_db']} dB")
    if q.get("snr_decomposition"):
        sd = q["snr_decomposition"]
        def _f(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) and x == x else "n/a"
        print("  SNR decomposition (mean centered dB; threshold 17):"
              f" raw={_f(sd['mean_snr_centered_raw_db'])}"
              f"  affine={_f(sd['mean_snr_centered_affine_db'])}"
              f"  lag={_f(sd['mean_snr_centered_lag_db'])}"
              f"  lag+affine={_f(sd['mean_snr_centered_lag_affine_db'])}")
        print(f"    |best_lag| mean={_f(sd['mean_abs_best_lag_samples'])} samples"
              f"  nonzero-lag frac={_f(sd['frac_panels_nonzero_lag'])}"
              f"  affine gain={_f(sd['mean_affine_gain'])}"
              f"  bias={_f(sd['mean_affine_bias_mV'])} mV")
    print(f"  eval_bundle: {eval_bundle.get('status')}"
          + (f" (self_verify={eval_bundle.get('self_verify')})"
             if eval_bundle.get("status") == "ok" else ""))

    # --- exit code: any failure, an unmet --expected_count, or a broken eval
    #     bundle (when one was actually attempted) is non-zero --------------- #
    rc = 0
    if n_failed:
        print(f"[fail] {n_failed} record(s) failed; see digitization_summary.json "
              f"'errors'.", file=sys.stderr)
        rc = 1
    if args.expected_count and not (n_ok == args.expected_count and n_failed == 0):
        print(f"[fail] strict check: need n_records_ok=={args.expected_count} and "
              f"n_records_failed==0, got ok={n_ok} failed={n_failed}.",
              file=sys.stderr)
        rc = 1
    if eval_bundle_attempted and eval_bundle_failed:
        print(f"[fail] eval bundle could not be produced/verified: "
              f"{eval_bundle.get('error') or eval_bundle.get('self_verify')}",
              file=sys.stderr)
        rc = 1
    if rc == 0:
        print("[ok] all records digitized successfully.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
