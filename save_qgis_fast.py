#!/usr/bin/env python3
"""Faster MintPy time-series point export for QGIS.

This is a project-local, CLI-compatible alternative to MintPy's save_qgis.py.
It keeps the same default fields, but reads data in row blocks and writes OGR
features inside transactions. For dense NISAR DS products, prefer GPKG output.
"""

from __future__ import annotations

import argparse
import errno
import os
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from osgeo import ogr, osr

from mintpy.save_qgis import read_bounding_box


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Keep example line breaks while still showing argument defaults."""

    def _get_help_string(self, action):
        if action.dest == "progress_bar":
            return action.help
        return super()._get_help_string(action)


AUTHOR_TEXT = """\
Author: 王帅
Affiliation: 中国矿业大学
Development date: 2026/4/27
Email: 2369989580@qq.com
"""

EXAMPLE = """\
example:
  save_qgis_fast.py timeseries_ERA5_ramp_demErr.h5 -g inputs/geometrygeo.h5
  save_qgis_fast.py timeseries_ERA5_ramp_demErr.h5 -g inputs/geometryRadar.h5
  save_qgis_fast.py timeseries_ERA5_ramp_demErr.h5 -g inputs/geometryRadar.h5 -b 200 150 400 350
  save_qgis_fast.py geo/geo_timeseries_ERA5_ramp_demErr.h5 -g geo/geo_geometryRadar.h5
  save_qgis_fast.py geo/geo_timeseries_demErr.h5 -g geo/geo_geometryRadar.h5 -o geo/geo_timeseries_demErr.gpkg
  save_qgis_fast.py geo/geo_timeseries_demErr.h5 -g geo/geo_geometryRadar.h5 --step 3 --min-coh 0.9 -o /tmp/qgis_thinned.gpkg
"""


@dataclass
class InputFiles:
    timeseries: str
    velocity: str
    coherence: str
    mask: str
    geometry: str


def _decode_date(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_attrs(fname: str) -> dict:
    with h5py.File(fname, "r") as h5:
        return {k: h5.attrs[k] for k in h5.attrs.keys()}


def _has_geo_attrs(attrs: dict) -> bool:
    return {"X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP"}.issubset(attrs.keys())


def _has_lat_lon_datasets(fname: str) -> bool:
    with h5py.File(fname, "r") as h5:
        return "latitude" in h5 and "longitude" in h5


def _can_locate_points(ts_file: str, geom_file: str) -> bool:
    """Return True if inputs contain enough information for QGIS lon/lat points."""
    if _has_geo_attrs(_read_attrs(ts_file)) or _has_geo_attrs(_read_attrs(geom_file)):
        return True
    return _has_lat_lon_datasets(geom_file)


def _find_dataset(h5: h5py.File, names: list[str], required: bool = True):
    for name in names:
        if name in h5:
            return h5[name]
    if required:
        raise KeyError(f"none of datasets exist: {', '.join(names)}")
    return None


def _default_output(ts_file: str) -> str:
    return str(Path(ts_file).with_suffix(".shp"))


def _find_geo_counterpart(args: argparse.Namespace) -> tuple[str, str] | None:
    ts_path = Path(args.ts_file)
    geom_path = Path(args.geom_file)
    ts_dir = ts_path.parent

    if ts_path.name.startswith("geo_"):
        geo_ts = ts_path
    else:
        geo_ts = ts_dir / "geo" / f"geo_{ts_path.name}"

    geom_candidates = []
    if geom_path.name.startswith("geo_"):
        geom_candidates.append(geom_path)
    else:
        geom_candidates.extend(
            [
                geo_ts.parent / f"geo_{geom_path.name}",
                geo_ts.parent / "geo_geometryRadar.h5",
                ts_dir / "geo" / "geo_geometryRadar.h5",
            ]
        )

    for geo_geom in geom_candidates:
        if geo_ts.is_file() and geo_geom.is_file() and _can_locate_points(str(geo_ts), str(geo_geom)):
            return str(geo_ts), str(geo_geom)
    return None


def resolve_geocoded_inputs(args: argparse.Namespace) -> None:
    """Switch radar-coordinate inputs to available geo products when needed."""
    if _can_locate_points(args.ts_file, args.geom_file):
        return

    counterpart = _find_geo_counterpart(args)
    if counterpart is None:
        msg = (
            "Can not get pixel-wise lon/lat from the input files.\n"
            f"  TimeSeries : {args.ts_file}\n"
            f"  Geometry   : {args.geom_file}\n"
            "The geometry file is radar-coordinate and does not contain latitude/longitude datasets.\n"
            "Please use geocoded MintPy files, for example:\n"
            "  save_qgis_fast.py geo/geo_timeseries_demErr.h5 -g geo/geo_geometryRadar.h5 -o geo/geo_timeseries_demErr.gpkg"
        )
        raise ValueError(msg)

    if args.pix_bbox is not None:
        msg = (
            "The input files are radar-coordinate and do not contain lon/lat, so save_qgis_fast.py "
            "can only auto-switch to geo products when no pixel bbox is supplied.\n"
            "Use the geo files directly with a bbox on the geo grid, or use -B/--geo-bbox."
        )
        raise ValueError(msg)

    if args.velocity_file or args.coherence_file or args.mask_file:
        msg = (
            "The input files are radar-coordinate and do not contain lon/lat. Auto-switching to geo "
            "products is disabled when --velocity-file/--coherence-file/--mask-file overrides are used."
        )
        raise ValueError(msg)

    geo_ts, geo_geom = counterpart
    print("input files are radar-coordinate and contain no lon/lat; using geocoded counterparts:")
    print(f"  TimeSeries : {geo_ts}")
    print(f"  Geometry   : {geo_geom}")
    args.ts_file = geo_ts
    args.geom_file = geo_geom


def _driver_from_output(out_file: str) -> tuple[str, str]:
    suffix = Path(out_file).suffix.lower()
    if suffix == ".gpkg":
        return "GPKG", "mintpy"
    if suffix == ".fgb":
        return "FlatGeobuf", "mintpy"
    if suffix in {".json", ".geojson"}:
        return "GeoJSON", "mintpy"
    return "ESRI Shapefile", "mintpy"


def gather_files(args: argparse.Namespace) -> InputFiles:
    ts_dir = os.path.dirname(args.ts_file)
    geo_prefix = "geo_" if os.path.basename(args.ts_file).startswith("geo_") else ""

    files = InputFiles(
        timeseries=args.ts_file,
        velocity=args.velocity_file or os.path.join(ts_dir, f"{geo_prefix}velocity.h5"),
        coherence=args.coherence_file or os.path.join(ts_dir, f"{geo_prefix}temporalCoherence.h5"),
        mask=args.mask_file or os.path.join(ts_dir, f"{geo_prefix}maskTempCoh.h5"),
        geometry=args.geom_file,
    )

    for fname in files.__dict__.values():
        if not os.path.isfile(fname):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fname)

    print("input files:")
    print(f"  TimeSeries : {files.timeseries}")
    print(f"  Velocity   : {files.velocity}")
    print(f"  Coherence  : {files.coherence}")
    print(f"  Mask       : {files.mask}")
    print(f"  Geometry   : {files.geometry}")
    return files


def create_layer(out_file: str, dates: list[str]):
    driver_name, layer_name = _driver_from_output(out_file)
    driver = ogr.GetDriverByName(driver_name)
    if driver is None:
        raise RuntimeError(f"OGR driver is not available: {driver_name}")

    if os.path.exists(out_file):
        print(f"output exists, overwriting: {out_file}")
        driver.DeleteDataSource(out_file)

    ds = driver.CreateDataSource(out_file)
    if ds is None:
        raise RuntimeError(f"failed to create output: {out_file}")

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer(layer_name, srs, geom_type=ogr.wkbPoint)
    if layer is None:
        raise RuntimeError(f"failed to create layer in output: {out_file}")

    field_defs = [
        ("CODE", ogr.OFTString, 8, 0),
        ("HEIGHT", ogr.OFTReal, 7, 2),
        ("H_STDEV", ogr.OFTReal, 5, 2),
        ("VEL", ogr.OFTReal, 8, 2),
        ("V_STDEV", ogr.OFTReal, 6, 2),
        ("COHERENCE", ogr.OFTReal, 5, 3),
        ("EFF_AREA", ogr.OFTInteger, 0, 0),
    ]

    for name, field_type, width, precision in field_defs:
        fd = ogr.FieldDefn(name, field_type)
        if width:
            fd.SetWidth(width)
        if precision:
            fd.SetPrecision(precision)
        layer.CreateField(fd)

    for date in dates:
        fd = ogr.FieldDefn(f"D{date}", ogr.OFTReal)
        fd.SetWidth(8)
        fd.SetPrecision(2)
        layer.CreateField(fd)

    layer_defn = layer.GetLayerDefn()
    field_idx = {layer_defn.GetFieldDefn(i).GetName(): i for i in range(layer_defn.GetFieldCount())}
    return ds, layer, layer_defn, field_idx, driver_name


def _geo_vectors(attrs: dict, box: tuple[int, int, int, int]):
    if not _has_geo_attrs(attrs):
        return None, None

    x0, y0, x1, y1 = box
    x_first = float(attrs["X_FIRST"])
    y_first = float(attrs["Y_FIRST"])
    x_step = float(attrs["X_STEP"])
    y_step = float(attrs["Y_STEP"])
    # Match MintPy's ut.get_lat_lon() convention: output point locations are
    # pixel centers, while X_FIRST/Y_FIRST describe the upper-left pixel corner.
    cols = np.arange(x0, x1, dtype=np.float64) + 0.5
    rows = np.arange(y0, y1, dtype=np.float64) + 0.5
    return x_first + cols * x_step, y_first + rows * y_step


def _lat_lon_block(lon_vec, lat_vec, lons_full, lats_full, row0: int, row1: int):
    if lon_vec is not None and lat_vec is not None:
        return lon_vec, lat_vec[row0:row1]
    return lons_full[row0:row1, :], lats_full[row0:row1, :]


def _as_float(values, scale=1.0):
    return float(values) * scale


def count_valid_points(args: argparse.Namespace, files: InputFiles, box: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = box
    total = 0

    with (
        h5py.File(files.velocity, "r") as vel_h5,
        h5py.File(files.coherence, "r") as coh_h5,
        h5py.File(files.mask, "r") as mask_h5,
        h5py.File(files.geometry, "r") as geom_h5,
    ):
        mask_ds = _find_dataset(mask_h5, ["mask"])
        coh_ds = _find_dataset(coh_h5, ["temporalCoherence", "coherence"])
        vel_ds = _find_dataset(vel_h5, ["velocity"])
        hgt_ds = _find_dataset(geom_h5, [args.height_dataset])

        for block_y0 in range(y0, y1, args.chunk_lines):
            block_y1 = min(block_y0 + args.chunk_lines, y1)
            mask = mask_ds[block_y0:block_y1, x0:x1] != 0
            coh = coh_ds[block_y0:block_y1, x0:x1]
            vel = vel_ds[block_y0:block_y1, x0:x1]
            hgt = hgt_ds[block_y0:block_y1, x0:x1]

            valid = mask & np.isfinite(coh) & np.isfinite(vel) & np.isfinite(hgt)
            if args.min_coh is not None:
                valid &= coh >= args.min_coh
            if args.step > 1:
                yy = np.arange(block_y0, block_y1)[:, None]
                xx = np.arange(x0, x1)[None, :]
                valid &= (yy % args.step == 0) & (xx % args.step == 0)

            total += int(np.count_nonzero(valid))
            if args.max_points and total >= args.max_points:
                return args.max_points

    return total


def write_points(args: argparse.Namespace, files: InputFiles, box: tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = box
    width = x1 - x0
    length = y1 - y0

    with h5py.File(files.timeseries, "r") as ts_h5:
        ts_ds = _find_dataset(ts_h5, ["timeseries"])
        dates = [_decode_date(d) for d in _find_dataset(ts_h5, ["date"])[:]]
        ts_attrs = {k: ts_h5.attrs[k] for k in ts_h5.attrs.keys()}

    out_file = args.out_file or _default_output(files.timeseries)
    ds, layer, layer_defn, field_idx, driver_name = create_layer(out_file, dates)

    geom_attrs = _read_attrs(files.geometry)
    coord_attrs = geom_attrs if _has_geo_attrs(geom_attrs) else ts_attrs
    lon_vec, lat_vec = _geo_vectors(coord_attrs, box)
    lats_full = lons_full = None
    if lon_vec is None or lat_vec is None:
        print("geometry has no geocoded X/Y attributes; falling back to MintPy lat/lon lookup")
        from mintpy.utils import utils as ut

        try:
            lats_full, lons_full = ut.get_lat_lon(ts_attrs, geom_file=files.geometry, box=box)
        except ValueError as exc:
            raise ValueError(
                "Can not get pixel-wise lon/lat. Use geocoded MintPy files, for example:\n"
                "  save_qgis_fast.py geo/geo_timeseries_demErr.h5 -g geo/geo_geometryRadar.h5 -o geo/geo_timeseries_demErr.gpkg"
            ) from exc

    print(f"output       : {out_file} ({driver_name})")
    print(f"pixel box    : x={x0}:{x1}, y={y0}:{y1} ({width} x {length})")
    if args.step > 1:
        print(f"decimation   : every {args.step} pixels in x/y")
    if args.min_coh is not None:
        print(f"coh filter   : temporalCoherence >= {args.min_coh}")
    if args.zero_first:
        print("zero-first   : enabled")
    if args.max_points:
        print(f"max points   : {args.max_points}")

    print("counting valid points for progress bar...")
    n_valid = count_valid_points(args, files, box)
    print(f"number of points with time-series: {n_valid}")

    date_fields = [field_idx[f"D{date}"] for date in dates]
    code_field = field_idx["CODE"]
    height_field = field_idx["HEIGHT"]
    h_std_field = field_idx["H_STDEV"]
    vel_field = field_idx["VEL"]
    v_std_field = field_idx["V_STDEV"]
    coh_field = field_idx["COHERENCE"]
    eff_area_field = field_idx["EFF_AREA"]

    counter = 0
    tx_counter = 0
    last_report = time.time()
    t0 = last_report
    transaction_open = False
    prog_bar = None
    if args.progress_bar and n_valid > 0:
        from mintpy.utils import ptime

        prog_bar = ptime.progressBar(maxValue=n_valid)

    def begin_transaction():
        nonlocal transaction_open
        if args.transaction_size > 0 and not transaction_open:
            layer.StartTransaction()
            transaction_open = True

    def commit_transaction():
        nonlocal transaction_open, tx_counter
        if transaction_open:
            layer.CommitTransaction()
            transaction_open = False
            tx_counter = 0

    with (
        h5py.File(files.timeseries, "r") as ts_h5,
        h5py.File(files.velocity, "r") as vel_h5,
        h5py.File(files.coherence, "r") as coh_h5,
        h5py.File(files.mask, "r") as mask_h5,
        h5py.File(files.geometry, "r") as geom_h5,
    ):
        ts_ds = _find_dataset(ts_h5, ["timeseries"])
        mask_ds = _find_dataset(mask_h5, ["mask"])
        coh_ds = _find_dataset(coh_h5, ["temporalCoherence", "coherence"])
        vel_ds = _find_dataset(vel_h5, ["velocity"])
        vel_std_ds = _find_dataset(vel_h5, ["velocityStd"], required=False)
        hgt_ds = _find_dataset(geom_h5, [args.height_dataset])

        begin_transaction()
        for block_y0 in range(y0, y1, args.chunk_lines):
            block_y1 = min(block_y0 + args.chunk_lines, y1)
            rel_y0 = block_y0 - y0
            rel_y1 = block_y1 - y0

            mask = mask_ds[block_y0:block_y1, x0:x1] != 0
            coh = coh_ds[block_y0:block_y1, x0:x1].astype(np.float64)
            vel = vel_ds[block_y0:block_y1, x0:x1].astype(np.float64)
            hgt = hgt_ds[block_y0:block_y1, x0:x1].astype(np.float64)
            if vel_std_ds is None:
                vel_std = np.zeros_like(vel, dtype=np.float64)
            else:
                vel_std = vel_std_ds[block_y0:block_y1, x0:x1].astype(np.float64)

            valid = mask & np.isfinite(coh) & np.isfinite(vel) & np.isfinite(hgt)
            if args.min_coh is not None:
                valid &= coh >= args.min_coh
            if args.step > 1:
                yy = np.arange(block_y0, block_y1)[:, None]
                xx = np.arange(x0, x1)[None, :]
                valid &= (yy % args.step == 0) & (xx % args.step == 0)

            rows, cols = np.nonzero(valid)
            if rows.size == 0:
                continue

            ts = ts_ds[:, block_y0:block_y1, x0:x1].astype(np.float64)
            if args.zero_first:
                ts -= ts[0:1, :, :]

            lon_block, lat_block = _lat_lon_block(lon_vec, lat_vec, lons_full, lats_full, rel_y0, rel_y1)
            geocoded_vectors = lon_vec is not None and lat_vec is not None

            for row, col in zip(rows, cols):
                if args.max_points and counter >= args.max_points:
                    break

                if geocoded_vectors:
                    lon = lon_block[col]
                    lat = lat_block[row]
                else:
                    lon = lon_block[row, col]
                    lat = lat_block[row, col]

                if not (np.isfinite(lon) and np.isfinite(lat)):
                    continue

                counter += 1
                tx_counter += 1
                feature = ogr.Feature(layer_defn)
                feature.SetField(code_field, hex(counter)[2:].zfill(8))
                feature.SetField(height_field, _as_float(hgt[row, col]))
                feature.SetField(h_std_field, 0.0)
                feature.SetField(vel_field, _as_float(vel[row, col], 1000.0))
                feature.SetField(v_std_field, _as_float(vel_std[row, col], 1000.0))
                feature.SetField(coh_field, _as_float(coh[row, col]))
                feature.SetField(eff_area_field, 1)

                for date_i, date_field in enumerate(date_fields):
                    feature.SetField(date_field, _as_float(ts[date_i, row, col], 1000.0))

                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint_2D(float(lon), float(lat))
                feature.SetGeometry(point)
                layer.CreateFeature(feature)
                feature = None

                if args.transaction_size > 0 and tx_counter >= args.transaction_size:
                    commit_transaction()
                    begin_transaction()

                if prog_bar is not None:
                    prog_bar.update(counter, every=args.progress_every, suffix=f"point {counter}/{n_valid}")
                else:
                    now = time.time()
                    if now - last_report >= args.report_seconds:
                        rate = counter / max(now - t0, 1e-6)
                        print(f"written {counter:,} points; {rate:,.0f} points/s; y={block_y0}:{block_y1}")
                        last_report = now

            if args.max_points and counter >= args.max_points:
                break

        commit_transaction()

    if prog_bar is not None:
        prog_bar.close()

    ds.FlushCache()
    ds = None
    elapsed = time.time() - t0
    rate = counter / max(elapsed, 1e-6)
    print(f"finished writing {counter:,} points in {elapsed:.1f} s ({rate:,.0f} points/s): {out_file}")
    return out_file


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert to QGIS compatible ps time-series\n\n" + AUTHOR_TEXT,
        epilog=EXAMPLE,
        formatter_class=HelpFormatter,
    )
    parser.add_argument("ts_file", help="time-series HDF5 file")
    parser.add_argument("-g", "--geom", dest="geom_file", required=True, help="geometry HDF5 file")
    parser.add_argument("-o", "--outshp", dest="out_file", help="Output shape/GPKG/vector file.")
    parser.add_argument(
        "-b",
        "--bbox",
        dest="pix_bbox",
        type=int,
        nargs=4,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="pixel bounding box in x0 y0 x1 y1",
    )
    parser.add_argument(
        "-B",
        "--geo-bbox",
        dest="geo_bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("S", "N", "W", "E"),
        help="geographic bounding box: south north west east",
    )
    parser.add_argument("--zf", "--zero-first", dest="zero_first", action="store_true", help="set first date displacement to zero")
    parser.add_argument("--step", type=int, default=1, help="write every Nth pixel in both x/y")
    parser.add_argument("--min-coh", type=float, default=None, help="minimum temporal coherence")
    parser.add_argument("--max-points", type=int, default=0, help="stop after this many points; 0 means no limit")
    parser.add_argument("--chunk-lines", type=int, default=64, help="number of image rows to read per block")
    parser.add_argument("--transaction-size", type=int, default=50000, help="OGR features per transaction; 0 disables transactions")
    parser.add_argument("--report-seconds", type=float, default=15.0, help="progress report interval")
    parser.add_argument("--progress-every", type=int, default=1000, help="update progress bar every N points")
    parser.add_argument("--no-progress-bar", dest="progress_bar", action="store_false", help="disable MintPy-style progress bar")
    parser.add_argument("--velocity-file", help="override velocity HDF5 file")
    parser.add_argument("--coherence-file", help="override temporal coherence HDF5 file")
    parser.add_argument("--mask-file", help="override mask HDF5 file")
    parser.add_argument("--height-dataset", default="height", help="height dataset name in geometry file")
    parser.set_defaults(progress_bar=True)
    return parser


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    if args.step < 1:
        raise ValueError("--step must be >= 1")
    if args.chunk_lines < 1:
        raise ValueError("--chunk-lines must be >= 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")

    resolve_geocoded_inputs(args)
    box = read_bounding_box(args.pix_bbox, args.geo_bbox, args.geom_file)
    files = gather_files(args)
    write_points(args, files, box)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
