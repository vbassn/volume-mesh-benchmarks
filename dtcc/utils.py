
import re

_MESH_PATTERN = re.compile(
    # r"^MESH_REPORT"
    r"\s+step=(?P<step>\d+)"
    r"\s+vertices=(?P<vertices>\d+)"
    r"\s+cells=(?P<cells>\d+)"
    r"\s+time_s=(?P<time>[\d.]+)"
    r"\s+min_ar=(?P<min_ar>[\d.]+)"
    r"\s+median_ar=(?P<median_ar>[\d.]+)"
    r"\s+max_ar=(?P<max_ar>[\d.]+)"
)

def parse_mesh_reports(text: str) -> dict[int, dict[str, float | int]]:
    results: dict[int, dict[str, float | int]] = {}
    for line in text.splitlines():
        m = _MESH_PATTERN.match(line)
        if not m:
            continue
        gd = m.groupdict()
        step = int(gd.pop("step"))
        results[step] = {
            "vertices": int(gd["vertices"]),
            "cells":    int(gd["cells"]),
            "time_s":   float(gd["time"]),
            "min_ar":   float(gd["min_ar"]),
            "median_ar":float(gd["median_ar"]),
            "max_ar":   float(gd["max_ar"]),
        }
    return results


def parse_mesh_report_txt(filename):
    """
    Parse a mesh_report.txt file and return a dictionary of metrics keyed by metric_step.
    For example, 'vertices_2': 45405, 'time_s_3': 1.319, etc.
    """
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split into key=value tokens
            tokens = line.split()
            step = None
            # first pass: find the step value
            for tok in tokens:
                if tok.startswith('step='):
                    _, step = tok.split('=', 1)
                    break
            if step is None:
                continue
            # second pass: parse other metrics
            for tok in tokens:
                key, val = tok.split('=', 1)
                if key == 'step':
                    continue
                # build dict key with step suffix
                dict_key = f"{key}_{step}"
                # convert to int or float
                try:
                    if key in ('vertices', 'cells'):
                        data[dict_key] = int(val)
                    else:
                        data[dict_key] = float(val)
                except ValueError:
                    data[dict_key] = val
    return data