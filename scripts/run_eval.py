"""Evaluate the Wiener + Viterbi pipeline on PPG-DaLiA.

Protocol matches Reiss et al. (2019): 8 s windows shifted by 2 s, MAE [bpm]
against the ECG-derived ground truth, reported per subject and per activity.

Usage:
  python scripts/run_eval.py <data_dir>                 # all 15 subjects
  python scripts/run_eval.py <data_dir> --grid          # tune alpha/sigma on S1-S3
  python scripts/run_eval.py <data_dir> --subjects S1,S2
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppg_hr.data import ACTIVITIES, SUBJECTS, load_subject, window_activity  # noqa: E402
from ppg_hr.pipeline import (  # noqa: E402
    ALPHA, SIGMA_BPM, estimate_hr, hr_from_spectra, subject_spectra,
)

TUNE_SUBJECTS = ("S1", "S2", "S3")
GRID_ALPHA = (1.0, 2.0, 4.0, 8.0)
GRID_SIGMA = (1.5, 2.5, 4.0)


def run_grid(data_dir):
    cache = {}
    for s in TUNE_SUBJECTS:
        rec = load_subject(data_dir, s)
        cache[s] = (*subject_spectra(rec), rec["hr"])
        print(f"spectra ready: {s}")
    print(f"\ngrid search on {','.join(TUNE_SUBJECTS)} (mean MAE [bpm]):")
    print("alpha\\sigma " + "".join(f"{sg:>8.1f}" for sg in GRID_SIGMA))
    for a in GRID_ALPHA:
        row = []
        for sg in GRID_SIGMA:
            maes = []
            for s in TUNE_SUBJECTS:
                p_ppg, p_acc, n_win, hr = cache[s]
                est, _ = hr_from_spectra(p_ppg, p_acc, a, sg)
                maes.append(np.mean(np.abs(est - hr[:n_win])))
            row.append(np.mean(maes))
        print(f"{a:>10.1f} " + "".join(f"{m:>8.2f}" for m in row))


def run_eval(data_dir, subjects, alpha, sigma_bpm, out_dir):
    rows = []
    act_err = {code: [] for code in ACTIVITIES}
    dump = {k: [] for k in ("subject", "est", "gt", "conf", "activity")}
    for s in subjects:
        rec = load_subject(data_dir, s)
        res = estimate_hr(rec, alpha=alpha, sigma_bpm=sigma_bpm)
        gt = rec["hr"][: res["n_win"]]
        err = np.abs(res["hr_est"] - gt)
        rows.append((s, float(err.mean())))
        acts = np.array([
            window_activity(rec["activity"], rec["fs_activity"], i)
            for i in range(res["n_win"])
        ])
        for e, a in zip(err, acts):
            act_err[a].append(e)
        dump["subject"].append(np.full(res["n_win"], s))
        dump["est"].append(res["hr_est"])
        dump["gt"].append(gt)
        dump["conf"].append(res["conf"])
        dump["activity"].append(acts)
        print(f"{s:>4}: MAE {err.mean():6.2f} bpm  (median {np.median(err):5.2f}, "
              f"windows {len(err)})")

    maes = np.array([m for _, m in rows])
    print(f"\noverall: MAE {maes.mean():.2f} +/- {maes.std():.2f} bpm "
          f"(alpha={alpha}, sigma={sigma_bpm})")
    print("\nper activity:")
    for code in sorted(act_err):
        if act_err[code]:
            e = np.array(act_err[code])
            print(f"  {ACTIVITIES[code]:>13}: MAE {e.mean():6.2f} bpm  (n={len(e)})")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mae_per_subject.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "mae_bpm", "alpha", "sigma_bpm"])
        for s, m in rows:
            w.writerow([s, f"{m:.3f}", alpha, sigma_bpm])
    npz = out_dir / "windows.npz"
    np.savez_compressed(npz, **{k: np.concatenate(v) for k, v in dump.items()})
    print(f"\nwrote {out} and {npz}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--sigma", type=float, default=SIGMA_BPM)
    ap.add_argument("--grid", action="store_true",
                    help=f"tune alpha/sigma on {','.join(TUNE_SUBJECTS)} only")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.grid:
        run_grid(args.data_dir)
        return
    subjects = SUBJECTS if args.subjects == "all" else tuple(args.subjects.split(","))
    run_eval(args.data_dir, subjects, args.alpha, args.sigma, Path(args.out))


if __name__ == "__main__":
    main()
