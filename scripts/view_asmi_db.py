#!/usr/bin/env python3
"""
View ASMI measurement data from asmi_data.db.

Usage:
  python scripts/view_asmi_db.py
  python scripts/view_asmi_db.py --campaign 6
  python scripts/view_asmi_db.py --campaign 6 --well A2
"""

import argparse
import struct
import sys
from pathlib import Path

# Run from ASMI_new
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "asmi_data.db"


def unpack_blob(blob: bytes) -> tuple[float, ...]:
    """Unpack little-endian doubles from BLOB."""
    n = len(blob) // 8
    return struct.unpack(f"<{n}d", blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=int, help="Filter by campaign ID")
    ap.add_argument("--well", type=str, help="Filter by well_id (e.g. A2)")
    ap.add_argument("--limit", type=int, default=5, help="Max campaigns to show (default 5)")
    args = ap.parse_args()

    import sqlite3
    conn = sqlite3.connect(DB_PATH)

    # List campaigns
    q = "SELECT id, description, created_at, status FROM campaigns"
    params = []
    if args.campaign:
        q += " WHERE id = ?"
        params.append(args.campaign)
    q += " ORDER BY id DESC"
    q += f" LIMIT {args.limit}"
    campaigns = conn.execute(q, params).fetchall()

    for cid, desc, created, status in campaigns:
        print(f"\n=== Campaign {cid}: {desc} ({created}) [{status}] ===")

        # Experiments (wells)
        exp_q = """
            SELECT e.id, e.well_id, e.created_at
            FROM experiments e
            WHERE e.campaign_id = ?
        """
        exp_params = [cid]
        if args.well:
            exp_q += " AND e.well_id = ?"
            exp_params.append(args.well.upper())
        exp_q += " ORDER BY e.well_id"
        exps = conn.execute(exp_q, exp_params).fetchall()

        for exp_id, well_id, exp_created in exps:
            print(f"\n  Well {well_id} (exp_id={exp_id})")

            # ASMI measurements
            rows = conn.execute(
                """
                SELECT id, z_positions, raw_forces, corrected_forces,
                       baseline_avg, baseline_std, force_exceeded, data_points
                FROM asmi_measurements
                WHERE experiment_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (exp_id,),
            ).fetchall()

            for row in rows:
                mid, z_blob, raw_blob, corr_blob, b_avg, b_std, exceeded, n_pts = row
                z_arr = unpack_blob(z_blob) if z_blob else ()
                raw_arr = unpack_blob(raw_blob) if raw_blob else ()
                corr_arr = unpack_blob(corr_blob) if corr_blob else ()
                print(f"    measurement_id={mid}: {n_pts} points, force_exceeded={bool(exceeded)}")
                print(f"    baseline: {b_avg:.4f} ± {b_std:.4f} N")
                if len(z_arr) > 0:
                    print(f"    Z range: {z_arr[0]:.3f} .. {z_arr[-1]:.3f} mm")
                    print(f"    first 3: Z={[f'{z:.3f}' for z in z_arr[:3]]}  F_corr={[f'{c:.3f}' for c in corr_arr[:3]]}")
                    if len(z_arr) > 5:
                        print(f"    last 3:  Z={[f'{z:.3f}' for z in z_arr[-3:]]}  F_corr={[f'{c:.3f}' for c in corr_arr[-3:]]}")

    conn.close()
    print()


if __name__ == "__main__":
    main()
