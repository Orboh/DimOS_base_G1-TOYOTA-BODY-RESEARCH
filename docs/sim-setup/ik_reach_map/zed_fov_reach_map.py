#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZEDカメラの画角を N×N 分割し、各セル×各距離の3D点へ G1 右腕 IK が届くかを地図にする。

【このスクリプトはリポジトリのコードを一切変更しない】既存実装は読み取りのみで使う:
  - dimos.robot.unitree.g1.harvest.ik_approach.IkApproachSkill … 到達判定の正本
      （収穫パイプライン Phase B と同じ判定器。None を返したら「届かない」）
  - docs/sim-setup/sim_dds_bridge.py の胸カメラ規約           … 画素→torso 変換の根拠
      L600  hfov = SIM_CAM_HFOV(既定90°), 640x360
      L617  lpos = SIM_CAM_LOCAL_POS(既定 0.08,0,0.20)  torso相対の取付位置
      L618  fwd  = SIM_CAM_LOCAL_FWD(既定 1,0,-0.35)    torso相対の視線方向
      L635- optical基底 = [camX, -camY, fwd]（bridge が cam_to_torso として publish する物と同一）
      L710  fx = fy = (W/2)/tan(hfov/2)（A/F の比で画角が決まるので単位は相殺される）

実行:
  cd /isaac-sim/workspace/DimOS_base_G1-TOYOTA-BODY-
  .venv/bin/python <このファイル>            # Isaac Sim は不要（pinocchio + URDF のみ）
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys

import numpy as np

REPO = os.environ.get("DIMOS_REPO", "/isaac-sim/workspace/DimOS_base_G1-TOYOTA-BODY-")
sys.path.insert(0, REPO)

# 判定ごとに info ログが出ると 500 行埋まるので抑制（結果は自前で集計して出す）
logging.disable(logging.INFO)

# --- 配色: dataviz スキルの reference palette --------------------------------
# sequential は「単一色相・値が大きいほど濃い」。虹色ランプは使わない。
BLUE_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
C_GOOD, C_BAD = "#008300", "#e34948"     # status（必ず ○/× 記号も併記し色だけに頼らない）
C_SURFACE, C_INK, C_INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def optical_basis(fwd_raw: np.ndarray) -> np.ndarray:
    """torso←optical の回転行列。sim_dds_bridge.py L635-674 と同一手順。"""
    fwd = np.asarray(fwd_raw, float)
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    cam_z = -fwd                                    # USDカメラは -Z が視線
    cam_x = np.cross(up, cam_z)
    cam_x /= np.linalg.norm(cam_x)
    cam_y = np.cross(cam_z, cam_x)
    return np.column_stack([cam_x, -cam_y, fwd])    # optical: X=右, Y=下, Z=前


def main() -> int:
    ap = argparse.ArgumentParser(description="ZED画角グリッド × 右腕IK到達性マップ")
    ap.add_argument("--grid", type=int, default=10, help="画角の分割数 N（N×N）")
    ap.add_argument("--depths", default="0.3,0.4,0.5,0.6,0.7", help="カメラからの距離[m]（光軸Z）")
    ap.add_argument("--radius-center", default=None,
                    help="半径を測る中心（torso_link 座標の x,y,z[m]）。既定は torso 原点。"
                         "IK の基準（pelvis）にするなら 0.00396,0,-0.044")
    ap.add_argument("--random", type=int, default=0,
                    help="画角内をランダムにN点サンプリングする（格子ではなく一様乱択）")
    ap.add_argument("--rand-min", type=float, default=0.05, help="ランダム時の前方距離の下限[m]")
    ap.add_argument("--rand-max", type=float, default=0.65, help="同 上限[m]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planes", default=None,
                    help="体に正対する平面までの距離[m]（--radius-center から前方 X）。"
                         "指定するとこちらを使う")
    ap.add_argument("--radii", default=None,
                    help="torso_link 原点（腰）からの距離[m]。指定するとこちらを使う。"
                         "画素の光線上で |p_torso| = R になる点を解く")
    ap.add_argument("--hfov", type=float, default=float(os.getenv("SIM_CAM_HFOV", "90")))
    ap.add_argument("--width", type=int, default=int(os.getenv("SIM_CAM_W", "640")))
    ap.add_argument("--height", type=int, default=int(os.getenv("SIM_CAM_H", "360")))
    ap.add_argument("--cam-pos", default=os.getenv("SIM_CAM_LOCAL_POS", "0.08,0,0.20"))
    ap.add_argument("--cam-fwd", default=os.getenv("SIM_CAM_LOCAL_FWD", "1,0,-0.35"))
    ap.add_argument("--standoff", type=float, default=0.0,
                    help="目標手前で止める量[m]。0=その点そのものに指先を置く（既定）")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    USE_PLANE = args.planes is not None
    USE_RADIUS = (args.radii is not None) and not USE_PLANE
    RC = np.array([float(x) for x in args.radius_center.split(",")]) if args.radius_center \
        else np.zeros(3)
    depths = [float(x) for x in ((args.planes or args.radii or args.depths).split(","))]
    lpos = np.array([float(x) for x in args.cam_pos.split(",")])
    fwd = np.array([float(x) for x in args.cam_fwd.split(",")])
    N, W, H = args.grid, args.width, args.height
    os.makedirs(args.out, exist_ok=True)

    R_opt = optical_basis(fwd)
    hfov = math.radians(args.hfov)
    fx = fy = (W / 2.0) / math.tan(hfov / 2.0)
    cx, cy = W / 2.0, H / 2.0

    from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill
    skill = IkApproachSkill(standoff_m=args.standoff)
    arm = skill._arm                      # 読み取りのみ（関節リミット余裕の算出用）
    lower, upper = np.asarray(arm.lower, float), np.asarray(arm.upper, float)
    rest = [0.0] * 29                     # verify_m2_reach_ik.py と同じ基準姿勢
    ws_x, ws_y, ws_z = skill._ws_x, skill._ws_y, skill._ws_z

    print("=" * 78)
    print("ZED画角 → 右腕IK 到達性マップ")
    print("=" * 78)
    print(f"カメラ : {W}x{H}px  HFOV={args.hfov}°  (VFOV={math.degrees(2*math.atan((H/2)/fy)):.1f}°)")
    print(f"         torso相対 取付={lpos.tolist()}  視線={fwd.tolist()}  fx=fy={fx:.1f}px")
    print(f"判定器 : IkApproachSkill(standoff={args.standoff}m)  position_only IK / 基準姿勢=全関節0")
    print(f"         合格条件 = IK収束 かつ 関節デルタ≤{skill._max_joint_delta_deg:.0f}° かつ リミット内")
    print(f"         ワークスペース箱 X{ws_x} Y{ws_y} Z{ws_z} [m]（この外は即NG）")
    _basis = (f"{RC.tolist()}（torso座標）から前方、体に正対する平面まで" if USE_PLANE
              else (f"中心 {RC.tolist()}（torso座標）からの距離" if USE_RADIUS
                    else "カメラからの光軸方向の深さ"))
    print(f"距離の基準: {_basis}")
    print(f"格子   : {N}x{N} セル × 距離 {depths} m = {N*N*len(depths)} 点\n")

    if args.random:
        rng = np.random.default_rng(args.seed)
        print(f"距離の基準: {RC.tolist()}（torso座標）から前方 "
              f"{args.rand_min}..{args.rand_max} m をランダムに {args.random} 点\n")
        rows = []
        n_try = 0
        while len(rows) < args.random and n_try < args.random * 20:
            n_try += 1
            u = float(rng.uniform(0, W))
            v = float(rng.uniform(0, H))
            D = float(rng.uniform(args.rand_min, args.rand_max))
            dvec = R_opt @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            if abs(dvec[0]) < 1e-9:
                continue
            s_hit = (RC[0] + D - lpos[0]) / dvec[0]
            if s_hit <= 0:
                continue
            t = lpos + dvec * s_hit
            p_eff = t - np.array([args.standoff, 0.0, 0.0])
            in_box = (ws_x[0] <= p_eff[0] <= ws_x[1] and ws_y[0] <= p_eff[1] <= ws_y[1]
                      and ws_z[0] <= p_eff[2] <= ws_z[1])
            res = skill.solve(t, rest)
            ok = res is not None
            if ok:
                qq = np.asarray(res.q_right, float)
                margin = float(np.min(np.minimum(qq - lower, upper - qq)))
                err, conv, dmax = res.err, res.converged, float(np.max(np.abs(qq)))
            else:
                err, conv, margin, dmax = float("nan"), False, float("nan"), float("nan")
            qcols = {f"q{i}": (round(float(res.q_right[i]), 6) if ok else "") for i in range(7)}
            rows.append(dict(depth=round(D, 4), row=len(rows), col=0,
                             u=round(u, 1), v=round(v, 1),
                             tx=round(t[0], 4), ty=round(t[1], 4), tz=round(t[2], 4),
                             reach_ok=int(ok), in_ws_box=int(in_box), converged=int(bool(conv)),
                             err_m=round(err, 5) if err == err else "",
                             joint_margin_rad=round(margin, 4) if margin == margin else "",
                             max_joint_rad=round(dmax, 4) if dmax == dmax else "", **qcols))
        n_ok = sum(r["reach_ok"] for r in rows)
        print(f"  IK が解ける点: {n_ok}/{len(rows)} ({100*n_ok/len(rows):.1f}%)")
        n_box = sum(1 for r in rows if not r["in_ws_box"])
        print(f"  ワークスペース箱の外: {n_box}/{len(rows)} ({100*n_box/len(rows):.1f}%)\n")
        with open(os.path.join(args.out, "right_arm_joint_order.txt"), "w") as f:
            f.write("\n".join(arm.joint_names) + "\n")
        with open(os.path.join(args.out, "right_arm_limits.csv"), "w", newline="") as f:
            lw = csv.writer(f); lw.writerow(["joint", "urdf_lower_rad", "urdf_upper_rad"])
            for nm, lo_, up_ in zip(arm.joint_names, lower, upper):
                lw.writerow([nm, round(float(lo_), 6), round(float(up_), 6)])
        csv_path = os.path.join(args.out, "reach_map.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"出力: {csv_path}")
        return 0

    rows = []
    for di, Z in enumerate(depths):
        for r in range(N):
            for c in range(N):
                u = (c + 0.5) * W / N
                v = (r + 0.5) * H / N
                # Z=1 のときの光線方向（torso 座標）
                dvec = R_opt @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
                if USE_PLANE:
                    # 体に正対する平面（torso の X が一定）との交点
                    x_plane = RC[0] + Z
                    if abs(dvec[0]) < 1e-9:
                        continue
                    s_hit = (x_plane - lpos[0]) / dvec[0]
                    if s_hit <= 0:
                        continue
                    t = lpos + dvec * s_hit
                elif USE_RADIUS:
                    # |lpos + dvec*s| = Z(=半径) を満たす s（カメラより前方の解）
                    o = lpos - RC                 # 中心を原点に移して球と交える
                    aa = float(dvec @ dvec)
                    bb = 2.0 * float(o @ dvec)
                    cc_ = float(o @ o) - Z * Z
                    disc = bb * bb - 4 * aa * cc_
                    if disc < 0:
                        continue                      # その方向にその半径の点は無い
                    s_hit = (-bb + math.sqrt(disc)) / (2 * aa)
                    if s_hit <= 0:
                        continue
                    t = lpos + dvec * s_hit
                else:
                    t = lpos + dvec * Z
                p_eff = t - np.array([args.standoff, 0.0, 0.0])
                in_box = (ws_x[0] <= p_eff[0] <= ws_x[1] and ws_y[0] <= p_eff[1] <= ws_y[1]
                          and ws_z[0] <= p_eff[2] <= ws_z[1])
                res = skill.solve(t, rest)
                ok = res is not None
                if ok:
                    q = np.asarray(res.q_right, float)
                    margin = float(np.min(np.minimum(q - lower, upper - q)))  # リミットまでの最小余裕[rad]
                    err, conv, dmax = res.err, res.converged, float(np.max(np.abs(q)))
                elif in_box:
                    # 箱の中なのに解けなかった → ソルバの残差を参考値として取り直す
                    _q, conv, err = arm.ik.solve(
                        __import__("pinocchio").SE3(arm.fk_root(np.zeros(7)).rotation,
                                                   arm.torso_to_root(p_eff)), np.zeros(7))
                    margin, dmax = float("nan"), float(np.max(np.abs(_q)))
                else:
                    err, conv, margin, dmax = float("nan"), False, float("nan"), float("nan")
                qcols = {f"q{i}": (round(float(res.q_right[i]), 6) if ok else "") for i in range(7)}
                rows.append(dict(depth=Z, row=r, col=c, u=round(u, 1), v=round(v, 1),
                                 tx=round(t[0], 4), ty=round(t[1], 4), tz=round(t[2], 4),
                                 reach_ok=int(ok), in_ws_box=int(in_box), converged=int(bool(conv)),
                                 err_m=round(err, 5) if err == err else "",
                                 joint_margin_rad=round(margin, 4) if margin == margin else "",
                                 max_joint_rad=round(dmax, 4) if dmax == dmax else "", **qcols))
        n_ok = sum(x["reach_ok"] for x in rows if x["depth"] == Z)
        print(f"  距離 {Z:.2f} m : 到達 {n_ok:3d}/{N*N} セル ({100*n_ok/(N*N):.0f}%)")

    with open(os.path.join(args.out, "right_arm_joint_order.txt"), "w") as f:
        f.write("\n".join(arm.joint_names) + "\n")   # q0..q6 の並び（Isaac 側は名前で対応付ける）
    with open(os.path.join(args.out, "right_arm_limits.csv"), "w", newline="") as f:
        lw = csv.writer(f)
        lw.writerow(["joint", "urdf_lower_rad", "urdf_upper_rad"])
        for nm, lo, up in zip(arm.joint_names, lower, upper):
            lw.writerow([nm, round(float(lo), 6), round(float(up), 6)])

    csv_path = os.path.join(args.out, "reach_map.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- 集計 ---------------------------------------------------------------
    score = np.zeros((N, N), int)          # 何個の距離で届いたか
    errmap = np.full((len(depths), N, N), np.nan)
    okmap = np.zeros((len(depths), N, N), bool)
    marg = np.full((len(depths), N, N), np.nan)
    for x in rows:
        di = depths.index(x["depth"])
        okmap[di, x["row"], x["col"]] = bool(x["reach_ok"])
        if x["reach_ok"]:
            score[x["row"], x["col"]] += 1
        if x["err_m"] != "":
            errmap[di, x["row"], x["col"]] = x["err_m"]
        if x["joint_margin_rad"] != "":
            marg[di, x["row"], x["col"]] = x["joint_margin_rad"]

    print(f"\n到達スコア（そのセルが何個の距離で届いたか / 最大 {len(depths)}）")
    print("      " + "".join(f"c{c:<3d}" for c in range(N)))
    for r in range(N):
        print(f"  r{r:<3d}" + "".join(f"{score[r,c]:<4d}" for c in range(N)))
    print("  ※ row0=画像上端, col0=画像左端。col が小さい＝画像左＝ロボットから見て左(+Y)")

    # 最良セル: 到達数 → 平均関節余裕（大きいほど無理がない）
    cand = []
    for r in range(N):
        for c in range(N):
            if score[r, c] == 0:
                continue
            m = np.nanmean(marg[:, r, c]) if np.any(~np.isnan(marg[:, r, c])) else float("nan")
            cand.append((score[r, c], m, r, c))
    cand.sort(key=lambda t: (-t[0], -(t[1] if t[1] == t[1] else -9)))
    print(f"\n最も余裕をもって腕を伸ばせるセル 上位10（到達距離数 → 関節リミット余裕）")
    print(f"  {'順':>2} {'セル':>8} {'到達':>5} {'関節余裕[rad]':>13}  {'代表 torso座標(X前,Y左,Z上)':>28}")
    for i, (s, m, r, c) in enumerate(cand[:10], 1):
        mid = [x for x in rows if x["row"] == r and x["col"] == c and x["reach_ok"]]
        mid = mid[len(mid) // 2]
        print(f"  {i:>2} r{r}c{c:<5} {s:>2}/{len(depths)} {m:>13.3f}   "
              f"({mid['tx']:.3f}, {mid['ty']:.3f}, {mid['tz']:.3f}) @{mid['depth']}m")

    # --- 図 -----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch

    SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_STEPS)
    C_GRID, C_NA = "#dddbd4", "#e8e7e3"   # 罫線 / 評価対象外セル
    nd = len(depths)

    def _style(a):
        a.set_facecolor(C_SURFACE)
        a.tick_params(colors=C_INK2, labelsize=8.5)
        for sp in a.spines.values():
            sp.set_color(C_GRID)

    def _cellgrid(a):
        for c in range(N + 1):
            a.axvline(c * W / N, color=C_SURFACE, lw=1.6)
        for r in range(N + 1):
            a.axhline(r * H / N, color=C_SURFACE, lw=1.6)

    # 図1: メイン = 到達スコア。sequential 単一色相を段階数ぶんに離散化（値が読める）
    steps = [BLUE_STEPS[int(round(i * (len(BLUE_STEPS) - 1) / nd))] for i in range(nd + 1)]
    cmap1 = ListedColormap(steps)
    norm1 = BoundaryNorm(np.arange(-0.5, nd + 1.5, 1.0), cmap1.N)

    fig, ax = plt.subplots(figsize=(9.2, 5.6), facecolor=C_SURFACE)
    _style(ax)
    im = ax.imshow(score, cmap=cmap1, norm=norm1, origin="upper",
                   extent=[0, W, H, 0], interpolation="nearest", aspect="auto")
    for r in range(N):
        for c in range(N):
            s = int(score[r, c])
            ax.text((c + .5) * W / N, (r + .5) * H / N,
                    str(s) if s else "x",                      # 0 は記号でも示す（色頼みにしない）
                    ha="center", va="center", fontsize=9,
                    color=("#ffffff" if s > nd * 0.55 else (C_BAD if s == 0 else C_INK2)),
                    fontweight=("bold" if s == 0 else "normal"))
    _cellgrid(ax)
    ax.set_title("Right-arm IK reachability over the ZED field of view",
                 fontsize=13, color=C_INK, pad=26, loc="left")
    ax.text(0, 1.035, f"cell = how many of the {nd} tested camera distances "
                      f"({', '.join(f'{d:g}' for d in depths)} m) the right hand can reach   |   "
                      f"x = none",
            transform=ax.transAxes, fontsize=9.5, color=C_INK2)
    ax.set_xlabel("image x [px]   (left edge = robot's left / +Y)", fontsize=9.5, color=C_INK2)
    ax.set_ylabel("image y [px]   (top = up)", fontsize=9.5, color=C_INK2)
    cb = fig.colorbar(im, ax=ax, ticks=range(nd + 1), fraction=0.030, pad=0.02, shrink=0.82)
    cb.set_label("distances reached", color=C_INK2, fontsize=9)
    cb.ax.tick_params(colors=C_INK2, labelsize=8.5)
    cb.outline.set_edgecolor(C_GRID)
    fig.tight_layout()
    f1 = os.path.join(args.out, "reach_score.png")
    fig.savefig(f1, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)

    # 図2: 距離ごとの小倍数。濃さ = 不足距離 err（濃いほど遠い）、記号で可否。
    #      ワークスペース箱の外は IK を評価していないので「欠測」として灰＋斜線にする
    #      （最大値の色で塗ると「最も届かない」と誤読されるため）。
    vmax = float(np.nanmax(errmap)) if np.any(~np.isnan(errmap)) else 1.0
    fig, axes = plt.subplots(1, nd, figsize=(2.9 * nd, 3.5), facecolor=C_SURFACE)
    axes = np.atleast_1d(axes)
    for di, Z in enumerate(depths):
        a = axes[di]
        _style(a)
        a.imshow(np.zeros((N, N)), cmap=ListedColormap([C_NA]), origin="upper",
                 extent=[0, W, H, 0], interpolation="nearest", aspect="auto")   # 欠測の下地
        a.imshow(np.ma.masked_invalid(errmap[di]), cmap=SEQ, vmin=0, vmax=vmax,
                 origin="upper", extent=[0, W, H, 0], interpolation="nearest", aspect="auto")
        for r in range(N):
            for c in range(N):
                good = okmap[di, r, c]
                na = np.isnan(errmap[di, r, c])
                a.text((c + .5) * W / N, (r + .5) * H / N,
                       "o" if good else ("/" if na else "x"),
                       ha="center", va="center", fontsize=7,
                       color=(C_GOOD if good else (C_INK2 if na else C_BAD)),
                       fontweight="bold")
        _cellgrid(a)
        a.set_title(f"{Z:g} m   —   {int(okmap[di].sum())}/{N*N} reachable",
                    fontsize=10, color=C_INK, loc="left", pad=6)
        a.set_xticks([]); a.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(vmin=0, vmax=vmax))
    cb = fig.colorbar(sm, ax=axes.tolist(), fraction=0.014, pad=0.012, shrink=0.72)
    cb.set_label("shortfall: how far the fingertip stops short [m]", color=C_INK2, fontsize=8.5)
    cb.ax.tick_params(colors=C_INK2, labelsize=8)
    cb.outline.set_edgecolor(C_GRID)
    fig.legend(handles=[Patch(facecolor=BLUE_STEPS[0], edgecolor=C_GRID, label="o  reachable"),
                        Patch(facecolor=BLUE_STEPS[8], edgecolor=C_GRID, label="x  out of reach (shaded by shortfall)"),
                        Patch(facecolor=C_NA, edgecolor=C_GRID, label="/  outside workspace box — IK not evaluated")],
               loc="lower left", bbox_to_anchor=(0.012, -0.015), ncol=3, frameon=False,
               fontsize=8.5, labelcolor=C_INK2)
    fig.suptitle("Reachability per camera distance", fontsize=11.5, color=C_INK, x=0.012, ha="left")
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.012, right=0.90)
    f2 = os.path.join(args.out, "reach_by_distance.png")
    fig.savefig(f2, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)

    print(f"\n出力:\n  {csv_path}\n  {f1}\n  {f2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
