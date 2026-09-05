# ZED-M setup on the AGX Orin (for `unitree-g1-okra-harvest-zed`)

> **この文書は 2026-06-23 時点の記録です**（`feat/jetson-gpu-inference-service` から救出）。
> 本文中の `feat/g1-okra-langgraph` ブランチや `/mnt/ssd/workspace/DimOS_base_G1-TOYOTA-BODY-`
> といったパス・ブランチ名は当時の Orin 上の構成で、現在の main とは一致しません。
> **SDK / JetPack / pyzed のバージョン対応とインストール手順が本題**です。
> 手元の x86 機での ZED SDK 導入は [`oda/ZED_M_Depth_check/ZED_SDK_Install_Guide.md`](../../../../../oda/ZED_M_Depth_check/ZED_SDK_Install_Guide.md) を参照（そちらは Orin 非対応）。

How to install the ZED SDK + `pyzed` on the G1's AGX Orin and run the
`unitree-g1-okra-harvest-zed` blueprint (ZED depth → YOLO 3D detection).

Verified 2026-06-23 on:

| | |
|---|---|
| Board | AGX Orin |
| JetPack | 6.2 (L4T **R36.5.0**, kernel 5.15-tegra, aarch64) |
| ZED SDK | **5.4.0** (`ZED_SDK_Tegra_L4T36.5_v5.4.0`) |
| pyzed | **5.4** (`pyzed-5.4-cp312-...`) |
| Camera | **ZED-M** (S/N 17243330, FW 1523), USB3 |
| DimOS venv | `/mnt/ssd/workspace/DimOS_base_G1-TOYOTA-BODY-/.venv` (Python **3.12**) |
| Wired control path | `ssh tbr@192.168.123.222` (NIC `eno1`) |

> The Orin's DimOS checkout lives on the SSD (`/mnt/ssd/workspace/DimOS_base_G1-TOYOTA-BODY-`),
> on branch `feat/g1-okra-langgraph`, venv created by `uv sync`. The old
> `~/dimos` (main) checkout was removed — there is one DimOS on this box now.

---

## 0. Match the SDK to JetPack

The installer is **L4T-version specific**. Confirm the board's L4T first:

```bash
cat /etc/nv_tegra_release      # -> R36 (release), REVISION: 5.0  => L4T 36.5
```

Then pick the matching ZED SDK download (here: `l4t36.5`). The current installer
URLs (302-redirect to the CDN):

- `https://download.stereolabs.com/zedsdk/5.4/l4t36.5/jetsons`  → `ZED_SDK_Tegra_L4T36.5_v5.4.0.zstd.run`

If the board's L4T differs, change the `l4t36.x` segment.

## 1. Give the Orin internet (it has none by default)

The Orin sits on the isolated `192.168.123.0/24` G1 LAN with no default route.
Two options — pick one.

**A. Share the laptop's internet over the wired G1 LAN (NAT).** On the **laptop**
(`wlp0s20f3` = WiFi/uplink, `enx...` = wired to the G1 LAN — adjust to yours):

```bash
sudo sysctl -w net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -o wlp0s20f3 -j MASQUERADE
sudo iptables -A FORWARD -i enx00e03a681095 -o wlp0s20f3 -j ACCEPT
sudo iptables -A FORWARD -i wlp0s20f3 -o enx00e03a681095 -m state --state RELATED,ESTABLISHED -j ACCEPT
```

On the **Orin** (laptop's G1-LAN IP is the gateway, e.g. `192.168.123.50`):

```bash
sudo ip route add default via 192.168.123.50
echo nameserver 8.8.8.8 | sudo tee -a /etc/resolv.conf
ping -c2 8.8.8.8        # confirm reachability
```

**B. Download on the laptop, copy over.** If the Orin can't get online, fetch
the installer on the laptop and `scp` it (the file is ~65–86 MB):

```bash
# laptop
wget -O ~/Downloads/ZED_SDK_Tegra_L4T36.5_v5.4.0.zstd.run \
  'https://stereolabs.sfo2.cdn.digitaloceanspaces.com/zedsdk/5.4/ZED_SDK_Tegra_L4T36.5_v5.4.0.zstd.run'
scp ~/Downloads/ZED_SDK_Tegra_L4T36.5_v5.4.0.zstd.run tbr@192.168.123.222:/mnt/ssd/
```

> The installer itself runs `apt-get` to pull dependencies, so the Orin needs
> internet (option A) **for the install step** even if you used B for the file.

## 2. Install the ZED SDK (silent)

On the **Orin**:

```bash
chmod +x /mnt/ssd/ZED_SDK_Tegra_L4T36.5_v5.4.0.zstd.run
sudo bash /mnt/ssd/ZED_SDK_Tegra_L4T36.5_v5.4.0.zstd.run -- silent skip_python skip_cuda
# installs to /usr/local/zed/ ; pulls apt deps; takes a few minutes
```

(`skip_python` here — we install `pyzed` explicitly in step 4 into the right venv.)

## 3. Fix permissions + register the shared libs

The installer leaves `/usr/local/zed/` as `root:zed 0770`, so a non-root import
fails with `ImportError: libsl_zed.so: cannot open shared object file`. Fix both
the read bit and the linker path:

```bash
sudo chmod -R a+rX /usr/local/zed/
echo /usr/local/zed/lib | sudo tee /etc/ld.so.conf.d/zed.conf
sudo ldconfig
ldconfig -p | grep zed        # should list libsl_zed.so / libsl_ai.so
```

(Optional, cleaner: `sudo usermod -aG zed tbr` and re-login instead of the chmod.)

## 4. Install `pyzed` into the DimOS venv (Python 3.12)

The SDK's helper downloads the wheel matching the interpreter you run it with.
The DimOS venv is **Python 3.12**, so run it with that interpreter (it has no
`pip`, so use `uv` to install the downloaded wheel):

```bash
cd /mnt/ssd/workspace/DimOS_base_G1-TOYOTA-BODY-
# downloads ~/pyzed-5.4-cp312-cp312-linux_aarch64.whl (pip step inside may warn — ignore)
sudo .venv/bin/python /usr/local/zed/get_python_api.py || true
uv pip install --python .venv/bin/python ~/pyzed-5.4-cp312-cp312-linux_aarch64.whl
```

## 5. Verify the camera

Plug the ZED-M into a **USB3** port, then:

```bash
lsusb | grep -i 2b03                       # STEREOLABS ZED-M camera
.venv/bin/python -c 'import pyzed.sl as sl; print(sl.Camera.get_sdk_version())'   # 5.4.0
```

Quick open test:

```python
.venv/bin/python - <<'PY'
import pyzed.sl as sl
cam = sl.Camera(); init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD720
print(cam.open(init))                       # ERROR_CODE.SUCCESS
print(cam.get_camera_information().serial_number)
cam.close()
PY
```

## 6. LCM multicast (needed before `dimos run`)

DimOS transports run over LCM multicast on `lo`. If multicast/route aren't set,
`dimos run` aborts in its preflight (`sudo ip link set lo multicast on` fails).
Enable them once per boot:

```bash
sudo ip link set lo multicast on
sudo ip route add 224.0.0.0/4 dev lo
```

## 7. YOLO weights (Git LFS)

The detector loads weights from `data/models_yolo/`. They're Git-LFS pointers; on
a fresh checkout the run fails with `models_yolo ... not found`. Either pull LFS
on the Orin, or copy from the laptop:

```bash
# laptop
rsync -av data/models_yolo/ tbr@192.168.123.222:/mnt/ssd/workspace/DimOS_base_G1-TOYOTA-BODY-/data/models_yolo/
```

## 8. Run

```bash
cd /mnt/ssd/workspace/DimOS_base_G1-TOYOTA-BODY-
ollama serve & ollama pull qwen3-vl:2b          # verify_harvest VLM (local)
ROBOT_INTERFACE=eno1 .venv/bin/dimos run unitree-g1-okra-harvest-zed
```

Expected boot log:

```
[ZED][INFO] [Init]  Camera successfully opened.   (HD720@15)
HarvestModule started — LIVE — real YOLO detect; depth=ZED; verify=Ollama:qwen3-vl:2b; ...
```

---

## Enabling NEURAL depth (optional, much denser depth)

`PERFORMANCE` is the default because `NEURAL` needs TensorRT + an optimized
engine. NEURAL fills depth holes on low-texture / single-colour surfaces, so a
near-field object (e.g. a banana at 30 cm) gets a dense, accurate pointcloud
where PERFORMANCE leaves only sparse edge points. Verified on the Orin
(2026-06-23): at 30 cm, PERFORMANCE gave the centre-pixel method a 0.45 m
fallback and ~4k surface points; NEURAL gave ~0.28 m and ~60k points, with all
three 3D methods agreeing.

This Orin ships TensorRT **runtime** only — NEURAL also needs the dev/bindings,
then a one-time engine optimization:

```bash
# 1. TensorRT dev + python bindings (same v10.3 as the runtime; NVIDIA L4T repo)
sudo apt-get update
sudo apt-get install -y python3-libnvinfer libnvinfer-dev   # adds NvInfer.h + libnvinfer.so + python3 tensorrt
# without these, ZED_Diagnostic prints "Cannot find TENSORRT" and -nrlo does nothing

# 2. Optimize (and download) the NEURAL depth engine (~6 min on Orin; jetson_clocks helps)
sudo jetson_clocks
/usr/local/zed/tools/ZED_Diagnostic -c -nrlo
# writes /usr/local/zed/resources/.neural_depth_*.model_optimized-...
# NOTE: the process can sit at 100% before exiting; the engine is already written.

# 3. use NEURAL: blueprint ZEDCamera(depth_mode="NEURAL"), or the scripts' --depth_mode NEURAL
```

NEURAL is heavier than PERFORMANCE (GPU load, lower FPS) — pick per task. Other
modes: `NEURAL_LIGHT` (faster, optimize with `-nrlo_light`), `NEURAL_PLUS`
(highest quality, `-nrlo_plus`).

## Gotchas (hit during bring-up)

- **NEURAL depth needs TensorRT.** With `depth_mode=NEURAL` and no TensorRT
  engine the SDK errors `NEURAL TRT NOT FOUND` then `CORRUPTED SDK INSTALLATION`
  and the camera fails to open. See "Enabling NEURAL depth" above; the blueprint
  pins `ZEDCamera(depth_mode="PERFORMANCE")` by default.
- **`libsl_zed.so` not found** → step 3 (permissions + `ldconfig`) wasn't done.
- **`No module named pip` in the venv** → it's a `uv` venv; install wheels with
  `uv pip install --python .venv/bin/python <wheel>`, not `python -m pip`.
- **`pyzed` ABI is interpreter-specific** → a `cp310` wheel won't import in the
  3.12 venv. Let `get_python_api.py` (run with `.venv/bin/python`) fetch the
  `cp312` wheel.
- **First run downloads a calibration + resource file** for the camera serial
  (needs internet on first launch); cached afterwards.
- **`Gravity alignment issues detected`** from the ZED is a benign warning while
  positional tracking re-initialises.

## What this blueprint does

`unitree-g1-okra-harvest-zed` = `ZEDCamera(PERFORMANCE)` + `HarvestModule(use_zed_depth=True)`.
The head/RGB `color_image` feeds YOLO; `depth_image` is wired to the module's
`depth_getter`, so each detection's 3D position uses **ZED metric depth at the
box-centre pixel** instead of the assumed-depth pinhole fallback (out-of-range /
NaN depth falls back to 0.45 m). Detection target is still the COCO `banana`
proxy (`target_classes="banana"`) until an okra-fine-tuned weight is loaded.

> ⚠️ **Not yet verified on the robot with a real target in frame.** Boot-to-flow
> works end-to-end, but the detect→3D→select→grasp path with an actual
> banana/okra in view has not been exercised on the G1 yet.
