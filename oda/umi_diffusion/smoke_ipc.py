#!/usr/bin/env python
"""Client-side IPC smoke test for umi_policy_server (Step 3, offline / dummy-cam).

Assumes a server is already running (start it with --dummy-cam). Sends a few predict
requests and checks the reply is a list of 6-dim [pos3, aa3] waypoints with a sane
inference time. Validates the ZMQ+msgpack contract + the 9->6 action decode path
end-to-end WITHOUT a GoPro or a robot.

Run (after starting the server):
  python oda/umi_diffusion/smoke_ipc.py            # any env with pyzmq+msgpack
"""
import sys
import time

import msgpack
import numpy as np
import zmq

ADDR = sys.argv[1] if len(sys.argv) > 1 else "tcp://127.0.0.1:5599"


def main():
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.setsockopt(zmq.SNDTIMEO, 5000)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(ADDR)

    # a plausible pre-grasp EE pose (ROOT frame), pos + axis-angle
    pos = [0.02, -0.22, 0.29]
    aa = [-2.0, 0.03, 0.03]

    ok = True
    for i in range(6):
        req = {"cmd": "predict", "t": time.time(), "eef_pos": pos, "eef_rot_aa": aa,
               "reset": (i == 0)}
        t0 = time.time()
        sock.send(msgpack.packb(req, use_bin_type=True))
        rep = msgpack.unpackb(sock.recv(), raw=False)
        rtt = (time.time() - t0) * 1e3
        if not rep.get("ok"):
            print(f"[{i}] SERVER ERROR: {rep.get('err')}")
            ok = False
            break
        acts = np.asarray(rep["actions"], dtype=float)
        assert acts.ndim == 2 and acts.shape[-1] == 6, f"expected (N,6), got {acts.shape}"
        print(f"[{i}] n={rep['n']} shape={acts.shape} infer={rep.get('infer_ms',0):.1f}ms "
              f"rtt={rtt:.1f}ms wp0_pos={np.round(acts[0,:3],4)} wp0_aa={np.round(acts[0,3:],4)}")
    sock.close(0)
    print("SMOKE_IPC", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
