"""
Bypass policy + IK to test if r_sho_pitch responds to commands at all.

Publishes target=1.0 rad (≈57°) to r_sho_pitch for 5 seconds and
prints the joint's actual position every 0.5s. Three outcomes:

  - position climbs from 0 toward 1.0  →  controller works, problem is upstream
                                          (policy/IK/replay script)
  - position stays at 0.0               →  controller broken, dig into URDF/PID
  - position oscillates / overshoots    →  PID tuned wrong but reachable
"""
import sys
import time
import threading

sys.path.append("/usr/lib/python3/dist-packages")
from gz.transport13 import Node
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model


JOINT = "r_sho_pitch"
TARGET = 1.0   # rad
DURATION = 5.0
SAMPLE_HZ = 30


state_pos = [0.0]
def on_joint(msg):
    for j in msg.joint:
        if j.name == JOINT:
            state_pos[0] = float(j.axis1.position)


def main():
    node = Node()
    if not node.subscribe(Model, "/world/grasp_world/model/ainex/joint_state", on_joint):
        sys.exit("subscribe failed")
    pub = node.advertise(f"/ainex/{JOINT}/cmd_pos", Double)
    if not pub.valid():
        sys.exit("advertise failed")

    print(f"[manual] subscribed + advertised. waiting 1s for handshake...")
    time.sleep(1.0)
    print(f"[manual] start: {JOINT} = {state_pos[0]:+.4f} rad")
    print(f"[manual] publishing target = {TARGET:+.2f} rad for {DURATION}s")

    t0 = time.time()
    last_print = 0.0
    msg = Double(); msg.data = TARGET
    while time.time() - t0 < DURATION:
        pub.publish(msg)
        if time.time() - last_print > 0.3:
            print(f"  t={time.time()-t0:4.1f}s  {JOINT}={state_pos[0]:+.4f} rad  "
                  f"err={TARGET-state_pos[0]:+.3f}")
            last_print = time.time()
        time.sleep(1.0 / SAMPLE_HZ)

    print(f"[manual] final: {JOINT} = {state_pos[0]:+.4f} rad")
    if abs(state_pos[0]) < 0.01:
        print("[manual] VERDICT: joint did not move → controller-side bug")
    elif abs(state_pos[0] - TARGET) < 0.1:
        print("[manual] VERDICT: joint reached target → controller works, "
              "issue is upstream (policy/IK/replay)")
    else:
        print("[manual] VERDICT: joint moved partially → PID too weak, raise p_gain further")


if __name__ == "__main__":
    main()
