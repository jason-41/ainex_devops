"""
End-to-end diagnostic. Publishes a HUGE target (1.5 rad) on r_sho_pitch
while simultaneously reading the joint's current angle, and reports.

Run this with Gazebo running. No replay/manual_test needed.
"""
import sys, time, threading

sys.path.append("/usr/lib/python3/dist-packages")
from gz.transport13 import Node
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model


JOINT = "r_sho_pitch"
TARGET = 1.5
DURATION = 6.0
HZ = 30


def main():
    samples = []
    sub_count_when_publishing = []

    def on_state(msg):
        for j in msg.joint:
            if j.name == JOINT:
                samples.append((time.time(), float(j.axis1.position)))

    node = Node()
    if not node.subscribe(Model, "/world/grasp_world/model/ainex/joint_state", on_state):
        sys.exit("subscribe FAILED")
    pub = node.advertise(f"/ainex/{JOINT}/cmd_pos", Double)
    if not pub.valid():
        sys.exit("advertise FAILED")

    print(f"[diag] handshake wait...")
    time.sleep(1.5)

    # Take a baseline reading (no command yet)
    if samples:
        baseline = samples[-1][1]
        print(f"[diag] baseline {JOINT} = {baseline:+.5f} rad")
    else:
        print("[diag] WARNING: no joint_state samples received during 1.5s wait!")
        print("[diag]   → subscriber not getting messages from Gazebo")
        sys.exit(1)

    # Drown the topic in target=1.5 messages for DURATION seconds
    print(f"[diag] publishing target={TARGET} rad at {HZ} Hz for {DURATION}s ...")
    samples.clear()
    msg = Double(); msg.data = TARGET
    t0 = time.time()
    pub_count = 0
    while time.time() - t0 < DURATION:
        ok = pub.publish(msg)
        pub_count += int(bool(ok))
        time.sleep(1.0 / HZ)

    print(f"[diag] published {pub_count} messages "
          f"(expected ~{int(DURATION*HZ)})")
    if pub_count < 10:
        print("[diag] FAIL: pub.publish() kept returning False — gz transport refusing")
        return

    # Wait a moment for state to settle, then sample
    time.sleep(0.5)

    if not samples:
        print("[diag] FAIL: no joint_state samples received after publishing")
        print("[diag]   → state subscription died, can't tell if joint moved")
        return

    positions = [pos for _, pos in samples]
    p_min, p_max = min(positions), max(positions)
    p_final = positions[-1]
    delta = p_final - baseline

    print()
    print(f"[diag] {JOINT} angle samples after {DURATION}s of target=1.5:")
    print(f"        baseline: {baseline:+.5f}")
    print(f"        min:      {p_min:+.5f}")
    print(f"        max:      {p_max:+.5f}")
    print(f"        final:    {p_final:+.5f}")
    print(f"        delta from baseline: {delta:+.5f} rad")
    print()

    if abs(delta) > 0.5:
        print("[diag] VERDICT: joint moves — controller works fine.")
        print("       So why didn't replay/manual_test work? Likely either")
        print("         (a) target was too small to overcome friction (bump action scale)")
        print("         (b) command msg shape mismatch (verify Double.data field)")
    elif abs(delta) > 0.01:
        print("[diag] VERDICT: joint moves a tiny bit but PID can't reach target.")
        print("       → friction / damping is enormous, raise p_gain dramatically (e.g. 500).")
    else:
        print("[diag] VERDICT: joint did NOT move at all despite huge target.")
        print("       → Plugin not applying force. Possible causes:")
        print("         1. plugin attached to wrong entity (check launch log)")
        print("         2. bullet-featherstone doesn't honor JointForceCmd")
        print("            (try reverting world to dartsim, but mesh collisions break)")
        print("         3. URDF dynamics block has insane damping/friction")
        print("         4. <static>true</static> somewhere is freezing the model")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
