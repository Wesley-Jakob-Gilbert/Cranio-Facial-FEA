"""Host-side logger + conversion for bench_daq.ino (OUTLINE; needs `pip install pyserial`).

  python validation/bench/host_logger.py log  /dev/ttyACM0 S1_trial1.csv     # stream raw counts
  python validation/bench/host_logger.py shunt                                 # print shunt table

Conversion uses shunt calibration, which calibrates the whole chain (bridge excitation,
dropping resistors, PGA gain, ADC reference) in one step:

  eps_sim = - R_g / (GF * (R_g + R_s))            (shunt R_s across the ACTIVE arm,
                                                  Micro-Measurements TN-514)
  k_ch    = eps_sim / (counts_shunted - counts_unshunted)        [strain per count]
  eps     = k_ch * (counts - counts_zero)

For 3-wire quarter bridges place the shunt at the gauge end of the leads (across the
gauge tabs/terminal), so lead-resistance desensitisation is included.
"""
from __future__ import annotations

import sys
import time


def eps_shunt(R_g=350.0, GF=2.0, R_s=174e3):
    return -R_g / (GF * (R_g + R_s))


def shunt_table():
    for R_s in (34.8e3, 69.8e3, 174e3, 348e3):
        print(f"R_s = {R_s/1e3:7.1f} k  ->  {eps_shunt(R_s=R_s)*1e6:9.1f} microstrain (350R, GF 2.0)")


def log(port, out, seconds=None):
    import serial
    s = serial.Serial(port, 115200, timeout=1)
    time.sleep(2)
    s.write(b"C"); time.sleep(3)
    s.write(b"S")
    t0 = time.time()
    with open(out, "w") as f:
        f.write(f"# started {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        while seconds is None or time.time() - t0 < seconds:
            line = s.readline().decode(errors="replace").strip()
            if line:
                f.write(f"{time.time():.3f},{line}\n")
    s.write(b"X")


if __name__ == "__main__":
    if sys.argv[1] == "shunt":
        shunt_table()
    elif sys.argv[1] == "log":
        log(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else None)
