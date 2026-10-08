#!/bin/sh
# Enumerate and characterize the ZCU104's on-board power sensors.
#
# Runs on the board, reads only sysfs, and changes nothing. Its output is the
# input to the sampler design: which rails exist, how they are named stably
# across boots, how fast they can be sampled, and whether the driver's power
# register agrees with bus voltage times current.
set -u

say() { printf '%s\n' "$*"; }
rule() { say "-------------------------------------------------------------"; }

read_attr() {
    # Echo an attribute's value, or nothing when it is absent or unreadable.
    [ -r "$1" ] && cat "$1" 2>/dev/null
}

say "DeepTile power-sensor probe"
say "host    : $(hostname 2>/dev/null)"
say "date    : $(date 2>/dev/null)"
say "kernel  : $(uname -r 2>/dev/null)"
say "boot id : $(read_attr /proc/sys/kernel/random/boot_id)"
rule

# ---------------------------------------------------------------------------
# 1. hwmon inventory
#
# hwmonN indices are assigned in probe order and are not stable across boots,
# so every device is reported with the identifiers that are: the driver name,
# the I2C bus-address pair, and the device-tree label when one exists.
# ---------------------------------------------------------------------------
say "== hwmon devices =="
found=0
for h in /sys/class/hwmon/hwmon*; do
    [ -d "$h" ] || continue
    found=$((found + 1))
    name=$(read_attr "$h/name")
    dev=$(readlink -f "$h/device" 2>/dev/null)
    i2c=$(basename "$dev" 2>/dev/null)
    label=$(read_attr "$dev/of_node/label")
    [ -n "$label" ] || label=$(read_attr "$h/device/of_node/label")
    shunt=$(read_attr "$h/shunt_resistor")
    interval=$(read_attr "$h/update_interval")

    say ""
    say "$(basename "$h")  name=${name:-?}  i2c=${i2c:-?}  label=${label:-<none>}"
    say "  sysfs      : $h"
    [ -n "$dev" ] && say "  device     : $dev"
    [ -n "$shunt" ] && say "  shunt_resistor : ${shunt} uOhm"
    [ -n "$interval" ] && say "  update_interval: ${interval} ms"

    # Channel values. mV, mA and uW are the hwmon units for these classes.
    for f in "$h"/in*_input "$h"/curr*_input "$h"/power*_input "$h"/temp*_input; do
        [ -r "$f" ] || continue
        base=$(basename "$f" _input)
        val=$(read_attr "$f")
        clabel=$(read_attr "$h/${base}_label")
        say "  ${base}=${val}${clabel:+  (${clabel})}"
    done
done
[ "$found" -eq 0 ] && say "  none found -- no hwmon class registered"
rule

# ---------------------------------------------------------------------------
# 2. Other measurement providers
#
# Some rails are behind PMBus regulators rather than INA226 shunt monitors, and
# the Xilinx AMS block reports on-die voltages that no INA sees.
# ---------------------------------------------------------------------------
say "== i2c devices =="
for d in /sys/bus/i2c/devices/*; do
    [ -e "$d" ] || continue
    n=$(read_attr "$d/name")
    say "  $(basename "$d")  ${n:-?}"
done
say ""
say "== iio devices (AMS / sysmon) =="
for d in /sys/bus/iio/devices/*; do
    [ -e "$d" ] || continue
    n=$(read_attr "$d/name")
    say "  $(basename "$d")  ${n:-?}"
done
rule

# ---------------------------------------------------------------------------
# 3. Power-register consistency
#
# The INA226 computes power internally from its calibration register, which the
# driver programs from the device tree's shunt resistance. Bus voltage times
# current is the same quantity derived from the raw registers: a mismatch means
# the calibration is not what the driver thinks. Agreement does NOT prove the
# shunt value matches the board -- a wrong shunt scales current and power alike.
# ---------------------------------------------------------------------------
say "== power1_input vs in2_input x curr1_input =="
say "(mV x mA = uW exactly, so the two columns are directly comparable)"
say ""
for h in /sys/class/hwmon/hwmon*; do
    [ -r "$h/power1_input" ] || continue
    [ -r "$h/in2_input" ] || continue
    [ -r "$h/curr1_input" ] || continue
    p=$(cat "$h/power1_input"); v=$(cat "$h/in2_input"); i=$(cat "$h/curr1_input")
    dev=$(readlink -f "$h/device" 2>/dev/null)
    echo "$p $v $i $(basename "$h") $(basename "${dev:-?}")" | awk '{
        expect = $2 * $3
        if (expect > 0) { d = 100.0 * ($1 - expect) / expect } else { d = 0 }
        printf "  %-8s %-10s reg=%9d uW   V*I=%9d uW   diff=%+7.2f %%\n",
               $4, $5, $1, expect, d
    }'
done
rule

# ---------------------------------------------------------------------------
# 4. Achievable sampling rate
#
# Each read is an I2C transaction on a shared bus, so the cost of a full sweep
# sets the floor on the sampling interval. This decides whether a single
# inference can be resolved or whether energy must be integrated over a run.
# ---------------------------------------------------------------------------
say "== sweep cost =="
rails=""
for h in /sys/class/hwmon/hwmon*; do
    [ -r "$h/power1_input" ] && rails="$rails $h/power1_input"
done
nrails=$(echo $rails | wc -w)
say "  rails with power1_input: $nrails"
if [ "$nrails" -gt 0 ]; then
    n=100
    t0=$(date +%s%N 2>/dev/null)
    k=0
    while [ "$k" -lt "$n" ]; do
        for f in $rails; do cat "$f" > /dev/null 2>&1; done
        k=$((k + 1))
    done
    t1=$(date +%s%N 2>/dev/null)
    if [ -n "$t0" ] && [ -n "$t1" ] && [ "$t1" -gt "$t0" ]; then
        echo "$t0 $t1 $n $nrails" | awk '{
            us_total = ($2 - $1) / 1000.0
            printf "  %d sweeps of %d rails in %.1f ms\n", $3, $4, us_total/1000.0
            printf "  per sweep : %.3f ms\n", us_total/1000.0/$3
            printf "  per rail  : %.3f ms\n", us_total/1000.0/$3/$4
            printf "  max sweep rate: %.1f Hz\n", 1000.0*$3/(us_total/1000.0)
        }'
    else
        say "  timing unavailable (date +%s%N unsupported)"
    fi
fi
rule

# ---------------------------------------------------------------------------
# 5. Idle drift
#
# A short quiet trace. Its spread is the noise floor any inference-attributable
# delta must clear to be meaningful.
# ---------------------------------------------------------------------------
say "== idle trace (20 samples, ~100 ms apart, total uW over all rails) =="
if [ "$nrails" -gt 0 ]; then
    k=0
    while [ "$k" -lt 20 ]; do
        tot=0
        for f in $rails; do
            v=$(cat "$f" 2>/dev/null || echo 0)
            tot=$((tot + v))
        done
        echo "$tot" | awk '{ printf "  %8.3f W\n", $1/1000000.0 }'
        k=$((k + 1))
        # usleep is absent on some images; sleep with a fraction is widely OK.
        sleep 0.1 2>/dev/null || sleep 1
    done
fi
rule
say "probe complete"
