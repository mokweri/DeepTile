# Measure power and energy

DeepTile measures energy per inference on the ZCU104 from the board's power
monitor, integrated over exactly the interval the host reports for throughput.
This guide explains what is measured and how to take a measurement you can
quote.

## What is measured

The primary instrument is the **INA226** monitor on the board's 12 V input. The
12 V input is the ZCU104's only external supply, so this reading covers the
whole board. Two IRPS5401 power controllers additionally report ten regulated
rails, including **VCCINT** (the FPGA fabric) and **VCC1V2** (the DDR4
interface). These are used only to attribute power to parts of the board, never
for the headline figure: they draw from the same 12 V input, so adding them to
it double-counts.

## Take a measurement

Measurement is a separate, privileged wrapper around an ordinary run, so the
host behaves identically whether or not it is measured.

1. **Check the sensors** after any BSP, kernel, or device-tree change. This is
   read-only:

    ```bash
    sudo scripts/hw/probe_power_sensors.sh
    ```

2. **Run the workload under the capture tool:**

    ```bash
    sudo scripts/hw/deeptile_power_capture.py --settle 5 --tail 10 \
        --run-record results/resnet50-power --out power.json -- \
        ./deeptile target.json models/resnet50/imagenet1k/int8-tqt/v1.0.0/graph.json \
            --dataset data/imagenet --output results/resnet50-power
    ```

The tool configures the sensor, captures an idle baseline before and after the
run, integrates power over the host's reported window, restores the sensor's
original settings, and writes a `deeptile.power.v1` record that cites the run
record by its SHA-256. By default the INA226 is sampled at 114 Hz with an
8.8 ms averaging window; `--rate fast` samples at about 1.5 kHz to resolve
structure within an inference. `--settle` and `--tail` set the idle baselines
in seconds; the reported evaluation uses 5 and 10.

!!! warning "Always pass `--run-record`"
    Without it, the whole command is integrated, including process start-up,
    XRT initialization, and weight upload. That overstates joules per image, and
    the record marks its window as `command` so readers can tell.

## What you get

For the host's window $[t_0, t_1]$ with $N$ images completed:

| Quantity | Definition |
|---|---|
| Total board energy | Integral of power over $[t_0, t_1]$ |
| Energy per image | Total board energy divided by $N$ |
| Dynamic power | Mean power during the run minus the idle baseline |
| Dynamic energy per image | Dynamic power × $(t_1 - t_0)$ ÷ $N$ |

Report both tiers together. Total board power is hard to game but includes fans,
network, and other infrastructure no accelerator design can influence. Dynamic
power isolates the workload but excludes that platform cost. Any
operations-per-watt figure must say which tier it uses.

## Good practice

- **Measure in a separate run.** Keep measured runs apart from the runs you
  report accuracy and throughput from.
- **Leave rail polling off** (the default) for any run whose throughput you
  report; polling the power controllers costs measurable CPU time. Enable it
  with `--rail-interval` only for characterization.
- **Watch the idle drift.** The board's fan speed follows die temperature and
  is powered from the measured rail, so the baseline shifts during a long run.
  A large difference between the idle readings before and after means the run
  was not thermally steady. Avoid measuring in the first minutes after power-on,
  when the fan runs at full speed.

## Limitations

- **The 12 V figure covers the whole board.** It cannot be broken down below
  the rails the power controllers expose.
- **Seven rails have no telemetry**, including the DDR4 termination supplies.
  The rail breakdown therefore understates memory power; attribute any
  shortfall to those rails and to regulator losses.
- **Energy is per inference, not per launch.** At the default rate one sample
  spans many short launches, so the capture integrates whole dataset passes;
  see [Execution and timing](../concepts/execution-and-timing.md) for how
  launches run.
