# Temperature quantization in Dan's TC-Lab records

All 21,614 recorded T1/T2 values in the twelve distinct ENV2 experiments lie
exactly on the two-decimal reporting grid

\[
T_{reported}=\operatorname{round}(0.3223\,n-50,2),\qquad n\in\mathbb{Z}.
\]

There are 158 distinct levels across both channels, from 19.94 to 70.54 C.
The inferred code labels span 217–374 without a gap in the pooled level set.
Adjacent pooled levels differ by 0.32 or 0.33 C: 121 increments of 0.32 C and
36 of 0.33 C. This is a 0.3223 C measurement grid reported to 0.01 C precision.
Printing a smooth measurement to two decimals alone would not create this
coarser lattice. Code labels are inferred from the candidate conversion, not
independently logged ADC codes.

The same conversion explains every individual run and both channels. It also
explains all 27,198 values across the fifteen CSV paths currently in `data/`.
Two of those paths are byte-identical aliases for ENV2 records; the additional
distinct file is `validation_experiment_sin_5_50.csv`. We do not count aliases
as independent experiments. Per-file raw SHA256 values and Git history are in
[summary.json](results/summary.json).

## Per-run evidence

The repeated-value fraction counts consecutive samples with identical reported
temperature. Plateau lengths are counts of samples, not elapsed seconds.

| ENV2 protocol / run | Samples per channel | Unique T1 / T2 levels | Repeated T1 / T2, percent | Longest T1 / T2 plateau, samples |
| --- | ---: | ---: | ---: | ---: |
| 30 s sine / 1 | 901 | 106 / 30 | 76.2 / 89.0 | 18 / 134 |
| 30 s sine / 2 | 899 | 105 / 30 | 77.3 / 88.1 | 16 / 82 |
| 30 s sine / 3 | 899 | 105 / 29 | 76.8 / 86.9 | 16 / 137 |
| 120 s sine / 1 | 900 | 110 / 32 | 58.4 / 87.2 | 20 / 114 |
| 120 s sine / 2 | 901 | 112 / 29 | 58.3 / 89.7 | 16 / 81 |
| 120 s sine / 3 | 901 | 111 / 30 | 59.0 / 89.0 | 15 / 65 |
| 300 s sine / 1 | 901 | 146 / 42 | 51.6 / 85.4 | 23 / 57 |
| 300 s sine / 2 | 901 | 142 / 36 | 52.7 / 88.6 | 22 / 83 |
| 300 s sine / 3 | 901 | 141 / 37 | 52.1 / 86.7 | 26 / 56 |
| Step / 1 | 901 | 123 / 38 | 79.0 / 87.0 | 81 / 85 |
| Step / 2 | 901 | 120 / 37 | 81.1 / 88.0 | 123 / 109 |
| Step / 3 | 901 | 110 / 33 | 81.3 / 87.7 | 60 / 93 |

All CSV temperature tokens have at most two decimal places. Decimal trailing
zeros are sometimes omitted. The rounding discrepancy between an observed
value and its candidate unrounded grid point is at most 0.005 C. Exact NumPy
float64 rounding matches all values here; a separate ±0.005 C bin check with
1e-6 C arithmetic tolerance also passes every value. This keeps possible
firmware float32/half-bin differences separate from lattice compatibility.

## Comparison with candidate conversion laws

These counts use the same twelve ENV2 files and 21,614 observations.

| Candidate reported grid | Exact two-decimal matches | Maximum reported mismatch, C |
| --- | ---: | ---: |
| `round(n * 0.3223 - 50, 2)` | 21,614 | 0 |
| `round(n * 330/1024 - 50, 2)` | 2,331 | 0.02 |
| `round(n * 330/1023 - 50, 2)` | 0 | 0.11 |
| `round(n * 500/1024 - 50, 2)` | 301 | 0.24 |
| `round(n * 0.3223, 2)`; zero-offset emulator lattice | 0 | 0.05 |

The exact 3.3 V/1024 coefficient is 0.322265625 C/count for a 10 mV/C sensor.
The data favor the rounded firmware coefficient 0.3223, even though their
difference is only about 0.01 percent. That arithmetic difference is far too
small to explain a 1 C thermal-model RMSE.

Under the inferred integer labels and rounding-to-nearest assumption, the
feasible step interval is [0.3222997033, 0.3223000000] C if the offset is fixed
at −50 C. Allowing a free offset gives [0.3222988506, 0.3223008850] C. These are
rounding-bin feasibility bounds, not statistical confidence intervals and not
a sensor calibration. They condition on the inferred labels; finer submultiple
lattices cannot be excluded by level membership alone.

## Hardware path and provenance

The historical acquisition notebook at Git revision
`b114b13f1416c44037daf933e1495110a8168633`,
`notebooks/tclab_experiment_validation.ipynb`, uses `with TCLab() as lab`,
`Historian(lab.sources)` and `data.to_csv(...)`. This is a physical-hardware
workflow. The saved loop does not cover every ENV2 filename, and it contains
no retained text output identifying the Python package or firmware version.
It is not a per-file execution log. The existing
[provenance audit](../review/data_provenance.json) records the distinction.

In the inspected official Python source, `TCLab.T1` and `TCLab.T2` request a
temperature from the device and convert its serial response to a Python
float. The emulator's separate `TCLabModel.quantize` method uses a zero-offset
floor lattice. That lattice has the same nominal spacing but a different
offset from these data. The hardware and emulator methods must be kept
separate. The inspected Python source SHA256 is
`f3eb0ad5cc3fb7f572c7b8a558c691548e569881ad16643986ba4d82264e98ad`.

The accompanying [sensor-calibration review](../sensor_calibration_review/README.md)
directly web-inspected the old official WebUSB sketch, internally version
1.3.0. It shows `analogRead(pin) * 0.3223 - 50.0` and
`analogReference(EXTERNAL)`. That demonstrates a hardware conversion matching
these data. The same review inspected an official APMonitor version 2.0.1
variant with ten-reading averaging, coefficient 0.322265625 and three-decimal
reporting. This variation is why the numerical lattice does not uniquely
identify Dan's firmware. These are mutable web listings, not byte-verified
acquisition-version sketches. An official-source download in this worker was
interrupted at an approval request; no downloaded firmware bytes or version
claim depend on that unfinished call.

The calibration worker also inspected the installed pip package `tclab` 1.0.0
without importing hardware code. Its source receipts verify the hardware
getter path and the Historian CSV writer, which performs no explicit decimal
formatting or temperature calibration. That installed version is evidence of
the inspected implementation, not Dan's historical package version. See its
[installed-package receipt](../sensor_calibration_review/installed_package_inspection.json)
for file hashes and distribution metadata.

Sources inspected October 25, 2026:

- [Official TCLab Python source](https://github.com/jckantor/TCLab/blob/master/tclab/tclab.py).
- [Official TCLab-sketch repository](https://github.com/jckantor/TCLab-sketch).
- [Official old WebUSB sketch](https://github.com/jckantor/TCLab-sketch/blob/master/TCLab-sketch-webusb-old/TCLab-sketch-webusb-old.ino).
- [TCLab documentation](https://tclab.readthedocs.io/_/downloads/en/latest/pdf/),
  which distinguishes the hardware class from the emulator and places firmware
  conversion on the Arduino side.

## Consequences and open questions

Quantization can produce plateaus, pulses at threshold crossings and apparent
serial structure when a smooth temperature traverses a coarse measurement
grid. Long plateaus in T2 are consistent with its slow movement through that
grid. They do not estimate an analogue-noise variance or prove that every
correlated residual is quantization error. Actual thermal drift, electronic
noise, serial reading times and model discrepancy may all affect transitions.

The 0.3223 C spacing limits an individual reading's resolution; it does not
impose a 0.3223 C lower bound on a sinusoidal amplitude estimated from many
samples. Sub-grid amplitudes need a measurement model and enough threshold
crossings or noise to carry information. The earlier fast-T2 uncertainty
finding remains relevant and should not be replaced with a resolution-only
argument.

A future observation likelihood should use the hardware's shifted grid and
documented reporting bins. Do not copy the emulator's zero-offset floor rule,
0.043 C analogue-noise default, or saturation limits into the hardware
likelihood without evidence. The remaining questions are the firmware/board
version, sensor calibration and reference-voltage accuracy, analogue-noise
behavior, and whether quantization-aware fitting materially changes inference.
No thermal refit or new hardware experiment was run for this review.

## Reproduction

Source was committed as `016101b` before running the audit with pinned Python,
one BLAS thread and the original unchanged CSVs:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/adowling/opt/anaconda3/bin/python \
  skunk_works/delay_study/hardware_quantization_review/audit.py \
  --output skunk_works/delay_study/hardware_quantization_review/results
```

The script requires a new output directory, saves source/config/raw hashes,
checks that raw hashes remain unchanged, and writes the full
[summary](results/summary.json) and [158-level table](results/ENV2_levels.csv).
The audit exited successfully. This note uses existing measured records and
does not identify their acquisition firmware from numerical agreement alone.
