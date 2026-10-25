# Sensor calibration: source review and implications for ENV2

Alex requested this review of physical TCLab measurements. Calibration error is
a hypothesis; no independent reference temperature or calibration record for
Dan's device has been found. No thermal model was refitted, package installed,
firmware flashed or hardware accessed. The approximately 1 C quantity discussed
in the coupled-model study is pooled T1 RMSE, not sine-amplitude error.

## Findings supported by inspected sources

| Question | Evidence | Consequence for this study |
| --- | --- | --- |
| Sensor type and nominal conversion | The official [sensor page](https://apmonitor.com/pdc/index.php/Main/TCLabSensor) identifies TMP36GZ analog sensors with T(C)=0.1 Vout(mV)-50. These are semiconductor sensors, not thermistors. | The nominal slope is 10 mV/C and the voltage at 0 C is 500 mV. Offset and sensitivity errors are distinct possibilities. |
| Factory calibration and tolerances | Analog Devices [TMP35/36/37 Rev H](https://www.analog.com/media/en/technical-documentation/data-sheets/tmp35_36_37.pdf), pp. 1, 3: typical accuracy is ±1 C at 25 C and ±2 C over -40 to 125 C. The G-grade limits are ±3 C at 25 C and ±4 C over the rated range. Typical linearity is 0.5 C. | The often quoted ±1/±2 C values are typical specifications, not measured errors or guaranteed limits for this device. Device grade and complete measurement chain matter. |
| Python hardware path | In the inspected upstream `tclab.py`, `TCLab.T1` and `TCLab.T2` send the corresponding command and parse the returned string as float. `scan()` calls these properties sequentially. | This source applies no Python-side temperature slope or offset correction in these getters. Firmware and any acquisition/postprocessing code own the conversion. Sequential reads should not be mistaken for exactly simultaneous ADC samples. |
| Calibration workflow | Official [Measure Temperature worksheet](https://www.apmonitor.com/pdc/uploads/Main/TCLab1.pdf), p.3, asks users to investigate manufacturing differences in slope/intercept, compare channels/room temperature and consider calibration offsets and ADC resolution. | The author documentation explicitly recognizes calibration questions. It does not give a measured calibration of ENV2 or a turnkey per-channel calibration API. |
| ADC/firmware version dependence | Official firmware repository supports multiple sketches. An inspected [APMonitor firmware listing](https://github.com/APMonitor/arduino/blob/master/0_Test_Device/Python/tclab_v2/tclab_v2.ino) changelog states version 1.6.0 averages ten temperature measurements; version 2.0.0 changes the reference calculation and adds binary temperature commands. | Do not infer averaging, reference voltage or reporting precision from the current package version alone. The board's actual firmware remains unknown. |

The [benchmark paper](https://apm.byu.edu/prism/uploads/Members/2020_park_tclab.pdf)
describes Uno/Leonardo hardware with a ten-bit ADC and TMP36GZ sensors. Its
terminology calls the sensors thermistors; the manufacturer and current sensor
page identify semiconductor analog sensors. Use the manufacturer specification
for accuracy qualifications and operating range.

## What was inspected and what was not

The cached upstream Python file has SHA-256
`f3eb0ad5cc3fb7f572c7b8a558c691548e569881ad16643986ba4d82264e98ad`.
The local source is `../_private/upstream/tclab_source.py`; it is ignored by Git.
`inspect_source.py` parses its AST without importing or executing the library
and saves method locations and hashes. It checks both hardware getters and
the hardware constructor. No per-channel calibration parameter is exposed by
the inspected hardware constructor or getters. This is a statement about those
paths, not proof that every historical TCLab interface lacks calibration.

[PyPI](https://pypi.org/project/tclab/) identifies the current release as 1.0.0,
uploaded September 11, 2023. Its wheel SHA-256 is
`40fa767e1ebfe49d8124f83774d0af1b3928cd3c753d8d9ae04e19118e202b82`.
The wheel was **not downloaded or byte-verified**: a metadata download failed
under restricted DNS, and the escalation call was interrupted before any file
was saved. A release-page checksum is not proof of inspecting the released
wheel.

The parent subsequently found an existing **pip-installed TCLab 1.0.0** in
`/Users/adowling/opt/anaconda3/envs/controls3/lib/python3.10/site-packages`.
Its hardware source, Historian, version file, METADATA, INSTALLER, WHEEL and
RECORD were read without imports or modification. All six nonempty RECORD
hashes for those inspected files match their bytes. The installed `tclab.py`
is byte-identical to the cached upstream file. The installed Historian
`to_csv` (lines 220–227) passes log values through `csv.writer` with no explicit
decimal rounding. The [installed package receipt](installed_package_inspection.json)
records hashes and limits. This completes inspection of a real installed
pip package while leaving independent wheel-archive provenance unverified.
Dan's acquisition package version remains unrecorded in the CSVs; the local
installation does not establish his 2024 environment.

Direct retrieval of the current standard `TCLab-sketch.ino` did not return
source through the available web endpoint. The alternative APMonitor listing
is version-specific source evidence, not verification of the firmware flashed
on Dan's unit. The quantization worker owns the raw data grid and historical
conversion comparison; see its section in the living discussion note.

Two directly inspected official firmware listings give concrete alternatives:

| Listing | Conversion/reference/reporting found in the code | Limit |
| --- | --- | --- |
| [Historical WebUSB-old sketch](https://github.com/jckantor/TCLab-sketch/blob/master/TCLab-sketch-webusb-old/TCLab-sketch-webusb-old.ino), internal version 1.3.0 | A single analogRead multiplied by 0.3223, then minus 50; external analog reference; channel pins 0 and 2. | A historical variant, not the recovered firmware of Dan's board. It demonstrates an offset grid directly from physical hardware. |
| [APMonitor tclab_v2 sketch](https://github.com/APMonitor/arduino/blob/master/0_Test_Device/Python/tclab_v2/tclab_v2.ino), internal version 2.0.1 | Ten reads averaged, active conversion 0.322265625 per count for 3.3 V; alternative 1.75 V conversion commented out; external reference; text responses specify three decimal places. | The changelog mentions 1.75 V support, but the active conversion is 3.3 V. Read code rather than inferring active settings from the changelog. |

The upstream [Historian.to_csv implementation](https://github.com/jckantor/TCLab/blob/master/tclab/historian.py)
passes its log to Python's `csv.writer` without a decimal-format string or
temperature transformation. This supports checking firmware output precision
before attributing the data grid to CSV export. The installed 1.0.0 source
independently confirms that behavior; acquisition-version code remains unknown.

## How calibration can enter the model

Write a channel measurement as y_i=a_i*T_i+b_i+e_i before quantization.
Here a_i is static sensitivity error, b_i static offset, and e_i contains other
measurement error. This is a diagnostic representation, not a newly fitted law.

| Mechanism | What can be absorbed | What can still change |
| --- | --- | --- |
| Common static offset | For a translation-invariant linear thermal model, shifting both temperatures, initial states and ambient by the same constant changes no temperature differences. The absolute ambient/offset split is unidentifiable without an external reference. | An ambient bound in indicated C does not create an independent reference. Nonlinear absolute-temperature radiation breaks this exact invariance unless calibration is propagated into the physical temperatures. |
| Common static positive slope | In a freely fitted SISO model, scaling temperature rise scales K. With freely transformed ambient/initial states, constant scaling does not change poles or phase. | Absolute physical interpretation and uncertainty change. Constrained physical laws or independently known ambient/power can prevent full absorption. A constant slope alone cannot supply the missing fast phase. |
| Different slopes or offsets by channel | A SISO gain can hide its own slope. Separate unrestricted output gains/offsets could absorb channel calibration. | Our symmetric two-output model does not have unrestricted channel calibration. Relative slopes can distort T2/T1 response and inferred coupling; unequal offsets cannot generally be absorbed by one common ambient per run. Relative calibration and physical asymmetry can trade off. |
| ADC reference error | For an ideal nominal sensor and ADC, if software assumes Vnom but actual reference is Vref, indicated y=(Vnom/Vref)*(T+50)-50. A constant error therefore produces a linked slope and intercept. | A shared ADC reference creates a common component, while sensor tolerances remain channel-specific. A reference varying with command or time can produce a dynamic systematic discrepancy. No such variation has been measured here. |
| Temperature-dependent sensor or conversion error | A local slope/offset approximation may be absorbed around a single operating point. | Curvature can change apparent gain across temperature and mimic heat-loss or power nonlinearity. Temperature-only regression cannot uniquely separate these causes. |
| Quantization and averaging | A grid can be characterized without knowing absolute calibration. Averaging noisy ADC readings may produce finer reported increments. | Quantization affects likelihood and uncertainty. It neither measures calibration bias nor proves the simulator generated the data. Averaging/windowing can add dynamics and must be traced to firmware. |
| Sensor thermal lag | A static calibration cannot absorb a frequency-dependent response. | Package mounting and heat transfer can introduce dynamics. Datasheet electronic turn-on time is not the thermal time constant of a mounted sensor/heater assembly. |

For scale, a **hypothetical** 1% error in Vnom/Vref gives 0.75 C indicated error
at 25 C and 1.20 C at 70 C under the ideal formula. Those are algebraic examples,
not voltage-reference tolerances or estimates for Dan's board.

## Online calibration search and evidence limits

Searches included `TCLab sensor calibration`, `TCLab calibrate temperature`,
`TCLab voltage reference`, `TCLab calibration TMP36`, and targeted GitHub issue
and sketch searches. The official worksheet is a verified calibration-specific
source. The benchmark paper gives nominal sensor/ADC descriptions; it does
not report a multipoint reference-thermometer calibration of the ENV2 device.

No independently verified, instrument-referenced TCLab sensor calibration study
was found in this bounded search. This is not an assertion that none exists.
Several papers titled model calibration or identification concern fitting
thermal dynamics, not calibrating sensor voltage against known temperature.
The Almeria curricula article DOI 10.1016/j.ifacol.2022.09.306 is one such lead;
its abstract does not establish sensor metrology, so no sensor accuracy result
is inferred from it. Generic Arduino/TMP36 forum anecdotes concern different
circuits and were not used to assign an error to this device. A Testing and
Calibration Laboratory also abbreviated TCLab is unrelated and was excluded.

## Evidence needed before a calibration claim

1. Recover the recorded board model, flashed sketch/version, package version,
   reference-voltage configuration and acquisition/postprocessing code. CSV
   precision alone cannot recover them uniquely.
2. A known common-temperature equilibrium can constrain relative channel
   agreement. Two agreeing sensors may share bias; two unequal readings may
   also reflect gradients or incomplete equilibration. Existing first readings
   do not establish traceable absolute accuracy.
3. To identify static offset and slope separately, use at least two independently
   known, equilibrated temperatures spanning the range; multiple points can
   test curvature. Include uncertainty in the reference and temperature
   uniformity. This is a proposed measurement requirement, not an experiment
   performed or authorized in this review.
4. Electrical checks of ADC reference and sensor voltages would separate
   conversion error from sensor transfer error. A calibrated independent
   dynamic temperature measurement would help separate thermal lag from
   static calibration. No hardware action was taken.

The present temperature-only data may support relative calibration sensitivity
analyses, but a lower fit error after adding calibration parameters would not
establish calibration as the cause of the remaining discrepancy.
