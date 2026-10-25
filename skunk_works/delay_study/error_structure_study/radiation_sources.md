# Sources for the radiation scale

Accessed October 25, 2026; primary institutional/manufacturer web pages inspected.
No source PDF bytes were downloaded or hashed. These sources establish plausible
surface sensitivity; they do not identify the material or finish of Dan's unit.

- [ASHRAE 2017 Fundamentals, chapter 4](https://handbook.ashrae.org/Handbooks/F17/IP/F17_Ch04/f17_ch04_ip.aspx),
  radiation and combined-convection sections, Table 5: total hemispherical
  emissivity is 0.03 for bright-dipped aluminum foil, 0.04 for alloy 6061, and
  0.87–0.92 for several black paints. The small-surface/large-enclosure gray
  approximation and view-factor qualifications bound the calculation here.
- [Gustavsen and Berdahl, LBNL institutional abstract, 2003](https://bies.lbl.gov/publications/spectral-emissivity-anodized-aluminum):
  measured normal total emissivity of specified anodized window-frame surfaces
  is 0.834–0.856, with much wider variation in internal cavities. The abstract
  was inspected; the paper's experimental methods were not. Normal total is
  not automatically hemispherical total, and window frames are not TCLab fins.
- [Fluke Process Instruments, Emissivity—Metals](https://www.flukeprocessinstruments.com/en-us/why-use-pyrometers%3F/emissivity-metals),
  aluminum rows: optical-band values depend on finish and wavelength. This
  pyrometer guide supports checking those distinctions; its spectral numbers
  are not inserted directly into the Stefan–Boltzmann calculation.

The calculation uses emissivities 0.05, 0.2, 0.5 and 0.9 as declared scenarios,
not a measured interval or fitted property. Area 0.0012 m2 and the example
convective coefficient 5 W/(m2 K) come from the previously inspected
[APMonitor radiative example](https://apmonitor.com/pdc/index.php/Main/TCLabRadiative).
The surface may be hotter than the recorded sensor. Effective area, view factors,
finish and surroundings remain unknown. Loss estimates scale with emissivity
and area under the stated gray-surface assumption.
