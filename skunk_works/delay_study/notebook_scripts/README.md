# Notebook migration

These 13 Python drivers preserve the ordered scientific code cells from the
complete workshop notebooks, including markdown as comments. Shell escapes in
Colab-only setup blocks become explicit `subprocess.run` calls. The source
notebooks remain the teaching artifacts; new delay-model research will live in
importable study modules rather than in either notebook cells or these drivers.

Run `python skunk_works/delay_study/export_notebooks.py` from the repository root
to regenerate them. The exporter checks equality of scientific abstract syntax
trees before writing a driver. `manifest.json` identifies every notebook and
script by SHA-256 and records each original code cell's hash.

Run drivers with the workshop environment, for example:

```
MPLBACKEND=Agg conda run -n pyomo-doe-maint python skunk_works/delay_study/notebook_scripts/pyomo_simulation.py
```

The standalone simulation driver was executed successfully during migration.
The remaining estimation/design drivers require the custom workshop Pyomo API
where used; exporting and parsing them does not establish runtime compatibility.
They intentionally execute top-level code and should not be imported as libraries.
They select the original notebook working directory for relative input paths.
Student exercise notebooks with missing answers are intentionally not exported.
