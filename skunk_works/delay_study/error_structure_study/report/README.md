# Working report

This is an intermediate four-page LaTeX/PDF draft. It is not the completed
remaining-error study. The first page states that status; pending profiles,
refitted coverage, transfer, selected sensitivities and final figures remain.
The final report also needs its concise Markdown companion.

The document opens in the built-in LaTeX editor as this project. A local TeX
Live build produces main.pdf. The current source also passed the desktop
compiler. Use two pdflatex passes if references or pagination require it.

check_numbers.py runs with the pinned scientific Python and verifies displayed
rounding against terminal independently audited fits, raw data and direct RMSE
arithmetic. check_layout.py uses the bundled PDF runtime and checks letter
dimensions, text extent and every annotation rule. Their saved JSON binds the
source, checker and PDF. A separate visual_review.json records inspected pages.
After editing, compile, rerender, run both checks and inspect the latest pages.

Requested layout: 0.25-inch left margin, 3.25-inch right margin and 9/32-inch
college-rule spacing. The body is five inches wide; 35 lines occupy the separate
right annotation column on each page. The PDF-writing marker was recorded once
before authoring through the PDF skill. Scientific completion requires the
full goal audit, not just successful compilation and layout checks.
