# Run this script from the root directory of this project

mkdir ./_build/
mkdir ./_build/html/
mkdir ./_build/html/_images
python ./scripts/process_notebooks.py
jb build ../controls
open ./_build/html/index.html