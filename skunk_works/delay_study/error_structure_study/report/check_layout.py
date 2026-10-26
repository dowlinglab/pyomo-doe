"""Verify tablet geometry; numerical/scientific report claims need separate audit."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import pdfplumber


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();rows=[]
    with pdfplumber.open(args.input) as pdf:
        for number,page in enumerate(pdf.pages,1):
            assert abs(page.width-612)<.01 and abs(page.height-792)<.01
            chars=[c for c in page.chars if c['text'].strip()]
            left=min(c['x0'] for c in chars);right=max(c['x1'] for c in chars)
            # Optical punctuation protrusion can extend a fraction of a point
            # beyond the geometric 18..378 pt body. It must not enter the rules.
            assert left>=17. and right<=379.,(number,left,right)
            rules=[l for l in page.lines if abs(l['x0']-388.8)<.1 and abs(l['x1']-594.)<.1 and abs(l['top']-l['bottom'])<.01]
            assert len(rules)==35,(number,len(rules))
            ys=sorted(l['top'] for l in rules);pitch_error=max(abs(b-a-20.25) for a,b in zip(ys[:-1],ys[1:]))
            assert pitch_error<.02
            vertical=[l for l in page.lines if abs(l['x0']-383.04)<.1 and abs(l['x1']-383.04)<.1]
            assert len(vertical)==1
            rows.append({'page':number,'text_left_pt':left,'text_right_pt':right,'ruled_lines':len(rules),'college_rule_pitch_error_pt':pitch_error})
    result={'status':'passed','pdf_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'checker_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256((args.input.parent/'main.tex').read_bytes()).hexdigest(),
        'python':platform.python_version(),'pdfplumber':pdfplumber.__version__,'per_page':rows,
        'requested_geometry':{'left_margin_in':.25,'right_margin_in':3.25,'body_width_in':5.,'rule_spacing_in':.28125},
        'scope':'Letter pages, body text extent, separate annotation column and 35 college-spaced rules on every page. Numerical claims, complete-goal status and visual typography require separate review.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
