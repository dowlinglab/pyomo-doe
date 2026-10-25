"""Conditional gray-surface radiation scales and nonlinear remainder checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad
HERE=Path(__file__).resolve().parent
SIGMA=5.670374419e-8


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=json.loads((HERE/'config.json').read_text())
    area=config['radiation_scenario_area_m2'];surroundings=config['radiation_surroundings_C']+273.15
    anchor=50+273.15;rows=[];largest=0.;derivative_error=0.
    for epsilon in config['radiation_scenario_emissivities']:
        secant_anchor=epsilon*SIGMA*(anchor+surroundings)*(anchor*anchor+surroundings*surroundings)
        for tc in config['radiation_scenario_surface_C']:
            t=tc+273.15
            flux=epsilon*SIGMA*(t**4-surroundings**4)
            secant=epsilon*SIGMA*(t+surroundings)*(t*t+surroundings*surroundings)
            tangent=4*epsilon*SIGMA*t**3
            # Independent integral of the differential conductance checks
            # subtraction/factorization, including the zero-difference limit.
            reference=quad(lambda z:4*epsilon*SIGMA*z**3,surroundings,t,epsabs=1e-10)[0]
            largest=max(largest,abs(flux-reference))
            h=.001
            fd=epsilon*SIGMA*((t+h)**4-(t-h)**4)/(2*h)
            derivative_error=max(derivative_error,abs(fd-tangent))
            assert abs(flux-(t-surroundings)*secant)<1e-10
            rows.append({'surface_C':tc,'emissivity_scenario':epsilon,'area_m2':area,
                'surroundings_C':surroundings-273.15,'net_flux_W_per_m2':flux,
                'heat_loss_W':area*flux,'equivalent_h_W_per_m2_K':secant,
                'differential_h_W_per_m2_K':tangent,
                'nonlinear_remainder_vs_50C_secant_W':area*(flux-secant_anchor*(t-surroundings)),
                'radiation_fraction_if_convective_h_5':secant/(5+secant)})
    assert largest<1e-9 and derivative_error<1e-6,(largest,derivative_error)
    frame=pd.DataFrame(rows);frame.to_csv(args.output/'scenarios.csv',index=False)
    result={'status':'passed','quadrature_max_error_W_per_m2':largest,
        'differential_conductance_FD_max_error_W_per_m2_K':derivative_error,
        'assumptions':{'surface':'gray, isothermal; large surroundings enclosure at 23 C',
        'emissivities':'0.05, 0.2, 0.5, 0.9 sensitivity assumptions, not measured metal properties',
        'area':'published illustrative 0.0012 m2; heat rates scale linearly with effective area',
        'nonlinear_remainder':'difference from radiation approximated by a linear loss through ambient, matched at 50 C',
        'surface_temperature':'scenario, not directly measured heater temperature; hotter cases are sensitivities',
        'comparison':'hconv=5 scenario only; fitted Ua can already include linearized radiation'}}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(frame[frame.surface_C.isin([50,70.54,100])].to_string(index=False),flush=True)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
