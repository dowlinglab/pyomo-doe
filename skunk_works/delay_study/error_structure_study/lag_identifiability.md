# What unequal lags can identify

For the positive symmetric linear model, with known command transformation and
shared calibrated output scale, the plus and minus temperature transfers are

\[
G_+(s)=\frac{K}{(1+Ls)D_+(s)},\qquad
G_-(s)=\frac{K}{(1+Ls)D_-(s)},
\]

where

\[
D_+(s)=t_A r_Ss^2+[t_A+r_S(1+r)]s+1,
\]
\[
D_-(s)=t_A r_Ss^2+[t_A+r_S(1+r+2q)]s+1+2q.
\]

Here r=Us/Ua and q=Ux/Ua. The observed channels give G+=G1+G2 and G-=G1-G2
when only heater 1 is driven. This argument assumes the ideal transfer functions
are known at all relevant frequencies; it is not an assertion about the three
measured sine frequencies.

The two thermal polynomials differ by 2q(1+rS s). If q>0, any shared thermal
root would therefore have to be s=-1/rS. Substitution gives D+(-1/rS)=-r,
which is nonzero for r>0. Thus the thermal denominators have no common factor.
The common factor of the two full denominators identifies the input-lag factor
1+Ls. Coincident input and thermal poles do not invalidate the greatest-common-
factor argument: at least one thermal denominator lacks that pole.

After removing that factor, normalize the thermal denominators with D+(0)=1.
Their constant difference identifies 2q. The linear-coefficient difference
identifies 2q*rS, so rS follows; the common quadratic coefficient gives tA*rS,
and the plus linear coefficient gives r. The DC gain of G+ gives K. Positive r
and q therefore permit generic structural separation of input and sensor lags
under these restricted assumptions.

At r=0 both thermal denominators contain 1+rS s. Sensor and input lag factors
then commute exactly, and swapping their labels leaves the forced transfers
unchanged. At small positive r the ideal model may be structurally identifiable
while the data give little practical distinction. In the fitted model r reaches
its lower bound; lowering that bound is therefore a required sensitivity.
The completed synthetic exchange check and the full-data profiles determine
what can be estimated, rather than the structural argument alone.

This is not a proof for unequal channel calibrations, asymmetric networks,
unknown temperature-dependent power, radiation, or an unknown static command
map. Initial-state transients and ambient forcing can add information beyond
the zero-state forced transfers. A weak direction in the observed information
matrix is a finite-data result in specified coordinates, not evidence that the
device literally has a zero sensor heat capacity.
