# Spin pumping & rectification effects
A bilayer structure consisting of a ferromagnetic metal (FM) on top of a non-magnetic metal (NM) with high spin-orgit coupling can be used
to observe spin pumping and spin-to-charge conversion. Upon ferromagnetic resonance (FMR) the precessing magnetization of the FM causes
spin angular momentum transfer into the adjacent NM in the form of a pure spin current - the so called spin pumping effect. The inverse
spin-Hall effect (ISHE) inside the NM converts the spin current into a charge current that can be detected with as a DC voltage.

However, due to induced microwave currents in the FM layer, other effects like anisotropic magnetoresistance (AMR) or anomalous Hall
effect (AHE) are rectified and contribute to the observed voltage signal. All those signals have the same lineshape as the ISHE effect and
special measurements have to be conducted in order to separate different effects. In-plane angular dependence measurement can differentiate between
different contributions and accurately identify the magnitude of the spin-pumping effect.

I wrote a FittingTool class that does all the fitting and plotting. An example of how to use it is demonstrated in the jupyter notebook.
The required file format is a csv-style file, and the column names can be controlled with a 'feature_names' instance attribute.
