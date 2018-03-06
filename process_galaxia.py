import ebf
import astropy.table as atpy
import numpy as np



F = ebf.read('stream_gap_sim.ebf')
dm = 5*np.log10(F['rad']*1e3)-5  
g,r,i = [F['sdss_%s'%_]+dm for _ in ['g','r','i']]

tab = atpy.Table()
for filt in ['g','r']:
    tab.add_column(atpy.Column(eval(filt),filt))
tab.write('stream_gap_mock.fits',overwrite=True)
