import ebf
import astropy.table as atpy
import numpy as np
import string

for name in ['30_deg_mock.ebf','60_deg_mock.ebf','90_deg_mock.ebf']:
	namebase = string.split(name,'.')[0]
	F = ebf.read(name)
	dm = 5*np.log10(F['rad']*1e3)-5  
	u,g,r,i,z = [F['sdss_%s'%_]+dm for _ in ['u','g','r','i','z']]

	tab = atpy.Table()
	for filt in ['u','g','r','i','z']:
	    tab.add_column(atpy.Column(eval(filt),filt))
	tab.write(namebase+'.fits',overwrite=True)
