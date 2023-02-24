import numpy as np
import pandas as pd
if __name__ == "__main__":
    tipsy = open("/Users/yangqian/Documents/UZHCS/AdvancedHighPerformanceComputing/Week1/b0-final.std", 'rb')
    header_type = np.dtype(
        [('time', '>f8'), ('N', '>i4'), ('Dims', '>i4'), ('Ngas', '>i4'), ('Ndark', '>i4'), ('Nstar', '>i4'),
         ('pad', '>i4')])
    gas_type = np.dtype(
        [('mass', '>f4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4'), ('vx', '>f4'), ('vy', '>f4'), ('vz', '>f4'),
         ('rho', '>f4'), ('temp', '>f4'), ('hsmooth', '>f4'), ('metals', '>f4'), ('phi', '>f4')])
    dark_type = np.dtype(
        [('mass', '>f4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4'), ('vx', '>f4'), ('vy', '>f4'), ('vz', '>f4'),
         ('eps', '>f4'), ('phi', '>f4')])
    star_type = np.dtype(
        [('mass', '>f4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4'), ('vx', '>f4'), ('vy', '>f4'), ('vz', '>f4'),
         ('metals', '>f4'), ('tform', '>f4'), ('eps', '>f4'), ('phi', '>f4')])

    header = np.fromfile(tipsy, dtype=header_type, count=1)
    header = dict(zip(header_type.names, header[0]))
    gas = np.fromfile(tipsy, dtype=gas_type, count=header['Ngas'])
    gas = pd.DataFrame(gas, columns=gas.dtype.names)
    dark = np.fromfile(tipsy, dtype=dark_type, count=header['Ndark'])
    dark = pd.DataFrame(dark, columns=dark.dtype.names)
    star = np.fromfile(tipsy, dtype=star_type, count=header['Nstar'])
    star = pd.DataFrame(star, columns=star.dtype.names)
    tipsy.close()

    print(header)
    print(dark.iloc[99])
    print("xmin: ", min(dark['x']), "xmax: ", max(dark['x']))
    print("ymin: ", min(dark['y']), "ymax: ", max(dark['y']))
    print("zmin: ", min(dark['z']), "zmax: ", max(dark['z']))
    print("Mass sum:", dark['mass'].sum())
