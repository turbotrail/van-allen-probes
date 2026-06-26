import cdflib

cdf_file = cdflib.CDF('uvw/2015/rbsp-a_magnetometer_uvw_emfisis-l2_20150101_v1.6.2.cdf')
print(cdf_file.cdf_info())
