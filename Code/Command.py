import Analysis as an

ad = an.Diagnostics("e")

# ad.fielsago()
# ad.rimgo()
# ad.SpotDiagram(type="square")
# ad.Viego()

rsi_value = ad.frt.rsi("f",5,"r",1)
for val in rsi_value:
    print(val)

# import RayTracing as rt

# ldl = rt.LensDataLoader(r'./LensDataCenter.xlsx')
# lens_data = ldl.load_rdn()
# prt = rt.ParaxialRayTracing(lens_data)
# frt = rt.FiniteRayTracing(lens_data, v_fio, wavelength, fields)
