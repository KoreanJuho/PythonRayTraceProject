import RayTracing as rt
import Analysis as an

ldl = rt.LensDataLoader(r'./LensDataCenter.xlsx')
lens_data= ldl.load_rdn()
fields = ldl.load_fields()

prt = rt.ParaxialRayTracing(lens_data)
value_fio = prt.fio("e")
value_fir = prt.fir(value_fio)

frt = rt.FiniteRayTracing(lens_data, value_fio, "e", fields)

# frt.map[10] = 5.5
# frt.setvig(Change = True)

# for a in frt.Vignetting_factors:
#     print(a)

# v_rsi = frt.rsi("r", 3, "f", 4)
# for a in v_rsi:
#     print(a)

# for a in frt.map:
#     print(a)

ad = an.Diagnostics("e")
ad.fielsago()
ad.rimgo()
ad.SpotDiagram()
ad.Viego()