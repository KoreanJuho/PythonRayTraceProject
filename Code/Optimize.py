import Analysis as an

ad = an.Diagnostics("e")

spotsize = ad.SpotDiagram(type="square", graph=False)

for s in spotsize:
    print(s)