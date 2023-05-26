import openpyxl as xl

# Load the workbook
wb = xl.load_workbook(r'./LensDataCenter.xlsx')

# Select sheets
RDN_Sheet = wb['RDN']
SurfaceProperty_Sheet = wb['SurfaceProperty']

# Initialize an empty list to store your lens data
lens_data = []

# The data starts from row 2 and continues until there's a row with no surface type
row = 2
while RDN_Sheet[f'A{row}'].value is not None:
    # Create a dictionary for each lens and append it to the list
    lens = {
        'Surface_number': row - 2,  # Surface numbering starts from 0
        'Surface_type': RDN_Sheet[f'C{row}'].value,
        'Y_radius': RDN_Sheet[f'D{row}'].value,
        'Thickness': RDN_Sheet[f'E{row}'].value,
    }
    
    # If surface type is "Asphere", retrieve aspheric coefficients from SurfaceProperty sheet
    if lens['surface_type'] == 'Asphere':
        lens['aspheric_coefficients'] = {}
        for coeff_row in range(2, 12):  # 2 to 11
            coeff_name = SurfaceProperty_Sheet[f'A{coeff_row}'].value
            coeff_value = SurfaceProperty_Sheet[f'B{row}'].value
            lens['aspheric_coefficients'][coeff_name] = coeff_value

    lens_data.append(lens)
    row += 1

# Now lens_data is a list of dictionaries, where each dictionary represents a lens
for lens in lens_data:
    print(lens)
