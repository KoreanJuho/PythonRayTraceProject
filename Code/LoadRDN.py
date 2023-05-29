import openpyxl
from openpyxl.utils import get_column_letter

class LensData:
    def __init__(self, LDC_path):
        self.LDC_path = LDC_path
        self.Index_path = r'./Tools/RefractiveIndexData.xlsx'
        self.Index_priority = ["CDGM", "HOYA", "SCHOTT", "HIKARI", "SUMITA", "OHARA"]
        self.lens_data = []
    
    def load_rdn(self):
        # Load the workbook
        wb = openpyxl.load_workbook(self.LDC_path)

        # Select sheets
        RDN_Sheet = wb['RDN']
        SurfaceProperty_Sheet = wb['SurfaceProperties']

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
            
            if RDN_Sheet[f'F{row}'].value is None:
                lens['RefractiveIndex'] = {wavelength: 1 for wavelength in ["C", "d", "e", "F", "g"]}
            else:
                lens_name = RDN_Sheet[f'F{row}'].value
                lens['RefractiveIndex'] = self.load_refractive_index(lens_name)
                    
            # Check if this is the stop surface
            if isinstance(RDN_Sheet[f'A{row}'].value, str) and RDN_Sheet[f'A{row}'].value.lower() == 'stop':
                self.stop_surface = lens['Surface_number']
            
            # If surface type is "Asphere", retrieve aspheric coefficients from SurfaceProperty sheet
            if lens['Surface_type'] == 'Asphere':
                lens['Aspheric_coefficients'] = {}

                # Scan first row of SurfaceProperty sheet to find column with this surface number
                for col in range(2, SurfaceProperty_Sheet.max_column + 1):
                    if SurfaceProperty_Sheet[f'{get_column_letter(col)}1'].value == lens['Surface_number']:
                        # We found the column for this surface's coefficients
                        col_letter = get_column_letter(col)
                        break
                else:
                    print(f"No aspheric coefficients found for surface number {lens['Surface_number']}")
                    col_letter = None

                if col_letter is not None:
                    # Get aspheric coefficients from the found column
                    for coeff_row in range(2, 12):  # 2 to 11
                        coeff_name = SurfaceProperty_Sheet[f'A{coeff_row}'].value
                        coeff_value = SurfaceProperty_Sheet[f'{col_letter}{coeff_row}'].value
                        lens['Aspheric_coefficients'][coeff_name] = coeff_value

            self.lens_data.append(lens)
            row += 1

        # The image surface is the last one
        self.image_surface = self.lens_data[-1]['Surface_number']

        # Now lens_data is a list of dictionaries, where each dictionary represents a lens
        for lens in self.lens_data:
            print(lens)

        # Print the stop and image surfaces
        print(f"Stop surface: {self.stop_surface}")
        print(f"Image surface: {self.image_surface}")
        
    def load_refractive_index(self, lens_name):
        # Load refractive index data for a specific lens name from the appropriate excel sheet.
        """
        Args:
            lens_name: str, the name of the lens to find the refractive index for.
        """

        refractive_index_data = {}

        wb_index = openpyxl.load_workbook(self.Index_path)
        
        # convert lens_name to lower case for comparison
        lens_name_lower = lens_name.lower()
        
        # loop over the sheets in priority order
        for sheet_name in self.Index_priority:
            sheet = wb_index[sheet_name]
            
            # loop over the rows in the sheet, starting from row 3
            for row in range(3, sheet.max_row):
                lens_name_in_sheet = sheet[f'A{row}'].value
                if lens_name_in_sheet is not None:
                    lens_name_in_sheet_lower = lens_name_in_sheet.lower()
                else:
                    break
                
                # if the lens name in the current row matches the lens_name argument
                if lens_name_in_sheet_lower == lens_name_lower:
                        
                        # loop over the wavelength columns and add the data to refractive_index_data
                        for column, wavelength in zip('BCDEF', ["C", "d", "e", "F", "g"]):
                            refractive_index_data[wavelength] = sheet[f'{column}{row}'].value

                        # if we found the lens, no need to look in the other sheets
                        return refractive_index_data

        # if we get here, we didn't find the lens in any of the sheets
        raise ValueError(f"Could not find refractive index data for lens '{lens_name}'")

        
lens = LensData(r'./LensDataCenter.xlsx')
lens.load_rdn()