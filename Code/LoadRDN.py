import openpyxl
from openpyxl.utils import get_column_letter
from typing import List, Dict, Optional, Union
from openpyxl.worksheet.worksheet import Worksheet

class LensDataLoader:
    def __init__(self):
        self.LDC_path = r'./LensDataCenter.xlsx'
        self.Index_path = r'./Tools/RefractiveIndexData.xlsx'
        self.Index_priority = ["CDGM", "HOYA", "SCHOTT", "HIKARI", "SUMITA", "OHARA"]
        self.lens_data: List[Dict[str, Union[str, int, float, Dict[str, float]]]] = []
        self.stop_surface: Optional[int] = None
        self.image_surface: Optional[int] = None
        self.load_rdn()
    
    def load_rdn(self):
        # Load the workbook
        wb = openpyxl.load_workbook(self.LDC_path)

        # Select sheets
        RDN_Sheet: Worksheet = wb['RDN']
        SurfaceProperty_Sheet: Worksheet = wb['SurfaceProperties']

        # The data starts from row 2 and continues until there's a row with no surface type
        row = 2
        while (cell_value := RDN_Sheet[f'A{row}'].value) is not None:
            lens_name = RDN_Sheet[f'F{row}'].value

            # Create a dictionary for each lens and append it to the list
            lens = {
                'Surface_number': row - 2,  # Surface numbering starts from 0
                'Surface_type': RDN_Sheet[f'C{row}'].value,
                'Y_radius': RDN_Sheet[f'D{row}'].value,
                'Thickness': RDN_Sheet[f'E{row}'].value,
                'RefractiveIndex': {wavelength: 1 for wavelength in ["C", "d", "e", "F", "g"]} 
                                  if lens_name is None else self.load_refractive_index(lens_name)
            }
            
            # Check if this is the stop surface
            if isinstance(cell_value, str) and cell_value.lower() == 'stop':
                self.stop_surface = lens['Surface_number']

            # If surface type is "Asphere", retrieve aspheric coefficients from SurfaceProperty sheet
            if lens['Surface_type'] == 'Asphere':
                lens['Aspheric_coefficients'] = self.load_aspheric_coefficients(SurfaceProperty_Sheet, lens['Surface_number'])

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

    def load_aspheric_coefficients(self, SurfaceProperty_Sheet: Worksheet, surface_number: int) -> Dict[str, float]:
        # Scan first row of SurfaceProperty sheet to find column with this surface number
        for col in range(2, SurfaceProperty_Sheet.max_column + 1):
            if SurfaceProperty_Sheet[f'{get_column_letter(col)}1'].value == surface_number:
                # We found the column for this surface's coefficients
                col_letter = get_column_letter(col)
                break
        else:
            print(f"No aspheric coefficients found for surface number {surface_number}")
            return {}
        
        # Get aspheric coefficients from the found column
        return {
            SurfaceProperty_Sheet[f'A{coeff_row}'].value: SurfaceProperty_Sheet[f'{col_letter}{coeff_row}'].value
            for coeff_row in range(2, 12)  # 2 to 11
        }

    def load_refractive_index(self, lens_name: str) -> Dict[str, float]:
        # Load refractive index data for a specific lens name from the appropriate excel sheet.
        refractive_index_data: Dict[str, float] = {}

        wb_index = openpyxl.load_workbook(self.Index_path)
        
        # convert lens_name to lower case for comparison
        lens_name_lower = lens_name.lower()
        
        # loop over the sheets in priority order
        for sheet_name in self.Index_priority:
            sheet = wb_index[sheet_name]
            
            # loop over the rows in the sheet, starting from row 3
            for row in range(3, sheet.max_row):
                lens_name_in_sheet = sheet[f'A{row}'].value
                if lens_name_in_sheet is not None and lens_name_in_sheet.lower() == lens_name_lower:
                    refractive_index_data = {
                        wavelength: sheet[f'{column}{row}'].value
                        for column, wavelength in zip('BCDEF', ["C", "d", "e", "F", "g"])
                    }
                    return refractive_index_data

        # if we get here, we didn't find the lens in any of the sheets
        raise ValueError(f"Could not find refractive index data for lens '{lens_name}'")

class ABCD:
    def __init__(self):
        self.A = 0.0
        self.B = 0.0
        self.C = 0.0
        self.D = 0.0
        self.B_Dummy = 0.0
        self.D_Dummy = 0.0

class ParaxialRayTracing:
    def __init__(self, lens_data: List[Dict[str, Union[str, int, float, Dict[str, float]]]]):
        self.lens_data = lens_data

    def calculate_group_system_matrix(self, start_surface: int, finish_surface: int, wavelength: str) -> ABCD:
        size = finish_surface - start_surface + 1
        ary_len = 2 * size

        group_gaussian_array = self.make_gaussian_array(ary_len, start_surface - 1, wavelength)
        group_system_matrix = self.calculate_abcd(group_gaussian_array, ary_len)
        return group_system_matrix

    def make_gaussian_array(self, ary_len: int, start: int, wavelength: str) -> List[float]:
        gaussian_array = [-1 * self.lens_data[0]['Thickness']]
        gaussian_array.extend([(self.lens_data[start + i // 2 + 1]['RefractiveIndex'][wavelength] - self.lens_data[start + i // 2]['RefractiveIndex'][wavelength]) / 
                            self.lens_data[start + i // 2 + 1]['Y_radius'] if i % 2 != 0 else 
                            -1 * self.lens_data[start + i // 2]['Thickness'] / self.lens_data[start + i // 2]['RefractiveIndex'][wavelength]
                            for i in range(1, ary_len)])
        return gaussian_array

    def calculate_abcd(self, gaussian_array: List[float], ary_len: int) -> ABCD:
        system_matrix = ABCD()
        system_matrix.A = self.calc_gaussian_bracket(gaussian_array, 1, ary_len - 1)
        system_matrix.B = self.calc_gaussian_bracket(gaussian_array, 0, ary_len - 1)
        system_matrix.C = self.calc_gaussian_bracket(gaussian_array, 1, ary_len)
        system_matrix.D = self.calc_gaussian_bracket(gaussian_array, 0, ary_len)

        gaussian_array[0] = 0
        system_matrix.B_Dummy = self.calc_gaussian_bracket(gaussian_array, 0, ary_len - 1)
        system_matrix.D_Dummy = self.calc_gaussian_bracket(gaussian_array, 0, ary_len)
        return system_matrix
    
    @staticmethod
    def calc_gaussian_bracket(gaussian_array: List[float], start: int, finish: int) -> float:
        gaussian_bracket = [0] * (finish + 1)
        for current_point in range(start, finish):
            if current_point == start:
                gaussian_bracket[current_point] = gaussian_array[start]
            elif current_point == start + 1:
                gaussian_bracket[current_point] = gaussian_array[start] * gaussian_array[start + 1] + 1
            else:
                gaussian_bracket[current_point] = (gaussian_bracket[current_point - 1] * 
                                                gaussian_array[current_point] + 
                                                gaussian_bracket[current_point - 2])
        return gaussian_bracket[finish - 1]


lens = LensDataLoader()
prt = ParaxialRayTracing(lens.lens_data)

# Specify the starting and ending surfaces and the wavelength for the tracing
start_surface = 3  # Replace with your starting surface
finish_surface = 4  # Replace with your ending surface
wavelength = "e"  # Replace with your desired wavelength

# Perform the tracing
abcd = prt.calculate_group_system_matrix(start_surface, finish_surface, wavelength)

# You can now access the results in the `abcd` object
print(abcd.A, abcd.B, abcd.C, abcd.D, abcd.B_Dummy, abcd.D_Dummy)