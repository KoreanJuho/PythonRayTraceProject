import openpyxl
from openpyxl.utils import get_column_letter
from typing import List, Dict, Optional, Union
from openpyxl.worksheet.worksheet import Worksheet
from numpy import arcsin, tan

class LensDataLoader:
    def __init__(self, LDC_path):
        self.LDC_path = LDC_path 
        self.Index_path = r'./Tools/RefractiveIndexData.xlsx'
        self.Index_priority = ["CDGM", "HOYA", "SCHOTT", "HIKARI", "SUMITA", "OHARA"]
        
    def load_rdn(self):
        # Load the workbook
        wb = openpyxl.load_workbook(self.LDC_path)

        lens_data = []
        
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
                                  if lens_name is None else self.__load_refractive_index(lens_name)
            }
            
            # Check if this is the stop surface
            if isinstance(cell_value, str) and cell_value.lower() == 'stop':
                stop_surface = lens['Surface_number']

            # If surface type is "Asphere", retrieve aspheric coefficients from SurfaceProperty sheet
            if lens['Surface_type'] == 'Asphere':
                lens['Aspheric_coefficients'] = self.__load_aspheric_coefficients(SurfaceProperty_Sheet, lens['Surface_number'])

            lens_data.append(lens)
            row += 1

        # The image surface is the last one
        image_surface = lens_data[-1]['Surface_number']
        
        return lens_data, stop_surface, image_surface

    def __load_aspheric_coefficients(self, SurfaceProperty_Sheet: Worksheet, surface_number: int) -> Dict[str, float]:
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
        
    def __load_refractive_index(self, lens_name: str) -> Dict[str, float]:
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

class ParaxialRayTracing:
    def __init__(self, lens_data: List[Dict[str, Union[str, int, float, Dict[str, float]]]], stop_surface: int, image_surface: int):
        self.lens_data = lens_data
        self.stop_surface = stop_surface
        self.image_surface = image_surface

    def calculate_system_matrix(self, start_surface: int, finish_surface: int, wavelength: str) -> Dict[str, float]:
        size = finish_surface - start_surface + 1
        ary_len = 2 * size
        
        system_matrix = {
            'A': 0.0,
            'B': 0.0,
            'C': 0.0,
            'D': 0.0,
            'B_dummy': 0.0,
            'D_dummy': 0.0,
        }

        gaussian_array = [-1 * self.lens_data[0]['Thickness']]
        gaussian_array.extend([(self.lens_data[start_surface + i // 2]['RefractiveIndex'][wavelength] - self.lens_data[start_surface - 1 + i // 2]['RefractiveIndex'][wavelength]) / 
                            self.lens_data[start_surface + i // 2]['Y_radius'] if i % 2 != 0 else 
                            -1 * self.lens_data[start_surface + i // 2 - 1]['Thickness'] / self.lens_data[start_surface + i // 2 - 1]['RefractiveIndex'][wavelength]
                            for i in range(1, ary_len)])
        
        system_matrix['A'] = self.__calc_gaussian_bracket(gaussian_array, 1, ary_len - 1)
        system_matrix['B'] = self.__calc_gaussian_bracket(gaussian_array, 0, ary_len - 1)
        system_matrix['C'] = self.__calc_gaussian_bracket(gaussian_array, 1, ary_len)
        system_matrix['D'] = self.__calc_gaussian_bracket(gaussian_array, 0, ary_len)

        gaussian_array[0] = 0
        system_matrix['B_dummy'] = self.__calc_gaussian_bracket(gaussian_array, 0, ary_len - 1)
        system_matrix['D_dummy'] = self.__calc_gaussian_bracket(gaussian_array, 0, ary_len)
        
        return system_matrix
    
    @staticmethod
    def __calc_gaussian_bracket(gaussian_array: List[float], start: int, finish: int) -> float:
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
    
    def calc_fio(self, fno: float, yim: float, wavelength: str) -> List[Dict[str, float]]:
        system_matrix = self.calculate_system_matrix(1, self.image_surface - 1 , wavelength)
        stop_system_matrix = self.calculate_system_matrix(1, self.stop_surface, wavelength)
        
        fio = [{'hmy': 0.0, 'umy': 0.0, 'hcy': 0.0, 'ucy': 0.0} for _ in range(self.image_surface + 1)]
        
        if self.lens_data[0]['Thickness'] >= 10000000000:
            fio[0]['hmy'] = 0.0
            fio[0]['umy'] = 0.0
            fio[1]['hmy'] = 1 / system_matrix['C'] / 2 / fno
        else:
            na = 1 / 2 / fno
            nao = - na / system_matrix['D']
            fio[0]['hmy'] = 0.0
            fio[0]['umy'] = tan(arcsin(nao))
            fio[1]['hmy'] = self.lens_data[0]['Thickness'] * fio[0]['umy']

        fio[0]['hcy'] = yim * system_matrix['D']
        fio[0]['ucy'] = stop_system_matrix['A'] / stop_system_matrix['B'] * fio[0]['hcy']
        fio[1]['hcy'] = stop_system_matrix['B_dummy'] / stop_system_matrix['A'] * fio[0]['ucy']
        
        for ray_type in ['my', 'cy']:
            for current_surface in range(1, self.image_surface + 1):
                if current_surface != 1:
                    fio[current_surface]['h'+ray_type] = fio[current_surface - 1]['h'+ray_type] + self.lens_data[current_surface - 1]['Thickness'] * fio[current_surface - 1]['u'+ray_type]

                fio[current_surface]['u'+ray_type] = (self.lens_data[current_surface - 1]['RefractiveIndex'][wavelength] * fio[current_surface - 1]['u'+ray_type] - fio[current_surface]['h'+ray_type] * (self.lens_data[current_surface]['RefractiveIndex'][wavelength] - self.lens_data[current_surface - 1]['RefractiveIndex'][wavelength]) / self.lens_data[current_surface]['Y_radius']) / self.lens_data[current_surface]['RefractiveIndex'][wavelength]

                if current_surface == (self.image_surface - 1) and ray_type == "my":
                    self.lens_data[current_surface]['Thickness'] = -1 * fio[current_surface]['h'+ray_type] / fio[current_surface]['u'+ray_type]

        return fio
    
    def calc_fir(self, fio):
        fir = {}
        
        fir['EFL'] = fio[0]['hmy'] / fio[self.image_surface-1]['hcy']
        fir['BFL'] = fio[self.image_surface-1]['hmy'] / fio[self.image_surface-1]['hcy']
        
        fir['ENP'] = - fio[1]['hcy'] / fio[0]['ucy']
        fir['EPD'] = 2 * (fio[1]['hmy'] + fir['ENP'] * fio[0]['umy'])
        fir['EXP'] = - fio[self.image_surface-1]['hcy'] / fio[self.image_surface-1]['ucy']
        fir['EXD'] = 2 * (fio[self.image_surface-1]['hmy'] + fir['EXP'] * fio[self.image_surface-1]['umy'])
        
        return fir

ldl = LensDataLoader(r'./LensDataCenter.xlsx')
lens_data , stop_surface, image_surface = ldl.load_rdn()

for lens in lens_data:
    print(lens)

print(f"Stop surface: {stop_surface}")
print(f"Image surface: {image_surface}")

prt = ParaxialRayTracing(lens_data, stop_surface, image_surface)
fio = prt.calc_fio(2.4, 5.25, "e")
fir = prt.calc_fir(fio)

for s in fio:
    print(s)