import RayTracing as rt
from math import ceil
import matplotlib.pyplot as plt

class Diagnostics:
    def __init__(self, central_wavelength):
        ldl = rt.LensDataLoader(r'./LensDataCenter.xlsx')
        self.lens_data= ldl.load_rdn()
        self.fields = ldl.load_fields()
        self.prt = rt.ParaxialRayTracing(self.lens_data)
        self.value_fio = self.prt.fio(central_wavelength)
        self.value_fir = self.prt.fir(self.value_fio)
        self.frt = rt.FiniteRayTracing(self.lens_data, self.value_fio, central_wavelength, self.fields)
        self.frt.setvig()
        self.central_wavelength = central_wavelength
        self.wavelength = list(self.lens_data["rdn"][1]["RefractiveIndex"].keys())
        
    def fielsa(self):
        def lsa():
            v_lsa = {
                "C": [0]*51,
                "d": [0]*51,
                "e": [0]*51,
                "F": [0]*51,
                "g": [0]*51,
                "Relative_Pupil_Height": [0]*51 }

            for wavelength in self.wavelength:
                if wavelength != "e":
                    system_matrix = self.prt.calculate_system_matrix(1, self.lens_data["img"]-1, wavelength)
                    v_lsa[wavelength][0] = system_matrix["A"] / system_matrix["C"] - self.value_fir["BFL"]

                for time in range(1, 51):
                    v_lsa["Relative_Pupil_Height"][time] = v_lsa["Relative_Pupil_Height"][time - 1] + 0.02
                    v_rsi = self.frt.rsi(0, v_lsa["Relative_Pupil_Height"][time], 0, 0, wavelength = wavelength)
                    v_lsa[wavelength][time] = -1 * v_rsi[self.lens_data["img"]]["Y"] * v_rsi[self.lens_data["img"]]["N"] / v_rsi[self.lens_data["img"]]["M"]
            return v_lsa
        
        def fie(wavelength):
            
            v_fie = {
                "Image_Height": [0]*51,
                "Xfo": [0]*51,
                "Yfo": [0]*51,
                "Distortion": [0]*51 }

            for time in range(1, 51):
                v_fie["Image_Height"][time] = v_fie["Image_Height"][time - 1] + 0.02

                s_rsi = self.frt.rsi(0.0005, 0, 0, v_fie["Image_Height"][time], wavelength = wavelength)
                t_rsi = self.frt.rsi(0, 0.0005, 0, v_fie["Image_Height"][time], wavelength = wavelength)
                c_rsi = self.frt.rsi(0, 0, 0, v_fie["Image_Height"][time], wavelength = wavelength)
                
                v_fie["Xfo"][time] = -1 * s_rsi[self.lens_data["img"]]["X"] * c_rsi[self.lens_data["img"]]["N"] * (1 - s_rsi[self.lens_data["img"]]["L"] ** 2) ** 0.5 / s_rsi[self.lens_data["img"]]["L"]
                v_fie["Yfo"][time] = -1 * (t_rsi[self.lens_data["img"]]["Y"] - c_rsi[self.lens_data["img"]]["Y"]) / (t_rsi[self.lens_data["img"]]["M"] / t_rsi[self.lens_data["img"]]["N"] - c_rsi[self.lens_data["img"]]["M"] / c_rsi[self.lens_data["img"]]["N"])
                v_fie["Distortion"][time] = (c_rsi[self.lens_data["img"]]["Y"] - self.value_fio[self.lens_data["img"]]["hcy"] * v_fie["Image_Height"][time]) / (self.value_fio[self.lens_data["img"]]["hcy"] * v_fie["Image_Height"][time]) * 100

            for time in range(51):
                v_fie["Image_Height"][time] *= self.value_fio[self.lens_data["img"]]["hcy"]
            
            return v_fie
        
        def plot_all(v_lsa, v_fie):
            plt.figure(figsize=(9, 6))

            def round_up_to_nearest_05(number, up):
                return ceil(number / up) * up

            # LSA Graph
            plt.subplot(1, 3, 1)  # First subplot in a grid of 1x3
            for wavelength in ["C", "d", "e", "F", "g"]:
                plt.plot(v_lsa[wavelength], v_lsa["Relative_Pupil_Height"], label=wavelength)
            plt.xlabel('FOCUS (MILLIMETERS)')
            plt.title('LONGITUDINAL\nSPHERICAL ABER.', pad=20, fontsize='large')
            vmax= round_up_to_nearest_05(max(max(abs(val) for val in v_lsa[key]) for key in ["C", "d", "e", "F", "g"]), 0.05)
            plt.xlim([-vmax, vmax])
            plt.ylim([0, 1])
            plt.xticks([-vmax, -vmax/2, 0, vmax/2, vmax])
            plt.yticks([0.25, 0.5, 0.75, 1])
            ax = plt.gca()  # get current axis
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_visible(False) # Hide the right and top spines
            ax.spines['top'].set_visible(False)
            ax.yaxis.tick_left() # Only show ticks on the left and bottom spines
            ax.xaxis.tick_bottom()
            plt.tick_params(axis = 'both', labelsize = 'x-small')
            plt.legend(fontsize='x-small')

            # Fie Graph
            plt.subplot(1, 3, 2)  # Second subplot in a grid of 1x3
            plt.plot(v_fie["Xfo"], v_fie["Image_Height"], label='Xfo', linestyle='-', color='teal')
            plt.plot(v_fie["Yfo"], v_fie["Image_Height"], label='Yfo', linestyle='--', color='teal')
            plt.xlabel('FOCUS (MILLIMETERS)')
            plt.title('ASTIGMATIC\nFIELD CURVES', pad=20, fontsize='large')
            vmax= round_up_to_nearest_05(max(max(abs(val) for val in v_fie[key]) for key in ["Xfo", "Yfo"]),0.1)
            plt.xlim([-vmax, vmax])
            y = v_fie["Image_Height"][50]
            plt.ylim([0, y])
            plt.xticks([-vmax, -vmax/2, 0, vmax/2, vmax])
            plt.yticks([y*0.25, y*0.5, y*0.75, y*1])
            ax = plt.gca()  # get current axis
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_visible(False) # Hide the right and top spines
            ax.spines['top'].set_visible(False)
            ax.yaxis.tick_left() # Only show ticks on the left and bottom spines
            ax.xaxis.tick_bottom()
            ax.set_ylabel('IMG HT', rotation=0, fontsize = 9)
            ax.yaxis.set_label_coords(0.6,1.002)
            plt.tick_params(axis = 'both', labelsize = 'x-small')
            plt.legend(fontsize='x-small')

            # Distortion Graph
            plt.subplot(1, 3, 3)  # Third subplot in a grid of 1x3
            plt.plot(v_fie["Distortion"], v_fie["Image_Height"], color='teal')
            plt.xlabel('% DISTORTION')
            plt.title('DISTORTION', pad=30, fontsize='large')
            vmax= round_up_to_nearest_05(max(abs(val) for val in v_fie["Distortion"]),1)
            plt.xlim([-vmax, vmax])
            y = v_fie["Image_Height"][50]
            plt.ylim([0, y])
            plt.xticks([-vmax, -vmax/2, 0, vmax/2, vmax])
            plt.yticks([y*0.25, y*0.5, y*0.75, y*1])
            ax = plt.gca()  # get current axis
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_visible(False) # Hide the right and top spines
            ax.spines['top'].set_visible(False)
            ax.yaxis.tick_left() # Only show ticks on the left and bottom spines
            ax.xaxis.tick_bottom()
            ax.set_ylabel('IMG HT', rotation=0, fontsize = 9)
            ax.yaxis.set_label_coords(0.6,1.002)
            plt.tick_params(axis = 'both', labelsize = 'x-small')

            # plt.tight_layout()  # Adjust spacing between subplots to minimize overlap
            plt.show()
        
        v_lsa = lsa()
        v_fie = fie(self.central_wavelength)
        plot_all(v_lsa, v_fie)
        
ad = Diagnostics("e")

ad.fielsa()