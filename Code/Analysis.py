import RayTracing as rt
from math import ceil
import matplotlib.pyplot as plt
from numpy import sqrt

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
        self.wavelengths = list(self.lens_data["rdn"][1]["RefractiveIndex"].keys())
        
    def fielsago(self):
        def lsa():
            v_lsa = {
                "C": [0]*51,
                "d": [0]*51,
                "e": [0]*51,
                "F": [0]*51,
                "g": [0]*51,
                "Relative_Pupil_Height": [0]*51 }

            for wavelength in self.wavelengths:
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
        
    def rimgo(self):
        
        def speedrsi(stp_x, stp_y, ojt_x, ojt_y, y0, wavelength = None):
            v_rsi = []
            vtc = {}
            vtc['X'] = self.value_fio[1]['hmy'] * stp_x
            vtc['Y'] = y0 + self.value_fio[1]['hmy'] * stp_y
            vtc['Z'] = 0
            vtc['L'] = vtc['X'] / sqrt(vtc['X']**2 + (vtc['Y'] - self.value_fio[0]['hcy'] * ojt_y) ** 2 + self.lens_data["rdn"][0]['Thickness'])
            vtc['M'] = (vtc['Y'] - self.value_fio[0]['hcy'] * ojt_y) / sqrt(vtc['X'] ** 2 + (vtc['Y'] - self.value_fio[0]['hcy'] * ojt_y) ** 2 + self.lens_data["rdn"][0]['Thickness']**2)
            vtc['N'] = sqrt(1 - vtc['L']**2 - vtc['M']**2)
            v_rsi.append(vtc)
            v_rsi = self.frt.Raytracing(v_rsi, wavelength)
            return v_rsi
        
        def clacrf():
            rayfan = []
            for current_field in range(len(self.fields)):
                vuy, vly = self.frt.Vignetting_factors[current_field]["vuy"], self.frt.Vignetting_factors[current_field]["vly"]
                interval = (2 - vuy - vly) / 100
                Relative_Pupil_Height = [-1 + vly] + [-1 + vly + interval*i for i in range(1, 101)]
                vux = self.frt.Vignetting_factors[current_field]["vux"]
                interval = (1-vux)/50
                xRelative_Pupil_Height = [0] + [interval*i for i in range(1, 51)]
                
                c_rsi = self.frt.rsi(0, 0, 0, self.fields[current_field]["Y_height"], wavelength = self.central_wavelength)
                central_y = c_rsi[self.lens_data["img"]]["Y"]
                central_x = c_rsi[self.lens_data["img"]]["X"]
                
                rf = {}
                trf = {}
                srf = {}
                for wavelength in self.wavelengths:
                    v_rsi = self.frt.rsi(0, 0, 0, self.fields[current_field]["Y_height"], wavelength)
                    y0 = v_rsi[0]["Y"]
                    rf1 = []
                    for current_pupil in Relative_Pupil_Height:
                        a_rsi = speedrsi(0, current_pupil, 0, self.fields[current_field]["Y_height"], y0, wavelength)
                        rf1.append(a_rsi[self.lens_data["img"]]["Y"] - central_y)
                    trf[wavelength] = rf1
                    rf["tan"] = trf
                    rf2 = []
                    for current_pupil in xRelative_Pupil_Height:
                        a_rsi = speedrsi(current_pupil, 0, 0, self.fields[current_field]["Y_height"], y0, wavelength)
                        rf2.append(a_rsi[self.lens_data["img"]]["X"] - central_x)
                    srf[wavelength] = rf2
                    rf["sagi"] = srf
                rayfan.append(rf)
            return rayfan, Relative_Pupil_Height, xRelative_Pupil_Height

        def plot(rayfan, Relative_Pupil_Height, xRelative_Pupil_Height):
            
            filen = len(self.fields)
            fig, axes = plt.subplots(nrows=filen, ncols=2, figsize=(6, 7), gridspec_kw={'width_ratios': [2, 1]})
            
            def plot_sub(ax, rh, st, fld):
                for wavelength in self.wavelengths:
                    ax.plot(rh, rayfan[fld - 1][st][wavelength], label=wavelength, linewidth=1)
                ax.spines['left'].set_position('zero')
                ax.spines['bottom'].set_position('zero')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([-0.025, 0.025])
                ax.tick_params(axis='both', labelsize='x-small')
            
            for curfld in range(filen):
                plot_sub(axes[curfld, 0], Relative_Pupil_Height, "tan", filen - curfld)
                plot_sub(axes[curfld, 1], xRelative_Pupil_Height, "sagi", filen - curfld)
            
            plt.tight_layout()
            plt.show()
        
        rayfan, Relative_Pupil_Height, xRelative_Pupil_Height = clacrf()
        plot(rayfan, Relative_Pupil_Height, xRelative_Pupil_Height)
        
ad = Diagnostics("e")
# ad.fielsago()
ad.rimgo()