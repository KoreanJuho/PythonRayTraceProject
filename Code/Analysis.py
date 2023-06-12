import RayTracing as rt
from math import ceil
import matplotlib.pyplot as plt
from numpy import sqrt, sin, cos, linspace
from math import pi

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
            color_list = ['red', 'gold', 'green', 'blue', 'mediumvioletred']
            for i, wavelength in enumerate(self.wavelengths):
                plt.plot(v_lsa[wavelength], v_lsa["Relative_Pupil_Height"], label=wavelength, color = color_list[i])
            plt.xlabel('FOCUS (MILLIMETERS)')
            plt.title('LONGITUDINAL\nSPHERICAL ABER.', pad=20, fontsize='large')
            vmax= round_up_to_nearest_05(max(max(abs(val) for val in v_lsa[key]) for key in self.wavelengths), 0.05)
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
            plt.plot(v_fie["Xfo"], v_fie["Image_Height"], label='Xfo', linestyle='-', color='green')
            plt.plot(v_fie["Yfo"], v_fie["Image_Height"], label='Yfo', linestyle='--', color='green')
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
            plt.plot(v_fie["Distortion"], v_fie["Image_Height"], color='green')
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
                        a_rsi = self.frt.speedrsi(0, current_pupil, 0, self.fields[current_field]["Y_height"], y0, wavelength)
                        rf1.append(a_rsi[self.lens_data["img"]]["Y"] - central_y)
                    trf[wavelength] = rf1
                    rf["tan"] = trf
                    rf2 = []
                    for current_pupil in xRelative_Pupil_Height:
                        a_rsi = self.frt.speedrsi(current_pupil, 0, 0, self.fields[current_field]["Y_height"], y0, wavelength)
                        rf2.append(a_rsi[self.lens_data["img"]]["X"] - central_x)
                    srf[wavelength] = rf2
                    rf["sagi"] = srf
                rayfan.append(rf)
            return rayfan, Relative_Pupil_Height, xRelative_Pupil_Height

        def plot(rayfan, Relative_Pupil_Height, xRelative_Pupil_Height):
            
            filen = len(self.fields)
            fig, axes = plt.subplots(nrows=filen, ncols=2, figsize=(6, 7), gridspec_kw={'width_ratios': [2, 1]})
            
            color_list = ['red', 'gold', 'green', 'blue', 'mediumvioletred']
            def plot_sub(ax, rh, st, fld):
                for i, wavelength in enumerate(self.wavelengths):
                    ax.plot(rh, rayfan[fld - 1][st][wavelength], label=wavelength, linewidth=1, color=color_list[i])
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
        
    def SpotDiagram(self, type = "ellipse", finish = None):
        def ellipse(current_field, finish):
            x = [0]
            vuy, vly = self.frt.Vignetting_factors[current_field]["vuy"], self.frt.Vignetting_factors[current_field]["vly"]
            vux = self.frt.Vignetting_factors[current_field]["vux"]
            y = [-1 * (vuy + vly) / 2]
            
            b = (2 - vuy - vly) / 2
            a = sqrt((1 - vux) ** 2 / (1 - (y[0] ** 2) / (b ** 2)))
            
            index = 1
            step = 30
            for radius in range(1, 11):
                if radius != 10:
                    step -= 3
                else:
                    step = 1
                for angle in range(0, 360, step):
                    x.append(a * radius / 10 * cos(angle * step * pi / 180))
                    y.append(b * radius / 10 * sin(angle * step * pi / 180))
                    index += 1
            xy = {"x": x, "y": y}
            
            wxy = {}
            for wavelength in self.wavelengths:
                v_rsi = self.frt.rsi(0, 0, 0, self.fields[current_field]["Y_height"], wavelength)
                y0 = v_rsi[0]["Y"]
                mxy = {}
                x, y = [], []
                for index, value in enumerate(xy["x"]):
                    v_rsi = self.frt.speedrsi(value, xy["y"][index], 0, self.fields[current_field]["Y_height"], y0, wavelength=wavelength, finish = finish, discard = True)
                    if v_rsi is not None:
                        x.append(v_rsi[finish]["X"])
                        y.append(v_rsi[finish]["Y"])
                mxy["x"] = x
                mxy["y"] = y
                wxy[wavelength] = mxy
            return wxy

        def square(current_field, finish):
            grid = 20
            vuy, vly = self.frt.Vignetting_factors[current_field]["vuy"], self.frt.Vignetting_factors[current_field]["vly"]
            vux = self.frt.Vignetting_factors[current_field]["vux"]
            x = linspace((-1+vux)*1.005, (1-vux)*1.005, grid)
            y = linspace((-1+vly)*1.005, (1-vuy)*1.005, grid)
            
            wxy = {}
            for wavelength in self.wavelengths:
                v_rsi = self.frt.rsi(0, 0, 0, self.fields[current_field]["Y_height"], wavelength)
                y0 = v_rsi[0]["Y"]
                mxy = {}
                spo_x, spo_y = [], []
                for i in range(len(x)):
                    for j in range(len(y)):
                        finish = self.frt.image_surface
                        v_rsi = self.frt.speedrsi(x[i], y[j], 0, self.fields[current_field]["Y_height"], y0, wavelength=wavelength, finish = finish, discard = True)
                        if v_rsi is not None:
                            spo_x.append(v_rsi[finish]["X"])
                            spo_y.append(v_rsi[finish]["Y"])
                mxy["x"] = spo_x
                mxy["y"] = spo_y
                wxy[wavelength] = mxy
            return wxy
        
        def spot_size(spo, finish):
            spotsize = []
            for current_field in range(len(self.fields)):
                if current_field == 0:
                    x0 = 0
                    y0 = 0
                else:
                    c_rsi = self.frt.rsi(0,0,0,self.fields[current_field]["Y_height"])
                    x0 = c_rsi[finish]["X"]
                    y0 = c_rsi[finish]["Y"]
                ss = []
                for wavelength in self.frt.wavelength:
                    for index in range(len(spo[current_field][wavelength]["x"])):
                        x1 = spo[current_field][wavelength]["x"][index]
                        y1 = spo[current_field][wavelength]["y"][index]
                        ss.append(sqrt((x1-x0)**2+(y1-y0)**2))
                avg = sum(ss)/len(ss)
                spotsize.append(avg)
            return spotsize
   
        def plot(spo,spotsize):
            def plot_sub(ax, spo, current_field, spotsize):
                for i, wavelength in enumerate(self.wavelengths):
                    ax.scatter(spo[current_field][wavelength]["x"], spo[current_field][wavelength]["y"], label=wavelength, color=color_list[i], s=1)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim([-0.08, 0.08])
                ax.set_aspect('equal')
                formatted_spotsize = format(spotsize[current_field], '.6f')
                text = f"Field: {current_field}\nSpot size: {formatted_spotsize}"
                ax.annotate(text, xy=(1, 0.5), xycoords='axes fraction', xytext=(5, 0), textcoords='offset points', ha='left', va='center')
        
            fig, axes = plt.subplots(nrows=len(self.fields), ncols=1, figsize=(6, 7))
            color_list = ['red', 'gold', 'green', 'blue', 'mediumvioletred']
            fldlen = len(self.fields)
            for curfld in range(fldlen):
                plot_sub(axes[curfld], spo, fldlen - curfld - 1, spotsize)
            
            plt.tight_layout()
            plt.show()
        
        spo = []
        if finish is None:
            finish = self.frt.image_surface
        
        for current_field in range(len(self.fields)):
            if type == "ellipse":
                spo.append(ellipse(current_field, finish))
            else:
                spo.append(square(current_field, finish))
        spotsize = spot_size(spo,finish)
        plot(spo,spotsize)
        
    def Viego(self):
        def Calc_Sag_Lens(current_surface, current_oal):
            
            def Calc_z(lens_height, current_surface):
                r_square = lens_height ** 2
                curvature = 1 / self.frt.lens_data[current_surface]["Y_radius"]
                if self.frt.lens_data[current_surface]['Surface_type'] == "Sphere":
                    k = 0
                else:
                    k = self.frt.lens_data[current_surface]["K"]
                z_prime = curvature * r_square / (1 + sqrt(1 - (1 + k) * curvature ** 2 * r_square))
                if self.frt.lens_data[current_surface]['Surface_type'] == "Asphere":
                    time = 2
                    for asp in self.frt.lens_data[current_surface]['Aspheric_coefficients'].values():
                        z_prime += asp * r_square ** time
                        time += 1
                return z_prime
            
            lens_height = max(self.frt.map[current_surface], self.frt.map[current_surface+1])
            step_size = 52
            if self.frt.lens_data[current_surface+1]['RefractiveIndex'][self.central_wavelength] != 1:
                lens_height = max(lens_height, self.frt.map[current_surface+1])
            
            if self.frt.lens_data[current_surface-1]['RefractiveIndex'][self.central_wavelength] != 1:
                lens_height = max(lens_height, self.frt.map[current_surface-1])
            
            interval = 2 * lens_height / (step_size // 2 - 1)
            
            z = []
            r = []
            
            for time in range(step_size // 2):
                current_height = -lens_height + interval * time
                if abs(current_height) <= self.frt.map[current_surface]:
                    z_prime = Calc_z(current_height, current_surface)
                elif time <= 7:
                    z_prime = Calc_z(-self.frt.map[current_surface], current_surface)
                else:
                    z_prime = Calc_z(self.frt.map[current_surface], current_surface)
                
                z.append(current_oal + z_prime)
                r.append(current_height)
            
            current_oal += self.frt.lens_data[current_surface]["Thickness"]
            
            for time in range(step_size // 2):
                current_height = lens_height - interval * time
                if abs(current_height) <= self.frt.map[current_surface + 1]:
                    z_prime = Calc_z(current_height, current_surface + 1)
                elif time <= 7:
                    z_prime = Calc_z(self.frt.map[current_surface + 1], current_surface + 1)
                else:
                    z_prime = Calc_z(-self.frt.map[current_surface + 1],current_surface + 1)
                
                z.append(current_oal + z_prime)
                r.append(current_height)
            
            z.append(z[0])
            r.append(r[0])
            
            zr = {"z":z, "r":r}
            
            return zr
        
        def stp(current_surface, current_oal):
            z = [current_oal, current_oal]
            lr = [-self.frt.map[current_surface], -self.frt.map[current_surface]-1]
            ur = [self.frt.map[current_surface], self.frt.map[current_surface]+1]
            stp = {"z":z , "lr":lr, "ur": ur}
            return stp

        def img(current_oal):
            z = [current_oal, current_oal]
            r = [-self.frt.map[current_surface]-0.1,self.frt.map[current_surface]+0.1]
            img = {"z":z, "r":r}
            return img
        
        def rays(Full_oal):
            rays = []
            for current_field in range(1, len(self.frt.fields)+1):
                for current_ep in range(1, 4):
                    oal = 0
                    z = []
                    r = []
                    if current_field != 1 or current_ep != 1:
                        v_rsi = self.frt.rsi("f", current_field,"r", current_ep)
                        ho = Full_oal / 10
                        z = [-ho, v_rsi[1]["Z"]]
                        r = [v_rsi[0]["Y"] - v_rsi[0]["M"] / v_rsi[0]["N"] * ho, v_rsi[1]["Y"]]
                        for current_surface in range(2, self.frt.image_surface+1):
                            z.append(oal + self.frt.lens_data[current_surface-1]["Thickness"] + v_rsi[current_surface]["Z"])
                            r.append(v_rsi[current_surface]["Y"])
                            oal = oal + self.frt.lens_data[current_surface-1]["Thickness"]
                        ray = {"z": z, "r": r}
                        rays.append(ray)
            return rays
        
        zr = []
        current_oal = 0
        for current_surface in range(1, self.frt.image_surface):
            if self.frt.lens_data[current_surface]['RefractiveIndex'][self.central_wavelength] != 1:
                zr.append(Calc_Sag_Lens(current_surface, current_oal))
            elif current_surface == self.frt.stop_surface:
                stp = stp(current_surface, current_oal)
            current_oal += self.frt.lens_data[current_surface]["Thickness"]
        img = img(current_oal)
        rays = rays(current_oal)
                
        for a in range(len(zr)):
            plt.plot(zr[a]["z"], zr[a]["r"], color = "black", linewidth = 1)
        plt.plot(stp["z"],stp["lr"], color = "black", linewidth = 1)
        plt.plot(stp["z"],stp["ur"], color = "black", linewidth = 1)
        plt.plot(img["z"],img["r"], color = "black", linewidth = 1)
        colorlst = ["red", "green", "blue", "brown", "magenta"]
        for i, ray in enumerate(rays):
            plt.plot(ray["z"], ray["r"], color = colorlst[(i+1)//3], linewidth = 1)
        
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()