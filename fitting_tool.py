import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize
import os
import re


class FittingTool:
    """
    
    """
    
    def __init__(self, folder, file_list, feature_names=['B_current', 'B', 'Vabs', 'Vishe', 'Vlockin', 'VRlockin', 'VThetalockin']):
        """
            Loads the data in the form of a list of pandas dataframes
            
            Args:
                folder - str; path to a folder with files.
                file_list - list; list of names of files from inside folder.
        """
        self.folder = folder
        self.file_list = file_list
        self.feature_names = feature_names
        self.data = self.load_data(self.folder, self.file_list)
    
    
    def load_data(self, folder, file_list):
        load = lambda x: pd.read_csv(os.path.join(folder, x), header=1, sep="\t", names=self.feature_names)
        data = [load(i) for i in file_list]
        data = [df.dropna(0) for df in data]
        return data
    
    
    @classmethod
    def get_ang_dep(cls, folder, plot_data=False):
        """
            Factory method to fit angular dependence
            
            Args:
                folder - str; path to a folder with angular dependence measurement files.
                    Each file is assumed to contain data for one angular point; Its name
                    should contain info about the angle in a format: _{ang}deg, e.g. _110deg
        """
        file_list, angles = cls._get_ang_dep_files(folder, os.listdir(folder))
        fitter = cls(folder, file_list)
        fitter.angles = angles
        
        fitter.fit_data()
        if plot_data:
            fitter.plot_data()
        fitter.save_params()
            
        fitter.fit_ang_dep()
        fitter.plot_ang_dep()
        
        return fitter
    
    
    @staticmethod
    def _get_ang_dep_files(folder, file_list):
        new_file_list = np.array([file for file in file_list if re.search(r'deg', file) is not None])
        angles = np.array([int(re.search(r'_(\d{1,3})deg', file).group(1)) for file in new_file_list])
        sorting_keys = np.argsort(angles)
        return new_file_list[sorting_keys].tolist(), angles[sorting_keys].tolist()
    

    @staticmethod    
    def _fit_voltage(x, *args):
        (M1, M2, Bres, lwidth, off) = args
        y = (M1 * (0.5 * lwidth)**2 / ((x - Bres)**2 + (0.5 * lwidth)**2)
             - M2 * (x - Bres) * np.abs(lwidth) / ((x - Bres)**2 + (0.5 * lwidth)**2) + off)
        return y
    
    
    @staticmethod
    def _fit_abs(x, *args):
        (M1, M2, Bres, lwidth, off) = args
        y = (-2 * M1 * (x - Bres) * (0.5 * lwidth)**2 / ((x - Bres)**2 + (0.5 * lwidth)**2)**2
             -2 * M2 * (0.5 * lwidth)**2 * ((x - Bres)**2 - (0.5 * lwidth)**2) / (((x - Bres)**2 + (0.5 * lwidth)**2)**2) + off)
        return y
    
    
    def fit_data(self, fit_voltage=True, voltage_init_args=[1.0, 0.1, 17.0, 2, -0.1],
                fit_abs=True, abs_init_args=[10, 1, 17.0, 2, -1.0]):
        
        if fit_voltage:
            print('Fitting voltage data')
            self.voltage_params = np.empty((len(self.file_list), 5))
            self.voltage_params_covariance = np.empty((len(self.file_list), 5))
            
            for i in range(len(self.file_list)):
                try:
                    B = self.data[i].B*1000
                    V = self.data[i].Vishe*1000000
                    voltage_init_args[0] = np.abs(V.min()-V.max())
                    voltage_init_args[4] = V.mean()
                    pars, parscov = scipy.optimize.curve_fit(self._fit_voltage, B, V, voltage_init_args)
                    self.voltage_params[i,:], self.voltage_params_covariance[i,:] = pars, np.sqrt(np.diag(parscov))
                    print('.', end="")
                except RuntimeError:
                    print('\nRuntime error in file: ', self.file_list[i])
                except:
                    print('\nUnexpected error in file: ', self.file_list[i])
                    raise
            
        if fit_abs:
            print('\nFitting absorption data')
            self.abs_params = np.empty((len(self.file_list), 5))
            self.abs_params_covariance = np.empty((len(self.file_list), 5))
            
            for i in range(len(self.file_list)):
                try:
                    B = self.data[i].B*1000
                    V = self.data[i].Vlockin*1000000
                    abs_init_args[0] = np.abs(V.min()-V.max())
                    abs_init_args[4] = V.mean()
                    pars, parscov = scipy.optimize.curve_fit(self._fit_abs, B, V, abs_init_args)
                    self.abs_params[i,:], self.abs_params_covariance[i,:] = pars, np.sqrt(np.diag(parscov))
                    print('.', end="")
                except RuntimeError:
                    print('\nRuntime error in file: ', self.file_list[i])
                except:
                    print('\nUnexpected error in file: ', self.file_list[i])
                    raise
                    
    
    def save_params(self):
        if hasattr(self, 'voltage_params'):
            np.save(os.path.join(self.folder, 'voltage_params'), self.voltage_params)
            np.save(os.path.join(self.folder, 'voltage_params_covariance'), self.voltage_params_covariance)
        if hasattr(self, 'abs_params'):
            np.save(os.path.join(self.folder, 'abs_params'), self.abs_params)
            np.save(os.path.join(self.folder, 'abs_params_covariance'), self.abs_params_covariance)
            
    
    def load_params(self):
        if os.path.exists(os.path.join(self.folder, 'voltage_params.npy')):
            self.voltage_params = np.load(os.path.join(self.folder, 'voltage_params.npy'))
            self.voltage_params_covariance = np.load(os.path.join(self.folder, 'voltage_params_covariance.npy'))
        else:
            print('voltage_params.npy does not exist')
        if os.path.exists(os.path.join(self.folder, 'abs_params.npy')):
            self.abs_params = np.load(os.path.join(self.folder, 'abs_params.npy'))
            self.abs_params_covariance = np.load(os.path.join(self.folder, 'abs_params_covariance.npy'))
        else:
            print('abs_params.npy does not exist')
            
    
    def plot_data(self):
        fig = plt.figure(figsize=(15,len(self.file_list)*6))
        for i in range(len(self.file_list)):

            ax = fig.add_subplot(len(self.file_list),2,(2*i+1))
            ax.plot(self.data[i].B*1000, self.data[i].Vishe*1000000)
            if hasattr(self, 'voltage_params'):
                ax.plot(self.data[i].B*1000, self._fit_voltage(self.data[i].B*1000, *self.voltage_params[i,:]))
                ax.annotate(('V_symm: ' + str(round(self.voltage_params[i,0], 2)) + ' $\pm$ ' + str(round(self.voltage_params_covariance[i,0], 2))
                             + ' $\mu V$\nV_antisymm: ' + str(round(self.voltage_params[i,1], 2)) + ' $\pm$ ' + str(round(self.voltage_params_covariance[i,1], 2))
                             + ' $\mu V$\nFWHM: ' + str(round(self.voltage_params[i,3], 4)) + ' $\pm$ ' + str(round(self.voltage_params_covariance[i,3], 2))
                             + ' mT'), (0.05, 0.5), xycoords='axes fraction', fontsize=12)
            ax.set_xlabel('B /mT', fontsize=15)
            ax.set_ylabel('V_device $/\mu V$', fontsize=15)
            ax.set_title(str(i)+'   ' + self.file_list[i][:-4], fontsize=15)

            ax = fig.add_subplot(len(self.file_list),2,(2*i+2))
            ax.plot(self.data[i].B*1000, self.data[i].Vlockin*1000000)
            if hasattr(self, 'abs_params'):
                ax.plot(self.data[i].B*1000, self._fit_abs(self.data[i].B*1000, *self.abs_params[i,:]))
                ax.annotate(('Res_symm: ' + str(round(self.abs_params[i,0], 2)) + ' $\mu V$' + '\nRes_antisymm: ' + str(round(self.abs_params[i,1], 2))
                             + ' $\mu V$\nFWHM: ' + str(round(self.abs_params[i,3], 4)) + ' mT'), (0.05, 0.7), xycoords='axes fraction', fontsize=12)
            ax.set_xlabel('B /mT', fontsize=15)
            ax.set_ylabel('V_lockin $/\mu V$', fontsize=15)
            
        plt.show()
        
    
    @staticmethod
    def _fit_ang_sym(angle, *args):
        (phy, phyz, Vamr_x, Vamr_z, Vahe, Vishe, ang_offset) = args
        y = (-np.sin(phy)*np.abs(Vamr_x)*np.sin(np.pi/2+angle-ang_offset)*np.cos(2*(np.pi/2+angle-ang_offset))
             + np.sin(phyz)*np.abs(Vamr_z)*np.sin(np.pi/2+angle-ang_offset)*np.sin(2*(np.pi/2+angle-ang_offset))
             - np.cos(phy)*np.abs(Vahe)*np.sin(np.pi/2+angle-ang_offset) + Vishe*(np.sin(np.pi/2+angle-ang_offset))**3)
        return y
    
    @staticmethod
    def _fit_ang_antisym(angle, *args):
        (phy, phyz, Vamr_x, Vamr_z, Vahe, Vishe, ang_offset) = args
        y = (np.cos(phy)*np.abs(Vamr_x)*np.sin(np.pi/2+angle-ang_offset)*np.cos(2*(np.pi/2+angle-ang_offset))
             - np.cos(phyz)*np.abs(Vamr_z)*np.sin(np.pi/2+angle-ang_offset)*np.sin(2*(np.pi/2+angle-ang_offset))
             - np.sin(phy)*np.abs(Vahe)*np.sin(np.pi/2+angle-ang_offset))
        return y
    
    def _fit_ang_total(self, angle_combo, *args):
        sym = self._fit_ang_sym(angle_combo[:len(angle_combo)//2], *args)
        antisym = self._fit_ang_antisym(angle_combo[len(angle_combo)//2:], *args)
        return np.append(sym, antisym)
    
    def fit_ang_dep(self, init_args=[-80*np.pi/180, -80*np.pi/180, 20.0, 1.0, 1.0,  0.0, 0.0]):
        angle_combo = np.append(self.angles, self.angles)
        angle_combo = angle_combo/180*np.pi 
        y_combo = np.append(self.voltage_params[:,0], self.voltage_params[:,1])
        
        self.ang_params, self.ang_params_covariance = scipy.optimize.curve_fit(self._fit_ang_total, angle_combo, y_combo, init_args)
        self.ang_params_covariance = np.sqrt(np.diag(self.ang_params_covariance))
    
    def plot_ang_dep(self):
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,6))
        ax1.plot(self.angles, self.voltage_params[:,0])
        ax2.plot(self.angles, self.voltage_params[:,1])
        if hasattr(self, 'ang_params'):
            angles = np.array(self.angles)
            fit_sym = self._fit_ang_sym(angles/180*np.pi, *self.ang_params)
            fit_antisym = self._fit_ang_antisym(angles/180*np.pi, *self.ang_params)
            ax1.plot(self.angles, fit_sym, 'r:')
            ax2.plot(self.angles, fit_antisym, 'r:')
            ax1.annotate(('V_AMR_x = ' + str(round(self.ang_params[2],3)) + ' $\pm$ ' + str(round(self.ang_params_covariance[2],3))
                          + 'uV\nV_AMR_z = ' + str(round(self.ang_params[3],3)) + ' $\pm$ ' + str(round(self.ang_params_covariance[3],3))
                          + 'uV\nV_AHE = ' + str(round(self.ang_params[4],3)) + ' $\pm$ ' + str(round(self.ang_params_covariance[4],3))
                          + ' uV\nV_ISHE = ' + str(round(self.ang_params[5],3)) + ' $\pm$ ' + str(round(self.ang_params_covariance[5],3))
                          + ' uV'), (0.05, 0.75), xycoords='axes fraction', fontsize=12)
        ax1.set_title('V_symmetric', fontsize=20)
        ax2.set_title('V_antisymmetric', fontsize=20)
        ax1.set_xlabel('In-plane angle /deg', fontsize=14)
        ax2.set_xlabel('In-plane angle /deg', fontsize=14)
        ax1.set_ylabel('V_symmetric /uV', fontsize=14)
        ax2.set_ylabel('V_antisymmetric /uV', fontsize=14)

        plt.savefig(os.path.join(self.folder, 'In-plane_ang_dep.png'))
        plt.show()