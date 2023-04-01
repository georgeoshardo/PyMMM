import tifffile
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from skimage.transform import rotate
from scipy.signal import find_peaks
from pystackreg import StackReg
import os
import warnings
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from itertools import cycle, product
from skimage.transform import warp
from PIL import Image
from copy import deepcopy
from tqdm.autonotebook import tqdm
from multipledispatch import dispatch

class Experiment:
    def __init__(self, directory, custom_filename_splitter=None, custom_img_path_generator = None, save_filetype = None, mean_amount = None):
        self.directory = directory
        self.extracted_dir = directory + "{}extracted{}".format(os.path.sep,os.path.sep)
        self.files = glob(self.extracted_dir + "/*")
        self.custom_img_path_generator = custom_img_path_generator

        if self.custom_img_path_generator:
            def img_path_generator(directory, FOV, channel, time, file_extension):
                FOV, channel, time = self.coordinate_converter(FOV, channel, time)
                return self.custom_img_path_generator(directory, FOV, channel, time, file_extension)
            self.img_path_generator = img_path_generator
        else:
            def img_path_generator(directory, FOV, channel, time, file_extension):
                FOV, channel, time = self.coordinate_converter(FOV, channel, time)
                return f"{directory}/{FOV}_{channel}_{time}.{file_extension}"
            self.img_path_generator = img_path_generator

        if custom_filename_splitter:
            def filename_splitter(filename):
                filename = filename.split(os.path.sep)[-1]
                return custom_filename_splitter(filename)
            
            self.filename_splitter = filename_splitter
        else:
            def filename_splitter(filename):
                FOV = filename.split(os.path.sep)[-1].split("_")[0]
                time = filename.split(os.path.sep)[-1].split("_")[2].split(".")[0]
                channel = filename.split(os.path.sep)[-1].split("_")[1]
                file_extension = filename.split(os.path.sep)[-1].split("_")[2].split(".")[-1]
                return FOV, time, channel, file_extension
            self.filename_splitter = filename_splitter

        self.FOVs = [self.filename_splitter(x)[0] for x in self.files]
        self.FOVs = sorted(list(set(self.FOVs)))
        self.FOVs = deepcopy(self.FOVs)
        self.num_FOVs = len(self.FOVs)
        self.times = [self.filename_splitter(x)[1] for x in self.files]
        self.times = sorted(list(set(self.times)))
        self._times = deepcopy(self.times)
        self.channels = [self.filename_splitter(x)[2] for x in self.files]
        self.channels = sorted(list(set(self.channels)))
        self.file_extension = self.filename_splitter(self.files[0])[3]

        # Checking the file type and assigning the correct imreader and imwriter functions.
        def png_imreader(filename):
            return np.array(Image.open(filename))
        def png_imwriter(filename, data):
            """
            write PNG
            """
            data = Image.fromarray(data)
            data.save(filename)
        def tiff_imreader(filename):
            return tifffile.imread(filename)
        def tiff_imwriter(filename, data):
            """
            write TIFF
            """
            tifffile.imwrite(filename, data)
        
        if "png" in self.file_extension.lower():
            self.imreader = png_imreader
            self.imwriter = png_imwriter
            self.reg_imreader = png_imreader
        elif "tif" in self.file_extension.lower():
            self.imreader = tiff_imreader
            self.imwriter = tiff_imwriter
            self.reg_imreader = tiff_imreader
        else:
            raise ValueError(f"Invalid file extension: {self.file_extension.lower()}")

        self.save_file_extension = self.file_extension
        if save_filetype:
            if "png" in save_filetype.lower():
                self.imwriter = png_imwriter
                self.reg_imreader = png_imreader
                self.save_file_extension = "png"
            elif "tif" in save_filetype.lower():
                self.imwriter = tiff_imreader
                self.reg_imreader = tiff_imreader
                self.save_file_extension = "tiff"

        self.dims = self.imreader(self.files[0]).shape
        self.dtype = self.imreader(self.files[0]).dtype
        self.experiment_name = os.path.basename(os.path.normpath(self.directory))  # gets last part of the directory
        self.registered_dir = self.directory + "{}registered{}".format(os.path.sep,os.path.sep)
        self.num_timepoints = len(self.times)
        self._mean_start = "END"
        if mean_amount:
            self.mean_amount = mean_amount
        else:
            warnings.warn("No mean_amount attribute set. Taking image means over all timepoints. Very slow!")
            self.mean_amount = self.num_timepoints
        self.PC_channel = None
        self.trench_y_offsets = None
        self.y_peaks = None
        self.pruned_experiment_trench_x_lims = None
        self.trench_directory = directory + "{}trenches{}".format(os.path.sep,os.path.sep)
        self.rotation = None
        self.mean_of_timestack = dict()
        
    def __str__(self):
        return f"""
            Experiment name: {self.experiment_name}
            Channels: {self.channels}
            Timepoints: {self.num_timepoints}
            FOVs: {len(self.FOVs)}
            Registered: {self.is_registered}
        """

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            name_eq = self.experiment_name == other.experiment_name
            fov_eq = all([x == y for x, y in zip(self.FOVs, other.FOVs)])
            channels_eq = all([x == y for x, y in zip(self.channels, other.channels)])
            times_eq = all([x == y for x, y in zip(self.times, other.times)])
            n_fov_eq = len(self.FOVs) == len(other.FOVs)
            n_times_eq = self.num_timepoints == other.num_timepoints
            n_channels_eq = len(self.channels) == len(other.channels)
            dims_eq = all([x == y for x, y in zip(self.dims, other.dims)])
            registered_eq = self.is_registered == other.is_registered
            return all(
                [name_eq, fov_eq, channels_eq, times_eq, n_fov_eq, n_times_eq, n_channels_eq, dims_eq, registered_eq])
        else:
            return False
    
    def coordinate_converter(self, FOV, channel, time):
        """
        
        :param FOV:
        :param channel:
        :param time:
        :return:
        """
        if isinstance(FOV, int):
            FOV = self.FOVs[FOV]
        if isinstance(channel, int):
            channel = self.channels[channel]
        if isinstance(time, int):
            time = self.times[time]
        return FOV, channel, time


    def trench_path_generator(self, directory, FOV, channel, trench, time, file_extension):
        FOV, channel, time = self.coordinate_converter(FOV, channel, time)
        return "{}/{}_{}_TR{}_{}.{}".format(directory, FOV, channel, str(trench).zfill(2), time, file_extension)

    @property
    def mean_start(self):
        return self._mean_start

    @mean_start.setter
    def mean_start(self, value):
        assert value in ["START", "END"], "mean_start should be START or END"
        self._mean_start = value

    def set_analysis_times(self, start, end):
        self.times = self._times[start:end]

    def discard_FOVs(self, FOVs):
        self.FOVs = [x for x in self.FOVs if x not in FOVs]

    @property
    def registration_channel(self):
        return self._registration_channel

    @registration_channel.setter
    def registration_channel(self, value):
        assert value in self.channels, f"Channel must be one of {self.channels}"
        self._registration_channel = value

    @property
    def PC_channel(self):
        return self._PC_channel

    @PC_channel.setter
    def PC_channel(self, value):
        assert value in self.channels + [None], f"Phase contrast channel must be one of {self.channels + [None]}"
        self._PC_channel = value

    @property
    def is_registered(self):
        try:
            if len(self.files) == len(glob(self.registered_dir + "/*")):
                self._is_registered = True
            elif self.set_analysis_times:
                self._is_registered = True
            else:
                self._is_registered = False
            return self._is_registered
        except:
            return False

    @is_registered.setter
    def is_registered(self, value):
        self._is_registered = value


    def mean_img_getter(self, FOV, channel, rotation = None, registered = False):
        if self._mean_start == "END":
            mean_times = self.times[-self.mean_amount:]
        elif self._mean_start == "START":
            mean_times = self.times[:self.mean_amount]
        else:
            mean_times = self.times[-self.mean_amount:]
        img_size = self.dims
        mean_img = np.zeros(img_size)
        mean_img_imgs = (self.get_image(FOV, channel, x, registered) for x in mean_times)
        for img in mean_img_imgs:
            mean_img += img / self.mean_amount
        if rotation:
            self.rotation = rotation
            mean_img = rotate(mean_img, rotation, preserve_range=True)
        return mean_img.astype(np.uint16)

    def get_mean_images(self, rotation=0, *, plot=False): 
        mean_images = dict()
        mean_images_ = Parallel(n_jobs=-1)(delayed(self.mean_img_getter)(FOV, self._registration_channel, rotation) for FOV in tqdm(self.FOVs))
        self.rotation = rotation
        for img, FOV in zip(mean_images_, self.FOVs):
            mean_images[FOV] = img
            #self.mean_of_timestack[FOV+self._registration_channel] = img

        print(
            f"Mean images for {len(self.FOVs)} FOVs with rotation of {rotation} deg calculated, use the mean_images method to return a dict of mean images")

        if plot:
            plt.figure(figsize=(10, 10))
            plt.imshow(mean_images[self.FOVs[0]], cmap="Greys_r")
            plt.title(self.FOVs[0])
            ax = plt.gca()
            ax.grid(which='both', color='w', linestyle='-', linewidth=2)
            plt.show()

        self.mean_images = mean_images


        
    def get_image(self, FOV, channel, time, registered=False, *, plot=False, rotation=False):
        if registered:
            #assert self.is_registered, "Experiment not registered"
            directory = self.img_path_generator(self.registered_dir, FOV, channel, time, self.save_file_extension)
            image = self.reg_imreader(directory)
        else:
            directory = self.img_path_generator(self.extracted_dir, FOV, channel, time, self.file_extension)
            image = self.imreader(directory)
        if rotation:
            image = rotate(image, rotation, preserve_range=True).astype(np.uint16)
        if plot:
            plt.figure(figsize=(10, 10))
            plt.imshow(image, cmap="Greys_r")
            plt.title(f"FOV: {FOV}, channel: {channel}, time: {time}")
            plt.show()
        

        
        return image


    def rotate_experiment(self, rotation, force=False, n_jobs = -1):
        rotation = self.rotation
        if hasattr(self, "mean_images"):
            pass
        else:
            raise Exception("You haven't called get_mean_images to calculate this experiment's mean images.")
        try:
            os.mkdir(self.directory + "/registered/")
            fields = product(self.FOVs, self.channels, self.times)
            Parallel(n_jobs=n_jobs)(delayed(self.rotate_image)(*field, rotation) for field in fields)
        except:
            if force:
                warnings.warn("Rerotating experiment")
                fields = product(self.FOVs, self.channels, self.times)
                Parallel(n_jobs=n_jobs)(delayed(self.rotate_image)(*field, rotation) for field in fields)
            elif (not force) and (self.is_registered):
                raise Exception("The experiment has already been rotated, to re-rotate use force=True")
            else:
                raise Exception(
                    "The registration directory exists, but the experiment does not seem to be fully registed. Check number of files")
    
            
    def rotate_image(self, FOV, channel, time, rotation, save=False):
        img = self.get_image(FOV, channel, time, registered = False)
        img = rotate(img, rotation, preserve_range=True).astype(np.uint16)
        if save:
            out_path = self.img_path_generator(self.registered_dir, FOV, channel, time, self.save_file_extension)
            self.imwriter(out_path, img)
            return None
        else:
            return img

    def register_experiment(self, force=False, mode = "mean", sum=False, n_jobs=-1, fiduciary = None, parallel_time = False, y_lims = (0,-1), x_lims = (0,-1)):
        if mode == "mean":
            if hasattr(self, "mean_images"):
                pass
            else:
                raise Exception("You haven't called get_mean_images to calculate this experiment's mean images.")
        try:
            os.mkdir(self.directory + "/registered/")
        except:
            pass
        if fiduciary:
            self.tmats = self.get_transformation_matrices(fiduciary, mode, y_lims, x_lims)
            Parallel(n_jobs=-1)(delayed(self.warp_and_save)(FOV, channel, time, tmat) for time, tmat in zip(self.times, self.tmats) for FOV in tqdm(self.FOVs) for channel in self.channels)
        else:
            for FOV in tqdm(self.FOVs):
                tmats = self.get_transformation_matrices(FOV, mode, y_lims, x_lims)
                Parallel(n_jobs=-1)(delayed(self.warp_and_save)(FOV, channel, time, tmat) for time, tmat in zip(self.times, tmats) for channel in self.channels)


    def warp_image(self, FOV, channel, time, tmats):
        mov = self.get_image(FOV, channel, time, registered=False, plot=False, rotation = self.rotation)
        mov = warp(mov, tmats, preserve_range=True, order=3)
        mov = mov.astype(np.uint16)
        return mov

    def warp_and_save(self, FOV, channel, time, tmats):
        out_path = self.img_path_generator(self.registered_dir, FOV, channel, time, self.save_file_extension)
        mov = self.warp_image(FOV, channel, time, tmats)
        self.imwriter(out_path, mov)

    def register_image(self, FOV, channel, time, sr, ref, y_lims, x_lims):
        mov = self.get_image(FOV, channel, time, registered=False, plot=False, rotation = self.rotation)
        tmats = sr.register(ref[y_lims[0]:y_lims[1],x_lims[0]:x_lims[1]], mov[y_lims[0]:y_lims[1],x_lims[0]:x_lims[1]])
        return tmats

    def get_transformation_matrices(self, FOV, mode, y_lims, x_lims):
        if mode == "mean":
            ref = self.mean_images[FOV]
        elif mode == "last":
            ref = self.get_image(FOV, self._registration_channel, self.times[-1], registered=False, rotation = self.rotation)
        elif mode == "first":
            ref = self.get_image(FOV, self._registration_channel, self.times[0], registered=False, rotation = self.rotation)
        elif type(mode) == int:
            ref = self.get_image(FOV, self._registration_channel, self.times[mode], registered=False, rotation = self.rotation)
        sr = StackReg(StackReg.RIGID_BODY)
        return Parallel(n_jobs=-1)(delayed(self.register_image)(FOV, self._registration_channel, time, sr, ref, y_lims, x_lims) for time in tqdm(self.times))

    def register_FOV(self, FOV, mode="mean", sum = False, tmats_list = None, parallel_time = False):
        # modes: mean, first, last, n, 
        if tmats_list:
            for time, tmats in zip(self.times,tmats_list):
                for channel in self.channels:
                    mov = self.register_image(FOV, channel, time, tmats)
                    out_path = self.img_path_generator(self.registered_dir, FOV, channel, time, self.save_file_extension)
                    self.imwriter(out_path, mov)
            return None
        tmats_list = []
        if mode == "mean":
            ref = self.mean_images[FOV]
            sr = StackReg(StackReg.RIGID_BODY)
            mov = self.get_image(FOV, self._registration_channel, self.times[0], registered=False, plot=False)
            tmats = sr.register(ref, mov)
            tmats_list.append(tmats)
            for channel in self.channels:
                mov = self.register_image(FOV, channel, self.times[0], tmats)
                out_path = self.img_path_generator(self.registered_dir, FOV, channel, self.times[0], self.save_file_extension)
                self.imwriter(out_path, mov)
        elif mode == "previous": # If aligning to the previous frame, write the first frame without any modification
            mov = self.get_image(FOV, self._registration_channel, self.times[0], registered=False, plot=False)
            if rotate: # Need to rotate the first image if rotation specified 
                mov = rotate(mov, self.rotation, preserve_range = True)
                mov = mov.astype(np.uint16)
            out_path = self.img_path_generator(self.registered_dir, FOV, self._registration_channel, self.times[0],
                                                self.save_file_extension)
            self.imwriter(out_path, mov)
            for channel in self.channels:
                if channel == self._registration_channel:
                    pass
                else:
                    mov = self.get_image(FOV, channel, self.times[0], registered=False, plot=False)
                    if rotate: # Need to rotate the first image if rotation specified 
                        mov = rotate(mov, self.rotation, preserve_range = True)
                        mov = mov.astype(np.uint16)
                    out_path = self.img_path_generator(self.registered_dir, FOV, channel, self.times[0], self.save_file_extension)
                    self.imwriter(out_path, mov)
        def register(time, prev_time):
            #for time, prev_time in list(zip(self.times[1:], self.times)):
                if sum:
                    mov = self.get_image(FOV, self.channels[0], time, registered=False, plot=False).astype(float)
                    for channel in self.channels[1:]:
                        mov += self.get_image(FOV, self.channels[0], time, registered=False, plot=False).astype(float)
                else:
                    mov = self.get_image(FOV, self._registration_channel, time, registered=False, plot=False)
                sr = StackReg(StackReg.RIGID_BODY)
                if mode == "previous":
                    if sum:
                        ref = self.get_image(FOV, self.channels[0], prev_time, registered=False, plot=False)
                        for channel in self.channels[1:]:
                            ref += self.get_image(FOV, channel, prev_time, registered=False, plot=False)
                    else:
                        ref = self.get_image(FOV, self._registration_channel, prev_time, registered=False, plot=False)
                tmats = sr.register(ref, mov)

                for channel in self.channels:
                    mov = self.register_image(FOV, channel, time, tmats)
                    out_path = self.img_path_generator(self.registered_dir, FOV, channel, time, self.save_file_extension)
                    self.imwriter(out_path, mov)
                return tmats
        if parallel_time:
            n_jobs = -1
        else:
            n_jobs = 1
        tmats_ = Parallel(n_jobs=n_jobs)(delayed(register)(time, prev_time ) for time, prev_time in list(zip(self.times[1:], self.times)))
        tmats_list += tmats_
        return tmats_list


    def get_mean_of_timestack(self, FOV, channel, *, plot=False):
        try:
            FOV, channel, _ = self.coordinate_converter(FOV, channel, 0)
            mean_img =  self.mean_of_timestack[FOV+channel]
        except:
            mean_img = self.mean_img_getter(FOV, channel, registered=self.is_registered)
            self.mean_of_timestack[FOV+channel] = mean_img.astype(np.uint16)



        if plot:
            plt.figure(figsize=(10, 10))
            plt.imshow(mean_img, cmap="Greys_r")
            plt.title(f"FOV: {FOV}, channel: {channel}, time: mean")
            plt.show()
        return mean_img

    def mean_t_x(self, FOV, channel, sigma=False, *, plot=False):
        mean_img = self.get_mean_of_timestack(FOV, channel)
        mean_img = mean_img.mean(axis=0)
        if sigma:
            mean_img = gaussian_filter1d(mean_img, sigma)
        if plot:
            plt.plot(mean_img)
            plt.title(f"FOV: {FOV}, Channel: {channel}, Gaussian sigma: {sigma}")
            plt.show()
        return mean_img

    def mean_t_y(self, FOV, channel, sigma=False, *, plot=False):
        mean_img = self.get_mean_of_timestack(FOV, channel)
        mean_img = mean_img.mean(axis=1)
        if sigma:
            mean_img = gaussian_filter1d(mean_img, sigma)
        if plot:
            plt.plot(mean_img)
            plt.title(f"FOV: {FOV}, Channel: {channel}, Gaussian sigma: {sigma}")
            plt.show()
        return mean_img

    def find_trench_peaks(self, FOV, channel=None, sigma=False, distance=40, height=0, prominence=0, threshold=0, *, plot=False):
        if channel is not None:
            pass
        else:
            channel = self._registration_channel
        mean_img = self.mean_t_x(FOV, channel, sigma)
        peaks, _ = find_peaks(mean_img, distance=distance, height=height, prominence=prominence, threshold=threshold)
        ### Would be good to add here a minimum x value and maximum x value for the peaks, such that trenches close to the edge of the image are stripped out.
        if plot:
            plt.plot(mean_img)
            plt.title(f"FOV: {FOV}, Channel: {channel}, Gaussian sigma: {sigma}")
            plt.plot(peaks, mean_img[peaks], "x", c="r")
            plt.show()
        return mean_img, peaks

    def find_all_trench_x_positions(self, 
                                    channel=None, 
                                    *, 
                                    sigma=False, 
                                    distance=40, 
                                    height=0, 
                                    prominence=0, 
                                    threshold=0, 
                                    shrink_scale = 2.2, 
                                    plot=False, 
                                    plot_save = False):
        if plot_save: 
            try:
                os.mkdir(self.directory + "/diagnostics/")
            except:
                pass
            try:
                os.mkdir(self.directory + "/diagnostics/trench_x_positions/")
            except:
                pass
        
        if channel is not None:
            pass
        else:
            channel = self._registration_channel

        peaks = Parallel(n_jobs=-1)(delayed(self.find_trench_peaks)(FOV, 
                                                                    channel, 
                                                                    sigma, 
                                                                    distance, 
                                                                    height, 
                                                                    prominence, 
                                                                    threshold, 
                                                                    plot=False) for FOV in self.FOVs)
        self.peaks = dict()
        for FOV, peak in zip(self.FOVs, peaks):
            self.peaks[FOV] = peak[1]
        
        #self.peaks = {FOV: self.find_trench_peaks(FOV, channel, sigma, distance, plot=False)[1] for FOV in self.FOVs}
        self.trench_spacing = np.mean([np.mean(np.diff(self.peaks[FOV])) for FOV in self.FOVs])
        experiment_trench_x_lims = {FOV:
                                        zip(self.peaks[FOV] - round(self.trench_spacing / shrink_scale), 
                                            self.peaks[FOV] + round(self.trench_spacing / shrink_scale)) for FOV in
                                    self.FOVs
                                    }

        if plot:
            subplots = self.num_FOVs
            cols = 2
            rows = round(np.ceil(subplots / cols))
            fig, axes = plt.subplots(nrows=rows, ncols=cols, dpi=80, figsize=(20, 20))
            color_cycler = cycle(["red", "green", "blue", "yellow", "orange", "purple", "white"])
            axes_flat = axes.flatten()

        pruned_experiment_trench_x_lims = {}
        for i, FOV in enumerate(self.FOVs):
            trench_x_lims = experiment_trench_x_lims[FOV]
            _trench_x_lims = []
            for L, R in trench_x_lims:
                if (L < 0) or (R > self.dims[1]):
                    pass
                else:
                    _trench_x_lims.append((L, R))
            if plot or plot_save:
                #mean_img = self.get_image(FOV, channel, 1, registered=self.is_registered)
                mean_img = self.get_mean_of_timestack(FOV, channel)
            if plot:
                axes_flat[i].imshow(mean_img, cmap="Greys_r")
                axes_flat[i].get_xaxis().set_ticks([])
                axes_flat[i].get_yaxis().set_ticks([])
                axes_flat[i].set_title(FOV)
                axes_flat[i].autoscale(enable=True)
                if self.y_peaks and self.trench_y_offsets:
                    axes_flat[i].axhline(self.y_peaks[FOV][0] - self.trench_y_offsets[0], color="r")
                    axes_flat[i].axhline(self.y_peaks[FOV][0] - self.trench_y_offsets[1], color="r")
                for (L, R), color in zip(_trench_x_lims, color_cycler):
                    axes_flat[i].axvspan(L, R, alpha=0.1, color=color)
                    axes_flat[i].axvline(x=L, color=color)
                    axes_flat[i].axvline(x=R, color=color)
                
            if plot_save:
                fig_save, axes_save = plt.subplots(nrows=1, ncols=1, dpi=80, figsize=(20, 20))
                color_cycler = cycle(["red", "green", "blue", "yellow", "orange", "purple", "white"])
                axes_save.imshow(mean_img, cmap="Greys_r")
                axes_save.get_xaxis().set_ticks([])
                axes_save.get_yaxis().set_ticks([])
                axes_save.set_title(FOV)
                axes_save.autoscale(enable=True)
                if self.y_peaks and self.trench_y_offsets:
                    axes_save.axhline(self.y_peaks[FOV][0] - self.trench_y_offsets[0], color="r")
                    axes_save.axhline(self.y_peaks[FOV][0] - self.trench_y_offsets[1], color="r")
                for (L, R), color in zip(_trench_x_lims, color_cycler):
                    axes_save.axvspan(L, R, alpha=0.1, color=color)
                    axes_save.axvline(x=L, color=color)
                    axes_save.axvline(x=R, color=color)
                plt.tight_layout()
                plt.savefig(self.directory + "/diagnostics/trench_x_positions/{}.png".format(str(FOV)))
                plt.close()
                
            
            pruned_experiment_trench_x_lims[FOV] = _trench_x_lims
            if plot:
                plt.tight_layout()
                plt.show()
        self.pruned_experiment_trench_x_lims = pruned_experiment_trench_x_lims
        return pruned_experiment_trench_x_lims

    def find_lane_peaks(self, FOV, channel=None, sigma=False, distance=1, height=1, *, plot=False):
        if channel is not None:
            pass
        else:
            channel = self._registration_channel
        mean_img = self.mean_t_y(FOV, channel, sigma)
        peaks, _ = find_peaks(mean_img, distance=distance, height=height)
        if plot:
            plt.plot(mean_img)
            plt.title(f"FOV: {FOV}, Channel: {channel}, Gaussian sigma: {sigma}")
            plt.plot(peaks, mean_img[peaks], "x", c="r")
            plt.show()
        return mean_img, peaks

    def find_all_trench_y_positions_PC(self, channel=None, *, sigma=False, distance=1, height=1, plot=False, plot_save = False):
        
        if plot_save: 
            try:
                os.mkdir(self.directory + "/diagnostics/")
            except:
                pass
            try:
                os.mkdir(self.directory + "/diagnostics/trench_y_positions/")
            except:
                pass
        
        if channel is not None:
            pass
        else:
            channel = self._registration_channel
        assert self.trench_y_offsets is not None, "Please set the trench_y_offsets attribute"
        assert self.trench_y_offsets[0] > self.trench_y_offsets[1], "Ensure the first offset is greater than the other"
        self.y_peaks = {
            FOV: self.find_lane_peaks(FOV, channel=channel, sigma=sigma, distance=distance, height=height, plot=False)[
                1] for FOV in self.FOVs}
        if plot:
            subplots = self.num_FOVs
            cols = 2
            rows = round(np.ceil(subplots / cols))
            fig, axes = plt.subplots(nrows=rows, ncols=cols, dpi=80, figsize=(20, 20))
            color_cycler = cycle(["red", "green", "blue", "yellow", "orange", "purple", "white"])
            axes_flat = axes.flatten()

        for i, FOV in enumerate(self.FOVs):
            if plot or plot_save:
                mean_img = self.get_mean_of_timestack(FOV, channel)
            if plot:
                axes_flat[i].imshow(mean_img, cmap="Greys_r")
                axes_flat[i].get_xaxis().set_ticks([])
                axes_flat[i].get_yaxis().set_ticks([])
                axes_flat[i].set_title(FOV)
                axes_flat[i].autoscale(enable=True)
                axes_flat[i].axhline(self.y_peaks[FOV][0] - self.trench_y_offsets[0], color="r")
                axes_flat[i].axhline(self.y_peaks[FOV][0] - self.trench_y_offsets[1], color="r")
                if self.pruned_experiment_trench_x_lims:
                    for (L, R), color in zip(self.pruned_experiment_trench_x_lims[FOV], color_cycler):
                        axes_flat[i].axvspan(L, R, alpha=0.1, color=color)
                        axes_flat[i].axvline(x=L, color=color)
                        axes_flat[i].axvline(x=R, color=color)
                        
            if plot_save:
                fig_save, axes_save = plt.subplots(nrows=1, ncols=1, dpi=80, figsize=(20, 20))
                color_cycler = cycle(["red", "green", "blue", "yellow", "orange", "purple", "white"])
                axes_save.imshow(mean_img, cmap="Greys_r")
                axes_save.get_xaxis().set_ticks([])
                axes_save.get_yaxis().set_ticks([])
                axes_save.set_title(FOV)
                axes_save.autoscale(enable=True)
                axes_save.axhline(self.y_peaks[FOV][0] - self.trench_y_offsets[0], color="r")
                axes_save.axhline(self.y_peaks[FOV][0] - self.trench_y_offsets[1], color="r")
                if self.y_peaks and self.trench_y_offsets:
                    axes_save.axhline(self.y_peaks[FOV][0] - self.trench_y_offsets[0], color="r")
                    axes_save.axhline(self.y_peaks[FOV][0] - self.trench_y_offsets[1], color="r")
                for (L, R), color in zip(self.pruned_experiment_trench_x_lims[FOV], color_cycler):
                    axes_save.axvspan(L, R, alpha=0.1, color=color)
                    axes_save.axvline(x=L, color=color)
                    axes_save.axvline(x=R, color=color)
                plt.tight_layout()
                plt.savefig(self.directory + "/diagnostics/trench_y_positions/{}.png".format(str(FOV)))
                plt.close()

        if plot:
            plt.tight_layout()
            plt.show()
        return self.y_peaks

    def get_trenches(self, FOV, channel, time, save = False):
        image = self.get_image(FOV, channel, time, registered=self.is_registered)
        y_pos = self.y_peaks[FOV][0]
        x_pos = self.pruned_experiment_trench_x_lims[FOV]
        trenches = []
        for i, (L, R) in enumerate(x_pos):
            trench = image[y_pos - self.trench_y_offsets[0]:y_pos - self.trench_y_offsets[1], L:R]
            trenches.append(trench)
        if save:
            for i, trench in enumerate(trenches):
                out_path = self.trench_path_generator(self.trench_directory, FOV, channel, i, time, self.save_file_extension)
                self.imwriter(out_path, trench)

    def extract_trenches(self, n_jobs=-1, force=False):
        fields = product(self.FOVs, self.channels, self.times)
        try:
            os.mkdir(self.directory + "/trenches/")
            Parallel(n_jobs=n_jobs)(delayed(self.get_trenches)(*field, True) for field in fields)
        except:
            if force:
                warnings.warn("Re-extracting trenches")
                Parallel(n_jobs=n_jobs)(delayed(self.get_trenches)(*field, True) for field in fields)
