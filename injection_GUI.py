from PIL import Image, ImageTk
import sys
import os
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
from keras.models import load_model
import cv2


class Inj_GUI(tkinter.Frame):
    def __init__(self, parent, filename=None):
        tkinter.Frame.__init__(self, parent)
        self.grid()
        self.path_label_cont = tkinter.StringVar()
        self.createwidgets()
        self.croprect_start = None
        self.croprect_end = None
        self.current_rect = None
        self.filename = filename
        self.folder_path = None
        self.norm_size = 64
        self.crop_frame = 10
        self.is_crop = 0
        self.tamper_number = 0
        procedures_path = os.path.join(os.path.abspath(__file__)[0:-16], "procedures")
        self.generator_layer_1 = load_model(os.path.join(procedures_path, "generator_layer_1.h5"))
        self.generator_layer_2 = load_model(os.path.join(procedures_path, "generator_layer_2.h5"))
        self.dicom_image = None
        self.prev_image = None
        self.min_hu = 300  # minimum hu level for display
        self.max_hu = 800  # maximum hu level for display
    def createwidgets(self):
        self.canvas = tkinter.Canvas(self, height=600, width=1000, relief=tkinter.SUNKEN)
        self.canvas.bind('<Button-1>', self.canvas_mouse1_callback)
        self.canvas.bind('<ButtonRelease-1>', self.canvas_mouseup1_callback)
        self.canvas.bind('<B1-Motion>', self.canvas_mouse1move_callback)
        self.button_load = tkinter.Button(self, text='load', activebackground='#F01', command=self.load_file)
        self.button_crop = tkinter.Button(self, text='crop', activebackground='#F01', command=self.start_crop)
        self.button_delete = tkinter.Button(self, text='delete', activebackground='#F01', command=self.delete_crop)
        self.button_save = tkinter.Button(self, text='save', activebackground='#F01', command=self.save_tamper)
        self.button_inject = tkinter.Button(self, text='inject', activebackground='#F01', command=self.inject)
        self.button_cancel = tkinter.Button(self, text='back', activebackground='#F01', command=self.back)
        self.button_set_path = tkinter.Button(self, text='save in:', activebackground='#F01', command=self.set_path)
        self.label_path = tkinter.Label(self, textvariable=self.path_label_cont, bg='white')
        self.scroll_x = tkinter.Scrollbar(self, orient=tkinter.HORIZONTAL)
        self.scroll_y = tkinter.Scrollbar(self, orient=tkinter.VERTICAL)
        self.scroll_x.config(command=self.canvas.xview)
        self.scroll_y.config(command=self.canvas.yview)
        self.scroll_y.grid(row=0, column=0, sticky='ns')
        self.canvas.grid(row=0, column=1, columnspan=5)
        self.scroll_x.grid(row=1, columnspan=5, sticky='ew')
        self.button_load.grid(row=2, column=1, padx=2, pady=2)
        self.button_crop.grid(row=2, column=2, padx=2, pady=2)
        self.button_delete.grid(row=3, column=2, padx=2)
        self.button_inject.grid(row=2, column=3, padx=2, pady=2)
        self.button_cancel.grid(row=3, column=3, padx=2)
        self.button_save.grid(row=2, column=4, padx=2, pady=2)
        self.button_set_path.grid(row=4, column=1)
        self.label_path.grid(row=4, column=2)

    def canvas_mouse1_callback(self, event):
        if self.is_crop:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            self.croprect_start = (x, y)

    def canvas_mouse1move_callback(self, event):
        if self.is_crop:
            if self.current_rect:
                self.canvas.delete(self.current_rect)
            x1 = self.croprect_start[0]
            y1 = self.croprect_start[1]
            x2 = self.canvas.canvasx(event.x)
            y2 = self.canvas.canvasy(event.y)
            a = int(min(y2 - y1, x2 - x1))
            box = (x1, y1, x1 + a, y1 + a)
            cr = self.canvas.create_rectangle(box, outline='red')
            self.current_rect = cr

    def canvas_mouseup1_callback(self, event):
        if self.is_crop:
            self.croprect_end = (event.x, event.y)
            self.is_crop = 0

    def loadimage(self):
        try:
            self.dicom_image = np.load(self.filename)
            self.prev_image = np.copy(self.dicom_image)

        except:
            tkinter.messagebox.showerror("Error", "Error loading file' ")
        else:
            self.display_to_canvas()

    def load_file(self):
        self.filename = None
        filenames = filedialog.askopenfilenames(master=self, defaultextension='*.*', multiple=1, parent=self,
                                                filetypes=(("all files", "*.*"), ("numpy files", "*.npy")),
                                                title='select image')
        if filenames:
            filename = filenames[0]
            self.filename = filename
            self.loadimage()

    def start_crop(self):
        if self.filename:
            if not self.current_rect:
                self.is_crop = 1
        else:
            tkinter.messagebox.showerror("Error", "please load new image")

    def delete_crop(self):
        self.canvas.delete(self.current_rect)
        self.croprect_start = None
        self.croprect_end = None
        self.current_rect = None

    def back(self):
        # back to last var of the image
        self.dicom_image = np.copy(self.prev_image)
        self.display_to_canvas()

    def save_tamper(self):
        if self.folder_path:
            self.tamper_number += 1
            save_path = self.generate_save_path()
            try:
                np.save(save_path, self.dicom_image)
            except:
                tkinter.messagebox.showerror("Error", "Error saving the tamper")
            else:
                self.delete_crop()
        else:
            tkinter.messagebox.showerror("Error", "for saving please make a crop and choose folder to save the crop in")

    def set_path(self):
        self.folder_path = filedialog.askdirectory()
        self.path_label_cont.set(self.folder_path)

    def generate_save_path(self):
        base_name = os.path.basename(self.filename)
        base_name_split = base_name.split('.')
        save_path = self.folder_path + '/' + '/' + base_name_split[0] + '_' + \
                    str(self.tamper_number)
        return save_path

    def inject(self):
        # injecting tumor to mammogram inside selected square
        if not self.current_rect:
            return None
        rect_cords = self.canvas.coords(self.current_rect)
        x1, y1, x2, y2 = rect_cords
        a = int(min(y2 - y1, x2 - x1))
        rect = self.dicom_image[int(y1):int(y1 + a), int(x1):int(x1 + a)]
        # normalize rect
        norm_rect = cv2.resize(rect, (self.norm_size, self.norm_size))
        condition_img = np.copy(norm_rect) / 1100
        condition_img[self.crop_frame:self.norm_size - self.crop_frame - 1, self.crop_frame:self.norm_size - 10 - 1] = 0
        input_layer_1 = np.expand_dims(condition_img, axis=-1)
        input_layer_1 = np.expand_dims(input_layer_1, axis=0)
        # implementing first GAN
        fake_layer_1 = self.generator_layer_1.predict(input_layer_1)
        # adding white gaussian noise to the middle of the first GAN output -#
        fake_layer_1[:, self.crop_frame:self.norm_size - self.crop_frame - 1,
        self.crop_frame:self.norm_size - self.crop_frame - 1] = \
            np.random.randn(1, self.norm_size - 2 * self.crop_frame - 1,
                            self.norm_size - 2 * self.crop_frame - 1, 1) * 0.003 + \
            fake_layer_1[:, self.crop_frame:self.norm_size - self.crop_frame - 1,
            self.crop_frame:self.norm_size - self.crop_frame - 1]
        # implementing second GAN
        fake_middle = self.generator_layer_2.predict(fake_layer_1)
        # pasting the second GAN output to the middle of the original image
        fake_middle = np.squeeze(fake_middle)
        fake_layer_2 = np.copy(condition_img)
        fake_layer_2[self.crop_frame:self.norm_size - self.crop_frame - 1, self.crop_frame:self.norm_size - 10 - 1] = \
            fake_middle
        # de-normalization of the image
        de_norm_fake = cv2.resize(fake_layer_2, (a, a)) * 1100
        # marge the tamper with the original mammogram
        self.prev_image = np.copy(self.dicom_image)
        self.dicom_image[int(y1):int(y1 + a), int(x1):int(x1 + a)] = self.marge_injection(de_norm_fake, rect)
        self.display_to_canvas()
        self.current_rect = None
        self.is_crop = 0

    def display_to_canvas(self):
        # displaying mammogram image to the canvas with color normalization
        mammo = np.copy(self.dicom_image)
        mammo[np.where(mammo > self.max_hu)] = self.max_hu
        mammo[np.where(mammo < self.min_hu)] = self.min_hu
        mammo = ((mammo - self.min_hu) / (self.max_hu - self.min_hu)) * 255
        img = Image.fromarray(mammo)
        self.photo_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tkinter.NW, image=self.photo_img)
        self.canvas.config(scrollregion=self.canvas.bbox(tkinter.ALL))

    def generate_distance_matrix(self, n):
        # Creates a nXn size matrix, where each element contains its distance from the center
        # The distance increases linearly from 0 to 1
        dis_arr = np.zeros((n, n))
        for i in range(n // 2):
            distance = n // 2 - i
            dis_arr[i, i:n - i] = distance
            dis_arr[n - 1 - i, i:n - i] = distance
            dis_arr[i:n - i, i] = distance
            dis_arr[i:n - i, n - 1 - i] = distance
        dis_arr = dis_arr / n
        return dis_arr

    def generante_weight_matrix(self, n, x):
        # Creates a nXn weight matrix, for weighted average
        distance_arr = self.generate_distance_matrix(n)
        G = 1 - np.power(distance_arr, 3)
        weigtht_matrix = np.divide(1, 1 + np.exp(np.divide(-(x - 400), 55))) * G
        return weigtht_matrix

    def marge_injection(self, fake, real):
        # marge the GAN output with the original image
        n = np.size(fake, 0)
        fake = fake + np.random.randn(n, n) * 10  # adding gaussian noise to the fake image
        weight_matrix = self.generante_weight_matrix(n, fake) # weighted average between real and fake image
        marge = np.multiply(fake, weight_matrix) + np.multiply(real, 1 - weight_matrix)

        return marge.astype(int)


def main():
    root = tkinter.Tk()
    filename = None
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    a = Inj_GUI(root, filename=filename)
    a.master.title('Mammo-CGAN')
    a.mainloop()


if __name__ == '__main__':
    main()
