import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import cv2
import tkFileDialog as tff
import numpy
import matplotlib.pyplot as plt
import ttk
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import Tkinter as tk
from PIL import Image
import os

batch_size = 32

def buat_model(ep_model):
    epochs = ep_model
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

    model_candi = Sequential()

    model_candi.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150), activation='relu', padding='same'))
    model_candi.add(Dropout(0.2))
    model_candi.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model_candi.add(MaxPooling2D(pool_size=(2, 2)))
    model_candi.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_candi.add(Dropout(0.2))
    model_candi.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_candi.add(MaxPooling2D(pool_size=(2, 2)))
    model_candi.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model_candi.add(Dropout(0.2))
    model_candi.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model_candi.add(MaxPooling2D(pool_size=(2, 2)))
    model_candi.add(Flatten())
    model_candi.add(Dropout(0.2))
    model_candi.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model_candi.add(Dropout(0.2))
    model_candi.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model_candi.add(Dropout(0.2))
    model_candi.add(Dense(6, activation='softmax'))

    model_candi.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model_candi


if __name__ == "__main__":
    classout = ['Candi Borobudur','Candi Kalasan','Candi Mendut','Candi Prambanan','Candi Sari','Candi Sewu']
    answer = 1
    while answer > 0:
        print("=====MENU DeepLearning-Candi=====")
        print("1. Training model ")
        print("2. Test model")
        print("3. Klasifikasi citra")
        print("4. Evaluasi Model")
        print("0. Exit")
        answer = int(input('Silahkan pilih --> '))

        if answer == 1:
            epochs_t = int(input('Masukkan jumlah epochs --> '))
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

            train_generator = train_datagen.flow_from_directory(
                'dataset/train',
                target_size=(150, 150),
                batch_size=batch_size,
                class_mode='categorical')

            model = buat_model(epochs_t)
            start = time.time()
            model.fit_generator(
                train_generator,
                steps_per_epoch=2000 // batch_size,
                epochs=epochs_t)
            model.save_weights('bobot_train_candi_10.h5')
            end = time.time()
            print("Model memerlukan %0.2f detik untuk training" % (end - start))
        elif answer == 2:
            test_datagen = ImageDataGenerator(rescale=1. / 255)

            validation_generator = test_datagen.flow_from_directory(
                'dataset/validation',
                target_size=(150, 150),
                batch_size=batch_size,
                class_mode='categorical')

            model = buat_model(50)

            model.load_weights('bobot_train_candi.h5')

            start = time.time()
            scores = model.evaluate_generator(validation_generator,
                                     steps=800 // batch_size)
            end = time.time()
            print("Akurasi : %.2f%%" % (scores[1] * 100))
            print("Model memerlukan %0.2f detik untuk test" % (end - start))
        elif answer == 3:
            top = tk.Tk()
            filename = "blank.jpg"

            tk.Tk.wm_title(top, "Deep Learning Candi : Klasifikasi Citra")

            label = tk.Label(top, text="Klasifikasi Citra")
            label.pack(pady=10, padx=10)


            def close_window():
                top.destroy()
                top.quit()
                os.system('clear')

            def prediksiCitra():
                model = buat_model(50)

                model.load_weights('bobot_train_candi_100.h5')
                global filename
                global ima

                im = cv2.resize(ima, (150, 150)).astype(numpy.float32)
                im[:, :, 0] -= 103.939
                im[:, :, 1] -= 116.779
                im[:, :, 2] -= 123.68
                im = im.transpose((2, 0, 1))
                im = numpy.expand_dims(im, axis=0)
                pr = model.predict_classes(im, verbose=0)
                labelp['text'] = "Klasifikasi --> " + classout[(pr[0])]
                # print ("Prediksi --> " + classout[(pr[0])])

            def unggahCitra():
                filename = tff.askopenfilename()

                global canvas
                global img
                global button2
                global ima
                global labelp
                labelp.pack_forget()
                button2.pack_forget()
                canvas.get_tk_widget().pack_forget()
                img = Image.open(filename)
                ima = cv2.imread(filename)
                fig = plt.figure(figsize=(5, 5), dpi=80)
                plt.imshow(img)
                canvas = FigureCanvasTkAgg(fig, top)
                canvas.show()
                canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                button2 = ttk.Button(top, text="Klasifikasi!", command=lambda *args: prediksiCitra())
                button2.pack(padx=50)
                labelp = tk.Label(top, text="Klasifikasi --> ")
                labelp.pack(pady=20, padx=50)


            button1 = ttk.Button(top, text="Unggah Citra", command=lambda *args: unggahCitra())
            button1.pack()

            img = Image.open(filename)
            ima = cv2.imread(filename)
            fig = plt.figure(figsize=(5, 5), dpi=80)
            plt.imshow(img)
            canvas = FigureCanvasTkAgg(fig, top)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            button2 = ttk.Button(top, text="Klasifikasi!", command=lambda *args: prediksiCitra())
            button2.pack(padx=50)
            labelp = tk.Label(top, text="Klasifikasi --> ")
            labelp.pack(pady=20, padx=50)

            top.protocol("WM_DELETE_WINDOW", close_window)
            top.mainloop()
        elif answer == 4:
            print(model.summary())



