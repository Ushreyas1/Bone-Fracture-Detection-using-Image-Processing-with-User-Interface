from asyncio import windows_events
import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import os
import cv2
import numpy as np
from pre_process import _reshape_img, get_model
import tkinter.ttk as ttk
import time
import shutil



def window2():
    window.destroy()
    window1 = tk.Tk()
    window1.title('second Design')
    window1.geometry("1550x900")
    #frame = LabelFrame(window1,text='Hello_World',bg='white',font=(30))
    #frame.pack(expand=True, fill=BOTH)
    window1.config(bg="#690c0c")
    def upload_file():
        global img
        f_types = [('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        print(filename)
        
        progress_bar = ttk.Progressbar(window1, orient="horizontal",mode="determinate", maximum=100, value=0)
        progress_bar.pack()
        progress_bar['value'] = 0
        shutil.copyfile(filename, "result_img/original.png")

    

        
#======================================================
        frame11 = LabelFrame(window1,bg='#690c0c',text='',font=(30))
        frame11.pack(expand=True,fill=BOTH)

        
        frame2 = LabelFrame(frame11,bg='#3366ff',fg='white',text='X-ray Image',font=("Times New Roman", 20))
        frame2.pack(expand=True, side=LEFT)
        img2 = Image.open("result_img/original.png")
        test2 = ImageTk.PhotoImage(img2)
        imglabe2 = tk.Label(frame2,  image=test2, compound='center')
        imglabe2.pack()
#======================================================
        
        window1.update()
        progress_bar['value'] += 20
        # img = ImageTk.PhotoImage(file=filename)
        # b2 =tk.Button(my_w,image=img) # using Button 
        # b2.grid(row=3,column=1)
        
        """
        Currently, `img_name` will be used to get resized image from `images/resized` folder
        and original image from `images/Fractured Bone` so it expects the same named image file
        to be available in both the folders.

        All names starting with F{n} are available in both the folders. 1<= n <=100
        """

        model_name= "ridge_model"
        img_name = os.path.basename(filename)
        #img_name="F1"
        print(img_name)
        img_file= 'images/resized/{}'.format(img_name)
        orig_img= 'images/Fractured Bone/{}'.format(img_name)

        #for image read
        try:
            img_t=cv2.imread(img_file,cv2.IMREAD_COLOR)
            img=cv2.imread(orig_img,cv2.IMREAD_COLOR)
            print(img)
            shape= img.shape
        except (AttributeError,FileNotFoundError):
            try:
                img_t=cv2.imread(img_file,cv2.IMREAD_COLOR)
                img=cv2.imread(orig_img,cv2.IMREAD_COLOR)
                shape=img.shape
            except (AttributeError,FileNotFoundError):
                img_t=cv2.imread(img_file,cv2.IMREAD_COLOR)
                img=cv2.imread(orig_img,cv2.IMREAD_COLOR)
                shape=img.shape

            #else: raise FileNotFoundError("No image file {img_file}.jpg or {img_file}.JPG".format(img_file=img_file))
        #else:
        #	raise FileNotFoundError("No image file {img_file}.jpg or {img_file}.JPG".format(img_file=img_file))


        #details of Imge
        print("\nShape: ",shape)
        print("\nSize: ",img.size)
        print("\nDType: ",img.dtype)

        #==============Manual edge ditect=====================
        def segment_img(_img,limit):
            for i in range(0,_img.shape[0]-1):
                for j in range(0,_img.shape[1]-1): 
                    if int(_img[i,j+1])-int(_img[i,j])>=limit:
                        _img[i,j]=0
                    elif(int(_img[i,j-1])-int(_img[i,j])>=limit):
                        _img[i,j]=0
            
            return _img
        #======================================================

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #for i in range(0,gray.shape[0]):
        #	for j in range(0,gray.shape[1]): 
        #		if (int(gray[i,j]))<=100:
        #			gray[i,j]=100

        #gray=segment_img(gray,15)
        cv2.imwrite("result_img/gconvert.png",gray)
        #gray.save()
        #cv2.imshow("GrayEdited",gray)
#======================================================
        # ClearButton.config(font=("Times New Roman", 20))
        frame3 = LabelFrame(frame11,bg='#3366ff',fg='white',text='Gray Converted',font=("Times New Roman", 20))
        frame3.pack(expand=True, side=LEFT)
        img11 = Image.open(r"result_img/gconvert.png")
        test3 = ImageTk.PhotoImage(img11)
        imglabel = tk.Label(frame3,  image=test3, compound='center')
        imglabel.pack()
#======================================================
        
        window1.update()
        progress_bar['value'] += 20
        
        median = cv2.medianBlur(gray,5)

        model= get_model(model_name)
        pred_thresh= model.predict([_reshape_img(img_t)])
        print(pred_thresh[0])
        bool,threshold_img=cv2.threshold(median,pred_thresh[0],255,cv2.THRESH_BINARY)
        #blur=cv2.GaussianBlur(threshold_img,(7,7),0)
        #cv2.imshow("threshold",threshold_img)

        
        cv2.imwrite("result_img/threshold.png",threshold_img)
        img1 = Image.open(r"result_img/threshold.png")
#======================================================        
        frame4 = LabelFrame(frame11,bg='#3366ff',fg='white',text='Threshold Image',font=("Times New Roman", 20))
        frame4.pack(expand=True, side=LEFT)    
        test4 = ImageTk.PhotoImage(img1)
        imglabe4 = tk.Label(frame4,  image=test4, compound='center')
        imglabe4.pack()
#======================================================        
        window1.update()
        progress_bar['value'] += 20

        initial=[]
        final=[]
        line=[]
        #count=[]
        #for i in range(0,256):
        #	count.append(0)

        for i in range(0,gray.shape[0]):
            tmp_initial=[]
            tmp_final=[]
            for j in range(0,gray.shape[1]-1):
                #count[gray[i,j]]+=1
                if threshold_img[i,j]==0 and (threshold_img[i,j+1])==255:
                    tmp_initial.append((i,j))
                    #img[i,j]=[255,0,0]
                if threshold_img[i,j]==255 and (threshold_img[i,j+1])==0:
                    tmp_final.append((i,j))
                    #img[i,j]=[255,0,0]
            
            x= [each for each in zip(tmp_initial,tmp_final)]
            x.sort(key= lambda each: each[1][1]-each[0][1])
            try:
                line.append(x[len(x)-1])
            except IndexError: pass

        #print(count)


        err= 15
        danger_points=[]

        #store distances
        dist_list=[]

        for i in range(1,len(line)-1):
            dist_list.append(line[i][1][1]-line[i][0][1])
            try:
                prev_= line[i-3]
                next_= line[i+3]

                dist_prev= prev_[1][1]-prev_[0][1]
                dist_next= next_[1][1]-next_[0][1]
                diff= abs(dist_next-dist_prev)
                if diff>err:
                    #print("Dist: {}".format(abs(dist_next-dist_prev)))
                    #print(line[i])
                    data=(diff, line[i])
                    #print(data)
                    if len(danger_points):
                        prev_data=danger_points[len(danger_points)-1]
                        #print(prev_data)
                        #print("here1....")
                        if abs(prev_data[0]-data[0])>2 or data[1][0]-prev_data[1][0]!=1:
                            #print("here2....")
                            print(data)
                            danger_points.append(data)
                    else:
                        print(data)
                        danger_points.append(data)
            except Exception as e:
                print(e)
                pass

            #print(each)
            start,end= line[i]
            #raise ZeroDivisionError
            mid=int((start[0]+end[0])/2),int((start[1]+end[1])/2)
            #img[mid[0],mid[1]]=[0,0,255]
        
            
        bone="Healthy Bone"
        for i in range(0,len(danger_points)-1,2):
            bone="Bone is Fractured"
            try:
                start_rect=danger_points[i][1][0][::-1]
                start_rect=(start_rect[0]-40, start_rect[1]-40)
          
                end_rect= danger_points[i+1][1][1][::-1]
                end_rect= (end_rect[0]+40, end_rect[1]+40)
                
                cv2.rectangle(img,start_rect,end_rect,(0,0,255),2)
            except:
                print("Pair not found")
          
        #blur= cv2.GaussianBlur(img,(5,5),0)

        import matplotlib.pyplot as plt
        import numpy as np

        #fig, (ax1, ax2)= plt.subplots(2,1)

        fig2, ax3= plt.subplots(1,1)

        #fig3, ax1= plt.subplots(1,1)
        
        x= np.arange(1,gray.shape[0]-1)
        y= dist_list
        print(x)
        print(y)
        #print(len(x),len(y))

        cv2.calcHist(gray,[0],None,[256],[0,256])
        
        #if(len(x)==len(y)):
            #ax1.plot(x,y)
        #else:
            #print("sdkjfhsdkj")
        
        #try:
            #ax1.plot(x,y)
        #except:
            #print("Could not plot")
        img= np.rot90(img)
        #ax2.imshow(img)
        cv2.imwrite("result_img/finalplot.png",img)
#======================================================        
        frame22 = LabelFrame(window1,bg='#690c0c',text='',font=(30))
        frame22.pack(expand=True, fill=BOTH)
        img5 = Image.open(r"result_img/finalplot.png")
        frame5 = LabelFrame(frame22,bg='#3366ff',fg='white',text='Fracture Detected',font=("Times New Roman", 20))
        frame5.pack(expand=True, side=LEFT)    
        test5 = ImageTk.PhotoImage(img5)
        imglabe5 = tk.Label(frame5,  image=test5, compound='center')
        imglabe5.pack()
  
        frame6 = LabelFrame(frame22,bg='white',fg='red',text='Fracture Detected Output', width=700, height=500,font=("Times New Roman", 20))
        frame6.pack(expand=True, side=LEFT)    
        imglabe5 = tk.Label(frame6,  text=str(bone), compound='center')
        imglabe5.config(fg="green")
        imglabe5.config(bg="white")
        imglabe5.pack()
        def exit():
            window1.destroy()
        
        def window3():
            window1.destroy()

            window2 = tk.Tk()
            window2.title('Bone')
            window2.geometry("1550x900")
            #frame = LabelFrame(window1,text='Hello_World',bg='white',font=(30))
            #frame.pack(expand=True, fill=BOTH)
            window2.config(bg="#3366ff")
            
            #image2 = Image.open(r"BoneD.jpg")
            #test2 = ImageTk.PhotoImage(image2)
            #imglabel2 = tk.Label(window2,  image=test2, compound='center')
            #imglabel2.pack()

            
            
            frame23 = LabelFrame(window2,bg='#3366ff',text='',font=(30))
            frame23.pack(expand=True, fill=BOTH)

            projectnamelable=tk.Label(frame23,  text="", compound='center')
            projectnamelable.pack()
            projectnamelable.config(font=("fantasy", 30))
            projectnamelable.config(fg="Red")
            projectnamelable.config(bg="#3366ff")

            image2 = Image.open(r"BoneD.jpg")
            test2 = ImageTk.PhotoImage(image2)
            imglabel2 = tk.Label(frame23,  image=test2, compound='center')
            imglabel2.pack()
            
            window2.mainloop()
 

        continueButton=tk.Button(frame22,text='Close',height=2, width=15,command=exit, bd=5, highlightthickness=0)
        continueButton.config(fg="white")
        continueButton.config(font=("Times New Roman", 10))
        continueButton.config(bg="#ff6699")
        continueButton.pack(side=tk.RIGHT,anchor="n",pady=20,padx=300)
            
            

        

        if(bone=='Bone is Fractured'):   
            continueButton=tk.Button(frame6,text='Health Care',command=window3)
            continueButton.config(fg="white")
            continueButton.config(font=("Times New Roman", 10))
            continueButton.config(bg="green")
            continueButton.pack(side=BOTTOM)
            
        
#======================================================        
        window1.update()
        progress_bar['value'] += 40
        window1.update()
        #count= range(256)
        #ax3.hist(count, 255, weights=count, range=[0,256])
        ax3.hist(gray.ravel(),256,[0,256])
        
        plt.show()
        window1.update()
        progress_bar['value'] += 20
        window1.update()



        

    test2 = ImageTk.PhotoImage(file="C:/bone/bb.jpg")
    frame = LabelFrame(window1,bg='#3366ff',text='',font=(30))
    frame.pack(expand=True, fill=BOTH)
    background_label = Label(frame, image=test2)
    background_label.place(x=0, y=0, relwidth=1, relheight=1) 

    
    
    browseButton=tk.Button(frame,text='Select Image',height=2, width=15,command=upload_file , bd=5, highlightthickness=0)
    browseButton.config(fg="white")
    browseButton.config(bg="green")
    browseButton.config(font=("Times New Roman", 10))
    browseButton.pack(side=tk.LEFT,anchor="n",pady=20,padx=350)

    
    ClearButton = tk.Button(frame, text='Back',height=2, width=15, command=exit, bd=5, highlightthickness=0)
    ClearButton.config(fg="white")
    ClearButton.config(bg="red")
    ClearButton.config(font=("Times New Roman", 10))
    ClearButton.pack(side=tk.RIGHT,anchor="n",pady=20,padx=300)
    
    window1.mainloop()

     


window = tk.Tk()
window.title('First Design')
window.geometry("1550x900")

  
test2 = ImageTk.PhotoImage(file="C:/bone/bb.jpg")
frame = LabelFrame(window,text='',bg='#f0f0f0',font=(30))
frame.pack(expand=True, fill=BOTH)
background_label = Label(frame, image=test2)
background_label.place(x=0, y=0, relwidth=1, relheight=1)  # This will make the image cover the entire frame


# headinglable=tk.Label(frame,  text="", compound='center')
# headinglable.pack()
# headinglable.config(font=("Courier", 25))
# headinglable.config(fg="BLACK")
# headinglable.config(bg="#050505")

projectnamelable=tk.Label(frame,  text="Bone Fracture Detection", compound='center')
projectnamelable.pack(pady=50)

projectnamelable.config(font=("Times New Roman", 50))
projectnamelable.config(fg="Red")
projectnamelable.config(bg="#050505")


# image2 = Image.open(r"BoneD.jpg")
# test2 = ImageTk.PhotoImage(image2)
# imglabel2 = tk.Label(frame,  image=test2, compound='center')
# imglabel2.pack()

# projectnamelable=tk.Label(frame,  text="", compound='center')
# projectnamelable.pack()
# projectnamelable.config(font=("fantasy", 25))
# #projectnamelable.config(fg="Red")
# projectnamelable.config(bg="#050505")


continueButton=tk.Button(frame, text='Continue',   height=2, width=15, command=window2, bd=5, highlightthickness=0)
continueButton.config(fg="white")
continueButton.config(bg="red")
continueButton.config(font=("Times New Roman", 10))
#continueButton.config(bg="#ff6699")
continueButton.pack(pady=150)
# height=2, width=15,

window.mainloop()






    



