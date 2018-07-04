#-------------------------------------------
#- Gradient Descent Polynomial Regression  -
#-------------------------------------------
#- By: SELLAM Abdellah                     -
#- PhD Student at USTHB-LRIA/ALGERIA       -
#-------------------------------------------
#- Creation Date: 04-07-2018               -
#-------------------------------------------
#- This program illustrates the use of     -
#- Gradient Descent to find a polynom that -
#- approximates a 2D-Points' dataset       -
#- Rendering is done with Tkinter and PIL  -
#- Polynomial Regression is done using     -
#- Tensorflow                              -
#-------------------------------------------
import time as tm
import random as rd
import tkinter as tk
from PIL import Image
from PIL import ImageTk
from tkinter import ttk
import tensorflow as tf
from PIL import ImageDraw
from threading import Timer

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid()
        self.createWidgets()

    def startRegression(self):
        #Disable the start button until regression is over
        self.start.config(state="disable")
        #The degree n of the polynomial model
        polyDegree=int(self.polyType.get())
        #Polynomial Model defined as: Y=W0+W1*X^1+...+Wn*X^n
        W=[]
        #Creating tensorflow polynomial model's variables
        for i in range(polyDegree+1):
            W.append(tf.Variable(self.W[i],dtype=tf.float32))
        #X data placeholder
        X=tf.placeholder(tf.float32)
        #Y data placeholder
        Y=tf.placeholder(tf.float32)
        #Creating tensorflow polynomial model's graph
        PolyModel=W[0]
        for i in range(polyDegree):
            PolyModel=PolyModel+W[i+1]*tf.pow(X,tf.constant(i+1,dtype=tf.float32))
        #Defining the loss function for the gradient descent algorithm
        SErr=tf.square(PolyModel-Y)
        Loss=tf.reduce_mean(SErr)
        #Gradient Descent Optimizer
        optimizer=tf.train.GradientDescentOptimizer(self.LearningRate)
        train=optimizer.minimize(Loss)
        #Makin a Tensorflow Running Session
        session = tf.Session()
        #A Tensorflow Global Variables' initializer (To initialize Polynomial
        # Model's weights)
        init = tf.global_variables_initializer()
        session.run(init)
        self.Epoch=0
        def TrainProc():
            while(self.Epoch<self.MaxEpochs):
                print("Epoch %d/%d"%(self.Epoch+1,self.MaxEpochs))
                session.run(train, {X: self.dataset[0], Y: self.dataset[1]})
                self.W,self.Loss=session.run([W,Loss],{X:self.dataset[0],Y:self.dataset[1]})
                for j in range(polyDegree+1):
                    self.WeightViews[j][2].set("%.02f"%(self.W[j]))
                    print("W%d: %.02f"%(j,self.W[j]))
                self.Epoch=self.Epoch+1
                tm.sleep(0.2)
                self.render()
            self.start.config(state="normal")
            TrainTimer.cancel()
        TrainTimer=Timer(0.2,TrainProc)
        TrainTimer.start()

    def makeWeights(self,polyDegree):
        for wv in self.WeightViews:
            wv[0].grid_forget()
            wv[1].grid_forget()
        self.W=[]
        self.WeightViews=[]
        for i in range(polyDegree+1):
            W=rd.random()
            self.W.append(W)
            entryText=tk.StringVar()
            self.WeightViews.append([ttk.Label(self,text="w%d"%(i)),ttk.Entry(self,textvariable=entryText),entryText])
            self.WeightViews[i][0].grid(row=2+i,column=1,padx=self.margin,pady=self.margin,sticky=tk.N+tk.E+tk.W)
            self.WeightViews[i][1].grid(row=2+i,column=2,padx=self.margin,pady=self.margin,sticky=tk.N+tk.E+tk.W)
            entryText.set("%.02f"%(W))
        self.render()

    def polyTypeChange(self,event):
        self.makeWeights(int(self.polyType.get()))

    def mouseDown(self,event):
        if event.x>0 and event.x<self.width and event.y>0 and event.y<self.height:
            X=(event.x-self.origin[0])/self.bsize
            Y=(event.y-self.origin[1])/self.bsize
            self.dataset[0].append(X)
            self.dataset[1].append(Y)
            self.render()

    def createWidgets(self):
        self.LearningRate=0.001
        self.MaxEpochs=100
        self.width=400
        self.height=400
        self.margin=8
        self.bsize=40
        self.psize=3
        self.origin=[200,200]
        self.dataset=[[],[]]
        self.WeightViews=[]
        self.polyType = ttk.Combobox(self,values=['1','2','3','4','5','6','7','8','9'])
        self.start = ttk.Button(self,text="Start Regression",command=self.startRegression)
        self.canvas = tk.Canvas(self, width=self.width,height=self.height)
        self.polyType.grid(row=0,column=1,columnspan=2,padx=self.margin,pady=self.margin,sticky=tk.S+tk.E+tk.W)
        self.start.grid(row=1,column=1,columnspan=2,padx=self.margin,pady=self.margin,sticky=tk.N+tk.E+tk.W)
        self.canvas.grid(row=0,column=0,rowspan=12,padx=self.margin,pady=self.margin)
        self.polyType.current(0)
        self.makeWeights(int(self.polyType.get()))
        self.polyType.bind("<<ComboboxSelected>>",self.polyTypeChange)
        self.canvas.bind("<Button-1>",self.mouseDown)
        self.render()

    #The following function does the drawing of the dataset of 2D-Points
    #Along with the Cartesian System's Grid
    def render(self):
        #Create a back-buffer (Double buffering to avoid fliker)
        self.back=Image.new(mode="RGB",size=(self.width,self.height),color="white")
        draw=ImageDraw.Draw(self.back)
        #Background Rectangle
        draw.polygon([(2,2),(self.width-2,2),(self.width-2,self.height-2),(2,self.height-2)],fill='white',outline='black')
        #Negative X Lines
        X=self.origin[0]-self.bsize
        while X>0:
            draw.line([(X,0),(X,self.height)],fill='gray')
            X=X-self.bsize
        #Positive X Lines
        X=self.origin[0]+self.bsize
        while X<self.width:
            draw.line([(X,0),(X,self.height)],fill='gray')
            X=X+self.bsize
        #X=0
        X=self.origin[0]
        draw.line([(X,0),(X,self.height)],fill='black')
        #Negative Y Lines
        Y=self.origin[1]-self.bsize
        while Y>0:
            draw.line([(0,Y),(self.width,Y)],fill='gray')
            Y=Y-self.bsize
        #Positive Y Lines
        Y=self.origin[1]+self.bsize
        while Y<self.height:
            draw.line([(0,Y),(self.width,Y)],fill='gray')
            Y=Y+self.bsize
        #Y=0
        Y=self.origin[1]
        draw.line([(0,Y),(self.width,Y)],fill='black')
        #Origin
        X=self.origin[0]
        Y=self.origin[1]
        draw.polygon([(X-self.psize,Y),(X,Y-self.psize),(X+self.psize),(Y,X),(Y+self.psize)],fill='blue',outline='blue')
        #Polynomial model
        X0=0
        Y0=0
        for X1 in range(1,self.width+1):
            x=(X1-self.origin[0])/self.bsize
            n=0
            y=0
            for w in self.W:
                y=y+w*pow(x,n)
                n=n+1
            Y1=int(self.origin[1]+y*self.bsize)
            draw.line([(X0,Y0),(X1,Y1)],fill='blue')
            X0=X1
            Y0=Y1
        #2D-Points' dataset
        N=min(len(self.dataset[0]),len(self.dataset[1]))
        for i in range(N):
            X=self.origin[0]+self.dataset[0][i]*self.bsize
            Y=self.origin[1]+self.dataset[1][i]*self.bsize
            draw.polygon([(X-self.psize,Y),(X,Y-self.psize),(X+self.psize,Y),(X,Y+self.psize)],fill='red',outline='red')
        self.BackTk=ImageTk.PhotoImage(self.back)
        self.canvas.create_image(200,200,image=self.BackTk)

app = Application()
app.master.title('Polynomial Regression Illustration')
app.mainloop()
