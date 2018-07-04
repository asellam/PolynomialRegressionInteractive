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
    #Initialization proc
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        #Gradient Descent Regression Learning Rate
        self.LearningRate=0.1
        #Maximum number of epochs the regression will last
        self.MaxEpochs=100
        #Width of the display canvas
        self.width=400
        #Height of the display canvas
        self.height=400
        #Margin between gui components
        self.margin=8
        #The scale: Number of physical pixels in a logical unit for 2D-Points'
        # coordinates
        self.bsize=40
        #Radius in pixels for point display
        self.psize=3
        #Position of the origin (0,0) in Physical pixels coordinates
        self.origin=[200,200]
        #Two lists, one for X coordinates and the second one is for Y coordinates
        self.dataset=[[],[]]
        #Polynomial Model Weights: Y=W0+W1*X^1+...+Wn*X^n
        self.WeightViews=[]
        #Tkinter stuff
        #Display the main window
        self.grid()
        #Create GUI components
        self.createWidgets()

    #OnClick event handler of the "Start Regression" button.
    #This will start the Polynomial Regression using tensorflow
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
        #Initialize Epochs count to 0
        self.Epoch=0
        #A timer core proc this will cause the regression code to run
        # in a seperate thread. Thus, the gui will not be blocked or grayed
        def TrainProc():
            #A while loop for regression epochs
            while(self.Epoch<self.MaxEpochs):
                #In this version Epochs Progress is displayed in the consol
                print("Epoch %d/%d"%(self.Epoch+1,self.MaxEpochs))
                #Run the train graph of tensorflow that is equivalent to a single
                #epoch of Regression
                session.run(train, {X: self.dataset[0], Y: self.dataset[1]})
                #Run tensorflow W and Loss to get new values of weights and loss
                self.W,self.Loss=session.run([W,Loss],{X:self.dataset[0],Y:self.dataset[1]})
                #Displaying This Epoch's loss in the consol
                print("    Loss: %.04f"%(self.Loss))
                #Displaying new weights in the gui
                for j in range(polyDegree+1):
                    self.WeightViews[j][2].set("%.02f"%(self.W[j]))
                #Increment Epochs count
                self.Epoch=self.Epoch+1
                #Sleep to give the user a chance to see changes in weights and loss
                tm.sleep(0.2)
                #Render the new polynom graph in TK Canvas
                self.render()
            #Enable the "Start Regression" button again
            self.start.config(state="normal")
            #Stop the timer once and for all
            TrainTimer.cancel()
        #Create a timer that fires a few moments later
        TrainTimer=Timer(0.2,TrainProc)
        #Start the Timer Thread
        TrainTimer.start()

    #This method will initialize weights and create gui component for their display
    #Must be Called each time the user selects a new type of polynom
    def makeWeights(self,polyDegree):
        #Destroy gui components for old weights
        for wv in self.WeightViews:
            wv[0].grid_forget()
            wv[1].grid_forget()
        #Clear weights' list
        self.W=[]
        self.WeightViews=[]
        #Make room for new weights
        for i in range(polyDegree+1):
            #picks a random number W
            W=rd.random()
            #add W to the list of weights
            self.W.append(W)
            #Tkinter stuff to create and initialize display for the weights
            # Make a TextVar linked to Tk Entry that displays this weight's values
            entryText=tk.StringVar()
            # Make a title tk label and a display tk entry for this weight
            self.WeightViews.append([ttk.Label(self,text="w%d"%(i)),ttk.Entry(self,textvariable=entryText),entryText])
            #Display the title tk label
            self.WeightViews[i][0].grid(row=2+i,column=1,padx=self.margin,pady=self.margin,sticky=tk.N+tk.E+tk.W)
            #Display the value tk entry
            self.WeightViews[i][1].grid(row=2+i,column=2,padx=self.margin,pady=self.margin,sticky=tk.N+tk.E+tk.W)
            #Display the initial weight value in the appropriate tk entry component
            entryText.set("%.02f"%(W))
        #Render the initial random polynom
        self.render()
        #This will make the regression Learning Rate smaller for polynoms with
        # higher degree
        self.LearningRate=pow(10,-polyDegree)

    #OnChange event handler for the list of polynom degrees
    def polyTypeChange(self,event):
        self.makeWeights(int(self.polyType.get()))

    #OnMouseDown event handler for the canvas component
    def mouseDown(self,event):
        #Compute logical coordinates of the soon-to-be-added point from screen
        #Physical coordinates
        X=(event.x-self.origin[0])/self.bsize
        Y=(event.y-self.origin[1])/self.bsize
        #Add the new 2d-Point to the dataset
        self.dataset[0].append(X)
        self.dataset[1].append(Y)
        #Render the new point on the canvas
        self.render()

    #Create GUI components
    def createWidgets(self):
        #A Combobox control that allows the user to select the degree of the
        # Polynomial Model
        self.polyType = ttk.Combobox(self,values=['1','2','3','4','5','6','7','8','9'])
        #A button to fire the Polynomial Regression
        self.start = ttk.Button(self,text="Start Regression",command=self.startRegression)
        #A canvas to display the 2D points, polynomial model and the cartesian
        # system's grid
        self.canvas = tk.Canvas(self, width=self.width,height=self.height)
        #Display the Polynomial Model degree's Combobox on the screen
        self.polyType.grid(row=0,column=1,columnspan=2,padx=self.margin,pady=self.margin,sticky=tk.S+tk.E+tk.W)
        #Display the start button on the screen
        self.start.grid(row=1,column=1,columnspan=2,padx=self.margin,pady=self.margin,sticky=tk.N+tk.E+tk.W)
        #Display the canvas on the screen
        self.canvas.grid(row=0,column=0,rowspan=12,padx=self.margin,pady=self.margin)
        #Set current Polynomial Model degree to 1
        self.polyType.current(0)
        #Make weights and their gui stuff (depending on the polynom degree)
        self.makeWeights(int(self.polyType.get()))
        #Add an OnChange event handler for the polynom degree Combobox
        self.polyType.bind("<<ComboboxSelected>>",self.polyTypeChange)
        #Add an OnMouseDown event handler for the canvas
        self.canvas.bind("<Button-1>",self.mouseDown)
        #Render the initial stuff on canvas (2D-points, polynomial model, ...)
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
        #SwapBuffers: FrontBuffer <- Backbuffer
        #Creates a Tkinter image equivalent to the PIL image that holds the Backbuffer
        # rendering result
        self.BackTk=ImageTk.PhotoImage(self.back)
        #Display The Backbuffer rendering result in the canvas (FrontBuffer)
        self.canvas.create_image(200,200,image=self.BackTk)

#Tkinter Application Class Stuff
app = Application()
app.master.title('Polynomial Regression Illustration')
app.mainloop()
