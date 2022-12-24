import pandas as pd
import tkinter as tk
#from tkinter import *
from tkinter import *
fields = ('maths', 'physics', 'chemistry', 'aptitude','branch')
def prediction(entries):
   m = float(entries['maths'].get())
   p = float(entries['physics'].get())
   c = float(entries['chemistry'].get())
   a = float(entries['aptitude'].get())
   result=clf_pf.predict([[m,p,c,a]])
   branch=["CSE","ECE","EEE","CHEMICAL","MECH"]
   entries['branch'].delete(0,END)
   entries['branch'].insert(0,branch[result[0]-1])
def makeform(root, fields):
   entries = {}
   for field in fields:
      row = Frame(root)
      lab = Label(row, width=22, text=field+": ", anchor='w')
      ent = Entry(row)
      ent.insert(0,"0")
      row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
      lab.pack(side = LEFT)
      ent.pack(side = RIGHT, expand = YES, fill = X)
      entries[field] = ent
   return entries


if __name__ == '__main__':
   csvfile = pd.read_csv('C://Users//ChandanA//OneDrive//Desktop//others//marks.csv')
   table = csvfile
   att_names = ['maths', 'physics', 'chemistry', 'aptitude']
   X = table[att_names]
   Y = table['branch']
   from sklearn.model_selection import train_test_split

   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

   from sklearn.naive_bayes import GaussianNB

   clf_pf = GaussianNB()
   clf_pf.fit(X_train, Y_train)

   from tkinter import ttk
   from tkinter import *

   root = tk.Tk()
   ents = makeform(root, fields)
   root.bind('<Return>', (lambda event, e = ents: fetch(e)))
   b1 = Button(root, text = 'give branch',
   command=(lambda e = ents: prediction(e)))
   b1.pack(side = LEFT, padx = 5, pady = 5)
   b2 = Button(root, text = 'Quit', command = root.quit)
   b2.pack(side = LEFT, padx = 5, pady = 5)
   root.mainloop()



