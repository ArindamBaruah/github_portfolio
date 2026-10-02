from tkinter import*
class mymenudemo:
    def __init__(self, root):
        #create menubar
        self.menubar = Menu(root)

        #attach the menubar to the root window
        root.config(menu=self.menubar)

        #create file menu
        self.filemenu=Menu(root, tearoff=0)

        #create menu items in file menu
        self.filemenu.add_command(label='New', command=self.donothing)
        self.filemenu.add_command(label='open', command=self.donothing)
        self.filemenu.add_command(label='save', command=self.donothing)

        #add a horizontal line as seperator
        self.filemenu.add_separator()

        #create another itme below seperator
        self.filemenu.add_command(label='Exit', command=root.destroy)

        #add menu with the name file
        self.menubar.add_cascade(label='File', menu=self.filemenu)

        #crete edit menu
        self.editmenu = Menu(root,tearoff=0)

        #create menu item in edit menu
        self.editmenu.add_command(label='Cut', command=self.donothing)
        self.editmenu.add_command(label='Copy', command=self.donothing)
        self.editmenu.add_command(label='paste', command=self.donothing)

        #add the edit menu with a name edit to the menubar
        self.menubar.add_cascade(label='Edit', menu=self.editmenu)

    def donothing(self):
        pass
root = Tk()

        #title for root window
root.title('A menu example')

        #crete object
obj=mymenudemo(root)

        #definie the size of root window
root.geometry('600x350')

root.mainloop()
