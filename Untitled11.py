###############CREATE CONNECTION WITH SQL########################################
# connect() it open /establish a neww connector, it returns object representing the connection as in this my db
import mysql.connector
mydb=mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="myrootpassword",
    )
print(mydb)

#to check connection is opened or not
if (mydb.is_connected()):
    print("connectd")
else:
    print(" not connectd")
###########################################################################################
#You can create Cursor object using the cursor() method of the Connection object/class
#this cursor method is used to given sql queries
#we need cursor object so we can call execute () method
#my_cursor is cursor object
########CREATE CURSOR############################
my_cursor=mydb.cursor()

############### COMMAND TO CREATE THE DATABASE###############################

#my_cursor.execute("CREATE DATABASE test")

#############################SHOW ALL DATABASE###############################
my_cursor.execute("SHOW DATABASES")
for db in my_cursor:
    print(db[0])                      # to see all the databases 
#mydb.close()     # to close the connection 


# In[37]:


####################OPEN CONNECTION WITH SPECIFILC DATABASE
import mysql.connector
mydb=mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="myrootpassword",
    database="classtest"    # here we are opening the connection with particular database
    )
mycursor = mydb.cursor()    # we are creating the cursor object so that we can call execute method


########################TO SEE ALL TABLES IN DATABASE######################################
mycursor.execute("SHOW TABLES")
for i in mycursor:
    print(i)

###############################CREATE TABLE #################################
'''
mycursor.execute("CREATE TABLE employee (  ename VARCHAR(255), eaddress VARCHAR(255))")
'''
###################################add unique column as primary key, if table already avaliable use alter table############
'''
mycursor.execute("ALTER TABLE employee ADD COLUMN uid INT AUTO_INCREMENT PRIMARY KEY")
'''
#################################################################################################
'''
mycursor.execute("INSERT INTO employee(ename,eaddress) VALUES ('monu','hsp')")
mydb.commit()
'''
###############################ANOTHER WAY TO INSERT THE ITEMS  with execute function #######################
'''
sq="INSERT INTO employee(ename,eaddress) VALUES (%s, %s)"
va=("bhatia", "jal")
mycursor.execute(sq,va)
mydb.commit()
'''
########insert many lines at a one time with execute many function ############
'''
sq="INSERT INTO employee(ename,eaddress) VALUES (%s, %s)"
va=[("bhatia", "jal"),("el","jal"),("ja","hiii")]
mycursor.executemany(sq,va)
mydb.commit()
'''



####################################

#######TO SELECT ALL RECORD FROM TABLE##############

mycursor.execute("SELECT * FROM employee")
myresult = mycursor.fetchall()
for x in myresult:
    print(x)

###############TO SELECT PARTICULAR COULMN ###############
'''
mycursor.execute("SELECT ename, eaddress FROM employee")
myresult = mycursor.fetchall()

for x in myresult:
    print(x)
'''


# In[ ]:


from tkinter import *
import mysql.connector
mydb=mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="myrootpassword",
    )
root=Tk()
Label(root,text="username").grid(row=0,column=1)
Label(root,text="password").grid(row=1, column=1)
Entry(root).grid(row=0,column=2)
Entry(root).grid(row=1,column=2)


def savedata():
    my_cursor=mydb.cursor()
    #my_cursor.execute("CREATE DATABASE Login")
    
    
    

Button(root, text="Save", command=savedata).grid(row=3,column=2)
root.mainloop()



# In[ ]:




