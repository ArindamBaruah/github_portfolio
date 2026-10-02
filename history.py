y_cord = int((screen_height/2) - (height/2))
root.geometry("{}x{}+{}+{}".format(width, height, x_cord, y_cord))
def single_value():
import tkinter as tk
from tkinter import simpledialog
answer = simpledialog.askstring('Input', 'What is 2 + 2?',parent=root)
if answer is not None:
lbl_message.config(text=answer)
else:
lbl_message.config(text='You did not enter an answer!')
from tkinter import font          # This gives access to fonts.
from tkinter import messagebox    # This gives access to message boxes.
from tkinter import simpledialog  # Used for single value data entry.
from tkinter import filedialog    # Returns the path to a file.
from tkinter import colorchooser
import tkinter as tk        
root = tk.Tk()
root.title("Using Dialog Boxes")
root.configure(bg='lightyellow')
menu_bar = tk.Menu(root)
message_menu = tk.Menu(menu_bar, tearoff=0)
message_menu.add_command(label='Single value dialog',command=single_value)
menu_bar.add_cascade(label='Messages',menu=message_menu,command=quit)
# The Exit menu item can now be added.
menu_bar.add_cascade(label='Exit',command=quit)
# This adds the frame that holds the message label.
message_frame = tk.Frame(root)
width, height = 500, 400
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
center_window_on_screen()
lbl_font = font.Font(family='Georgia',size='18',weight='bold')
lbl_message = tk.Label(message_frame,text='This is where messages appear.',font=lbl_font,bg='brown', fg='lightyellow',wraplength=500)
lbl_message.pack()
message_frame.pack(pady=10)
root.config(menu=menu_bar)
root.mainloop()
from ccp import center_window_on_screen
from ccp import single_value
from ccp import center_window_on_screen
from ccp import single_value
from ccp import center_window_on_screen
from ccp import single_value
from tkinter import font          # This gives access to fonts.
%runfile C:/Users/RISHI/.spyder-py3/temp.py --wdir
answer

## ---(Wed Sep 11 02:04:27 2024)---
from temp import predict_churn
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df = pd.read_csv("C:\\Users\\Rishi\\Downloads\\Telco-Customer-Churn.csv")
df.sample(6)
df.drop("customerID",axis="columns",inplace=True)
df[pd.to_numeric(df.TotalCharges,errors="coerce").isnull()]
df.shape
df1=df[df.TotalCharges!=" "]
df1.shape
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
df1.replace("No internet service","No",inplace=True)
df1.replace("No phone service","No",inplace=True)
Y_N_columns=["Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling","Churn"]
for col in Y_N_columns: df1[col].replace({"Yes":1,"No":0},inplace=True)
df1.sample(6)
df1["gender"].replace({"Female":1,"Male":0},inplace=True)
df2=pd.get_dummies(data=df1, columns=["InternetService","Contract","PaymentMethod"])
df2.sample(6)
df2.dtypes
scale_columns=["tenure","MonthlyCharges","TotalCharges"]
import sklearn
from sklearn.preprocessing import MinMaxScaler
df2[scale_columns] = MinMaxScaler().fit_transform(df2[scale_columns])
pd.set_option("display.max_columns", None)
df2.sample(6)
X=df2.drop("Churn",axis="columns")
Y=df2["Churn"]
from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=5)
#X_train.shape
#X_test.shape
import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([keras.layers.Dense(20, input_shape=(26,), activation="relu"),keras.layers.Dense(15, activation="relu"),keras.layers.Dense(1, activation="sigmoid")])
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#model.fit(X_train,Y_train,epochs=100)
#model.evaluate(X_test,Y_test)
from imblearn.combine import SMOTEENN
sm = SMOTEENN()
X_resampled, Y_resampled = sm.fit_resample(X,Y)
Xr_train,Xr_test,Yr_train,Yr_test=train_test_split(X_resampled, Y_resampled,test_size=0.2)
model.fit(Xr_train,Yr_train,epochs=100)
model.evaluate(Xr_test,Yr_test)
Xr_test = Xr_test.reset_index(drop = True)
y_pred = model.predict(Xr_test)
predict_churn(y_pred[0])
from temp import predict_churn
predict_churn(y_pred[0])
from temp import predict_churn
predict_churn(y_pred[0])
from temp import predict_churn
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df = pd.read_csv("C:\\Users\\Rishi\\Downloads\\Telco-Customer-Churn.csv")
df.sample(6)
df.drop("customerID",axis="columns",inplace=True)
df[pd.to_numeric(df.TotalCharges,errors="coerce").isnull()]
df.shape
df1=df[df.TotalCharges!=" "]
df1.shape
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
df1.replace("No internet service","No",inplace=True)
df1.replace("No phone service","No",inplace=True)
Y_N_columns=["Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling","Churn"]
for col in Y_N_columns: df1[col].replace({"Yes":1,"No":0},inplace=True)
df1.sample(6)
df1["gender"].replace({"Female":1,"Male":0},inplace=True)
df2=pd.get_dummies(data=df1, columns=["InternetService","Contract","PaymentMethod"])
df2.sample(6)
df2.dtypes
scale_columns=["tenure","MonthlyCharges","TotalCharges"]
import sklearn
from sklearn.preprocessing import MinMaxScaler
df2[scale_columns] = MinMaxScaler().fit_transform(df2[scale_columns])
pd.set_option("display.max_columns", None)
df2.sample(6)
X=df2.drop("Churn",axis="columns")
Y=df2["Churn"]
from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=5)
#X_train.shape
#X_test.shape
import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([keras.layers.Dense(20, input_shape=(26,), activation="relu"),keras.layers.Dense(15, activation="relu"),keras.layers.Dense(1, activation="sigmoid")])
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#model.fit(X_train,Y_train,epochs=100)
#model.evaluate(X_test,Y_test)
from imblearn.combine import SMOTEENN
sm = SMOTEENN()
X_resampled, Y_resampled = sm.fit_resample(X,Y)
Xr_train,Xr_test,Yr_train,Yr_test=train_test_split(X_resampled, Y_resampled,test_size=0.2)
model.fit(Xr_train,Yr_train,epochs=100)
model.evaluate(Xr_test,Yr_test)
Xr_test = Xr_test.reset_index(drop = True)
y_pred = model.predict(Xr_test)
predict_churn(y_pred[0])
y_pred[0]
y_pred[10]
y_pred[100]
predict_churn(y_pred[0])
predict_churn(y_pred[100])
from temp import predict_churn
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df = pd.read_csv("C:\\Users\\Rishi\\Downloads\\Telco-Customer-Churn.csv")
df.sample(6)
df.drop("customerID",axis="columns",inplace=True)
df[pd.to_numeric(df.TotalCharges,errors="coerce").isnull()]
df.shape
df1=df[df.TotalCharges!=" "]
df1.shape
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
df1.replace("No internet service","No",inplace=True)
df1.replace("No phone service","No",inplace=True)
Y_N_columns=["Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling","Churn"]
for col in Y_N_columns: df1[col].replace({"Yes":1,"No":0},inplace=True)
df1.sample(6)
df1["gender"].replace({"Female":1,"Male":0},inplace=True)
df2=pd.get_dummies(data=df1, columns=["InternetService","Contract","PaymentMethod"])
df2.sample(6)
df2.dtypes
scale_columns=["tenure","MonthlyCharges","TotalCharges"]
import sklearn
from sklearn.preprocessing import MinMaxScaler
df2[scale_columns] = MinMaxScaler().fit_transform(df2[scale_columns])
pd.set_option("display.max_columns", None)
df2.sample(6)
X=df2.drop("Churn",axis="columns")
Y=df2["Churn"]
from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=5)
#X_train.shape
#X_test.shape
import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([keras.layers.Dense(20, input_shape=(26,), activation="relu"),keras.layers.Dense(15, activation="relu"),keras.layers.Dense(1, activation="sigmoid")])
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#model.fit(X_train,Y_train,epochs=100)
#model.evaluate(X_test,Y_test)
from imblearn.combine import SMOTEENN
sm = SMOTEENN()
X_resampled, Y_resampled = sm.fit_resample(X,Y)
Xr_train,Xr_test,Yr_train,Yr_test=train_test_split(X_resampled, Y_resampled,test_size=0.2)
model.fit(Xr_train,Yr_train,epochs=100)
model.evaluate(Xr_test,Yr_test)
Xr_test = Xr_test.reset_index(drop = True)
y_pred = model.predict(Xr_test)
predict_churn(y_pred[0])
y_pred[0]

## ---(Thu Sep 12 00:34:57 2024)---
%runfile C:/Users/RISHI/.spyder-py3/temp.py --wdir
from tkinter import simpledialog
class predict_churn(simpledialog.Dialog):
def init(self, parent, title=None):
self.result = None
super().init(parent, title=title)
def body(self, frame):
tk.Label(frame, text="Enter your name:").grid(row=0, column=0)
self.entry_name = tk.Entry(frame)
self.entry_name.grid(row=0, column=1)
return self.entry_name
def apply(self):
self.result = self.entry_name
def on_button_click():
dialog = predict_churn(root, title="Custom Dialog")
dialog.result
tk.Button(root, text="Open Dialog", command=on_button_click).pack(pady=20)
from temp import predict_churn
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df = pd.read_csv("C:\\Users\\Rishi\\Downloads\\Telco-Customer-Churn.csv")
df.sample(6)
df.drop("customerID",axis="columns",inplace=True)
df[pd.to_numeric(df.TotalCharges,errors="coerce").isnull()]
df.shape
df1=df[df.TotalCharges!=" "]
df1.shape
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
df1.replace("No internet service","No",inplace=True)
df1.replace("No phone service","No",inplace=True)
Y_N_columns=["Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling","Churn"]
for col in Y_N_columns: df1[col].replace({"Yes":1,"No":0},inplace=True)
df1.sample(6)
df1["gender"].replace({"Female":1,"Male":0},inplace=True)
df2=pd.get_dummies(data=df1, columns=["InternetService","Contract","PaymentMethod"])
df2.sample(6)
df2.dtypes
scale_columns=["tenure","MonthlyCharges","TotalCharges"]
import sklearn
from sklearn.preprocessing import MinMaxScaler
df2[scale_columns] = MinMaxScaler().fit_transform(df2[scale_columns])
pd.set_option("display.max_columns", None)
df2.sample(6)
X=df2.drop("Churn",axis="columns")
Y=df2["Churn"]
from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=5)
#X_train.shape
#X_test.shape
import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([keras.layers.Dense(20, input_shape=(26,), activation="relu"),keras.layers.Dense(15, activation="relu"),keras.layers.Dense(1, activation="sigmoid")])
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#model.fit(X_train,Y_train,epochs=100)
#model.evaluate(X_test,Y_test)
from imblearn.combine import SMOTEENN
sm = SMOTEENN()
X_resampled, Y_resampled = sm.fit_resample(X,Y)
Xr_train,Xr_test,Yr_train,Yr_test=train_test_split(X_resampled, Y_resampled,test_size=0.2)
model.fit(Xr_train,Yr_train,epochs=100)
model.evaluate(Xr_test,Yr_test)
Xr_test = Xr_test.reset_index(drop = True)
y_pred = model.predict(Xr_test)
predict_churn(y_pred)
from temp import predict_churn
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df = pd.read_csv("C:\\Users\\Rishi\\Downloads\\Telco-Customer-Churn.csv")
df.sample(6)
df.drop("customerID",axis="columns",inplace=True)
df[pd.to_numeric(df.TotalCharges,errors="coerce").isnull()]
df.shape
df1=df[df.TotalCharges!=" "]
df1.shape
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
df1.replace("No internet service","No",inplace=True)
df1.replace("No phone service","No",inplace=True)
Y_N_columns=["Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling","Churn"]
for col in Y_N_columns: df1[col].replace({"Yes":1,"No":0},inplace=True)
df1.sample(6)
df1["gender"].replace({"Female":1,"Male":0},inplace=True)
df2=pd.get_dummies(data=df1, columns=["InternetService","Contract","PaymentMethod"])
df2.sample(6)
df2.dtypes
scale_columns=["tenure","MonthlyCharges","TotalCharges"]
import sklearn
from sklearn.preprocessing import MinMaxScaler
df2[scale_columns] = MinMaxScaler().fit_transform(df2[scale_columns])
pd.set_option("display.max_columns", None)
df2.sample(6)
X=df2.drop("Churn",axis="columns")
Y=df2["Churn"]
from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=5)
#X_train.shape
#X_test.shape
import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([keras.layers.Dense(20, input_shape=(26,), activation="relu"),keras.layers.Dense(15, activation="relu"),keras.layers.Dense(1, activation="sigmoid")])
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#model.fit(X_train,Y_train,epochs=100)
#model.evaluate(X_test,Y_test)
from imblearn.combine import SMOTEENN
sm = SMOTEENN()
X_resampled, Y_resampled = sm.fit_resample(X,Y)
Xr_train,Xr_test,Yr_train,Yr_test=train_test_split(X_resampled, Y_resampled,test_size=0.2)
model.fit(Xr_train,Yr_train,epochs=100)
model.evaluate(Xr_test,Yr_test)
Xr_test = Xr_test.reset_index(drop = True)
y_pred = model.predict(Xr_test)
import tkinter as tk
from tkinter import simpledialog
application_window = tk.Tk()
answer = simpledialog.askinteger("Input", "Enter CustomerId",parent=application_window)
pred[answer]
%runfile C:/Users/RISHI/.spyder-py3/ccp.py --wdir
import ipynb-py-convert
pip install ipynb-py-convert
!pip install ipynb-py-convert
import ipynb-py-convert
ipynb-py-convert spatial.ipynb spatial.py
ls
ipynb-py-convert spatial.ipynb spatial.py
nbconvert --to script spatial.ipynb
ipynb-py-convert --to script spatial.ipynb
ipynb-py-convert
import ipynb-py-convert
!pip install jupyter
!pip install nbconvert
jupyter nbconvert --to OPTIONS spatial.ipynb
jupyter nbconvert --spatial.ipynb
nbconvert --spatial.ipynb
import nbconvert
nbconvert --spatial.ipynb
import jupyter
jupyter nbconvert --to OPTIONS spatial.ipynb
nbconvert --spatial.ipynb
jupyter nbconvert --spatial.ipynb
jupyter nbconvert spatial.ipynb
nbconvert spatial.ipynb
ipynb-py-convert spatial.ipynb spatial.py
jupyter nbconvert --spatial.ipynb
jupyter nbconvert --to OPTIONS spatial.ipynb
%matplotlib auto
%matplotlib inline

## ---(Thu Sep 12 20:08:55 2024)---
import os
import PIL
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch
import pytorch
!pip install torch
import torch
import pytorch
import torch
import pytorch
!pip show pandas
!pip show numpy
import numpy as np
import pytorch
import torch
import pandas
!pip install torch
from sympy import torch
import torch
from torch.utils.data.dataset import Dataset
import torch
!pip install torch
import torch
ls
import torch
python setup.py develop && python -c "import torch"
python -c "import torch"
import torch
import pandas
import torch
!pip uninstall torch
import torch

## ---(Thu Sep 12 21:05:07 2024)---
import torch
import os
import PIL
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.utils.data.sampler as sampler
from torch import nn, optim
import torch.nn.functional as F
import random
import numpy as np
!pip install numpy
!pip install pickle,random
!pip install pickle and random
!pip install random
!pip install random2
!pip install pickle5
!pip install pickle4
!pip install PIL
!pip install PILLOW
!pip install cv2
!pip install opencv-python
!pip install torchvision
import os
import PIL
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.utils.data.sampler as sampler
from torch import nn, optim
!pip install matplotlib
import os
import PIL
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.utils.data.sampler as sampler
from torch import nn, optim
import torch.nn.functional as F
training_file = "D:\\MEGA BACKUP\\PC\\Nielit\\gtsrb\\train.p"
validation_file = "D:\\MEGA BACKUP\\PC\\Nielit\\gtsrb\\valid.p"
testing_file = "D:\\MEGA BACKUP\\PC\\Nielit\\gtsrb\\test.p"
with open(training_file, mode='rb') as f:
with open(training_file, mode='rb') as f:train = pickle.load(f)
with open(validation_file, mode='rb') as f:valid = pickle.load(f)
with open(testing_file, mode='rb') as f:test = pickle.load(f)
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
n_train = len(X_train)
n_valid = len(X_valid)
n_test = len(X_test)
image_shape = X_train[0].shape[:-1]
n_classes = len(set(y_train))
print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
class PickledDataset(Dataset):
def __init__(self, file_path, transform=None):
with open(file_path, mode='rb') as f:
data = pickle.load(f)
self.features = data['features']
self.labels = data['labels']
self.count = len(self.labels)
self.transform = transform
def __getitem__(self, index):
feature = self.features[index]
if self.transform is not None:
feature = self.transform(feature)
return (feature, self.labels[index])
def __len__(self):
return self.count
### Data exploration visualization.
fig, ax = plt.subplots()

## ---(Sun Sep 15 15:35:03 2024)---
import os
import PIL
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.utils.data.sampler as sampler
from torch import nn, optim
import torch.nn.functional as F
training_file = "D:\\MEGA BACKUP\\PC\\Nielit\\gtsrb\\train.p"
validation_file = "D:\\MEGA BACKUP\\PC\\Nielit\\gtsrb\\valid.p"
testing_file = "D:\\MEGA BACKUP\\PC\\Nielit\\gtsrb\\test.p"
with open(training_file, mode='rb') as f:train = pickle.load(f)
with open(validation_file, mode='rb') as f:valid = pickle.load(f)
with open(testing_file, mode='rb') as f:test = pickle.load(f)
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
n_train = len(X_train)
n_valid = len(X_valid)
n_test = len(X_test)
image_shape = X_train[0].shape[:-1]
n_classes = len(set(y_train))
print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
fig, ax = plt.subplots()
ax.bar(range(n_classes), np.bincount(y_train), 0.5, color='r')
ax.set_xlabel('Signs')
ax.set_ylabel('Count')
ax.set_title('The count of each sign')
plt.show()
plt.figure(figsize=(16, 16))
plt.show()
for c in range(n_classes):
%runcell -i 0 C:/Users/RISHI/.spyder-py3/spatial.py
import os
import PIL
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.utils.data.sampler as sampler
from torch import nn, optim
import torch.nn.functional as F


training_file = "D:\\MEGA BACKUP\\PC\\Nielit\\gtsrb\\train.p"
validation_file = "D:\\MEGA BACKUP\\PC\\Nielit\\gtsrb\\valid.p"
testing_file = "D:\\MEGA BACKUP\\PC\\Nielit\\gtsrb\\test.p"

with open(training_file, mode='rb') as f:train = pickle.load(f)
with open(validation_file, mode='rb') as f:valid = pickle.load(f)
with open(testing_file, mode='rb') as f:test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


n_train = len(X_train)

n_valid = len(X_valid)

n_test = len(X_test)


image_shape = X_train[0].shape[:-1]

n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


class PickledDataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, mode='rb') as f:
            data = pickle.load(f)
            self.features = data['features']
            self.labels = data['labels']
            self.count = len(self.labels)
            self.transform = transform
    
    def __getitem__(self, index):
        feature = self.features[index]
        if self.transform is not None:
            feature = self.transform(feature)
        return (feature, self.labels[index])
    
    def __len__(self):
        return self.count


### Data exploration visualization.
fig, ax = plt.subplots()
ax.bar(range(n_classes), np.bincount(y_train), 0.5, color='r')
ax.set_xlabel('Signs')
ax.set_ylabel('Count')
ax.set_title('The count of each sign')
plt.show()

plt.figure(figsize=(16, 16))
for c in range(n_classes):
    i = random.choice(np.where(y_train == c)[0])
    plt.subplot(8, 8, c+1)
    plt.axis('off')
    plt.title('class: {}'.format(c))
    plt.imshow(X_train[i])

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func
    
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

class BaselineNet(nn.Module):
    def __init__(self, gray=False):
        super(BaselineNet, self).__init__()
        input_chan = 1 if gray else 3
        self.conv1 = nn.Conv2d(input_chan, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 43)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

torch.manual_seed(1)


train_dataset = PickledDataset(training_file, transform=transforms.ToTensor())
valid_dataset = PickledDataset(validation_file, transform=transforms.ToTensor())
test_dataset = PickledDataset(testing_file, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(x, y):
    return x.to(device), y.to(device, dtype=torch.int64)

train_loader = WrappedDataLoader(train_loader, to_device)
valid_loader = WrappedDataLoader(valid_loader, to_device)
test_loader = WrappedDataLoader(test_loader, to_device)


model = BaselineNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
n_epochs = 20


def loss_batch(model, loss_func, x, y, opt=None):
    loss = loss_func(model(x), y)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    return loss.item(), len(x)
def valid_batch(model, loss_func, x, y):
    output = model(x)
    loss = loss_func(output, y)
    pred = torch.argmax(output, dim=1)
    correct = pred == y.view(*pred.shape)
    
    return loss.item(), torch.sum(correct).item(), len(x)
def fiit(epochs, model, loss_func, opt, train_dl, valid_dl, dl):
    for epoch in range(epochs):
        # Train model
        model.train()
        losses, nums = zip(*[loss_batch(model, loss_func, x, y, opt) for x, y in train_dl])
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        # Validation model
        model.eval()
        with torch.no_grad():
            losses, corrects, nums = zip(*[valid_batch(model, loss_func, x, y) for x, y in valid_dl])
            valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            valid_accuracy = np.sum(corrects) / np.sum(nums) * 100
            losses, corrects, nums = zip(*[valid_batch(model, loss_func, x, y) for x, y in dl])
            test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            test_accuracy = np.sum(corrects) / np.sum(nums) * 100
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"Test loss: {test_loss:.6f}\t"
                  f"Test accruacy: {test_accuracy:.3f}%"
                  f"Train loss: {train_loss:.6f}\t"
                  f"Validation loss: {valid_loss:.6f}\t",
                  f"Validation accruacy: {valid_accuracy:.3f}%")
def evaluate(model, loss_func, dl):
    model.eval()
    with torch.no_grad():
        losses, corrects, nums = zip(*[valid_batch(model, loss_func, x, y) for x, y in dl])
        test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        test_accuracy = np.sum(corrects) / np.sum(nums) * 100
    
    print(f"Test loss: {test_loss:.6f}\t"
          f"Test accruacy: {test_accuracy:.3f}%")


fiit(n_epochs, model, criterion, optimizer, train_loader, valid_loader, test_loader)
evaluate(model, criterion, test_loader)


Test_loss = [3.509771,3.476337,3.443507,3.264597,2.344808,1.638526,1.367384,1.272794,1.142709,1.117169,0.991350,1.007908,0.996129,0.990292, 0.923320,0.921272,0.937243,0.931465,0.902250,0.942475]
epoch=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]


with open('model_pickle','wb') as f:
    pickle.dump(model,f)


import pickle
with open('model_pickle','rb') as f:
    mp = pickle.load(f)


plt.plot(Test_loss,epoch,'b-o',label='loss over 20 epochs');
plt.xlabel('Loss')
plt.ylabel('Epoch')
plt.legend()
plt.show()



class CLAHE_GRAY:
    def __init__(self, clipLimit=2.5, tileGridSize=(4, 4)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize
    
    def __call__(self, im):
        img_y = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)[:,:,0]
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        img_y = clahe.apply(img_y)
        img_output = img_y.reshape(img_y.shape + (1,))
        return img_output
clahe = CLAHE_GRAY()
plt.figure(figsize=(16, 16))
for c in range(n_classes):
    i = random.choice(np.where(y_train == c)[0])
    plt.subplot(8, 8, c+1)
    plt.axis('off')
    plt.title('class: {}'.format(c))
    plt.imshow(clahe(X_train[i]).squeeze(), cmap='gray')


data_transforms = transforms.Compose([
    CLAHE_GRAY(),
    transforms.ToTensor()
])

train_dataset = PickledDataset(training_file, transform=data_transforms)
valid_dataset = PickledDataset(validation_file, transform=data_transforms)
test_dataset = PickledDataset(testing_file, transform=data_transforms)

train_loader = WrappedDataLoader(DataLoader(train_dataset, batch_size=64, shuffle=True), to_device)
valid_loader = WrappedDataLoader(DataLoader(valid_dataset, batch_size=64, shuffle=False), to_device)
test_loader = WrappedDataLoader(DataLoader(test_dataset, batch_size=64, shuffle=False), to_device)


model = BaselineNet(gray=True).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
fiit(n_epochs, model, criterion, optimizer, train_loader, valid_loader, test_loader)
evaluate(model, criterion, test_loader)


Test_loss2 = [3.487602,3.460745,3.431897,3.118102,1.922260,1.356172,1.198058,0.969637,0.904229,0.818955,0.803091,0.749072,0.738959,0.728072,0.700556,0.665253,0.683440,0.631554,0.653302,0.660642]
epoch=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]


plt.plot(Test_loss2,epoch,'b-o',label='loss over 20 epochs');
plt.xlabel('Loss')
plt.ylabel('Epoch')
plt.legend()
plt.show()



import pickle
with open('model_pickle1','wb') as f:
    pickle.dump(model,f)


import pickle
with open('model_pickle1','rb') as f:
    mp = pickle.load(f)


def preprocess(path):
    if not os.path.exists(f"{path}/train_gray.p"):
        for dataset in ['train', 'valid', 'test']:
            with open(f"{path}/{dataset}.p", mode='rb') as f:
                data = pickle.load(f)
                X = data['features']
                y = data['labels']
            
            clahe = CLAHE_GRAY()
            for i in range(len(X)):
                X[i] = clahe(X[i])
            
            X = X[:, :, :, 0]
            with open(f"{path}/{dataset}_gray.p", "wb") as f:
                pickle.dump({"features": X.reshape(
                    X.shape + (1,)), "labels": y}, f)


preprocess('D:\\MEGA BACKUP\\PC\\Nielit\\gtsrb')
training_file = 'D:\\MEGA BACKUP\\PC\\Nielit\\gtsrb\\train_gray.p'
validation_file = "D:\\MEGA BACKUP\\PC\\Nielit\\gtsrb\\valid_gray.p"
testing_file = "D:\\MEGA BACKUP\\PC\\Nielit\\gtsrb\\test_gray.p"


train_dataset = PickledDataset(training_file, transform=transforms.ToTensor())
valid_dataset = PickledDataset(validation_file, transform=transforms.ToTensor())
test_dataset = PickledDataset(testing_file, transform=transforms.ToTensor())

train_loader = WrappedDataLoader(DataLoader(train_dataset, batch_size=64, shuffle=True), to_device)
valid_loader = WrappedDataLoader(DataLoader(valid_dataset, batch_size=64, shuffle=False), to_device)
test_loader = WrappedDataLoader(DataLoader(test_dataset, batch_size=64, shuffle=False), to_device)


def extend_dataset(dataset):
    X = dataset.features
    y = dataset.labels
    num_classes = 43
    
    X_extended = np.empty([0] + list(dataset.features.shape)[1:], dtype=dataset.features.dtype)
    y_extended = np.empty([0], dtype = dataset.labels.dtype)
    
    horizontally_flippable = [11, 12, 13, 15, 17, 18, 22, 26, 30, 35]
    vertically_flippable = [1, 5, 12, 15, 17]
    both_flippable = [32, 40]
    cross_flippable = np.array([
        [19, 20],
        [33, 34],
        [36, 37],
        [38, 39],
        [20, 19],
        [34, 33],
        [37, 36],
        [39, 38]
    ])
    
    for c in range(num_classes):
        X_extended = np.append(X_extended, X[y==c], axis=0)  
        
        if c in horizontally_flippable:
            X_extended = np.append(X_extended, X[y==c][:,:,::-1,:], axis=0)
        if c in vertically_flippable:
            X_extended = np.append(X_extended, X[y==c][:,::-1,:,:], axis=0)
        if c in cross_flippable[:,0]:
            flip_c = cross_flippable[cross_flippable[:,0]==c][0][1]
            X_extended = np.append(X_extended, X[y==flip_c][:,:,::-1,:], axis=0)
        if c in both_flippable:
            X_extended = np.append(X_extended, X[y==c][:,::-1,::-1,:], axis=0)
        
        y_extended = np.append(y_extended, np.full(X_extended.shape[0]-y_extended.shape[0], c, dtype=y_extended.dtype))
    
    dataset.features = X_extended
    dataset.labels = y_extended
    dataset.count = len(y_extended)
    
    return dataset
train_dataset = extend_dataset(train_dataset)
train_loader = WrappedDataLoader(DataLoader(train_dataset, batch_size=64, shuffle=True), to_device)
### Data exploration visualization.
fig, ax = plt.subplots()
ax.bar(range(n_classes), np.bincount(train_dataset.labels), 0.5, color='r')
ax.set_xlabel('Signs')
ax.set_ylabel('Count')
ax.set_title('The count of each sign')
plt.show()

plt.figure(figsize=(16, 16))
for c in range(n_classes):
    i = random.choice(np.where(train_dataset.labels == c)[0])
    plt.subplot(8, 8, c+1)
    plt.axis('off')
    plt.title('class: {}'.format(c))
    plt.imshow(train_dataset.features[i].squeeze(), cmap='gray')


model = BaselineNet(gray=True).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
fiit(n_epochs, model, criterion, optimizer, train_loader, valid_loader, test_loader)
evaluate(model, criterion, test_loader)


Test_loss3 = [3.545526,3.328307,1.698942,1.180225,0.987692,0.824676,0.775697,0.714043,0.667964,0.626187,0.585065,0.594630, 0.608713,0.597221,0.582876,0.556315,0.542433,0.500757,0.557533,0.527474]
epoch=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]


plt.plot(Test_loss3,epoch,'b-o',label='loss over 20 epochs');
plt.xlabel('Loss')
plt.ylabel('Epoch')
plt.legend()
plt.show()



with open('model_pickle2','wb') as f:
    pickle.dump(model,f)


import pickle
with open('model_pickle2','rb') as f:
    mp = pickle.load(f)


train_dataset = extend_dataset(PickledDataset(training_file))

class_sample_count = np.bincount(train_dataset.labels)
weights = 1 / np.array([class_sample_count[y] for y in train_dataset.labels])
samp = sampler.WeightedRandomSampler(weights, 43 * 20000)

train_loader = WrappedDataLoader(DataLoader(train_dataset, batch_size=64, sampler=samp), to_device)
balanced_y_train = torch.LongTensor([]).to(device)

with torch.no_grad():
    for _, y in train_loader:
        balanced_y_train = torch.cat((balanced_y_train, y))

fig, ax = plt.subplots()
ax.bar(range(n_classes), np.bincount(balanced_y_train.cpu().numpy()), 0.5, color='r')
ax.set_xlabel('Signs')
ax.set_ylabel('Count')
ax.set_title('The count of each sign')
plt.show()



train_data_transforms = transforms.Compose([
    transforms.ToPILImage(),
%%debug
train_data_transforms = transforms.Compose([transforms.ToPILImage(),transforms.RandomApply([transforms.RandomRotation(20, resample=PIL.Image.BICUBIC),transforms.RandomAffine(0, translate=(0.2, 0.2), resample=PIL.Image.BICUBIC),transforms.RandomAffine(0, shear=20, resample=PIL.Image.BICUBIC),transforms.RandomAffine(0, scale=(0.8, 1.2), resample=PIL.Image.BICUBIC)]),transforms.ToTensor()])
%debugfile C:/Users/RISHI/.spyder-py3/spatial.py --wdir
%debugfile C:/Users/RISHI/.spyder-py3/spatial.py --wdir
%debugfile C:/Users/RISHI/.spyder-py3/spatial.py --wdir
%debugfile C:/Users/RISHI/.spyder-py3/spatial.py --wdir

## ---(Sun Sep 22 23:41:48 2024)---
%%debug
#import modules
!pip install yahoo_fin
from yahoo_fin import options as op
#input ticker
ticker = 'BANKNIFTY'
#get expiration dates
expirationDates = op.get_expiration_dates(ticker)
ticker = 'NIFTY'
#get expiration dates
expirationDates = op.get_expiration_dates(ticker)
ticker = 'PLTR'
#get expiration dates
expirationDates = op.get_expiration_dates(ticker)
!pip install requests-html
from yahoo_fin import options as op
ticker = 'NIFTY BANK'
#get expiration dates
expirationDates = op.get_expiration_dates(ticker)
ticker = 'PLTR'
#get expiration dates
expirationDates = op.get_expiration_dates(ticker)
!pip install requests-html
from yahoo_fin import options as op
#input ticker
ticker = 'PLTR'
#get expiration dates
expirationDates = op.get_expiration_dates(ticker)
ticker = 'PLTR'
from yahoo_fin import options as op
expirationDates = op.get_expiration_dates(ticker)
from yahoo_fin import options as op
ticker = 'PLTR'
expirationDates = op.get_expiration_dates(ticker)

## ---(Mon Sep 23 00:14:59 2024)---
from yahoo_fin import options as op
#input ticker
ticker = 'PLTR'
#get expiration dates
expirationDates = op.get_expiration_dates(ticker)
import yahoo_fin
expirationDates = op.get_expiration_dates(ticker)
import requests_html
from requests_html import session
!pip install lxml_html_clean
from requests_html import session
import requests_html 
from yahoo_fin import options as op
#input ticker
ticker = 'PLTR'
#get expiration dates
expirationDates = op.get_expiration_dates(ticker)
callData = op.get_calls(ticker, date = expirationDates[0])
putData = op.get_puts(ticker, date = expirationDates[0])
from yahoo_fin import options as op
#input ticker
ticker = 'NIFTY BANK'
#get expiration dates
expirationDates = op.get_expiration_dates(ticker)
expirationDates
ticker = 'AAPL'
#get expiration dates
expirationDates = op.get_expiration_dates(ticker)
expirationDates
callData = op.get_calls(ticker, date = expirationDates[0])
from yahoo_fin import options as op
#input ticker
ticker = 'AAPL'
#get expiration dates
expirationDates = op.get_expiration_dates(ticker)
expirationDates
from yahoo_fin import options as op
#input ticker
ticker = 'AAPL'
#get expiration dates
expirationDates = op.get_expiration_dates(ticker)
ticker = '^NSEBANK'
#get expiration dates
expirationDates = op.get_expiration_dates(ticker)