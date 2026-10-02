def predict_churn(pred,data):
    import tkinter as tk
    from tkinter import simpledialog
    application_window = tk.Tk()
    answer = simpledialog.askinteger("Input", f"Please Enter CustomerId from the given data \n {data}",parent=application_window)
    if pred[answer] > 0.75:
        print('Satisfied')
    else:
        print('Not Satisfied')

