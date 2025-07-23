import tkinter as tk

def say_hello():
    label.config(text="Hello, MCNP Tools!")

root = tk.Tk()
root.title("MCNP Tools Test")

label = tk.Label(root, text="Click the button")
label.pack(pady=10)

button = tk.Button(root, text="Say Hello", command=say_hello)
button.pack(pady=10)

root.mainloop()