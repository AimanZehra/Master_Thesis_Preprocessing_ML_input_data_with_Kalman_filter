import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
def select_input_file():
    input_file = filedialog.askopenfilename(title="Select Input File", filetypes=[("Text files", "*.txt")])
    input_path.set(input_file)
def select_output_directory():
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    output_path.set(output_dir)
def process_file():
    input_file = input_path.get()
    output = output_path.get()
    if not input_file or not output:
        messagebox.showerror("Error", "Please select both an input file and output directory.")
        return
    if not input_file.endswith('.txt'):
        messagebox.showerror("Error", "Please select a valid .txt file.")
        return
    print(f"Processing {input_file}")
    with open(input_file, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        values = line.split()[16:] 
        data.append(values)
    df = pd.DataFrame(data)
    output_file_name = os.path.splitext(os.path.basename(input_file))[0]  
    output_file_path = os.path.join(output, f"Extracted_{output_file_name}.txt")
    with open(output_file_path, 'w') as f_out:
        for row in df.itertuples(index=False, name=None):
            f_out.write(' '.join(map(str, row)) + '\n') 
    messagebox.showinfo("Done", f"The file {input_file} has been processed!")
root = tk.Tk()
root.title("File Processor")
input_path = tk.StringVar()
output_path = tk.StringVar()
tk.Label(root, text="Input File:").grid(row=0, column=0, padx=10, pady=10)
tk.Entry(root, textvariable=input_path, width=50).grid(row=0, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=select_input_file).grid(row=0, column=2, padx=10, pady=10)
tk.Label(root, text="Output Directory:").grid(row=1, column=0, padx=10, pady=10)
tk.Entry(root, textvariable=output_path, width=50).grid(row=1, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=select_output_directory).grid(row=1, column=2, padx=10, pady=10)
tk.Button(root, text="Process File", command=process_file).grid(row=2, column=1, padx=10, pady=20)
root.mainloop()