import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, scrolledtext
from scipy.signal import find_peaks  # To detect peaks
import threading

# Kalman Filter with State-Space Model
def kalman_filter_state_model(signal, temperature, sampling_frequency, piston_radius, process_variance=1e-4, measurement_variance=1e-1):
    n_iterations = len(signal)
    
    # State transition matrix
    A = np.array([[1, 1/sampling_frequency, 0.5/(sampling_frequency**2)], 
                  [0, 1, 1/sampling_frequency], 
                  [0, 0, 1]])
    
    # Observation matrix
    H = np.array([[1, 0, 0]])
    
    # Initial state estimates: pressure, velocity, acceleration
    x = np.array([signal[0], 0, 0])
    
    # Process and measurement covariance matrices
    Q = process_variance * np.eye(3)  
    R = np.array([[measurement_variance]])  
    
    # Initialize error covariance matrix
    P = np.eye(3)
    
    # Storage for the estimated sound pressure and smoothed signal
    estimated_pressure = np.zeros(n_iterations)
    smoothed_signal = np.zeros(n_iterations)
    
    speed_of_sound = 331.3 + 0.606 * temperature
    air_density = 1.225
    
    for t in range(n_iterations):
        # Prediction step
        x = A @ x  
        P = A @ P @ A.T + Q  
        
        # Measurement update step
        z = np.array([signal[t]])  
        y = z - H @ x  
        S = H @ P @ H.T + R  
        K = P @ H.T @ np.linalg.inv(S)  
        x = x + K @ y  
        P = (np.eye(3) - K @ H) @ P  
        
        # Calculate sound pressure
        distance = calculate_distance(t, temperature, sampling_frequency)
        if distance > 0:
            estimated_pressure[t] = (4 * piston_radius * air_density / (distance * speed_of_sound**2)) * x[0]
        
        # Store the smoothed signal (which is the first state)
        smoothed_signal[t] = x[0]
    
    return estimated_pressure, smoothed_signal

# Distance Calculation
def calculate_distance(peak_index, temperature, sampling_frequency):
    speed_of_sound = 331.3 + 0.606 * temperature
    time_delay = peak_index / sampling_frequency
    distance = 0.5 * speed_of_sound * time_delay
    return distance

# Yanowitz-Bruckstein Peak Detection
def yanowitz_bruckstein_thresholding(signal, window_size=100, threshold_init_factor=0.5, min_peak_prominence=0.1):
    threshold = threshold_init_factor * np.max(signal)

    for i in range(window_size, len(signal) - window_size):
        window = signal[i - window_size:i + window_size]
        local_max = np.max(window)

        if local_max > threshold:
            threshold = local_max - min_peak_prominence

        if signal[i] >= threshold and signal[i] == local_max:
            if all(signal[i] >= signal[j] for j in range(i - window_size, i + window_size) if j != i):
                return i

    return None

# Niblack's Local Thresholding
def niblacks_local_thresholding(signal, window_size=2000, k=2.7):
    n = len(signal)
    for i in range(window_size, n - window_size):
        local_segment = signal[i - window_size:i + window_size]
        local_mean = np.mean(local_segment)
        local_std = np.std(local_segment)
        threshold = local_mean + k * local_std

        if signal[i] > threshold:
            if all(signal[i] >= signal[j] for j in range(i - window_size, i + window_size) if j != i):
                return i

    return None

# Signal Processor Application
class SignalProcessorApp:
    def __init__(self, master):
        self.master = master
        master.title("Signal Processor")

        # Index to keep track of which row is being processed
        self.current_row_index = 0

        # Widgets for file selection and parameter entry
        tk.Label(master, text="File Path:").grid(row=0, column=0)
        self.file_path = tk.Entry(master, width=70)
        self.file_path.grid(row=0, column=1)
        tk.Button(master, text="Browse", command=self.load_file).grid(row=0, column=2)

        tk.Label(master, text="Temperature (C):").grid(row=1, column=0)
        self.temperature = tk.Entry(master)
        self.temperature.insert(0, "20")
        self.temperature.grid(row=1, column=1)

        tk.Label(master, text="Sampling Frequency (Hz):").grid(row=2, column=0)
        self.sampling_frequency = tk.Entry(master)
        self.sampling_frequency.insert(0, "1953125")
        self.sampling_frequency.grid(row=2, column=1)

        tk.Label(master, text="Piston Radius (m):").grid(row=3, column=0)
        self.piston_radius = tk.Entry(master)
        self.piston_radius.insert(0, "0.05")
        self.piston_radius.grid(row=3, column=1)

        tk.Button(master, text="Start Processing", command=self.start_processing).grid(row=4, column=1)

        # ScrolledText widget for logging
        self.log = scrolledtext.ScrolledText(master, width=70, height=10, state='disabled')
        self.log.grid(row=6, columnspan=3, sticky='we')

        # Plotting area
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().grid(row=5, columnspan=3)

        # Initialize data storage
        self.pressure_df = None
        self.niblack_data = []
        self.yanowitz_data = []

        # Handle window close event
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def log_message(self, message):
        self.log.config(state='normal')
        self.log.insert(tk.END, message + "\n")
        self.log.config(state='disabled')
        self.log.see(tk.END)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if path:
            self.file_path.delete(0, tk.END)
            self.file_path.insert(0, path)
            self.log_message(f"Loaded file: {path}")
            # Load data assuming it's space-separated or comma-separated
            self.df = pd.read_csv(path, sep=r'\s+', header=None)
            self.pressure_df = pd.DataFrame(index=self.df.index, columns=self.df.columns)

    def start_processing(self):
        processing_thread = threading.Thread(target=self.process_all_rows)
        processing_thread.start()

    def process_all_rows(self):
        if not hasattr(self, 'df'):
            self.log_message("No file loaded.")
            return

        try:
            temperature = float(self.temperature.get())
            sampling_frequency = int(self.sampling_frequency.get())
            piston_radius = float(self.piston_radius.get())

            for i in range(len(self.df)):
                self.log_message(f"Processing row {i+1}/{len(self.df)}")
                data_row = self.df.iloc[i].to_numpy()

                # Estimate sound pressure and get smoothed signal using Kalman filter state-space model
                estimated_pressure, smoothed_signal = kalman_filter_state_model(data_row, temperature, sampling_frequency, piston_radius)

                # Detect Niblack peak and store data around it
                peak_index_niblack = niblacks_local_thresholding(estimated_pressure)
                if peak_index_niblack is not None:
                    start_index = max(peak_index_niblack - 500, 0)
                    end_index = min(peak_index_niblack + 500, len(data_row))
                    self.niblack_data.append(data_row[start_index:end_index])
                    self.log_message(f"Row {i}: Niblack peak detected at index: {peak_index_niblack}")

                # Detect Yanowitz-Bruckstein peak and store data around it
                peak_index_yb = yanowitz_bruckstein_thresholding(estimated_pressure)
                if peak_index_yb is not None:
                    start_index = max(peak_index_yb - 500, 0)
                    end_index = min(peak_index_yb + 500, len(data_row))
                    self.yanowitz_data.append(data_row[start_index:end_index])
                    self.log_message(f"Row {i}: Yanowitz-Bruckstein peak detected at index: {peak_index_yb}")

                # Store sound pressure for later analysis
                self.pressure_df.iloc[i] = estimated_pressure

                # Update the plot
                self.update_plot(data_row, smoothed_signal, estimated_pressure, peak_index_yb, peak_index_niblack)

            self.log_message("All rows processed.")
            self.save_niblack_data()
            self.save_yanowitz_data()

        except Exception as e:
            self.log_message(f"Error during processing: {str(e)}")

    def update_plot(self, raw_signal, smoothed_signal, estimated_pressure, peak_index_yb, peak_index_niblack):
        try:
            # Plot 1: Signal Amplitude (Raw and Smoothed Signal)
            self.ax1.clear()
            self.ax1.plot(raw_signal, label='Raw Signal', color='lightgray')
            self.ax1.plot(smoothed_signal, label='Smoothed Signal', color='blue')
            
            # Plot Yanowitz-Bruckstein peak in Plot 1
            if peak_index_yb is not None:
                self.ax1.plot(peak_index_yb, smoothed_signal[peak_index_yb], 'rx', markersize=12, label="Yanowitz-Bruckstein Peak")
            
            # Plot Niblack peak in Plot 1
            if peak_index_niblack is not None:
                self.ax1.plot(peak_index_niblack, smoothed_signal[peak_index_niblack], 'gx', markersize=12, label="Niblack Peak")
            
            self.ax1.set_xlabel('Time')
            self.ax1.set_ylabel('Signal Amplitude')
            self.ax1.legend()

            # Plot 2: Estimated Sound Pressure (with Envelope over Peaks)
            self.ax2.clear()

            # Plot the full time range
            self.ax2.plot(estimated_pressure, label='Estimated Sound Pressure', color='lightpink')

            # Detect peaks in the positive side of the signal
            positive_peaks, _ = find_peaks(estimated_pressure)

            # Interpolate a smooth envelope using the detected peaks
            if len(positive_peaks) > 0:
                envelope = np.interp(np.arange(len(estimated_pressure)), positive_peaks, estimated_pressure[positive_peaks])

                # Plot the envelope
                self.ax2.plot(envelope, label='Envelope', color='black', linestyle='-')

            # Plot Yanowitz-Bruckstein peak in Plot 2
            if peak_index_yb is not None:
                self.ax2.plot(peak_index_yb, estimated_pressure[peak_index_yb], 'rx', markersize=12, label="Yanowitz-Bruckstein Peak")
            
            # Plot Niblack peak in Plot 2
            if peak_index_niblack is not None:
                self.ax2.plot(peak_index_niblack, estimated_pressure[peak_index_niblack], 'gx', markersize=12, label="Niblack Peak")
            
            self.ax2.set_ylim([-0.003, 0.003])  # Adjust based on your data
            self.ax2.set_xlim(200, len(estimated_pressure) - 1)
            self.ax2.set_xlabel('Time')
            self.ax2.set_ylabel('Sound Pressure')
            self.ax2.legend()

            self.canvas.draw()
        except Exception as e:
            self.log_message(f"Error during plotting: {str(e)}")

    def save_niblack_data(self):
        try:
            output_path_niblack = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")], title="Save Niblack Data")
            if output_path_niblack:
                with open(output_path_niblack, 'w') as file:
                    for data in self.niblack_data:
                        np.savetxt(file, [data], fmt='%d', delimiter=' ')
                self.log_message(f"Niblack data saved to: {output_path_niblack}")
        except Exception as e:
            self.log_message(f"Error saving Niblack data: {str(e)}")

    def save_yanowitz_data(self):
        try:
            output_path_yanowitz = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")], title="Save Yanowitz-Bruckstein Data")
            if output_path_yanowitz:
                with open(output_path_yanowitz, 'w') as file:
                    for data in self.yanowitz_data:
                        np.savetxt(file, [data], fmt='%d', delimiter=' ')
                self.log_message(f"Yanowitz-Bruckstein data saved to: {output_path_yanowitz}")
        except Exception as e:
            self.log_message(f"Error saving Yanowitz-Bruckstein data: {str(e)}")

    def on_closing(self):
        try:
            if self.niblack_data:
                self.save_niblack_data()
            if self.yanowitz_data:
                self.save_yanowitz_data()
        finally:
            self.master.quit()
            self.master.destroy()

# Main execution block
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessorApp(root)
    root.mainloop()