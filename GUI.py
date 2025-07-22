

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import json
import os
import sys
import subprocess
from He3_Plotter import (
    SINGLE_SOURCE_YIELD,
    THREE_SOURCE_YIELD,
    AREA,
    VOLUME,
    run_analysis_type_1,
    run_analysis_type_2,
    run_analysis_type_3,
    run_analysis_type_4,
    select_file,
    select_folder
)

CONFIG_FILE = "config.json"

# Redirector for print statements to output console and plot file list
class TextRedirector:
    def __init__(self, widget, listbox):
        self.widget = widget
        self.listbox = listbox
    def write(self, string):
        self.widget.insert("end", string)
        self.widget.see("end")
        if "Saved:" in string:
            # Extract the path after "Saved:"
            path = string.split("Saved:", 1)[1].strip()
            self.listbox.insert("end", path)
    def flush(self):
        pass

class He3PlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("He3 Plotter GUI")
        self.root.iconbitmap("/Users/ioanhughes/Documents/PhD/MCNP/Code/icon.icns")
        self.root.geometry("600x500")

        self.neutron_yield = tk.StringVar(value="single")
        self.analysis_type = tk.StringVar(value="1")
        self.plot_listbox = None

        self.build_interface()
        self.load_config()

        # Redirect stdout to output console and listbox
        sys.stdout = TextRedirector(self.output_console, self.plot_listbox)

    def build_interface(self):
        # Create tabs
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True)

        # Create tab frames
        self.analysis_tab = ttk.Frame(self.tabs)
        self.help_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.analysis_tab, text="Analysis")
        self.tabs.add(self.help_tab, text="How to Use")

        # Yield frame
        yield_frame = ttk.LabelFrame(self.analysis_tab, text="Neutron Source Selection")
        yield_frame.pack(fill="x", padx=10, pady=5)
        # Replace radiobuttons with checkboxes for neutron sources
        self.source_vars = {
            "Small tank (1.25e6)": tk.BooleanVar(),
            "Big tank (2.5e6)": tk.BooleanVar(),
            "Graphite stack (7.5e6)": tk.BooleanVar()
        }
        for label, var in self.source_vars.items():
            ttk.Checkbutton(yield_frame, text=label, variable=var).pack(anchor="w", padx=10)

        # Analysis type frame
        analysis_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Type")
        analysis_frame.pack(fill="x", padx=10, pady=5)
        self.analysis_type_map = {
            "Efficiency & Neutron Rates": "1",
            "Thickness Comparison": "2",
            "Source Position Alignment": "3",
            "Photon Tally Plot": "4"
        }
        self.analysis_combobox = ttk.Combobox(
            analysis_frame,
            values=list(self.analysis_type_map.keys()),
            state="readonly"
        )
        self.analysis_combobox.set("Efficiency & Neutron Rates")
        self.analysis_combobox.pack(padx=10, pady=5)
        self.analysis_combobox.bind("<<ComboboxSelected>>", self.update_analysis_type)

        # Button frame
        button_frame = ttk.Frame(self.analysis_tab)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis_threaded).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clear Output", command=self.clear_output).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clear Saved Plots", command=self.clear_saved_plots).pack(side="left", padx=5)
        # CSV export toggle
        self.save_csv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(button_frame, text="Save CSVs", variable=self.save_csv_var).pack(side="left", padx=5)

        # Output console frame
        output_frame = ttk.LabelFrame(self.analysis_tab, text="Output Console")
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.output_console = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=8)
        self.output_console.pack(fill="both", expand=True)

        # File output list frame
        file_frame = ttk.LabelFrame(self.analysis_tab, text="Saved Plots")
        file_frame.pack(fill="both", expand=False, padx=10, pady=5)
        self.plot_listbox = tk.Listbox(file_frame, height=4)
        self.plot_listbox.pack(fill="both", expand=True)
        self.plot_listbox.bind("<Double-Button-1>", self.open_selected_plot)

        # Populate help tab with instructions
        help_label = tk.Label(self.help_tab, text="How to Use He3 Plotter", font=("Arial", 14, "bold"))
        help_label.pack(pady=10)

        help_text = (
            "Step 1: Select Neutron Sources\n"
            "• Tick the boxes for the neutron sources included in your setup:\n"
            "    – Small tank (1.25e6 n/s)\n"
            "    – Big tank (2.5e6 n/s)\n"
            "    – Graphite stack (7.5e6 n/s)\n\n"
            
            "Step 2: Choose Analysis Type\n"
            "• Select one of the following options from the dropdown menu:\n"
            "    – Efficiency & Neutron Rates:\n"
            "        Analyse a single MCNP output file. Outputs include incident/detected neutron rates and efficiency.\n"
            "    – Thickness Comparison:\n"
            "        Compare simulated and experimental data across different moderator thicknesses.\n"
            "    – Source Position Alignment:\n"
            "        Analyse changes in detection due to varying source positions.\n"
            "    – Photon Tally Plot:\n"
            "        Plot gamma tally data from Tally 34 in a single MCNP file.\n\n"

            "Step 3: Run the Analysis\n"
            "• Click 'Run Analysis' to begin.\n"
            "• Saved plots will be listed below and stored in a 'plots' subfolder.\n\n"

            "File Naming Requirements:\n"
            "• Thickness Comparison: filenames must end with the moderator thickness as '_10cmo', '_5cmo', etc. (Program may stuggle if the thickness is a decimal.)\n"
            "• Source Position Alignment: filenames should be named displacement values like '5_0cm', '10_0cm', etc. Where the underscore can be used instead of a decimal place due to file extention continuity.\n"
            "These identifiers are required for the programme to extract geometry information automatically.\n\n"

            "CSV Output Information:\n"
            "• If 'Save CSVs' is checked, CSV files will be saved in a separate 'csvs' folder alongside 'plots'.\n"
            "• Analysis 1 (Efficiency & Neutron Rates): saves neutron tallies, photon tally, and a summary CSV.\n"
            "• Analysis 2 (Thickness Comparison): saves a CSV of simulated vs experimental thickness data.\n"
            "• Analysis 3 (Source Position Alignment): saves a CSV of displacement vs detected rate data.\n"
            "• Analysis 4 (Photon Tally Plot): saves a CSV of photon tally data.\n"
        )

        help_box = scrolledtext.ScrolledText(self.help_tab, wrap=tk.WORD, height=25)
        help_box.insert("1.0", help_text)
        help_box.configure(state="disabled")
        help_box.pack(fill="both", expand=True, padx=10, pady=10)

    def update_analysis_type(self, event=None):
        selected_description = self.analysis_combobox.get()
        self.analysis_type.set(self.analysis_type_map[selected_description])

    def clear_output(self):
        self.output_console.delete("1.0", tk.END)

    def clear_saved_plots(self):
        self.plot_listbox.delete(0, tk.END)

    def open_selected_plot(self, event):
        selection = self.plot_listbox.curselection()
        if selection:
            file_path = self.plot_listbox.get(selection[0])
            if os.path.exists(file_path):
                try:
                    if sys.platform.startswith("darwin"):
                        subprocess.run(["open", file_path])
                    elif sys.platform.startswith("linux"):
                        subprocess.run(["xdg-open", file_path])
                    elif sys.platform.startswith("win"):
                        os.startfile(file_path)
                except Exception as e:
                    print(f"Failed to open file: {e}")

    # Removed preview_selected_plot method (inline plot preview functionality)

    def save_config(self):
        config = {
            "neutron_yield": self.neutron_yield.get(),
            "analysis_type": self.analysis_type.get(),
            "sources": {label: var.get() for label, var in self.source_vars.items()}
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    self.neutron_yield.set(config.get("neutron_yield", "single"))
                    self.analysis_type.set(config.get("analysis_type", "1"))
                    # Update the combobox to reflect the loaded analysis_type
                    for desc, val in self.analysis_type_map.items():
                        if val == self.analysis_type.get():
                            self.analysis_combobox.set(desc)
                            break
                    # Restore neutron sources selection
                    sources = config.get("sources", {})
                    for label, var in self.source_vars.items():
                        var.set(sources.get(label, False))
            except Exception as e:
                print(f"Failed to load config: {e}")

    def run_analysis_threaded(self):
        # Neutron source selection logic
        selected_sources = {
            "Small tank (1.25e6)": 1.25e6,
            "Big tank (2.5e6)": 2.5e6,
            "Graphite stack (7.5e6)": 7.5e6
        }
        yield_value = sum(val for label, val in selected_sources.items() if self.source_vars[label].get())
        if yield_value == 0:
            print("No neutron sources selected. Please select at least one.")
            return
        analysis = self.analysis_type.get()

        # Gather inputs for each analysis
        if analysis == "1":
            file_path = select_file("Select MCNP Output File")
            if not file_path:
                print("Analysis cancelled.")
                return
            args = (1, file_path, yield_value)

        elif analysis == "2":
            folder_path = select_folder("Select Folder with Simulated Data")
            if not folder_path:
                print("Analysis cancelled.")
                return
            lab_data_path = select_file("Select Experimental Lab Data CSV")
            if not lab_data_path:
                print("Analysis cancelled.")
                return
            args = (2, folder_path, lab_data_path, yield_value)

        elif analysis == "3":
            folder_path = select_folder("Select Folder with Simulated Source Position CSVs")
            if not folder_path:
                print("Analysis cancelled.")
                return
            args = (3, folder_path, yield_value)

        elif analysis == "4":
            file_path = select_file("Select MCNP Output File for Gamma Analysis")
            if not file_path:
                print("Analysis cancelled.")
                return
            args = (4, file_path)

        else:
            messagebox.showerror("Error", "Invalid analysis type selected.")
            return

        # Start background processing
        t = threading.Thread(target=self.process_analysis, args=(args,))
        t.daemon = True
        t.start()

    def process_analysis(self, args):
        self.save_config()
        # Apply CSV export preference
        import He3_Plotter
        He3_Plotter.EXPORT_CSV = self.save_csv_var.get()
        try:
            if args[0] == 1:
                _, file_path, yield_value = args
                run_analysis_type_1(file_path, AREA, VOLUME, yield_value)
            elif args[0] == 2:
                _, folder_path, lab_data_path, yield_value = args
                run_analysis_type_2(folder_path, lab_data_path, AREA, VOLUME, yield_value)
            elif args[0] == 3:
                _, folder_path, yield_value = args
                run_analysis_type_3(folder_path, AREA, VOLUME, yield_value)
            elif args[0] == 4:
                _, file_path = args
                run_analysis_type_4(file_path)
        except Exception as e:
            print(f"Error during analysis: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = He3PlotterApp(root)
    # Force the window to the front on macOS
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.mainloop()