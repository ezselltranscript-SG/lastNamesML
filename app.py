#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interfaz gráfica para el servicio de corrección de apellidos en documentos Word
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import subprocess
from datetime import datetime

class SurnameCorrectiorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Corrector de Apellidos - Servicio de Documentos Word")
        self.root.geometry("700x500")
        self.root.resizable(True, True)
        
        # Variables
        self.model_path = tk.StringVar(value="models/random_forest_model.pkl")
        self.surname_model_path = tk.StringVar(value="")
        self.reference_data = tk.StringVar(value="data/processed_surnames.csv")
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar(value="data/corrected_word")
        self.confidence = tk.DoubleVar(value=0.7)
        self.surname_confidence = tk.DoubleVar(value=0.6)
        self.use_ner = tk.BooleanVar(value=True)
        self.places_exclusion_file = tk.StringVar(value="")
        self.words_exclusion_file = tk.StringVar(value="")
        self.processing = False
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Servicio de Corrección de Apellidos", 
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=10, sticky="w")
        
        # Model path
        ttk.Label(main_frame, text="Modelo:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(main_frame, textvariable=self.model_path, width=50).grid(row=1, column=1, sticky="ew", pady=5)
        ttk.Button(main_frame, text="Examinar", command=self.browse_model).grid(row=1, column=2, padx=5, pady=5)
        
        # Reference data
        ttk.Label(main_frame, text="Datos de referencia:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(main_frame, textvariable=self.reference_data, width=50).grid(row=2, column=1, sticky="ew", pady=5)
        ttk.Button(main_frame, text="Examinar", command=self.browse_reference).grid(row=2, column=2, padx=5, pady=5)
        
        # Input file
        ttk.Label(main_frame, text="Documento Word:").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(main_frame, textvariable=self.input_file, width=50).grid(row=3, column=1, sticky="ew", pady=5)
        ttk.Button(main_frame, text="Examinar", command=self.browse_input).grid(row=3, column=2, padx=5, pady=5)
        
        # Output directory
        ttk.Label(main_frame, text="Carpeta de salida:").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=4, column=1, sticky="ew", pady=5)
        ttk.Button(main_frame, text="Examinar", command=self.browse_output).grid(row=4, column=2, padx=5, pady=5)
        
        # Confidence threshold
        ttk.Label(main_frame, text="Umbral de confianza:").grid(row=5, column=0, sticky="w", pady=5)
        confidence_frame = ttk.Frame(main_frame)
        confidence_frame.grid(row=5, column=1, sticky="ew", pady=5)
        
        confidence_scale = ttk.Scale(
            confidence_frame, 
            from_=0.0, 
            to=1.0, 
            orient=tk.HORIZONTAL, 
            variable=self.confidence, 
            length=300,
            command=self.update_confidence_label
        )
        confidence_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.confidence_label = ttk.Label(confidence_frame, text="0.7")
        self.confidence_label.pack(side=tk.RIGHT, padx=5)
        
        # Surname model path (opcional)
        ttk.Label(main_frame, text="Modelo de detección (opcional):").grid(row=5, column=0, sticky="w", pady=5)
        ttk.Entry(main_frame, textvariable=self.surname_model_path, width=50).grid(row=5, column=1, sticky="ew", pady=5)
        ttk.Button(main_frame, text="Examinar", command=self.browse_surname_model).grid(row=5, column=2, padx=5, pady=5)
        
        # Confidence threshold for correction
        ttk.Label(main_frame, text="Umbral de confianza (corrección):").grid(row=6, column=0, sticky="w", pady=5)
        confidence_frame = ttk.Frame(main_frame)
        confidence_frame.grid(row=6, column=1, sticky="ew", pady=5)
        
        confidence_scale = ttk.Scale(
            confidence_frame, 
            from_=0.0, 
            to=1.0, 
            orient=tk.HORIZONTAL, 
            variable=self.confidence, 
            length=300,
            command=self.update_confidence_label
        )
        confidence_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.confidence_label = ttk.Label(confidence_frame, text="0.7")
        self.confidence_label.pack(side=tk.RIGHT, padx=5)
        
        # Confidence threshold for surname detection
        ttk.Label(main_frame, text="Umbral de confianza (detección):").grid(row=7, column=0, sticky="w", pady=5)
        surname_confidence_frame = ttk.Frame(main_frame)
        surname_confidence_frame.grid(row=7, column=1, sticky="ew", pady=5)
        
        surname_confidence_scale = ttk.Scale(
            surname_confidence_frame, 
            from_=0.0, 
            to=1.0, 
            orient=tk.HORIZONTAL, 
            variable=self.surname_confidence, 
            length=300,
            command=self.update_surname_confidence_label
        )
        surname_confidence_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.surname_confidence_label = ttk.Label(surname_confidence_frame, text="0.6")
        self.surname_confidence_label.pack(side=tk.RIGHT, padx=5)
        
        # Use NER checkbox
        ner_frame = ttk.Frame(main_frame)
        ner_frame.grid(row=8, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Checkbutton(ner_frame, text="Usar reconocimiento de entidades nombradas (NER)", variable=self.use_ner).pack(side=tk.LEFT)
        
        # Exclusion lists
        ttk.Label(main_frame, text="Lista de exclusión (lugares):").grid(row=9, column=0, sticky="w", pady=5)
        ttk.Entry(main_frame, textvariable=self.places_exclusion_file, width=50).grid(row=9, column=1, sticky="ew", pady=5)
        ttk.Button(main_frame, text="Examinar", command=self.browse_places_exclusion).grid(row=9, column=2, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Lista de exclusión (palabras):").grid(row=10, column=0, sticky="w", pady=5)
        ttk.Entry(main_frame, textvariable=self.words_exclusion_file, width=50).grid(row=10, column=1, sticky="ew", pady=5)
        ttk.Button(main_frame, text="Examinar", command=self.browse_words_exclusion).grid(row=10, column=2, padx=5, pady=5)
        
        # Process button
        self.process_button = ttk.Button(
            main_frame, 
            text="Procesar Documento", 
            command=self.process_document,
            style="Accent.TButton"
        )
        self.process_button.grid(row=11, column=0, columnspan=3, pady=20)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progreso")
        progress_frame.grid(row=12, column=0, columnspan=3, sticky="ew", pady=10)
        
        self.progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=100, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Listo para procesar")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.pack(fill=tk.X, padx=10, pady=5)
        
        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="Registro")
        log_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=10)
        
        # Make the log frame expandable
        main_frame.rowconfigure(8, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Log text widget with scrollbar
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Configure style for accent button
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 11, "bold"))
        
    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Seleccionar modelo",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)
            
    def browse_reference(self):
        path = filedialog.askopenfilename(
            title="Seleccionar datos de referencia",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if path:
            self.reference_data.set(path)
            
    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Seleccionar documento Word",
            filetypes=[("Word documents", "*.docx"), ("All files", "*.*")]
        )
        if path:
            self.input_file.set(path)
            
    def browse_output(self):
        path = filedialog.askdirectory(title="Seleccionar carpeta de salida")
        if path:
            self.output_dir.set(path)
            
    def update_confidence_label(self, value):
        self.confidence_label.config(text=f"{float(value):.1f}")
        
    def update_surname_confidence_label(self, value):
        self.surname_confidence_label.config(text=f"{float(value):.1f}")
        
    def browse_surname_model(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar modelo de detección de apellidos",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            self.surname_model_path.set(filename)
            
    def browse_places_exclusion(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar lista de exclusión de lugares",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.places_exclusion_file.set(filename)
            
    def browse_words_exclusion(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar lista de exclusión de palabras comunes",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.words_exclusion_file.set(filename)
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        
    def process_document(self):
        # Validate inputs
        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("Error", "El archivo del modelo no existe")
            return
            
        if not os.path.exists(self.reference_data.get()):
            messagebox.showerror("Error", "El archivo de datos de referencia no existe")
            return
            
        if not self.input_file.get() or not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "Seleccione un documento Word válido")
            return
            
        if not self.input_file.get().lower().endswith('.docx'):
            messagebox.showerror("Error", "El archivo de entrada debe ser un documento Word (.docx)")
            return
            
        # Disable UI during processing
        self.processing = True
        self.process_button.config(state=tk.DISABLED)
        self.progress.start()
        self.status_var.set("Procesando documento...")
        
        # Start processing in a separate thread
        threading.Thread(target=self.run_processing, daemon=True).start()
        
    def run_processing(self):
        try:
            # Ensure output directory exists
            os.makedirs(self.output_dir.get(), exist_ok=True)
            
            # Verificar que el documento existe
            if not os.path.exists(self.input_file.get()):
                raise FileNotFoundError(f"El documento Word no existe: {self.input_file.get()}")
                
            # Verificar que el modelo existe
            if not os.path.exists(self.model_path.get()):
                raise FileNotFoundError(f"El archivo del modelo no existe: {self.model_path.get()}")
                
            # Verificar que los datos de referencia existen
            if not os.path.exists(self.reference_data.get()):
                raise FileNotFoundError(f"El archivo de datos de referencia no existe: {self.reference_data.get()}")
            
            # Build command
            cmd = [
                sys.executable,
                "word_service.py",
                "--model-path", self.model_path.get(),
                "--reference-data", self.reference_data.get(),
                "--input-file", self.input_file.get(),
                "--output-dir", self.output_dir.get(),
                "--confidence-threshold", str(self.confidence.get()),
                "--surname-confidence-threshold", str(self.surname_confidence.get())
            ]
            
            # Add optional parameters if provided
            if self.surname_model_path.get():
                cmd.extend(["--surname-model-path", self.surname_model_path.get()])
                
            # Add NER flag
            if self.use_ner.get():
                cmd.append("--use-ner")
                
            # Add exclusion lists if provided
            if self.places_exclusion_file.get():
                cmd.extend(["--places-exclusion-file", self.places_exclusion_file.get()])
                
            if self.words_exclusion_file.get():
                cmd.extend(["--words-exclusion-file", self.words_exclusion_file.get()])
            
            # Log command
            self.log(f"Ejecutando: {' '.join(cmd)}")
            
            # Run the command and capture output
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read and log output
            for line in process.stdout:
                self.log(line.strip())
                
            # Wait for process to complete
            process.wait()
            
            # Check result
            if process.returncode == 0:
                self.status_var.set("Procesamiento completado con éxito")
                messagebox.showinfo(
                    "Éxito", 
                    f"Documento procesado correctamente.\n\n"
                    f"Los archivos de salida se encuentran en:\n{self.output_dir.get()}"
                )
            else:
                self.status_var.set("Error en el procesamiento")
                messagebox.showerror(
                    "Error", 
                    "Ocurrió un error durante el procesamiento. "
                    "Revise el registro para más detalles."
                )
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log(f"ERROR: {str(e)}")
            self.log(f"DETALLES: {error_details}")
            self.status_var.set("Error en el procesamiento")
            
            # Imprimir el error en la terminal también
            print("\n" + "=" * 50)
            print("ERROR DETECTADO:")
            print(f"Error: {str(e)}")
            print("\nDetalles completos del error:")
            print(error_details)
            print("=" * 50 + "\n")
            
            messagebox.showerror("Error", f"Error: {str(e)}\n\nRevise el registro para más detalles y la terminal para información completa.")
            
        finally:
            # Re-enable UI
            self.root.after(0, self.reset_ui)
            
    def reset_ui(self):
        self.processing = False
        self.process_button.config(state=tk.NORMAL)
        self.progress.stop()
        

if __name__ == "__main__":
    # Check if required files exist
    model_path = "models/random_forest_model.pkl"
    if not os.path.exists(model_path):
        print(f"ERROR: Modelo no encontrado en {model_path}")
        print("Primero debe entrenar el modelo ejecutando train_model.py")
        sys.exit(1)
        
    # Create output directory
    os.makedirs("data/corrected_word", exist_ok=True)
    
    # Start the app
    root = tk.Tk()
    app = SurnameCorrectiorApp(root)
    root.mainloop()
