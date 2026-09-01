# 🧠 FetalBrain AI - Fetal Brain Abnormality Detection System

An AI-powered deep learning diagnostic system utilizing **Swin Transformer** for automated detection and classification of fetal brain abnormalities from ultrasound images.

---

## 📁 Project Structure

```text
final_fetal_detection_updated/
│
├── fetal_brain_swin_best.pth   # Pre-trained Swin Transformer PyTorch Model Checkpoint
├── requirements.txt            # Python Dependencies
├── README.md                   # Complete Setup & Run Instructions
├── run_app.bat                 # 1-Click Startup Script for Windows
├── run_app.sh                  # 1-Click Startup Script for macOS/Linux
│
├── webapp/                     # Flask Web Application
│   ├── app.py                  # Main Flask Server & Inference Pipeline
│   ├── requirements.txt        # Web App Dependencies
│   ├── instance/               # SQLite Database (auto-generated)
│   ├── static/                 # CSS, JavaScript, and Uploaded Images
│   └── templates/              # HTML Templates (UI)
│
├── datasets/                   # Sample Ultrasound Test Datasets
├── test.py                     # Standalone Desktop GUI Application (Tkinter)
└── train.ipynb                 # Model Training & Evaluation Jupyter Notebook
```

---

## 🚀 Quick Start Guide (Step-by-Step)

### 📌 Prerequisites
Make sure **Python 3.9, 3.10, or 3.11** is installed on your system.
- Download Python: https://www.python.org/downloads/
- *Note for Windows*: During installation, make sure to check **"Add Python to PATH"**.

---

### Step 1: Unzip the Project
Extract the zip file to your preferred folder (e.g., `Desktop` or `Documents`).

---

### Step 2: Open Terminal / Command Prompt in the Project Folder
- **Windows**: Open the unzipped folder, click the address bar, type `cmd`, and hit `Enter`.
- **macOS / Linux**: Open Terminal and `cd` into the unzipped project folder.

---

### Step 3: Create & Activate a Virtual Environment (Recommended)

#### On Windows:
```cmd
python -m venv venv
venv\Scripts\activate
```

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 4: Install Project Dependencies
Run the following command to install all required libraries:

```bash
pip install -r requirements.txt
```

> **Optional (GPU Support)**:
> If your system has an NVIDIA GPU and you want CUDA acceleration, install PyTorch with CUDA from [pytorch.org](https://pytorch.org):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

### Step 5: Run the Web Application

#### Option A: Running with Python
```bash
python webapp/app.py
```
*(Or navigate into `webapp` and run `python app.py`)*

#### Option B (Windows 1-Click):
Double-click `run_app.bat` in the project root folder.

---

### Step 6: Access the Application in Your Browser
Once the server starts, open your web browser (Chrome, Edge, Firefox, Brave) and go to:

👉 **`http://127.0.0.1:5000`** *(or `http://localhost:5000`)*

---

## 🖥️ Alternative: Running the Desktop GUI (Tkinter)
If you want to test the model using a standalone desktop GUI instead of the web browser:

```bash
python test.py
```
1. Click **"Select Ultrasound Image"**.
2. Select any scan from the `datasets/` folder.
3. View instant AI predictions with confidence scores.

---

## 🔑 How to Use the Web Application

1. **Register an Account**: Click **Register** in the top navbar and create a user profile.
2. **Login**: Log in with your email and password.
3. **Analyze Ultrasound Scan**:
   - Go to **Analyze** in the navbar.
   - Enter patient information (Name, Age, Gestational Age, etc.).
   - Upload a fetal brain ultrasound scan (`.png`, `.jpg`, `.jpeg`, `.bmp`).
   - Click **Run AI Analysis**.
4. **View & Download Report**:
   - Inspect the classification result and confidence score.
   - Click **Download Medical PDF Report** to generate a formatted clinical report.
5. **History & Profile**:
   - Access **Profile** to view past patient analyses and re-download reports anytime.

---

## 🛠️ Troubleshooting & FAQ

#### 1. `ModuleNotFoundError: No module named '...'`
Ensure your virtual environment is activated and run:
```bash
pip install -r requirements.txt
```

#### 2. `Address already in use` or Port 5000 busy
If port 5000 is occupied by another application, run on a different port:
- Open `webapp/app.py`, scroll to the bottom, and change `port=5000` to `port=5050`.
- Then visit `http://127.0.0.1:5050`.

#### 3. Where is the database stored?
The application uses SQLite. The database is automatically initialized at `webapp/instance/fetal_brain.db` on first run.

---

## 📄 License & Attribution
Developed for Fetal Brain Abnormality Detection Research.
