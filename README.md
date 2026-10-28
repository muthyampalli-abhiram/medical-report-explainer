<!-- Use only synthetic or anonymized reports. Never commit real patient data. -->

# Medical Report Explainer

## Critical Thresholds & Medical Descriptions
> [!IMPORTANT]
> The critical threshold values and plain-language test descriptions used in this project are illustrative demo content and must be reviewed by a qualified clinician before any real clinical use.

Medical Report Explainer is a local-first application designed to explain medical laboratory reports in clear, plain language. It empowers users to understand diagnostic biomarkers and monitor health trends across multiple test results over time. The application is built with a privacy-by-design approach, keeping health metrics and documents strictly on the local machine.

## Setup & Running

### 1. Create Virtual Environment
```bash
python -m venv .venv
```

Activate the virtual environment:
- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (CMD)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Development Server
```bash
uvicorn app.main:app --reload
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000) with interactive documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4. Run Tests
```bash
pytest
```

## Using the Web Interface

### Starting and Accessing the Interface
1. Start the server using Uvicorn:
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
2. Open your web browser and navigate to:
   [http://127.0.0.1:8000](http://127.0.0.1:8000)

### Supported File Types
- **Lab Reports ("My Reports" tab)**: `.pdf` (digital PDF reports), `.png`, `.jpg`, `.jpeg` (photos/scans up to 10 MB).
- **Prescriptions ("Prescription" tab)**: `.pdf` (digital PDF prescriptions).

### Privacy & Synthetic Data Reminder
> [!WARNING]
> Always use **synthetic, sample, or anonymized reports** for testing and demonstration. Never upload documents containing sensitive personal health identifiers or real patient data. All processing is local-first, files are deleted immediately after reading, and only general test names are ever processed by language models if enabled.
