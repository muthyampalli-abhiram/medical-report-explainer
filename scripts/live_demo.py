"""Live demonstration script running the FastAPI server and querying it via requests."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import requests

PORT = 8089
BASE_URL = f"http://127.0.0.1:{PORT}"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
TABLE_PDF = SAMPLES_DIR / "report_table.pdf"
LINES_PDF = SAMPLES_DIR / "report_lines.pdf"


def run_live_demo() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        temp_db = Path(tmpdir) / "live_demo.db"

        # Prepare environment
        env = os.environ.copy()
        env["DB_PATH"] = str(temp_db)
        env["LLM_PROVIDER"] = "fake"

        # Start uvicorn server in a subprocess
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ]
        proc = subprocess.Popen(cmd, env=env)

        try:
            # Wait for server to start
            for _ in range(30):
                try:
                    r = requests.get(f"{BASE_URL}/health", timeout=1)
                    if r.status_code == 200:
                        break
                except requests.RequestException:
                    time.sleep(0.2)
            else:
                raise RuntimeError("Server failed to start within timeout.")

            print("=" * 70)
            print("1. UPLOAD report_table.pdf (First Report)")
            print("=" * 70)
            with open(TABLE_PDF, "rb") as f:
                upload1_resp = requests.post(
                    f"{BASE_URL}/api/reports/upload",
                    files={"file": ("report_table.pdf", f, "application/pdf")},
                    data={"profile_id": 1},
                )
            print(f"Status Code: {upload1_resp.status_code}")
            data1 = upload1_resp.json()
            report1_id = data1["report_id"]
            print(f"Report ID: {report1_id}")
            print(f"Date: {data1['report_date']}, Lab: {data1['lab_name']}")
            print(f"Results: {data1['result_count']}, Abnormal: {data1['abnormal_count']}, Critical: {data1['critical_count']}")
            print()

            print("=" * 70)
            print("2. UPLOAD report_lines.pdf (Second Report for Trends)")
            print("=" * 70)
            with open(LINES_PDF, "rb") as f:
                upload2_resp = requests.post(
                    f"{BASE_URL}/api/reports/upload",
                    files={"file": ("report_lines.pdf", f, "application/pdf")},
                    data={"profile_id": 1},
                )
            print(f"Status Code: {upload2_resp.status_code}")
            data2 = upload2_resp.json()
            report2_id = data2["report_id"]
            print(f"Report ID: {report2_id}")
            print(f"Date: {data2['report_date']}, Lab: {data2['lab_name']}")
            print(f"Results: {data2['result_count']}, Abnormal: {data2['abnormal_count']}, Critical: {data2['critical_count']}")
            print()

            print("=" * 70)
            print("3. GET /api/profiles/1/trends/HbA1c")
            print("=" * 70)
            trend_resp = requests.get(f"{BASE_URL}/api/profiles/1/trends/HbA1c")
            print(f"Status Code: {trend_resp.status_code}")
            trend_points = trend_resp.json()
            print("HbA1c Trend Points:")
            for pt in trend_points:
                print(f"  - Date: {pt['report_date']} | Value: {pt['value']} {pt['unit']} | Flag: {pt['flag']}")
            print()

            print("=" * 70)
            print(f"4. GET /api/reports/{report1_id}/questions")
            print("=" * 70)
            q_resp = requests.get(f"{BASE_URL}/api/reports/{report1_id}/questions")
            print(f"Status Code: {q_resp.status_code}")
            q_data = q_resp.json()
            print("Questions for Doctor:")
            for i, q in enumerate(q_data["questions"], 1):
                print(f"  {i}. {q['text']}")
                print(f"     [Reason: {q['reason']}]")
            print("\nGeneral Questions:")
            for gq in q_data["general_questions"]:
                print(f"  - {gq}")
            if q_data["trend_notes"]:
                print("\nTrend Notes:")
                for tn in q_data["trend_notes"]:
                    print(f"  - {tn}")
            print()

            print("=" * 70)
            print(f"5. GET /api/reports/{report1_id}/explanation")
            print("=" * 70)
            exp_resp = requests.get(f"{BASE_URL}/api/reports/{report1_id}/explanation")
            print(f"Status Code: {exp_resp.status_code}")
            exp_data = exp_resp.json()
            print(f"Overall Notice: {exp_data['overall_notice']}")
            print(f"Disclaimer: {exp_data['disclaimer']}")
            print("\nExplanations:")
            for exp in exp_data["explanations"]:
                print(f"  - {exp['test_name']} ({exp['flag']}): {exp['result_sentence']}")
                print(f"    What it measures: {exp['what_it_measures']} (fallback: {exp['used_fallback']})")
            print()

            print("=" * 70)
            print("6. POST /api/prescriptions/parse (Prescription Text PDF)")
            print("=" * 70)
            prescription_pdf = SAMPLES_DIR / "prescription_text.pdf"
            with open(prescription_pdf, "rb") as f:
                rx_resp = requests.post(
                    f"{BASE_URL}/api/prescriptions/parse",
                    files={"file": ("prescription_text.pdf", f, "application/pdf")},
                )
            print(f"Status Code: {rx_resp.status_code}")
            rx_data = rx_resp.json()
            print("Extracted Medications:")
            for med in rx_data["medications"]:
                print(f"  - {med['name_as_written']} ({med['form']}, {med['strength']}): {med['frequency_raw']} -> {med['frequency_plain']}")
                if med['instructions_raw']:
                    print(f"    Instructions: {med['instructions_raw']}")
            print(f"Disclaimer: {rx_data['disclaimer']}")
            print()

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    run_live_demo()
