import os
import sys
import io
import json
import sqlite3

# Set working directory to project root
sys.path.insert(0, os.path.abspath("webapp"))

from webapp.app import app, init_db, get_db

def run_tests():
    print("=== Starting FetalBrain Trace & Evaluation Integration Test ===")
    init_db()
    
    client = app.test_client()
    
    # 1. Test Evaluation Page
    print("\n[Test 1] Testing /evaluation page...")
    res = client.get("/evaluation")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    html = res.data.decode("utf-8")
    assert "Model Performance Evaluation" in html, "Missing evaluation title"
    assert "Methodological Reconciliation" in html, "Missing reconciliation section"
    assert "15 × 15 Categorical Confusion Matrix" in html, "Missing confusion matrix"
    print("[OK] /evaluation passed!")
    
    # 2. Register & Login Test Doctor
    print("\n[Test 2] Creating and logging in test doctor...")
    with app.app_context():
        db = get_db()
        existing = db.execute("SELECT id FROM users WHERE email = 'tester@fetalbrain.org'").fetchone()
        if not existing:
            cur = db.execute(
                "INSERT INTO users (name, email, phone, organisation, password_hash, status) VALUES (?, ?, ?, ?, ?, ?)",
                ("Dr. Trace Tester", "tester@fetalbrain.org", "+919876543210", "AIT Hospital", "dummyhash", "approved")
            )
            db.commit()
            uid = cur.lastrowid
        else:
            uid = existing["id"]

    with client.session_transaction() as sess:
        sess["user_id"] = uid
        sess["name"] = "Dr. Trace Tester"
        sess["email"] = "tester@fetalbrain.org"
        sess["role"] = "doctor"
        sess["is_admin"] = False
        
    # 3. Test Prediction & Deterministic Trace Generation
    print("\n[Test 3] Uploading test ultrasound scan to /predict...")
    sample_img_dir = os.path.join("datasets", "normal")
    sample_img_name = [f for f in os.listdir(sample_img_dir) if f.endswith((".jpg", ".png"))][0]
    sample_img_path = os.path.join(sample_img_dir, sample_img_name)
    
    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()
        
    data = {
        "file": (io.BytesIO(img_bytes), sample_img_name),
        "patient_name": "Test Subject 01",
        "patient_email": "patient01@example.com",
        "patient_phone": "+91 9876543210",
        "patient_message": "Seminar Audit Verification Scan"
    }
    
    pred_res = client.post("/predict", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert pred_res.status_code == 200, f"Expected 200, got {pred_res.status_code}"
    pred_html = pred_res.data.decode("utf-8")
    assert "Developer Trace" in pred_html, "Developer Trace button missing from predict.html"
    assert "Softmax Probabilities" in pred_html, "Softmax Probabilities label missing from predict.html"
    print("[OK] /predict upload and result page passed!")
    
    # 4. Check Database Trace Persistence
    print("\n[Test 4] Verifying database records in predictions and prediction_traces...")
    with app.app_context():
        db = get_db()
        last_pred = db.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 1").fetchone()
        assert last_pred is not None, "No prediction found in DB"
        pred_id = last_pred["id"]
        
        trace_row = db.execute("SELECT * FROM prediction_traces WHERE prediction_id = ?", (pred_id,)).fetchone()
        assert trace_row is not None, f"No trace record found in prediction_traces for pred_id {pred_id}"
        
        trace_dict = json.loads(trace_row["trace_data"])
        assert trace_dict["prediction_id"] == pred_id, "Mismatch prediction_id in trace dict"
        assert "source" in trace_dict, "Missing source in trace dict"
        assert "validation" in trace_dict, "Missing validation in trace dict"
        assert "channels" in trace_dict, "Missing channels in trace dict"
        assert "normalization" in trace_dict, "Missing normalization in trace dict"
        assert "stages" in trace_dict, "Missing stages in trace dict"
        assert "classification" in trace_dict, "Missing classification in trace dict"
        assert "gradcam" in trace_dict, "Missing gradcam in trace dict"
        assert "execution" in trace_dict, "Missing execution in trace dict"
        print(f"[OK] Trace record for prediction #{pred_id} verified in SQLite!")
        
    # 5. Test /prediction/<id>/trace Full Page
    print(f"\n[Test 5] Requesting full Developer Trace page /prediction/{pred_id}/trace...")
    trace_res = client.get(f"/prediction/{pred_id}/trace")
    assert trace_res.status_code == 200, f"Expected 200, got {trace_res.status_code}"
    trace_html = trace_res.data.decode("utf-8")
    assert "Source Image & Photometric Integrity Checks" in trace_html, "Panel 1 missing"
    assert "RGB Conversion & 3-Channel Channel Dissection" in trace_html, "Panel 2 missing"
    assert "Resize & Spatial Resampling" in trace_html, "Panel 3 missing"
    assert "Tensor Contract & Normalization Proof" in trace_html, "Panel 4 missing"
    assert "Swin-Tiny Hierarchical Feature Extraction Stepper" in trace_html, "Panel 5 missing"
    assert "Classification Evidence & Complete 15-Class Distribution" in trace_html, "Panel 6 missing"
    assert "Explainable AI (Grad-CAM) Visual Matrix" in trace_html, "Panel 7 missing"
    assert "Reproducibility & Verification Audit Trail" in trace_html, "Panel 8 missing"
    print("[OK] All 8 panels rendered in trace page successfully!")
    
    # 6. Test /prediction/<id>/trace/json Endpoint
    print(f"\n[Test 6] Requesting /prediction/{pred_id}/trace/json...")
    json_res = client.get(f"/prediction/{pred_id}/trace/json")
    assert json_res.status_code == 200, f"Expected 200, got {json_res.status_code}"
    assert json_res.mimetype == "application/json", f"Expected application/json, got {json_res.mimetype}"
    parsed_json = json.loads(json_res.data.decode("utf-8"))
    assert parsed_json["prediction_id"] == pred_id, "Invalid JSON payload prediction_id"
    print("[OK] Trace JSON download endpoint passed!")
    
    # 7. Test PDF Generation
    print(f"\n[Test 7] Requesting PDF report /profile/export/pdf/{pred_id}...")
    pdf_res = client.get(f"/profile/export/pdf/{pred_id}")
    assert pdf_res.status_code == 200, f"Expected 200, got {pdf_res.status_code}"
    assert pdf_res.mimetype == "application/pdf", f"Expected application/pdf, got {pdf_res.mimetype}"
    assert len(pdf_res.data) > 1000, "PDF byte length suspiciously small"
    # 8. Test Admin Prediction Upload (Foreign Key & user_id=0 Safety)
    print("\n[Test 8] Testing upload by Administrator (user_id=0)...")
    with client.session_transaction() as sess:
        sess["user_id"] = "admin"
        sess["name"] = "System Administrator"
        sess["email"] = "admin@fetalbrain.org"
        sess["role"] = "admin"
        sess["is_admin"] = True
        
    admin_data = {
        "file": (io.BytesIO(img_bytes), sample_img_name),
        "patient_name": "Admin Test Patient",
        "patient_email": "adminpatient@example.com",
        "patient_phone": "+91 9999988888",
        "patient_message": "Admin foreign key verification"
    }
    admin_pred_res = client.post("/predict", data=admin_data, content_type="multipart/form-data", follow_redirects=True)
    assert admin_pred_res.status_code == 200, f"Expected 200, got {admin_pred_res.status_code}"
    admin_pred_html = admin_pred_res.data.decode("utf-8")
    assert "FOREIGN KEY constraint failed" not in admin_pred_html, "Foreign key constraint failed on Admin upload"
    assert "Developer Trace" in admin_pred_html, "Developer Trace button missing from admin prediction result"
    print("[OK] Admin scan upload and foreign key constraint verified!")

    print("\n=======================================================")
    print(" ALL TESTS PASSED SUCCESSFULLY! FULL PIPELINE VERIFIED! ")
    print("=======================================================")

if __name__ == "__main__":
    run_tests()
