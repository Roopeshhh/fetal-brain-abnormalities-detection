import os
import sqlite3
import uuid
import smtplib
import json
import urllib.request
import mimetypes
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta

import torch
import timm
from torchvision import transforms
from PIL import Image
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, g, Response
)
import csv
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps

# ==========================================
# CONFIG
# ==========================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fetalbrain-ai-secret-key-change-in-production")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(BASE_DIR, "instance", "fetal_brain.db")
MODEL_PATH = os.path.join(PARENT_DIR, "fetal_brain_swin_best.pth")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==========================================
# DATABASE
# ==========================================

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
    return g.db


def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with app.app_context():
        db = get_db()
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                phone TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                result_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                patient_name TEXT,
                patient_email TEXT,
                patient_phone TEXT,
                patient_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS support_tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT,
                category TEXT,
                subject TEXT,
                message TEXT NOT NULL,
                attachment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        try:
            db.execute("ALTER TABLE users ADD COLUMN name TEXT")
            db.commit()
        except sqlite3.OperationalError:
            pass
        for column in ["patient_name", "patient_email", "patient_phone", "patient_message"]:
            try:
                db.execute(f"ALTER TABLE predictions ADD COLUMN {column} TEXT")
                db.commit()
            except sqlite3.OperationalError:
                pass
        db.commit()


app.teardown_appcontext(close_db)

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")
try:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=len(classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded. {len(classes)} classes. Acc: {checkpoint.get('accuracy', 'N/A')}")
except Exception as e:
    print("MODEL LOAD ERROR:", e)
    model = None
    classes = []

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# HELPERS
# ==========================================

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def allowed_file(filename):
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXTENSIONS


def predict_image(filepath):
    image = Image.open(filepath).convert("RGB")
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        predicted_class = classes[pred.item()]
    return predicted_class.replace("_", " ").title(), round(confidence.item() * 100, 2)

# Web3Forms API Configuration
WEB3FORMS_KEY = os.environ.get("WEB3FORMS_ACCESS_KEY", "684d2858-47fb-48e5-bacb-9885f10cc2b5")

def send_web3forms_ticket(name, email, phone, category, subject, message):
    access_key = os.environ.get("WEB3FORMS_ACCESS_KEY", WEB3FORMS_KEY)
    if not access_key:
        print("[Web3Forms Note] Set WEB3FORMS_ACCESS_KEY to activate Web3Forms email delivery.")
        return False

    try:
        url = "https://api.web3forms.com/submit"
        full_message = f"Doctor/User Name: {name}\nDoctor Email: {email}\nPhone Number: {phone or 'Not Provided'}\nCategory: {category or 'General Support'}\n\nIssue Details:\n{message}"

        payload = {
            "access_key": access_key,
            "name": name,
            "email": email,
            "subject": f"[FetalBrain Support - {category}] {subject or 'New Support Ticket'}",
            "message": full_message,
            "from_name": "FetalBrain AI Support Desk"
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "FetalBrainAI/1.0"}
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            res = json.loads(response.read().decode("utf-8"))
            print("[Web3Forms API Response]:", res)
            if res.get("success"):
                print("[Web3Forms] SUCCESS! Email Delivered to dhanushre08@gmail.com!")
                return True
            else:
                print("[Web3Forms API Error Message]:", res.get("message"))
    except Exception as e:
        print("[Web3Forms Exception]:", str(e))
    return False

def send_support_email(name, email, phone, category, subject, message):
    receiver_email = os.environ.get("RECIPIENT_EMAIL", "dhanushre08@gmail.com")
    smtp_server = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("MAIL_PORT", 587))
    smtp_user = os.environ.get("MAIL_USERNAME")
    smtp_password = os.environ.get("MAIL_PASSWORD")

    email_body = f"""
==================================================
NEW CLINICAL / TECHNICAL SUPPORT TICKET SUBMITTED
==================================================
Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Doctor/User Name: {name}
Doctor Email: {email}
Phone Number: {phone or 'Not Provided'}
Category: {category or 'General Support'}
Subject: {subject or 'No Subject'}

MESSAGE / ISSUE DESCRIPTION:
--------------------------------------------------
{message}
"""

    print("--------------------------------------------------")
    print("SUPPORT TICKET DISPATCH LOG:")
    print(email_body)
    print("--------------------------------------------------")

    web3_sent = send_web3forms_ticket(name, email, phone, category, subject, message)

    if not web3_sent and smtp_user and smtp_password:
        try:
            msg = MIMEMultipart()
            msg["From"] = smtp_user
            msg["To"] = receiver_email
            msg["Subject"] = f"[FetalBrain Support - {category}] {subject or 'New Support Ticket'}"
            msg.attach(MIMEText(email_body, "plain"))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, receiver_email, msg.as_string())
            server.quit()
            print("Support Ticket Email sent successfully via SMTP!")
        except Exception as e:
            print("SMTP Send Error:", str(e))

# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        category = request.form.get("category", "General Support").strip()
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()

        if not name or not email or not message:
            flash("Please fill in your name, email address, and issue description.", "danger")
            return redirect(url_for("contact"))

        db = get_db()
        db.execute(
            """INSERT INTO support_tickets
               (name, email, phone, category, subject, message)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, email, phone, category, subject, message)
        )
        db.commit()

        send_support_email(name, email, phone, category, subject, message)

        contact_info = f"at {email}" + (f" or {phone}" if phone else "")
        flash(
            f"Thank you, Dr. {name}! Your support ticket has been submitted to our engineering team. "
            f"We will review your request and contact you {contact_info} shortly.",
            "success"
        )
        return redirect(url_for("contact"))

    return render_template("contact.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        phone = request.form["phone"].strip()
        password = request.form["password"]
        confirm = request.form["confirm_password"]

        if not name or not email or not phone or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        if len(password) < 4:
            flash("Password must be at least 4 characters.", "danger")
            return render_template("register.html")

        db = get_db()
        if db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone():
            flash("Email already registered.", "danger")
            return render_template("register.html")

        pw_hash = generate_password_hash(password)
        db.execute(
            "INSERT INTO users (name, email, phone, password_hash) VALUES (?, ?, ?, ?)",
            (name, email, phone, pw_hash)
        )
        db.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.permanent = True
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            session["email"] = user["email"]
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/profile")
@login_required
def profile():
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    predictions = db.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 50",
        (session["user_id"],)
    ).fetchall()
    return render_template("profile.html", user=user, predictions=predictions)


def get_user_predictions():
    db = get_db()
    return db.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC",
        (session["user_id"],)
    ).fetchall()


def get_prediction_owned(prediction_id):
    db = get_db()
    return db.execute(
        "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
        (prediction_id, session["user_id"])
    ).fetchone()


def build_csv_response(predictions):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["#", "Patient Name", "Email", "Phone", "Message", "Result", "Confidence (%)", "Date"])
    for i, p in enumerate(predictions, 1):
        writer.writerow([
            i,
            p["patient_name"] or "",
            p["patient_email"] or "",
            p["patient_phone"] or "",
            p["patient_message"] or "",
            p["result_class"],
            p["confidence"],
            p["created_at"]
        ])

    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=fetal_brain_report.csv"}
    )


def build_pdf_response(predictions, title="FetalBrain AI Report"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=title
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.alignment = TA_CENTER
    heading_style = styles["Heading2"]

    elements = [Paragraph(title, title_style), Spacer(1, 4 * mm)]
    elements.append(Paragraph("FetalBrain AI Detection Report", heading_style))
    elements.append(Spacer(1, 6 * mm))

    data = [["#", "Patient", "Email", "Phone", "Result", "Confidence", "Date"]]
    for i, p in enumerate(predictions, 1):
        data.append([
            str(i),
            p["patient_name"] or "-",
            p["patient_email"] or "-",
            p["patient_phone"] or "-",
            p["result_class"],
            f'{p["confidence"]}%',
            str(p["created_at"])
        ])

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e7e8e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)

    return Response(
        buffer.getvalue(),
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment; filename=fetal_brain_report.pdf"}
    )


@app.route("/profile/export/csv")
@login_required
def export_csv():
    return build_csv_response(get_user_predictions())


@app.route("/profile/export/pdf")
@login_required
def export_pdf():
    return build_pdf_response(get_user_predictions())


@app.route("/profile/export/csv/<int:prediction_id>")
@login_required
def export_single_csv(prediction_id):
    p = get_prediction_owned(prediction_id)
    if not p:
        flash("Prediction not found.", "danger")
        return redirect(url_for("profile"))
    return build_csv_response([p])


@app.route("/profile/export/pdf/<int:prediction_id>")
@login_required
def export_single_pdf(prediction_id):
    p = get_prediction_owned(prediction_id)
    if not p:
        flash("Prediction not found.", "danger")
        return redirect(url_for("profile"))
    return build_pdf_response([p], title="FetalBrain AI - Single Prediction Report")


@app.route("/analyze")
@login_required
def analyze():
    result = session.pop("last_result", None)
    return render_template(
        "predict.html",
        prediction=result["class"] if result else None,
        confidence=result["confidence"] if result else None,
        image_file=result["image"] if result else None
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if model is None:
        flash("Model not loaded. Contact admin.", "danger")
        return redirect(url_for("analyze"))

    if "file" not in request.files:
        flash("No file selected.", "danger")
        return redirect(url_for("analyze"))

    file = request.files["file"]
    if not file.filename:
        flash("No file selected.", "danger")
        return redirect(url_for("analyze"))

    if not allowed_file(file.filename):
        flash("Allowed file types: JPG, JPEG, PNG, BMP.", "danger")
        return redirect(url_for("analyze"))

    try:
        ext = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(UPLOAD_DIR, unique_name)
        file.save(save_path)

        result_class, confidence = predict_image(save_path)

        patient_name = request.form.get("patient_name", "").strip()
        patient_email = request.form.get("patient_email", "").strip()
        patient_phone = request.form.get("patient_phone", "").strip()
        patient_message = request.form.get("patient_message", "").strip()

        db = get_db()
        db.execute(
            """INSERT INTO predictions
               (user_id, filename, original_name, result_class, confidence,
                patient_name, patient_email, patient_phone, patient_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session["user_id"], unique_name, file.filename, result_class, confidence,
             patient_name, patient_email, patient_phone, patient_message)
        )
        db.commit()

        session["last_result"] = {
            "class": result_class,
            "confidence": confidence,
            "image": unique_name
        }
        return redirect(url_for("analyze"))

    except Exception as e:
        flash(f"Prediction error: {str(e)}", "danger")
        return redirect(url_for("analyze"))


# ==========================================
# ENTRY
# ==========================================

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
