import customtkinter as ctk
import pandas as pd
import joblib
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tkinter import filedialog

from data_loader import load_data


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "riskguard_pipeline.joblib"


def load_dataset():
    try:
        df = load_data()

        status_label.configure(
            text=f"Dataset loaded successfully!\nRows: {df.shape[0]}\nColumns: {df.shape[1]}"
        )

        maintenance_label.configure(
            text="System Status: Dataset loaded successfully"
        )

    except Exception as e:
        status_label.configure(
            text=f"Error loading dataset:\n{e}"
        )


def upload_applicant_file():
    try:
        file_path = filedialog.askopenfilename(
            title="Select applicant CSV file",
            filetypes=[("CSV files", "*.csv")]
        )

        if not file_path:
            return

        applicant_df = pd.read_csv(file_path)

        model = joblib.load(MODEL_PATH)

        applicant_df = applicant_df.drop(
            columns=[
                "ID",
                "Interest_rate_spread",
                "Upfront_charges",
                "rate_of_interest",
                "Status"
            ],
            errors="ignore"
        )

        prediction = model.predict(applicant_df)[0]
        probability = model.predict_proba(applicant_df)[0]

        approval_confidence = probability[0] * 100
        risk_confidence = probability[1] * 100

        if prediction == 0:
            result = (
                f"Estimated Result: Likely Approved\n"
                f"Confidence: {approval_confidence:.2f}%"
            )
        else:
            result = (
                f"Estimated Result: Higher Risk / May Not Be Approved\n"
                f"Risk Confidence: {risk_confidence:.2f}%"
            )

        status_label.configure(
            text=f"Applicant file processed successfully!\n{result}"
        )

        maintenance_label.configure(
            text="System Status: Last prediction completed successfully"
        )

    except Exception as e:
        status_label.configure(
            text=f"Error processing applicant file:\n{e}"
        )


def show_income_histogram():
    df = load_data()

    df["income"].hist(bins=30)
    plt.xlim(0, 100000)

    plt.title("Income Distribution")
    plt.xlabel("Income")
    plt.ylabel("Frequency")
    plt.show(block=False)
    plt.pause(0.1)


def show_scatter_plot():
    df = load_data()

    plt.scatter(df["income"], df["loan_amount"])

    plt.title("Income vs Loan Amount")
    plt.xlabel("Income")
    plt.ylabel("Loan Amount")
    plt.show(block=False)
    plt.pause(0.1)


def show_correlation_heatmap():
    df = load_data()

    columns_to_drop = [
        "ID",
        "rate_of_interest",
        "Interest_rate_spread",
        "Upfront_charges"
    ]

    df = df.drop(columns=columns_to_drop, errors="ignore")

    correlation = df.select_dtypes(include=["float64", "int64"]).corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(correlation, cmap="coolwarm", aspect="auto")
    plt.colorbar()

    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation.columns)), correlation.columns)

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def show_model_status():
    if MODEL_PATH.exists():
        maintenance_label.configure(
            text=f"System Status: Model found\nModel Path: {MODEL_PATH.name}"
        )
    else:
        maintenance_label.configure(
            text="System Status: Model file not found"
        )


app = ctk.CTk()

app.title("RiskGuard")
app.geometry("1050x720")
app.resizable(False, False)


main_frame = ctk.CTkScrollableFrame(app, corner_radius=20)
main_frame.pack(padx=35, pady=35, fill="both", expand=True)


title_label = ctk.CTkLabel(
    main_frame,
    text="RiskGuard",
    font=("Arial", 42, "bold")
)
title_label.pack(pady=(25, 5))


subtitle_label = ctk.CTkLabel(
    main_frame,
    text="Machine Learning Loan Risk Decision Support Dashboard",
    font=("Arial", 18),
    text_color="gray"
)
subtitle_label.pack(pady=(0, 25))


top_button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
top_button_frame.pack(pady=5)


load_button = ctk.CTkButton(
    top_button_frame,
    text="Load Training Dataset",
    command=load_dataset,
    width=240,
    height=42,
    font=("Arial", 15, "bold"),
    corner_radius=12
)
load_button.grid(row=0, column=0, padx=10, pady=10)


upload_button = ctk.CTkButton(
    top_button_frame,
    text="Upload Applicant CSV",
    command=upload_applicant_file,
    width=240,
    height=42,
    font=("Arial", 15, "bold"),
    corner_radius=12
)
upload_button.grid(row=0, column=1, padx=10, pady=10)


model_button = ctk.CTkButton(
    top_button_frame,
    text="Check Model Status",
    command=show_model_status,
    width=240,
    height=42,
    font=("Arial", 15, "bold"),
    corner_radius=12
)
model_button.grid(row=0, column=2, padx=10, pady=10)


visual_label = ctk.CTkLabel(
    main_frame,
    text="Data Exploration Visualizations",
    font=("Arial", 20, "bold")
)
visual_label.pack(pady=(25, 5))


visual_button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
visual_button_frame.pack(pady=5)


hist_button = ctk.CTkButton(
    visual_button_frame,
    text="Income Histogram",
    command=show_income_histogram,
    width=220,
    height=40,
    corner_radius=12
)
hist_button.grid(row=0, column=0, padx=10, pady=10)


scatter_button = ctk.CTkButton(
    visual_button_frame,
    text="Income vs Loan Amount",
    command=show_scatter_plot,
    width=220,
    height=40,
    corner_radius=12
)
scatter_button.grid(row=0, column=1, padx=10, pady=10)


heatmap_button = ctk.CTkButton(
    visual_button_frame,
    text="Correlation Heatmap",
    command=show_correlation_heatmap,
    width=220,
    height=40,
    corner_radius=12
)
heatmap_button.grid(row=0, column=2, padx=10, pady=10)


result_card = ctk.CTkFrame(main_frame, corner_radius=16, width=760, height=190)
result_card.pack(pady=25)
result_card.pack_propagate(False)


result_title = ctk.CTkLabel(
    result_card,
    text="Prediction Output",
    font=("Arial", 20, "bold")
)
result_title.pack(pady=(20, 8))


status_label = ctk.CTkLabel(
    result_card,
    text="No applicant file uploaded yet.",
    font=("Arial", 17),
    justify="center",
    wraplength=680
)
status_label.pack(pady=10)


maintenance_card = ctk.CTkFrame(main_frame, corner_radius=16, width=760, height=95)
maintenance_card.pack(pady=10)
maintenance_card.pack_propagate(False)


maintenance_title = ctk.CTkLabel(
    maintenance_card,
    text="Monitoring and Maintenance",
    font=("Arial", 18, "bold")
)
maintenance_title.pack(pady=(12, 2))


maintenance_label = ctk.CTkLabel(
    maintenance_card,
    text="System Status: Waiting for action",
    font=("Arial", 14),
    text_color="gray",
    wraplength=700
)
maintenance_label.pack(pady=5)


footer_label = ctk.CTkLabel(
    main_frame,
    text="Security Notice: Applicant data is processed locally and is not uploaded to external servers.",
    font=("Arial", 13),
    text_color="gray",
    wraplength=750
)
footer_label.pack(pady=(15, 5))


app.mainloop()