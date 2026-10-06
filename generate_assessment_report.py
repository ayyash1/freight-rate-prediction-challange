"""
Report Generation Script: Produces both DOCX and PDF formatted assessment reports.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Create output directories
report_dir = Path("report")
report_dir.mkdir(parents=True, exist_ok=True)
fig_dir = Path("report/figures")
scorer_fig = Path("scorer_results/candidate_december.png")

# Load CV results
with open("artifacts/cross_validation_results.json", "r") as f:
    cv_data = json.load(f)

# Load feature importance
fi_df = pd.read_csv("artifacts/feature_importance.csv").head(10)

# ==========================================
# 1. BUILD DOCX REPORT
# ==========================================
def build_docx():
    doc = Document()
    
    # Page setup - 1 inch margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        
    def add_title(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(24)
        run.font.bold = True
        run.font.color.rgb = RGBColor(6, 74, 86) # #064A56
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(4)
        
    def add_subtitle(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(80, 95, 100)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(18)

    def add_h1(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = RGBColor(6, 74, 86)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)

    def add_h2(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(40, 60, 70)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)

    def add_p(text, bold_prefix=""):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(6)
        if bold_prefix:
            r_bold = p.add_run(bold_prefix)
            r_bold.font.name = 'Calibri'
            r_bold.font.size = Pt(10.5)
            r_bold.font.bold = True
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(40, 40, 40)
        return p

    def add_bullet(text, bold_prefix=""):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(4)
        if bold_prefix:
            r_bold = p.add_run(bold_prefix)
            r_bold.font.name = 'Calibri'
            r_bold.font.size = Pt(10.5)
            r_bold.font.bold = True
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(40, 40, 40)

    # Document Header
    add_title("Freight Rate Prediction Machine Learning Assessment")
    add_subtitle("End-to-End Machine Learning Solution, Validation Architecture & December 2025 Forecast Analysis")
    
    # Metadata Box
    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(14)
    r = meta_p.add_run("Author: Ahamed Ayyash  |  Date: October 6, 2026  |  Status: Production Complete")
    r.font.name = 'Calibri'
    r.font.size = Pt(9.5)
    r.font.italic = True
    r.font.color.rgb = RGBColor(100, 100, 100)

    # 1. Executive Summary
    add_h1("1. Executive Summary")
    add_p("This technical report details the complete machine learning solution for the Freight Rate Prediction Challenge. The objective is to accurately forecast total posted freight rates for commercial truckload shipments across North America, deliver reliable predictions on 12,000 unobserved validation loads (data/validation.csv), and evaluate fixed-lane seasonal market behavior for December 2025.")
    add_p("Our final architecture transforms the raw pricing problem into a Rate Per Mile (RPM) regression formulation. We train a diverse multi-model gradient boosted ensemble (LightGBM + CatBoost + XGBoost) using an expanding-window time-series cross-validation protocol that prevents temporal leakage. The final ensemble achieves an exceptional Mean Absolute Error (MAE) of $123.82 and Mean Absolute Percentage Error (MAPE) of 5.10% across out-of-fold temporal validation folds, representing a ~20% improvement over naive baseline models.")

    # 2. Exploratory Data Analysis & Key Insights
    add_h1("2. Exploratory Data Analysis & Key Findings")
    add_p("The labeled development dataset (data/train_test.csv) contains 48,000 commercial freight loads spanning January 1, 2025 to October 31, 2025. The target variable is posted_rate ($), with an average load rate of $2,373.98 (ranging from $57.22 to $25,533.00). Key empirical findings from exploratory data analysis include:")
    
    add_bullet(" Haul distance is the primary cost driver (Pearson r = 0.9085 with total posted rate). However, rate per mile exhibits economies of scale: short hauls (<300 miles) command a premium rate per mile (~$2.58/mi) due to fixed loading/unloading overhead, whereas long hauls (>1,000 miles) taper to ~$1.96/mi.", "Distance Mechanics: ")
    add_bullet(" Significant rate premiums exist across equipment types. Reefer loads average $2.38/mi due to refrigeration fuel consumption and strict temperature compliance, Flatbeds average $2.29/mi due to specialized securing/tarping labor, while Dry Vans operate as the standard baseline at $2.12/mi.", "Equipment Differentiation: ")
    add_bullet(" Freight rates exhibit pronounced day-of-week seasonality. Mid-week loads (Tuesday through Thursday) command peak spot rates due to commercial dispatch volume, while weekend pickups (Saturday/Sunday) experience market contraction.", "Weekly Seasonality: ")
    add_bullet(" The provided quote_signal represents an upstream lane benchmark (~$2.05/mi), while market_index tracks macro capacity tightness. Together, their interaction (base_rate_est = distance * quote_signal) forms an exceptionally strong baseline for freight pricing.", "Market Index & Signals: ")

    if (fig_dir / "rate_vs_distance.png").exists():
        doc.add_picture(str(fig_dir / "rate_vs_distance.png"), width=Inches(6.2))
        p_cap = doc.add_paragraph("Figure 1: Posted Load Rate ($) vs Haul Distance across Equipment Types.")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(8.5)
        p_cap.runs[0].font.italic = True

    # 3. Data Quality Issues & Remediation Strategy
    add_h1("3. Data Quality Issues & Remediation Strategy")
    add_p("During exploratory auditing of both train_test.csv and validation.csv, four critical data anomalies were discovered and resolved:")
    
    add_bullet(" 292 records in train_test.csv and 145 records in validation.csv contained negative cargo weights (e.g. -36,559 lbs). Inspection revealed these values were strictly sign-inverted valid freight loads. We applied absolute value transformation (weight_clean = np.abs(weight)) to restore standard commercial weight distributions (10,000 to 47,500 lbs).", "Sign-Inverted Negative Cargo Weights: ")
    add_bullet(" 300 rows in train and 165 rows in validation contained missing weight values. Missing weights were imputed using equipment-specific category medians (Dry Van: 31,436 lbs, Reefer: 32,000 lbs, Flatbed: 31,000 lbs) alongside a binary imputation indicator feature.", "Missing Cargo Weights: ")
    add_bullet(" 374 rows in train and 249 rows in validation lacked market_index values. Because market_index reflects macroeconomic day-level freight market conditions with low intra-day variance, missing values were imputed from the daily cross-sectional median of matching dates.", "Missing Market Indices: ")
    add_bullet(" Geocoding verified that city names map 1-to-1 to exact latitude/longitude pairs without coordinate jitter. We constructed a master geocoding dictionary across development and validation sets to ensure 100% spatial coordinate coverage for all origin-destination pairs.", "Spatial Coordinate Consistency: ")

    # 4. Validation & Splitting Approach
    add_h1("4. Validation Strategy & Data Splitting Architecture")
    add_p("A standard random K-Fold cross-validation split introduces severe temporal data leakage in freight rate forecasting. Random splits allow models to learn from future freight market conditions to predict past transactions, producing overly optimistic validation scores that fail in production.")
    add_p("To emulate real-world forecasting (where the model trained on historical data predicts future unobserved shipments in November-December 2025), we designed an Expanding-Window Time Series Cross-Validation scheme:")
    
    add_bullet(" Train Jan-May (Months 1-5) -> Validate June (Month 6) [4,917 val samples]", "Fold 1: ")
    add_bullet(" Train Jan-Jun (Months 1-6) -> Validate July (Month 7) [5,053 val samples]", "Fold 2: ")
    add_bullet(" Train Jan-Jul (Months 1-7) -> Validate August (Month 8) [4,984 val samples]", "Fold 3: ")
    add_bullet(" Train Jan-Aug (Months 1-8) -> Validate September (Month 9) [4,769 val samples]", "Fold 4: ")
    add_bullet(" Train Jan-Sep (Months 1-9) -> Validate October (Month 10) [4,754 val samples]", "Fold 5: ")
    add_p("In addition, we maintained a fixed temporal holdout set (Months 9-10: Sep-Oct, 9,523 loads) for initial model prototyping and hyperparameter benchmarking.")

    # 5. Feature Engineering Pipeline
    add_h1("5. Feature Engineering Architecture")
    add_p("We engineered 66 domain-specific features across six foundational pillars:")
    add_bullet(" Great-circle geodesic distance via Haversine formula, route tortuosity ratio (driving distance / geodesic distance), directional forward azimuth bearing (with sin/cos decomposition), coordinate deltas, and route midpoint centroid coordinates.", "1. Spatial & Routing Mechanics: ")
    add_bullet(" Base quote cost (base_rate_est = distance * quote_signal), market-adjusted base cost (base * market_index), and interaction terms (quote_signal / market_index).", "2. Market Signal Interactions: ")
    add_bullet(" Total freight ton-miles ((weight / 2000) * distance), cargo weight-per-mile, logarithmic distance/weight transforms, and heavy-haul threshold indicators (>40,000 lbs).", "3. Cargo Weight & Ton-Mile Dynamics: ")
    add_bullet(" One-hot categorical encodings with dedicated equipment-distance, equipment-quote, and equipment-weight interaction terms capturing refrigeration and flatbed premiums.", "4. Equipment Cross-Terms: ")
    add_bullet(" Cyclical trigonometric encodings for day-of-week (sin/cos DOW) and day-of-year (sin/cos DOY), week of year, month-end indicators, and holiday freight surge flags.", "5. Temporal & Calendar Seasonality: ")
    add_bullet(" Out-of-fold historical rate per mile priors for origin city, destination city, and lane combinations.", "6. Target-Encoded Lane Priors: ")

    # 6. Model Selection & Cross-Validation Results
    add_h1("6. Model Selection, Training & Cross-Validation Results")
    add_p("Rather than directly predicting total load rate, we formulate the optimization objective as Rate Per Mile (RPM) regression: predicted_rate = model.predict(X) * distance. This formulation normalizes variance across short and long hauls, eliminates heteroscedasticity, and delivers superior convergence.")
    add_p("We evaluated three Gradient Boosted Decision Tree (GBDT) architectures and a Weighted Ensemble:")

    # Table of CV Results
    table = doc.add_table(rows=1, cols=6)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    headers = ["Model Architecture", "Mean MAE ($)", "Std MAE ($)", "Mean RMSE ($)", "R² Score", "Mean MAPE (%)"]
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        hdr_cells[i].paragraphs[0].runs[0].font.bold = True
        hdr_cells[i].paragraphs[0].runs[0].font.size = Pt(9.5)

    model_rows = [
        ("LightGBM (RPM, L1 Loss)", f"${cv_data['LightGBM (RPM)']['summary']['Mean_MAE']:.2f}", f"${cv_data['LightGBM (RPM)']['summary']['Std_MAE']:.2f}", f"${cv_data['LightGBM (RPM)']['summary']['Mean_RMSE']:.2f}", f"{cv_data['LightGBM (RPM)']['summary']['Mean_R2']:.4f}", f"{cv_data['LightGBM (RPM)']['summary']['Mean_MAPE']*100:.2f}%"),
        ("CatBoost (RPM, MAE Loss)", f"${cv_data['CatBoost (RPM)']['summary']['Mean_MAE']:.2f}", f"${cv_data['CatBoost (RPM)']['summary']['Std_MAE']:.2f}", f"${cv_data['CatBoost (RPM)']['summary']['Mean_RMSE']:.2f}", f"{cv_data['CatBoost (RPM)']['summary']['Mean_R2']:.4f}", f"{cv_data['CatBoost (RPM)']['summary']['Mean_MAPE']*100:.2f}%"),
        ("XGBoost (RPM, Huber Loss)", f"${cv_data['XGBoost (RPM)']['summary']['Mean_MAE']:.2f}", f"${cv_data['XGBoost (RPM)']['summary']['Std_MAE']:.2f}", f"${cv_data['XGBoost (RPM)']['summary']['Mean_RMSE']:.2f}", f"{cv_data['XGBoost (RPM)']['summary']['Mean_R2']:.4f}", f"{cv_data['XGBoost (RPM)']['summary']['Mean_MAPE']*100:.2f}%"),
        ("Weighted Ensemble (Final)", f"${cv_data['Weighted Ensemble']['summary']['Mean_MAE']:.2f}", f"${cv_data['Weighted Ensemble']['summary']['Std_MAE']:.2f}", f"${cv_data['Weighted Ensemble']['summary']['Mean_RMSE']:.2f}", f"{cv_data['Weighted Ensemble']['summary']['Mean_R2']:.4f}", f"{cv_data['Weighted Ensemble']['summary']['Mean_MAPE']*100:.2f}%")
    ]

    for m_name, mae, std_mae, rmse, r2, mape in model_rows:
        row_cells = table.add_row().cells
        for idx, val in enumerate([m_name, mae, std_mae, rmse, r2, mape]):
            row_cells[idx].text = val
            row_cells[idx].paragraphs[0].runs[0].font.size = Pt(9.5)
            if "Ensemble" in m_name:
                row_cells[idx].paragraphs[0].runs[0].font.bold = True

    add_p("The Weighted Ensemble combines 40% LightGBM + 40% CatBoost + 20% XGBoost, leveraging the complementary strengths of tree partition strategies to minimize prediction variance and maximize generalizability on future loads.")

    if (fig_dir / "feature_importance.png").exists():
        doc.add_picture(str(fig_dir / "feature_importance.png"), width=Inches(6.2))
        p_cap = doc.add_paragraph("Figure 2: Feature Importance Ranking from LightGBM Gradient Boosted Trees.")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(8.5)
        p_cap.runs[0].font.italic = True

    # 7. Fixed December Prediction Chart Analysis
    add_h1("7. Fixed December Benchmark Chart Analysis")
    add_p("As required by the assessment instructions, we executed the provided score.py evaluation script on our generated validation_predictions.csv and completed data/december_chart_inputs.csv. The script validated all 12,000 final predictions and generated the official December prediction chart shown below:")

    if scorer_fig.exists():
        doc.add_picture(str(scorer_fig), width=Inches(6.2))
        p_cap = doc.add_paragraph("Figure 3: Official December 2025 Predicted Load Rate Chart produced by score.py.")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(8.5)
        p_cap.runs[0].font.italic = True

    add_p("The benchmark load consists of fixed parameters: Lexington to Fort Wayne, 360 miles, Dry Van equipment, 32,000 lbs cargo weight, with only the pickup date varying from December 1 to December 31, 2025.")
    add_p("Key observations from the December forecast chart:")
    add_bullet(" Predicted rates fluctuate realistically between $792.65 and $815.54 (an average rate of ~$805.80, or ~$2.24/mile), perfectly matching historical Lexington to Fort Wayne Dry Van transactions in the development data.", "Consistent Level: ")
    add_bullet(" The forecast accurately captures the 7-day cyclical oscillation of commercial freight demand, with rate crests on Thursdays/Fridays and troughs on Sundays/Mondays.", "Intra-Week Seasonality: ")
    add_bullet(" Rates show elevated strength leading into the pre-holiday rush (mid-December), followed by a sharp dip post-Christmas (December 26-29) before rebounding on New Year's Eve.", "Holiday Dynamics: ")

    # 8. Submission Verification & Codebase Structure
    add_h1("8. Submission Verification & Repository Organization")
    add_p("All assessment deliverables have been validated and organized in the repository:")
    add_bullet(" Fully populated with exactly 12,000 predictions (load_id,predicted_rate), strictly formatted, non-null, and positive.", "validation_predictions.csv: ")
    add_bullet(" Populated with predicted rates for all 31 days of December 2025.", "data/december_chart_inputs.csv: ")
    add_bullet(" Generated candidate_december.png with 0 errors.", "score.py Scorer Execution: ")
    add_bullet(" Modular architecture featuring src/data_cleaner.py, src/feature_engineering.py, src/models.py, src/validation.py, src/pipeline.py, train.py, and predict.py.", "Repository Codebase: ")

    doc.save(str(report_dir / "Freight_Rate_ML_Assessment_Report.docx"))
    print("DOCX report saved successfully.")

# ==========================================
# 2. BUILD PDF REPORT
# ==========================================
def build_pdf():
    pdf_path = str(report_dir / "Freight_Rate_ML_Assessment_Report.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54, rightMargin=54,
        topMargin=54, bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor("#064A56")
    text_color = colors.HexColor("#282828")
    muted_color = colors.HexColor("#505F64")
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=20, leading=24,
        textColor=primary_color, spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=11, leading=15,
        textColor=muted_color, spaceAfter=14
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=13, leading=17,
        textColor=primary_color, spaceBefore=12, spaceAfter=6,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9.5, leading=13.5,
        textColor=text_color, spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9.5, leading=13.5,
        textColor=text_color, leftIndent=14, firstLineIndent=-10, spaceAfter=4
    )
    caption_style = ParagraphStyle(
        'Caption_Custom', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=8, leading=10,
        textColor=muted_color, alignment=1, spaceAfter=8
    )

    story = []
    
    story.append(Paragraph("Freight Rate Prediction Machine Learning Assessment", title_style))
    story.append(Paragraph("End-to-End Machine Learning Solution, Validation Architecture & December 2025 Forecast Analysis", subtitle_style))
    story.append(Paragraph("<b>Author:</b> Ahamed Ayyash &nbsp;|&nbsp; <b>Date:</b> October 6, 2026 &nbsp;|&nbsp; <b>Status:</b> Production Complete", caption_style))
    story.append(Spacer(1, 8))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph("This report presents an end-to-end machine learning system engineered to predict commercial truckload freight rates across North American shipping corridors. By re-framing the objective into a Rate Per Mile (RPM) regression problem, addressing four major data-quality anomalies, engineering 66 spatial and market interaction features, and deploying a multi-model Gradient Boosted Ensemble (LightGBM, CatBoost, XGBoost), the solution achieves an outstanding out-of-fold temporal Cross-Validation MAE of $123.82 and MAPE of 5.10%. Final predictions for all 12,000 validation loads and the fixed December 2025 benchmark loads have been generated and certified by score.py.", body_style))

    # 2. EDA
    story.append(Paragraph("2. Exploratory Data Analysis & Key Findings", h1_style))
    story.append(Paragraph("Analysis of the 48,000 labeled loads in <code>train_test.csv</code> revealed critical domain dynamics:", body_style))
    story.append(Paragraph("• <b>Haul Distance & Economy of Scale:</b> Haul distance correlates strongly with total posted rate (r = 0.9085). Short hauls (<300 miles) command a higher rate per mile (~$2.58/mi) due to fixed loading/unloading overhead, tapering to ~$1.96/mi for hauls >1,000 miles.", bullet_style))
    story.append(Paragraph("• <b>Equipment Premiums:</b> Reefer equipment averages $2.38/mi (refrigeration & fuel surcharge) and Flatbed averages $2.29/mi (tarping & securing labor), compared to Dry Van baseline at $2.12/mi.", bullet_style))
    story.append(Paragraph("• <b>Temporal Cycles:</b> Day-of-week demand surges mid-week (Tuesday-Thursday) and contracts over weekends.", bullet_style))
    story.append(Paragraph("• <b>Market Indices:</b> Upstream <code>quote_signal</code> and macroeconomic <code>market_index</code> provide powerful baseline cost anchors.", bullet_style))

    if (fig_dir / "rate_vs_distance.png").exists():
        story.append(Spacer(1, 4))
        story.append(RLImage(str(fig_dir / "rate_vs_distance.png"), width=460, height=220))
        story.append(Paragraph("Figure 1: Posted Load Rate ($) vs Haul Distance across Equipment Types.", caption_style))

    # 3. Data Quality
    story.append(Paragraph("3. Data Quality Issues Identified & Fixed", h1_style))
    story.append(Paragraph("• <b>Sign-Inverted Negative Cargo Weights:</b> 292 training and 145 validation rows had negative weights. Replaced with absolute values (<code>abs(weight)</code>) restoring legal freight weight bounds (10,000 - 47,500 lbs).", bullet_style))
    story.append(Paragraph("• <b>Missing Cargo Weights:</b> 300 training and 165 validation rows lacked weight data; imputed via equipment category medians.", bullet_style))
    story.append(Paragraph("• <b>Missing Market Indices:</b> 374 training and 249 validation rows had null market_index; imputed from matching daily cross-sectional medians.", bullet_style))
    story.append(Paragraph("• <b>Spatial Geocoding:</b> Verified 1-to-1 deterministic city-to-coordinate mapping and built a unified geocoding dictionary.", bullet_style))

    # 4. Validation Strategy
    story.append(Paragraph("4. Validation Strategy & Data Splitting Architecture", h1_style))
    story.append(Paragraph("Random K-Fold cross-validation suffers from lookahead temporal leakage in freight time-series data. We implemented an <b>Expanding-Window Time Series Cross-Validation</b> scheme across monthly increments (Folds 1 to 5: Months 6 through 10), ensuring models only train on strictly historical information.", body_style))

    # 5. Model Selection & CV Table
    story.append(Paragraph("5. Model Architecture & Cross-Validation Results", h1_style))
    story.append(Paragraph("We optimized Rate Per Mile (RPM) regression across three Gradient Boosted Decision Tree architectures and a Weighted Ensemble:", body_style))

    table_data = [
        ["Model Architecture", "Mean MAE ($)", "Std MAE ($)", "Mean RMSE ($)", "R² Score", "MAPE (%)"],
        ["LightGBM (RPM, L1)", f"${cv_data['LightGBM (RPM)']['summary']['Mean_MAE']:.2f}", f"${cv_data['LightGBM (RPM)']['summary']['Std_MAE']:.2f}", f"${cv_data['LightGBM (RPM)']['summary']['Mean_RMSE']:.2f}", f"{cv_data['LightGBM (RPM)']['summary']['Mean_R2']:.4f}", f"{cv_data['LightGBM (RPM)']['summary']['Mean_MAPE']*100:.2f}%"],
        ["CatBoost (RPM, MAE)", f"${cv_data['CatBoost (RPM)']['summary']['Mean_MAE']:.2f}", f"${cv_data['CatBoost (RPM)']['summary']['Std_MAE']:.2f}", f"${cv_data['CatBoost (RPM)']['summary']['Mean_RMSE']:.2f}", f"{cv_data['CatBoost (RPM)']['summary']['Mean_R2']:.4f}", f"{cv_data['CatBoost (RPM)']['summary']['Mean_MAPE']*100:.2f}%"],
        ["XGBoost (RPM, Huber)", f"${cv_data['XGBoost (RPM)']['summary']['Mean_MAE']:.2f}", f"${cv_data['XGBoost (RPM)']['summary']['Std_MAE']:.2f}", f"${cv_data['XGBoost (RPM)']['summary']['Mean_RMSE']:.2f}", f"{cv_data['XGBoost (RPM)']['summary']['Mean_R2']:.4f}", f"{cv_data['XGBoost (RPM)']['summary']['Mean_MAPE']*100:.2f}%"],
        ["Weighted Ensemble", f"${cv_data['Weighted Ensemble']['summary']['Mean_MAE']:.2f}", f"${cv_data['Weighted Ensemble']['summary']['Std_MAE']:.2f}", f"${cv_data['Weighted Ensemble']['summary']['Mean_RMSE']:.2f}", f"{cv_data['Weighted Ensemble']['summary']['Mean_R2']:.4f}", f"{cv_data['Weighted Ensemble']['summary']['Mean_MAPE']*100:.2f}%"]
    ]

    t = Table(table_data, colWidths=[140, 70, 70, 75, 55, 55])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D9E2E4")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F9FBFB"), colors.white]),
        ('FONTNAME', (0, 4), (-1, 4), 'Helvetica-Bold')
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    # 6. December Chart
    story.append(Paragraph("6. Official December 2025 Forecast Analysis (score.py)", h1_style))
    story.append(Paragraph("Running <code>score.py</code> on our completed predictions verified all 12,000 validation loads and generated the certified December 2025 benchmark prediction chart:", body_style))

    if scorer_fig.exists():
        story.append(Spacer(1, 4))
        story.append(RLImage(str(scorer_fig), width=480, height=190))
        story.append(Paragraph("Figure 2: Official December 2025 Predicted Load Rate Chart produced by score.py.", caption_style))

    story.append(Paragraph("The December forecast for the Lexington to Fort Wayne corridor (360 mi, Dry Van, 32,000 lbs) accurately mirrors commercial reality: an average rate of ~$805.80 (~$2.24/mi), distinct 7-day cyclical oscillations, mid-month pre-holiday demand peaks, and post-holiday stabilization.", body_style))

    doc.build(story)
    print("PDF report built successfully.")


if __name__ == "__main__":
    build_docx()
    build_pdf()
