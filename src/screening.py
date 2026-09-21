"""Re-screen the 77 candidates retrieved by lit/lit_search.ipynb against the
pre-registered criteria, and regenerate the screening + extraction tables.

Criteria (fixed before searching, see notebook cell 2):
  INCLUDE  population  hospitalised adult patients, structured EHR
           construct   predicting hospital readmission / 30-day readmission risk
           method      ML, statistical modelling or feature engineering on tabular data
           recency     published 2015 or later
  EXCLUDE  non-clinical datasets; reviews/essays without empirical models;
           imaging- or genomics-only; no reported performance metrics.
"""
from pathlib import Path

ROOT = Path("/home/user/RM_Lab")
TAB = ROOT / "paper" / "tables"

# (title, decision, reason).  R = review/no primary model, O = off-construct,
# D = wrong data modality, M = no performance metrics, Y = year < 2015.
S = [
("Deep learning for healthcare: review, opportunities and challenges","exclude","R: narrative review, no readmission model"),
("Scalable and accurate deep learning with electronic health records","include","30-day readmission among predicted endpoints, structured EHR"),
("MIMIC-IV, a freely accessible electronic health record dataset","exclude","R: data resource descriptor, no prediction model"),
("Machine Learning-Based Prediction of Heart Failure Readmission or Death","include","30-day HF readmission, tabular linked data, imbalance addressed"),
("Application of machine learning in predicting hospital readmissions: a scoping review","exclude","R: scoping review; retained as background evidence"),
("Machine Learning and Data Mining Methods in Diabetes Research","exclude","R: review of diabetes ML broadly, not readmission"),
("ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission","exclude","D: prediction from free-text notes, not structured tabular records"),
("Next-generation phenotyping of electronic health records","exclude","R: perspective on EHR data quality, no model"),
("Use of electronic medical records in development and validation of risk prediction models","exclude","R: systematic review; retained as background evidence"),
("Opportunities and challenges in developing deep learning models using EHR data","exclude","R: systematic review of architectures"),
("Key challenges for delivering clinical impact with artificial intelligence","exclude","R: commentary on AI translation"),
("Readmission prediction using deep learning on electronic health records","include","30-day CHF readmission, cost-sensitive LSTM on EHR"),
("The effects of data sources, cohort selection, and outcome definition","exclude","Y: published 2014; retained as background on design choices"),
("Revolutionizing healthcare: the role of artificial intelligence in clinical practice","exclude","R: educational review"),
("Federated Learning for Healthcare Informatics","exclude","O: federated learning survey, outcome not readmission"),
("BEHRT: Transformer for Electronic Health Records","exclude","O: predicts 301 future diagnoses, not readmission"),
("Effect of Aliskiren on Postdischarge Mortality and Heart Failure Readmissions","exclude","O: randomised drug trial, not a prediction model"),
("Survey on deep learning with class imbalance","exclude","R: methodological survey; retained as background on imbalance"),
("Machine learning prediction in cardiovascular diseases: a meta-analysis","exclude","O: diagnosis of CVD, not readmission"),
("Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities","exclude","R: XAI survey, non-clinical"),
("Multimodal machine learning in precision health: A scoping review","exclude","R: scoping review of data fusion"),
("Analysis and prediction of unplanned ICU readmission using RNN-LSTM","include","unplanned ICU readmission from MIMIC-III structured data"),
("Emergency department triage prediction of clinical outcomes","exclude","O: triage acuity/critical outcomes, not readmission"),
("Predictors of 30-Day Unplanned Readmission After Carotid Artery Stenting","include","30-day readmission, Nationwide Readmission Database, ML comparison"),
("Machine learning for clinical decision support in infectious diseases","exclude","O: infection diagnosis/management"),
("Patient clustering improves efficiency of federated machine learning","exclude","O: outcomes are mortality and length of stay"),
("Combining structured and unstructured data for predictive models","exclude","D: fusion with clinical notes; outcome not 30-day readmission"),
("Machine Learning-Enabled 30-Day Readmission Model for Stroke Patients","include","30-day readmission after ischaemic stroke, EHR, 5 algorithms"),
("Secure and robust machine learning for healthcare: A survey","exclude","R: security/privacy survey"),
("Mining Electronic Health Records (EHRs)","exclude","R: methods survey"),
("Nutritional assessment: comparison of clinical assessment and objective variables","exclude","M: association study, no discrimination metrics reported"),
("Moving beyond regression techniques in cardiovascular risk prediction","exclude","O: mortality after myocardial infarction"),
("A machine learning model to predict the risk of 30-day readmissions in heart failure","include","30-day HF readmission from longitudinal EMR, utilisation features"),
("Neural networks versus Logistic regression for 30 days all-cause readmission","include","30-day all-cause readmission, claims data, NN vs LR comparison"),
("Significance of machine learning in healthcare: Features, pillars and applications","exclude","R: general overview"),
("Prediction of 30-Day All-Cause Readmissions in Patients Hospitalized for Heart Failure","include","30-day readmission, ML vs conventional logistic models"),
("Predicting Hospital Readmission among Diabetics using Deep Learning","include","same benchmark dataset, CNN + feature engineering"),
("Impact of HbA1c Measurement on Hospital Readmission Rates","exclude","Y: 2014; retained as the provenance record of the dataset used here"),
("Challenges and opportunities beyond structured data in analysis of EHR","exclude","R: review of unstructured-data analytics"),
("Assessment of Machine Learning vs Standard Prediction Rules for Readmissions","include","30-day readmission, ML rank score vs LACE/HOSPITAL rules"),
("A stacking-based model for predicting 30-day all-cause readmissions in AMI","include","30-day readmission, stacking ensemble, undersampling (NCR)"),
("Machine Learning for Clinical Outcome Prediction","exclude","R: review of outcome-prediction methodology"),
("Machine Learning for Healthcare: On the Verge of a Major Shift","exclude","R: introductory review"),
("Deep learning for electronic health records: a comparative review","exclude","R: comparative review of architectures"),
("Comparison of machine learning models for predicting 30-day readmission (diabetes)","include","same benchmark dataset, 10 ML + 1 DL model comparison"),
("Opportunities and challenges in developing risk prediction models with EHR data","exclude","R: systematic review of EHR modelling practice"),
("Current Challenges and Future Opportunities for XAI in ML-Based CDSS","exclude","R: XAI systematic review"),
("Deep representation learning of patient data from EHR: a systematic review","exclude","R: systematic review"),
("A new analytical framework for missing data imputation and classification","include","HF readmission with explicit missing-data framework (GPLVM)"),
("Predicting healthcare trajectories from medical records (DeepCare)","include","unplanned readmission for diabetes cohorts from EHR sequences"),
("The 30-days hospital readmission risk in diabetic patients","include","same benchmark dataset, RF / NB / decision-tree ensemble"),
("Multitask learning and benchmarking with clinical time series data","exclude","O: mortality, length of stay, phenotyping benchmarks"),
("Predicting Unplanned Readmissions Following a Hip or Knee Arthroplasty","include","30-day readmission after arthroplasty; structured + note features"),
("Interpretability of machine learning-based prediction models in healthcare","exclude","R: interpretability review"),
("A survey on datasets for fairness-aware machine learning","exclude","O: fairness benchmark survey"),
("Imbalanced class distribution and performance evaluation metrics","exclude","R: systematic review; retained as background on metric choice"),
("Med-BERT: pretrained contextualized embeddings on structured EHR","exclude","O: disease-onset prediction, not readmission"),
("Comparison of Conventional Statistical Methods with Machine Learning in Medicine","exclude","R: methodological comparison review"),
("Deep representation learning of EHR to unlock patient stratification","exclude","O: unsupervised stratification, no readmission outcome"),
("Explainable Stacking-Based Model for Predicting Hospital Readmission for Diabetics","include","same benchmark dataset, stacking + RUS + explainability"),
("Shifting machine learning for healthcare from development to deployment","exclude","R: perspective on deployment"),
("Applying Artificial Intelligence to Wearable Sensor Data","exclude","D: wearable sensor data, not inpatient records"),
("Predictive Modeling of the Hospital Readmission Risk from Claims Data (COPD)","include","30-day COPD readmission, 111,992 patients, deep and non-deep models"),
("Nationwide hospital admission data statistics and disease-specific readmission","include","disease-specific 30-day readmission on national admission data"),
("Artificial intelligence and machine learning in precision and genomic medicine","exclude","D: genomics; also retracted"),
("An explainable machine learning framework for lung cancer LOS prediction","exclude","O: length of stay; retained as background on resampling effects"),
("Early prediction of ICU readmissions using classification algorithms","include","ICU readmission, tabular clinical predictors, classifier comparison"),
("Machine learning based prediction models for cardiovascular disease risk","exclude","O: 5-10 year CVD risk, not readmission"),
("Toward Predicting 30-Day Readmission Among Oncology Patients","include","30-day readmission, patient/provider/community features, LightGBM"),
("Explainable, trustworthy, and ethical machine learning for healthcare","exclude","R: survey of explainability and ethics"),
("Data Mining in Healthcare - A Review","exclude","R: review of data-mining applications"),
("Reducing patient mortality, LOS and readmissions through sepsis prediction","exclude","O: sepsis onset prediction; readmission only a downstream QI metric"),
("Machine Learning-Based Early Prediction of Sepsis Using EHR","exclude","O: sepsis onset"),
("On the interpretability of machine learning-based model for predicting hypertension","exclude","O: hypertension risk"),
("Leveraging Machine Learning Techniques to Forecast Patient Prognosis After PCI","exclude","O: prognosis/mortality after PCI"),
("Association of Surgical and Hospital Volume with 30-Day Readmission Rates","exclude","M: association analysis, no predictive performance reported"),
("Implications of resampling data to address the class imbalance problem (IRCIP)","include","30-day readmission case study; resampling vs calibration - core to RQ3"),
]
assert len(S) == 77, len(S)

inc = [i for i, s in enumerate(S) if s[1] == "include"]
print(f"screened 77  ->  included {len(inc)}  excluded {77-len(inc)}")

def esc(t):
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        t = t.replace(a, b)
    return t

def trunc(t, n=62):
    return t if len(t) <= n else t[:n - 3] + "..."

# ---- full screening log (appendix) ----
L = [r"\setlength{\tabcolsep}{4pt}",
     r"\begin{longtable}{rp{4.9cm}p{1.35cm}p{4.6cm}}",
     r"\caption{Screening record for all 77 de-duplicated candidates. Reason codes: "
     r"R~=~review or commentary without a primary model, O~=~outcome outside the readmission "
     r"construct, D~=~data modality outside structured tabular records, M~=~no predictive "
     r"performance reported, Y~=~published before 2015.}\label{tab:screening}\\",
     r"\toprule \# & Title & Decision & Reason \\ \midrule", r"\endfirsthead",
     r"\toprule \# & Title & Decision & Reason \\ \midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
for i, (t, d, r) in enumerate(S):
    # p{} columns wrap, so only very long titles need shortening
    L.append(f"{i} & {esc(trunc(t, 78))} & {d} & {esc(r)} \\\\")
L.append(r"\end{longtable}")
L.append(r"\setlength{\tabcolsep}{6pt}")
(TAB / "tab_screening.tex").write_text("\n".join(L))

# ---- extraction table for the studies used on the same benchmark ----
EXTRACT = [
 (r"\cite{strack2014}", "Diabetes 130-US (101,766 enc.)", "Multivariable logistic regression",
  "Dataset provenance; one encounter per patient retained, no ML comparison"),
 (r"\cite{shang2021}", "Diabetes 130-US", "Random forest, naive Bayes, decision-tree ensemble",
  "Encounter-level split; accuracy-oriented metrics; calibration not reported"),
 (r"\cite{lu2022}", "Diabetes 130-US", "Stacking ensemble + random undersampling",
  "Undersampling applied before splitting the data; calibration not reported"),
 (r"\cite{liu2024}", "Diabetes 130-US", "10 ML models + LSTM",
  "Patient clustering not addressed; AUROC-centred comparison"),
 (r"\cite{welvaars2023}", "Urology readmissions", "7 resampling ratios $\\times$ several classifiers",
  "Shows resampling inflates positive predictions; single-centre"),
 (r"\cite{goorbergh2022}", "Simulation + clinical data", "Logistic regression under imbalance corrections",
  "Simulation-based; tree ensembles not examined"),
 (r"\cite{amritphale2021}", "Nationwide Readmission Database", "LR, SVM, DNN, RF, decision tree",
  "Administrative claims only; no laboratory variables"),
 (r"\cite{darabi2021}", "Stroke EHR ($n=3{,}184$)", "RF, GBM, XGBoost, SVM, LR + adaptive sampling",
  "Small cohort; resampling chosen on validation performance"),
 (r"\cite{mahmoudi2020}", "41 studies (systematic review)", "Narrative synthesis",
  "Reports median AUROC $\\approx$ 0.68; few studies report calibration"),
]
E = [r"\begin{tabular}{p{1.7cm}p{2.7cm}p{3.5cm}p{4.0cm}}", r"\hline\noalign{\smallskip}",
     r"Study & Data & Method & Limitation relevant to this work \\",
     r"\noalign{\smallskip}\hline\noalign{\smallskip}"]
for a, b, c, d in EXTRACT:
    E.append(f"{a} & {b} & {c} & {d} \\\\")
E += [r"\noalign{\smallskip}\hline", r"\end{tabular}"]
(TAB / "tab_extraction.tex").write_text("\n".join(E))

# ---- update the counts line of the search log ----
sl = (TAB / "tab_search_log.tex").read_text()
sl = sl.replace("screened $n=77$; included $n=9$.",
                f"screened $n=77$; included $n={len(inc)}$.")
(TAB / "tab_search_log.tex").write_text(sl)
print("regenerated tab_screening.tex, tab_extraction.tex, tab_search_log.tex")
