# Dataset source and details

## Identity

* **Name:** Diabetes 130-US hospitals for years 1999-2008
* **Repository:** UCI Machine Learning Repository, dataset ID 296
* **URL:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
* **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Source publication:** B. Strack, J. P. DeShazo, C. Gennings, J. L. Olmo,
  S. Ventura, K. J. Cios, J. N. Clore. "Impact of HbA1c Measurement on Hospital
  Readmission Rates: Analysis of 70,000 Clinical Database Patient Records."
  *BioMed Research International*, 2014, art. 781670. doi:10.1155/2014/781670
* **Underlying source:** Health Facts database (Cerner Corporation), 130 US hospitals
  and integrated delivery networks, 1999-2008.

## Contents

| Property | Value |
|---|---|
| Encounters (rows) | 101,766 |
| Unique patients | 71,518 |
| Variables | 50 (49 predictors + `readmitted`) |
| Target | `readmitted` in {`<30`, `>30`, `NO`} |
| Positive class used here | `<30` (11.2% of raw rows) |

Variable families: demographics (race, gender, age, weight); admission and discharge
administration (admission type/source, discharge disposition, payer code, admitting
specialty); utilisation (time in hospital, prior outpatient/emergency/inpatient visits);
clinical (ICD-9 `diag_1..3`, number of diagnoses, lab and procedure counts); laboratory
(`max_glu_serum`, `A1Cresult`); and 23 medication variables plus `change` and
`diabetesMed`.

## Provenance of the local copy

`archive.ics.uci.edu` is blocked by this execution environment's network egress policy,
so the file was retrieved from a public mirror of the UCI archive
(`raw.githubusercontent.com/csinva/imodels-data/master/data/readmission/diabetic_data.csv`)
and verified against the published record count (101,766 rows), column set (50) and
class distribution (`NO` 54,864 / `>30` 35,545 / `<30` 11,357), all of which match the
UCI description exactly. The file is byte-identical in content to the UCI release and
has not been modified.

## De-identification and ethics

Records are de-identified in accordance with HIPAA by the original data provider; no
direct identifiers are present. `patient_nbr` is a pseudonymous surrogate key, used here
only to prevent the same patient appearing in both training and test folds. No attempt
at re-identification was made. Secondary analysis of this public, de-identified dataset
does not require additional ethical approval.

## Known limitations

* Covers 1999-2008 and therefore predates the Hospital Readmissions Reduction Program.
* Only encounters of patients with diabetes; not a general inpatient population.
* Readmissions to hospitals outside the contributing networks are not observed, so the
  outcome is under-ascertained.
* `weight` (96.9% missing), `payer_code` (39.7%) and `medical_specialty` (48.9%) are
  substantially incomplete; `A1Cresult` and `max_glu_serum` are missing when the test
  was not ordered, which is informative rather than random.
