# Predicting Atrial Fibrillation Risk from AF-Free ECG Recordings

Code for the Master's thesis *Predicting Atrial Fibrillation Risk from AF-Free ECG
Recordings: An Open, End-to-End Deep Learning Pipeline*.
A Dilated ResNet with temporal attention is trained on AF-free Holter ECG windows
from the [SHDB-AF](https://physionet.org/content/shdb-af/1.0.1/) database to
distinguish patients with paroxysmal AF (PAF) from patients without AF, using
5-fold patient-level cross-validation.

**Thesis (DiVA):** [Malmö University, 2026](https://urn.kb.se/resolve?urn=urn:nbn:se:mau:diva-88253)

**Two-page project summary:** [Project_Summary.docx](Project_Summary.docx)

## Contents

| Notebook | Purpose |
| --- | --- |
| `Script_1_Dataset_Preprocessing_GitHub.ipynb` | Filters the patient pool, extracts AF-free regions from the rhythm annotations, slices them into 60 s windows and builds the patient-level train/test split (40 patients: 20 PAF, 20 non-AF) plus a secondary test set from the remaining patients. |
| `Script_2_Training_GitHub.ipynb` | Loads the ECG signal of every window, preprocesses and caches it, and trains one model per cross-validation fold. |
| `Script_3_Evaluation_GitHub.ipynb` | Evaluates the fold models: window- and patient-level metrics, robustness diagnostics and confidence intervals. |

## 1. Download the dataset

The data come from the SHDB-AF database on PhysioNet:
<https://physionet.org/content/shdb-af/1.0.1/>

Use one of the three options below.

- [Download the ZIP file](https://physionet.org/content/shdb-af/get-zip/1.0.1/) (7.3 GB)
- Download the files using your terminal:

  ```bash
  wget -r -N -c -np https://physionet.org/files/shdb-af/1.0.1/
  ```

- Download the files using AWS command line tools:

  ```bash
  aws s3 sync --no-sign-request s3://physionet-open/shdb-af/1.0.1/ DESTINATION
  ```

Keep the files exactly as downloaded. The folder that contains
`AdditionalData.csv` and the WFDB records is the only path the scripts need:

```text
<your SHDB-AF folder>/          <- this is DATA_ROOT
├── AdditionalData.csv
├── 001.atr
├── 001.dat
├── 001.hea
├── 001.qrs
├── 002.atr
└── ...
```

With `wget`, this folder is `physionet.org/files/shdb-af/1.0.1/`. With the AWS
command, it is `DESTINATION`. With the ZIP file, it is the extracted folder that
contains `AdditionalData.csv`.

## 2. Set the data path

Each notebook has a **USER SETTING** cell near the top. Set `DATA_ROOT` to your
SHDB-AF folder, and use the same value in all three notebooks:

```python
DATA_ROOT = "/path/to/shdb-af/1.0.1"
```

- **Google Colab:** upload the folder to Google Drive and use a path such as
  `"/content/drive/MyDrive/shdb-af/1.0.1"`. Drive is mounted automatically when
  `DATA_ROOT` starts with `/content/drive`.
- **Local Jupyter:** use any local path, e.g. `"C:/data/shdb-af/1.0.1"` or
  `"/home/<user>/data/shdb-af/1.0.1"`.

You don't need to change any other path. All outputs are written to
`DATA_ROOT/outputs/`. The remaining settings in each **CONFIGURATION** cell
reproduce the thesis results.

## 3. Run the notebooks

Run the notebooks in order, each one top to bottom (*Runtime → Run all* in Colab):

1. **Script 1: Dataset Preprocessing.** Runs on a CPU. In Step G you may be
   asked to type a recording ID for any unselected patient with more than one
   valid recording.
2. **Script 2: Training.** Use a GPU runtime (*Runtime → Change runtime type*).
3. **Script 3: Evaluation.** A GPU runtime is recommended.

Each notebook checks its inputs before doing any work. If `DATA_ROOT` is wrong
or a previous script has not been run yet, it stops with a clear message.

### Output layout

```text
DATA_ROOT/outputs/
├── preprocessing/                         # Script 1
│   ├── step1_output.json
│   ├── step2_output.json
│   ├── step3_output.json
│   ├── patient_window_summary.{txt,json,pdf}
│   ├── recording_window_summary.{txt,json,pdf}
│   ├── train_test_split.json              # read by Scripts 2 and 3
│   └── secondary_test_set.json
├── cache/                                 # Script 2
│   └── preprocessed_dataset_all_40.pkl
└── results/
    └── dilated_resnet_attention/          # Scripts 2 and 3
        ├── model_fold_{0..4}.keras
        ├── cv_metadata.json
        └── metrics, plots, reports and confidence intervals (Script 3)
```

> **Note:** Script 2 reuses `cache/preprocessed_dataset_all_40.pkl` when it
> exists. If you re-run Script 1 or change the preprocessing settings, delete
> this file first.

## Requirements

The notebooks are written for Google Colab, where all libraries except `wfdb`
and `reportlab` are preinstalled. The notebooks install those two with `%pip`.
For a local Jupyter setup, install:

```bash
pip install wfdb reportlab numpy pandas scipy scikit-learn tensorflow matplotlib seaborn
```

- TensorFlow ≥ 2.11 (for `AdamW`)
- scikit-learn ≥ 1.0 (for `StratifiedGroupKFold`)
- `reportlab` is optional. Without it, only the PDF summaries of Script 1 are skipped.

The notebooks use notebook syntax (`%pip`), so run them in Colab or Jupyter,
not as plain Python scripts.

## Reproducibility

- All random seeds are fixed (`42`) in each notebook's CONFIGURATION cell.
- Script 1 lists the 40 selected patients and the recording used for each
  patient. It also checks the resulting window counts against the expected
  values.
- Script 2 saves its full configuration to `cv_metadata.json`, so each trained
  model can be traced back to the settings that produced it.

## Contact

Questions, bugs and suggestions about the code are best raised as an
[issue](../../issues) in this repository. For collaboration or reuse enquiries:
[linkedin.com/in/fotismagoulas](https://www.linkedin.com/in/fotismagoulas).

## Citation

If you use this code, please cite the thesis:

```text
Magoulas, F. M. (2026). Predicting Atrial Fibrillation Risk from AF-Free ECG Recordings :
An Open, End-to-End Deep Learning Pipeline (Dissertation).
Retrieved from https://urn.kb.se/resolve?urn=urn:nbn:se:mau:diva-88253
```

Or, in short form: F. M. Magoulas, 'Predicting Atrial Fibrillation Risk from AF-Free ECG
Recordings : An Open, End-to-End Deep Learning Pipeline', Dissertation, 2026.

## Dataset citation

If you use the SHDB-AF data, please cite:

- Tsutsui, K., Biton Brimer, S., & Behar, J. (2025). SHDB-AF: a Japanese Holter
  ECG database of atrial fibrillation (version 1.0.1). PhysioNet.
  <https://doi.org/10.13026/n6yq-fq90>
- Tsutsui, K., Brimer, S.B., Ben-Moshe, N. et al. SHDB-AF: a Japanese Holter ECG
  database of atrial fibrillation. *Sci Data* 12, 454 (2025).
  <https://doi.org/10.1038/s41597-025-04777-4>
- Pollard, T., et al. (2026). PhysioNet as a global platform for biomedical
  research. *Nature Health*. <https://doi.org/10.1038/s44360-026-00096-z>

The SHDB-AF database is distributed under the
[Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).
