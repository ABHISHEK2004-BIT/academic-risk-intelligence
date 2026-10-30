Academic Risk Intelligence — M.Tech IALM Project
An educational machine-learning project that uses a Decision Tree classifier to study an illustrative student academic-risk dataset. It includes a Jupyter Notebook for the complete practical workflow and a Streamlit dashboard for interactive exploration and prediction.
1. Project files
File	Description
IALM_EXP3_Professional.ipynb	Practical notebook: theory, data analysis, entropy/information gain, model training, validation, tuning, evaluation and explainability.
app.py	Interactive Streamlit dashboard.
student_academic_risk_1000.csv	Dataset containing 1,000 synthetic records.
requirements.txt	Required Python packages.


Keep the CSV file in the same folder as app.py and the notebook.
2. Important dataset note
The supplied CSV is synthetic teaching data, not real student information. The Academic_Risk target was generated using an illustrative rule with random noise. It is not a validated measure of academic performance or risk.

Before submitting this project, confirm that your instructor permits synthetic datasets and that this dataset/topic is sufficiently different from the choices made by other students. Do not use this model to make decisions about real students.
3. Requirements
- Windows 10/11
- Python installed and available from Command Prompt
- Internet access for installing packages, unless the required packages are already installed
Recommended: use a supported Python environment such as Python 3.11 or 3.13 if available. Package compatibility depends on the Python version installed on your computer.
4. Open the project folder
Open Command Prompt and move to the folder containing the project files. For example:
D:
cd \IALMproject
If your project is stored in a different folder, replace D:\IALMproject with its actual location.
5. Create and activate a virtual environment (recommended)
Run these commands in Command Prompt:
python -m venv .venv
.venv\Scripts\activate
After activation, the terminal normally displays (.venv) at the beginning of the prompt.

If you use PowerShell instead of Command Prompt, activate it with:
.\.venv\Scripts\Activate.ps1
6. Install the required packages
Use Python to invoke pip:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Windows Device Guard / Application Control note: If Windows blocks pip.exe, do not run pip install directly. The python -m pip form may work because it invokes pip through the selected Python interpreter. If Windows also blocks this command or package installation, contact your system administrator/IT support to install or approve the packages. Do not disable organizational security controls.

To check whether the main packages are available:
python -c "import pandas, numpy, matplotlib, sklearn, streamlit; print('All main packages are available')"
7. Run the Jupyter Notebook
From the project folder, run:
python -m notebook IALM_EXP3_Professional.ipynb
If the notebook module is unavailable, install the requirements first. Open the notebook in Jupyter and execute the cells from top to bottom. Make sure the CSV remains beside the notebook.
8. Run the Streamlit dashboard
From the project folder, run:
python -m streamlit run app.py
Use python -m streamlit rather than streamlit run if Windows says that streamlit is not recognized. This project environment has Streamlit 1.58.0 available through Python, so the module form is the appropriate launch command.

When Streamlit starts, it prints a local address. Open this address in your browser, usually:
http://localhost:8501
To stop the dashboard, return to the terminal and press Ctrl+C.
9. Dashboard sections
1. Overview — dataset summary and system workflow.
2. Data Quality & EDA — missing values, duplicate rows, target distribution and feature exploration.
3. Model Performance — classification results and evaluation views.
4. Explainability — decision-tree structure, feature importance and decision rules.
5. Live Prediction — enter feature values to obtain a model prediction.
The dashboard trains its own configured decision-tree pipeline when it loads. The notebook contains the more extensive model-comparison and hyperparameter-tuning workflow.
10. Notebook methodology
The notebook follows this sequence:
1. Define the problem statement and Decision Tree theory.
2. Explain entropy and information gain.
3. Load the dataset and inspect its structure.
4. Perform data-quality checks and exploratory data analysis.
5. Separate input features and target variable.
6. Split data into stratified training and testing sets.
7. Encode categorical features using a preprocessing pipeline.
8. Compare Decision Tree criteria such as entropy and Gini.
9. Perform stratified cross-validation and hyperparameter tuning on training data.
10. Evaluate the selected model on the held-out test set.
11. Interpret feature importance, decision rules and an individual prediction path.
12. Discuss limitations and conclusions.
11. Reproducibility and limitations
The project uses fixed random seeds where specified to make the exercise more reproducible. Results can vary with package versions or changes to the dataset. The dataset is synthetic, and performance on it does not demonstrate that the model will work on real institutional data.