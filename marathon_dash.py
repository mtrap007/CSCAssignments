import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay

# -- Utility functions --
def time_to_sec(t):
    try:
        h, m, s = map(int, t.split(':'))
        return h*3600 + m*60 + s
    except:
        return np.nan

def sec_to_hms(sec):
    td = timedelta(seconds=int(sec))
    tot = int(td.total_seconds())
    h = tot//3600
    m = (tot%3600)//60
    s = tot%60
    return f"{h:02d}:{m:02d}:{s:02d}"

@st.cache_data
def load_data(results_file, weather_file=None):
    res = pd.read_csv(results_file)
    if weather_file:
        weather = pd.read_csv(weather_file)
    else:
        weather = None
    return res, weather

# -- Streamlit App --
st.set_page_config(page_title="Chicago Marathon Dashboard", layout="wide")
st.title("Chicago Marathon Analysis & Forecasting")

# Sidebar: file upload and options
st.sidebar.header("Data Inputs & Settings")
results_file = st.sidebar.file_uploader("Upload Marathon Results CSV", type=["csv"] )
weather_file = st.sidebar.file_uploader("Upload Weather CSV (optional)", type=["csv"] )

if not results_file:
    st.warning("Please upload the marathon results CSV to proceed.")
    st.stop()

# Load data
res, weather = load_data(results_file, weather_file)
res['Finish Time'] = res['Finish Time'].astype(str)
res['finish_sec'] = res['Finish Time'].apply(time_to_sec)
res['Name'] = res['Name'].fillna('Unknown')
res = res[res['Gender'].isin(['M','F'])]

# Tabs for navigation
tabs = st.tabs(["Data Exploration", "Model Training", "Forecast & Interpretation"])

# -- Tab 1: Data Exploration --
with tabs[0]:
    st.header("Data Exploration")
    if st.expander("Show raw data").checkbox("Display Data"):  # checkbox inside expander
        st.dataframe(res.head())

    st.subheader("Participants by Gender")
    gender_counts = res['Gender'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%", startangle=140)
    ax1.axis('equal')
    st.pyplot(fig1)

    st.subheader("Year-wise Participant Counts")
    year_count = res['Year'].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    year_count.plot(kind='bar', ax=ax2)
    ax2.set_xlabel('Year'); ax2.set_ylabel('Count')
    st.pyplot(fig2)

    st.subheader("Age Distribution & Bins")
    ages = res['Age'].dropna()
    fig3, ax3 = plt.subplots()
    ages.plot.hist(bins=30, ax=ax3)
    ax3.set_xlabel('Age'); ax3.set_ylabel('Frequency')
    st.pyplot(fig3)

    bins = list(range(15,85,5))
    labels = [f"{b}–{b+4}" for b in bins[:-1]]
    res['age_bin'] = pd.cut(res['Age'], bins=bins, labels=labels, right=False)
    age_counts = res['age_bin'].value_counts().sort_index()
    fig4, ax4 = plt.subplots(figsize=(8,4))
    age_counts.plot(kind='bar', ax=ax4)
    ax4.set_xlabel('Age Bin'); ax4.set_ylabel('Runners')
    st.pyplot(fig4)

    st.subheader("Finish Time vs Age")
    fig5, ax5 = plt.subplots()
    sns.regplot(x='Age', y=res['finish_sec']/3600, data=res, scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax5)
    ax5.set_xlabel('Age'); ax5.set_ylabel('Finish Time (hrs)')
    st.pyplot(fig5)

# -- Tab 2: Model Training --
with tabs[1]:
    st.header("Model Training & Evaluation")
    # Prepare aggregated features
    grp = res.groupby(['Year','Gender'])
    df = grp['finish_sec'].mean().rename('avg_finish_sec').to_frame()
    df['n_finishers'] = grp.size()
    df['avg_age'] = grp['Age'].mean()
    df['sd_age'] = grp['Age'].std()
    year_gender = df.reset_index()
    year_gender['male'] = (year_gender['Gender']=='M').astype(int)
    year_gender['Year_x_male'] = year_gender['Year'] * year_gender['male']
    features = ['Year','male','Year_x_male','n_finishers','avg_age','sd_age']
    X = year_gender[features].values
    y = year_gender['avg_finish_sec'].values

    split_ratio = st.slider("Test Set Proportion", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split_ratio, random_state=42)

    st.subheader("Select Model & Hyperparameters")
    model_choice = st.selectbox("Model", ['OLS','Ridge','Lasso','RandomForest'])
    if model_choice == 'Ridge': alpha = st.number_input("Alpha", 0.0001, 10.0, value=0.001)
    elif model_choice == 'Lasso': alpha = st.number_input("Alpha", 0.0001, 1.0, value=0.1)
    elif model_choice == 'RandomForest': n_estimators = st.slider("n_estimators", 10, 500, 100)

    if st.button("Train & Evaluate"):
        if model_choice == 'OLS': model = LinearRegression()
        elif model_choice == 'Ridge': model = Ridge(alpha=alpha)
        elif model_choice == 'Lasso': model = Lasso(alpha=alpha)
        else: model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.metric("MAE (sec)", f"{mae:.1f}")
        st.metric("RMSE (sec)", f"{rmse:.1f}")
        st.metric("R²", f"{r2:.3f}")
        # Show fitted vs actual
        fig6, ax6 = plt.subplots()
        ax6.scatter(y_test/3600, y_pred/3600, alpha=0.7)
        ax6.plot([y_test.min()/3600, y_test.max()/3600], [y_test.min()/3600, y_test.max()/3600], 'k--')
        ax6.set_xlabel('Actual (hrs)'); ax6.set_ylabel('Predicted (hrs)')
        ax6.set_title('Actual vs Predicted')
        st.pyplot(fig6)

# -- Tab 3: Forecast & Interpretation --
with tabs[2]:
    st.header("Forecast & Interpretation")
    st.markdown("**Forecast next 10 years (2026–2035) for average finish times**")
    model = LinearRegression().fit(X_train, y_train)  # use OLS by default
    last_year = year_gender['Year'].max()
    future_years = np.arange(last_year+1, last_year+11)
    cov_feats = ['n_finishers','avg_age','sd_age']

    # Forecast covariates
    cov_forecasts = {}
    for feat in cov_feats:
        lr_f = LinearRegression().fit(year_gender['Year'].values.reshape(-1,1), year_gender[feat].values)
        cov_forecasts[feat] = lr_f.predict(future_years.reshape(-1,1))

    # Prepare dataframe for female & male
    forecasts = []
    for gender_val, label in [(0,'F'), (1,'M')]:
        for i, yr in enumerate(future_years):
            row = {
                'Year': yr,
                'male': gender_val,
                'Year_x_male': yr*gender_val,
                **{feat: cov_forecasts[feat][i] for feat in cov_feats}
            }
            forecasts.append(row)
    df_fut = pd.DataFrame(forecasts)
    Xf = df_fut[features].values
    df_fut['pred_sec'] = model.predict(Xf)
    df_fut['pred_hms'] = df_fut['pred_sec'].apply(sec_to_hms)
    st.dataframe(df_fut[['Year','male','pred_hms']].assign(Gender=lambda df: df['male'].map({0:'F',1:'M'})))

    st.subheader("Forecast Plot")
    fig7, ax7 = plt.subplots(figsize=(10,4))
    for gender_val, color, marker in [(0,'tab:blue','o'),(1,'tab:orange','s')]:
        sub = year_gender[year_gender['male']==gender_val]
        ax7.plot(sub['Year'], sub['avg_finish_sec']/3600, color=color, marker=marker, label=f"Historical {'Female' if gender_val==0 else 'Male'}")
        fut = df_fut[df_fut['male']==gender_val]
        ax7.plot(fut['Year'], fut['pred_sec']/3600, linestyle='--', color=color, label=f"Forecast {'Female' if gender_val==0 else 'Male'}")
    ax7.set_xlabel('Year'); ax7.set_ylabel('Avg Finish Time (hrs)')
    ax7.legend(); ax7.grid(alpha=0.3)
    st.pyplot(fig7)

    # LIME Explanation
    st.subheader("LIME Explanation for a Test Instance")
    if 'model' in locals():
        explainer = LimeTabularExplainer(training_data=X_train, feature_names=features, mode='regression')
        idx = st.number_input("Select test instance index", 0, len(X_test)-1, 0)
        exp = explainer.explain_instance(data_row=X_test[int(idx)], predict_fn=model.predict, num_features=len(features))
        fig8 = exp.as_pyplot_figure()
        st.pyplot(fig8)

    # Partial Dependence
    st.subheader("Partial Dependence Plots")
    if st.button("Show PDP for 'avg_age' & 'n_finishers'"):
        fig9, ax9 = plt.subplots(figsize=(8,4))
        PartialDependenceDisplay.from_estimator(model, X_train, ['avg_age','n_finishers'], feature_names=features, ax=ax9)
        st.pyplot(fig9)
