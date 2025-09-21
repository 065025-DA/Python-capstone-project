# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from io import BytesIO

st.set_page_config(page_title="HR Analytics — Enhanced Dashboard", layout="wide", initial_sidebar_state="expanded")

# -----------------------
# Helpers
# -----------------------
@st.cache_data(show_spinner=False)
def read_csv_file(uploaded_file):
    return pd.read_csv(uploaded_file)

def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

def derive_age(dob_series, ref_date=None):
    ref = ref_date or pd.Timestamp.today()
    dob = safe_to_datetime(dob_series)
    return ((ref - dob).dt.days // 365).astype("Int64")

def derive_years_at_company(start_series, exit_series=None, ref_date=None):
    start = safe_to_datetime(start_series)
    exit_dt = safe_to_datetime(exit_series) if exit_series is not None else pd.Series(pd.NaT, index=start.index)
    ref = ref_date or pd.Timestamp.today()
    end = exit_dt.fillna(ref)
    return ((end - start).dt.days // 365).astype("Int64")

def df_is_empty_or_none(df):
    return (df is None) or (df is not None and df.shape[0] == 0)

def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if not s.empty else np.nan

def pct(value):
    return f"{round(float(value),2)}%" if not pd.isna(value) else "N/A"

def show_no_data_warning():
    st.warning("⚠️ No data available after applying filters. Try clearing filters or upload datasets with matching columns.")

# -----------------------
# Sidebar - Uploads & Global Filters
# -----------------------
st.sidebar.title("🔁 Upload datasets")
st.sidebar.write("Upload the CSVs. Column names must match the expected schema (see README).")

uploaded_emp = st.sidebar.file_uploader("Employee data (employee_data.csv)", type="csv", key="emp")
uploaded_survey = st.sidebar.file_uploader("Engagement survey (employee_engagement_survey_data.csv)", type="csv", key="survey")
uploaded_recruit = st.sidebar.file_uploader("Recruitment (recruitment_data.csv)", type="csv", key="recruit")
uploaded_train = st.sidebar.file_uploader("Training (training_and_development_data.csv)", type="csv", key="train")

# Load
emp_df = survey_df = recruit_df = train_df = None

try:
    if uploaded_emp:
        emp_df = read_csv_file(uploaded_emp)
        # rename known EmpID -> Employee ID
        if "EmpID" in emp_df.columns:
            emp_df = emp_df.rename(columns={"EmpID": "Employee ID"})
        emp_df.columns = [c.strip() for c in emp_df.columns]
except Exception as e:
    st.sidebar.error(f"Error loading employee CSV: {e}")

try:
    if uploaded_survey:
        survey_df = read_csv_file(uploaded_survey)
        survey_df.columns = [c.strip() for c in survey_df.columns]
except Exception as e:
    st.sidebar.error(f"Error loading survey CSV: {e}")

try:
    if uploaded_recruit:
        recruit_df = read_csv_file(uploaded_recruit)
        recruit_df.columns = [c.strip() for c in recruit_df.columns]
except Exception as e:
    st.sidebar.error(f"Error loading recruitment CSV: {e}")

try:
    if uploaded_train:
        train_df = read_csv_file(uploaded_train)
        train_df.columns = [c.strip() for c in train_df.columns]
except Exception as e:
    st.sidebar.error(f"Error loading training CSV: {e}")

# -----------------------
# Merge Employee + Survey + Training
# -----------------------
merged_df = None
if emp_df is not None:
    merged_df = emp_df.copy()

    # derive Age, YearsAtCompany, Attrition
    if "DOB" in merged_df.columns:
        merged_df["Age"] = derive_age(merged_df["DOB"])
    if "StartDate" in merged_df.columns:
        merged_df["YearsAtCompany"] = derive_years_at_company(merged_df["StartDate"], merged_df.get("ExitDate"))
    if "ExitDate" in merged_df.columns:
        merged_df["Attrition"] = merged_df["ExitDate"].notna().astype("int64")

    # merge engagement (left)
    if survey_df is not None and "Employee ID" in survey_df.columns:
        merged_df = merged_df.merge(survey_df, on="Employee ID", how="left", suffixes=("", "_survey"))

    # aggregate training per employee if training uploaded
    if train_df is not None and "Employee ID" in train_df.columns:
        tr = train_df.copy()
        # normalize duration/cost columns detection
        dur_col = None
        for candidate in ["Training Duration(Days)", "Training Duration (Days)", "TrainingDuration"]:
            if candidate in tr.columns:
                dur_col = candidate
                break
        cost_col = "Training Cost" if "Training Cost" in tr.columns else None
        outcome_col = "Training Outcome" if "Training Outcome" in tr.columns else None

        def summarize_training(g):
            return pd.Series({
                "Train_Count": g.shape[0],
                "Train_AvgDays": pd.to_numeric(g[dur_col], errors="coerce").mean() if dur_col else np.nan,
                "Train_TotalCost": pd.to_numeric(g[cost_col], errors="coerce").sum() if cost_col else np.nan,
                "Train_SuccessRate": g[outcome_col].astype(str).str.lower().isin(["completed","passed"]).mean() if outcome_col else np.nan
            })

        agg_train = tr.groupby("Employee ID").apply(summarize_training).reset_index()
        merged_df = merged_df.merge(agg_train, on="Employee ID", how="left")

# -----------------------
# Global Filters (auto-populate; do not auto-apply unless selected)
# -----------------------
st.sidebar.markdown("---")
st.sidebar.header("🔎 Global Filters")

def unique_vals(colname):
    vals = []
    if merged_df is not None and colname in merged_df.columns:
        vals = merged_df[colname].dropna().unique().tolist()
    elif recruit_df is not None and colname in recruit_df.columns:
        vals = recruit_df[colname].dropna().unique().tolist()
    return sorted([v for v in vals if pd.notna(v)])

# Candidate filter columns based on your datasets
dept_options = unique_vals("DepartmentType") or unique_vals("Division")
gender_options = unique_vals("GenderCode") or unique_vals("Gender")
state_options = unique_vals("LocationCode") or unique_vals("State")

department_filter = st.sidebar.multiselect("Department", options=dept_options, default=[])
gender_filter = st.sidebar.multiselect("Gender", options=gender_options, default=[])
state_filter = st.sidebar.multiselect("Location / State", options=state_options, default=[])

# numeric sliders (only create if data present)
age_range = None
if merged_df is not None and "Age" in merged_df.columns:
    mn, mx = int(merged_df["Age"].dropna().min()), int(merged_df["Age"].dropna().max())
    age_range = st.sidebar.slider("Age range", mn, mx, (mn, mx))

tenure_range = None
if merged_df is not None and "YearsAtCompany" in merged_df.columns:
    mn, mx = int(merged_df["YearsAtCompany"].dropna().min()), int(merged_df["YearsAtCompany"].dropna().max())
    tenure_range = st.sidebar.slider("Years at Company", mn, mx, (mn, mx))

desired_salary_range = None
if recruit_df is not None and "Desired Salary" in recruit_df.columns:
    smin, smax = int(pd.to_numeric(recruit_df["Desired Salary"], errors="coerce").dropna().min()), int(pd.to_numeric(recruit_df["Desired Salary"], errors="coerce").dropna().max())
    desired_salary_range = st.sidebar.slider("Desired Salary (applicants)", smin, smax, (smin, smax))

st.sidebar.markdown("---")
if st.sidebar.button("Reset filters"):
    # reset is simply clearing selections; reload is fine if you want full reset
    st.experimental_rerun()

# -----------------------
# Apply filters (only when selections made)
# -----------------------
def apply_global_filters(df):
    if df is None:
        return None
    d = df.copy()
    # department
    if department_filter:
        if "DepartmentType" in d.columns:
            d = d[d["DepartmentType"].isin(department_filter)]
        elif "Division" in d.columns:
            d = d[d["Division"].isin(department_filter)]
    # gender
    if gender_filter:
        if "GenderCode" in d.columns:
            d = d[d["GenderCode"].isin(gender_filter)]
        elif "Gender" in d.columns:
            d = d[d["Gender"].isin(gender_filter)]
    # state/location
    if state_filter:
        if "LocationCode" in d.columns:
            d = d[d["LocationCode"].isin(state_filter)]
        elif "State" in d.columns:
            d = d[d["State"].isin(state_filter)]
    # age
    if age_range and "Age" in d.columns:
        d = d[d["Age"].between(age_range[0], age_range[1])]
    # tenure
    if tenure_range and "YearsAtCompany" in d.columns:
        d = d[d["YearsAtCompany"].between(tenure_range[0], tenure_range[1])]
    return d

# -----------------------
# Main layout: Tabs
# -----------------------
tabs = st.tabs(["Overview", "Employee", "Engagement", "Training", "Recruitment"])

# -----------------------
# Overview Tab
# -----------------------
with tabs[0]:
    st.header("🔎 Overview (Merged)")
    if merged_df is None:
        st.info("Please upload employee_data.csv (and optionally engagement/training CSVs) to see merged insights.")
    else:
        df = apply_global_filters(merged_df)
        if df_is_empty_or_none(df):
            show_no_data_warning()
        else:
            # KPIs
            c1, c2, c3, c4, c5 = st.columns(5)
            headcount = int(df["Employee ID"].nunique()) if "Employee ID" in df.columns else df.shape[0]
            attrition = None
            if "Attrition" in df.columns:
                attrition = df["Attrition"].dropna().mean() * 100
            avg_tenure = safe_mean(df.get("YearsAtCompany", pd.Series(dtype=float)))
            avg_age = safe_mean(df.get("Age", pd.Series(dtype=float)))
            # performance: pick first performance-like column
            perf_cols = [c for c in df.columns if "Performance" in c or "Rating" in c]
            avg_perf = safe_mean(df[perf_cols[0]]) if perf_cols else np.nan
            c1.metric("Headcount", headcount)
            c2.metric("Attrition Rate", pct(attrition) if attrition is not None else "N/A")
            c3.metric("Avg Tenure (yrs)", round(avg_tenure,1) if not pd.isna(avg_tenure) else "N/A")
            c4.metric("Avg Age", round(avg_age,1) if not pd.isna(avg_age) else "N/A")
            c5.metric("Avg Performance", round(avg_perf,1) if not pd.isna(avg_perf) else "N/A")

            st.markdown("---")

            # Department bar
            left, right = st.columns([2, 1])
            with left:
                st.subheader("Employees by Department")
                if "DepartmentType" in df.columns:
                    dept = df["DepartmentType"].value_counts().reset_index()
                    dept.columns = ["Department", "Count"]
                    fig = px.bar(dept, x="Department", y="Count", text="Count", hover_data={"Department":True,"Count":True})
                    st.plotly_chart(fig, use_container_width=True)
                elif "Division" in df.columns:
                    div = df["Division"].value_counts().reset_index()
                    div.columns = ["Division", "Count"]
                    fig = px.bar(div, x="Division", y="Count", text="Count")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No Department/Division column found.")

                # Attrition by department
                st.subheader("Attrition rate by Department")
                if "DepartmentType" in df.columns and "Attrition" in df.columns:
                    dept_attr = df.groupby("DepartmentType")["Attrition"].mean().reset_index().sort_values("Attrition", ascending=False)
                    fig = px.bar(dept_attr, x="DepartmentType", y="Attrition", labels={"Attrition":"Attrition Rate"}, hover_data={"Attrition":":.2f"})
                    fig.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need DepartmentType and Attrition for this chart.")

            with right:
                st.subheader("Gender split")
                if "GenderCode" in df.columns:
                    g = df["GenderCode"].value_counts().reset_index()
                    g.columns = ["Gender", "Count"]
                    fig = px.pie(g, names="Gender", values="Count", hole=0.4, hover_data=["Count"])
                    st.plotly_chart(fig, use_container_width=True)
                elif "Gender" in df.columns:
                    g = df["Gender"].value_counts().reset_index()
                    g.columns = ["Gender", "Count"]
                    st.plotly_chart(px.pie(g, names="Gender", values="Count", hole=0.4), use_container_width=True)
                else:
                    st.info("No Gender column detected.")

                st.subheader("Age distribution")
                if "Age" in df.columns:
                    st.plotly_chart(px.histogram(df, x="Age", nbins=20, title="Age distribution"), use_container_width=True)
                else:
                    st.info("DOB missing — Age can't be derived.")

            # Performance vs Tenure scatter
            if perf_cols and "YearsAtCompany" in df.columns:
                st.subheader("Performance vs Tenure")
                fig = px.scatter(df, x="YearsAtCompany", y=perf_cols[0], color="DepartmentType" if "DepartmentType" in df.columns else None,
                                 hover_data=["Employee ID", "Title"], title=f"{perf_cols[0]} vs Tenure")
                st.plotly_chart(fig, use_container_width=True)

            # Correlation heatmap on numeric columns
            num = df.select_dtypes(include=[np.number]).drop(columns=["Employee ID"], errors="ignore")
            if num.shape[1] >= 2:
                st.subheader("Numeric correlations")
                corr = num.corr()
                st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto"), use_container_width=True)

            # Download
            st.markdown("---")
            st.download_button("📥 Download merged & filtered dataset (CSV)", data=to_csv_bytes(df), file_name="merged_filtered.csv", mime="text/csv")

# -----------------------
# Employee Tab
# -----------------------
with tabs[1]:
    st.header("🧾 Employee Details")
    if merged_df is None:
        st.info("Upload employee_data.csv to view employee details.")
    else:
        df_e = apply_global_filters(merged_df)
        if df_is_empty_or_none(df_e):
            show_no_data_warning()
        else:
            # Top job titles
            st.subheader("Top Job Titles")
            if "Title" in df_e.columns:
                top = df_e["Title"].value_counts().reset_index()
                top.columns = ["Title", "Count"]
                fig = px.bar(top.head(20), x="Title", y="Count", hover_data=["Count"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Title column not found.")

            # Tenure histogram
            st.subheader("Tenure distribution")
            if "YearsAtCompany" in df_e.columns:
                st.plotly_chart(px.histogram(df_e, x="YearsAtCompany", nbins=20), use_container_width=True)
            else:
                st.info("YearsAtCompany not available.")

            # Termination types
            st.subheader("Termination types")
            if "TerminationType" in df_e.columns:
                term = df_e["TerminationType"].value_counts().reset_index()
                term.columns = ["Type", "Count"]
                st.plotly_chart(px.bar(term, x="Type", y="Count"), use_container_width=True)
            else:
                st.info("TerminationType column not detected.")

# -----------------------
# Engagement Tab
# -----------------------
with tabs[2]:
    st.header("📋 Engagement & Surveys")
    # prefer merged (includes survey) else survey DF
    df_s = None
    if merged_df is not None and "Engagement Score" in merged_df.columns:
        df_s = apply_global_filters(merged_df)
    elif survey_df is not None:
        df_s = apply_global_filters(survey_df)
    if df_is_empty_or_none(df_s):
        st.info("No engagement data available (upload survey CSV or merge).")
    else:
        # KPIs
        s1, s2, s3 = st.columns(3)
        s1_metric = safe_mean(df_s["Engagement Score"]) if "Engagement Score" in df_s.columns else np.nan
        s2_metric = safe_mean(df_s.get("Satisfaction Score", pd.Series(dtype=float)))
        s3_metric = safe_mean(df_s.get("Work-Life Balance Score", pd.Series(dtype=float)))
        s1.metric("Avg Engagement", round(s1_metric,2) if not pd.isna(s1_metric) else "N/A")
        s2.metric("Avg Satisfaction", round(s2_metric,2) if not pd.isna(s2_metric) else "N/A")
        s3.metric("Avg WLB", round(s3_metric,2) if not pd.isna(s3_metric) else "N/A")

        st.markdown("---")
        # Distribution
        if "Engagement Score" in df_s.columns:
            st.subheader("Engagement distribution")
            st.plotly_chart(px.histogram(df_s, x="Engagement Score", nbins=10), use_container_width=True)
        # Satisfaction vs WLB
        if "Satisfaction Score" in df_s.columns and "Work-Life Balance Score" in df_s.columns:
            st.subheader("Satisfaction vs Work-Life Balance")
            fig = px.scatter(df_s, x="Satisfaction Score", y="Work-Life Balance Score",
                             hover_data=["Employee ID"] if "Employee ID" in df_s.columns else None)
            st.plotly_chart(fig, use_container_width=True)
        # trend
        if "Survey Date" in df_s.columns:
            st.subheader("Engagement trend")
            df_s["_sd"] = safe_to_datetime(df_s["Survey Date"])
            trend = df_s.groupby(pd.Grouper(key="_sd", freq="M"))["Engagement Score"].mean().reset_index()
            trend = trend.dropna()
            if not trend.empty:
                st.plotly_chart(px.line(trend, x="_sd", y="Engagement Score"), use_container_width=True)

# -----------------------
# Training Tab
# -----------------------
with tabs[3]:
    st.header("📚 Training & Development")
    # If merged has aggregated training, use that; else raw train_df
    if merged_df is not None and "Train_Count" in merged_df.columns:
        df_t = apply_global_filters(merged_df)
        if df_is_empty_or_none(df_t):
            show_no_data_warning()
        else:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Participation Rate", pct(df_t["Train_Count"].gt(0).mean()*100))
            p2.metric("Avg Training Days", round(safe_mean(df_t.get("Train_AvgDays", pd.Series(dtype=float))),2) if not pd.isna(safe_mean(df_t.get("Train_AvgDays", pd.Series(dtype=float)))) else "N/A")
            p3.metric("Avg Training Cost", round(safe_mean(df_t.get("Train_TotalCost", pd.Series(dtype=float))),2) if not pd.isna(safe_mean(df_t.get("Train_TotalCost", pd.Series(dtype=float)))) else "N/A")
            p4.metric("Avg Success Rate", pct(df_t["Train_SuccessRate"].mean()*100) if "Train_SuccessRate" in df_t.columns else "N/A")

            st.markdown("---")
            if "DepartmentType" in df_t and "Train_SuccessRate" in df_t:
                grp = df_t.groupby("DepartmentType")["Train_SuccessRate"].mean().reset_index()
                st.plotly_chart(px.bar(grp, x="DepartmentType", y="Train_SuccessRate", title="Training Success by Dept"), use_container_width=True)
            else:
                st.info("Upload training CSV for program-level charts.")

    elif train_df is not None:
        df_t_raw = apply_global_filters(train_df)
        if df_is_empty_or_none(df_t_raw):
            show_no_data_warning()
        else:
            t1, t2, t3 = st.columns(3)
            t1.metric("Total Trainings", int(df_t_raw.shape[0]))
            t2.metric("Avg Duration (days)", round(safe_mean(df_t_raw.get("Training Duration(Days)", pd.Series(dtype=float))),2) if not pd.isna(safe_mean(df_t_raw.get("Training Duration(Days)", pd.Series(dtype=float)))) else "N/A")
            t3.metric("Total Cost", round(safe_mean(df_t_raw.get("Training Cost", pd.Series(dtype=float))) * df_t_raw.shape[0],2) if not pd.isna(safe_mean(df_t_raw.get("Training Cost", pd.Series(dtype=float)))) else "N/A")

            st.markdown("---")
            if "Training Outcome" in df_t_raw.columns:
                out = df_t_raw["Training Outcome"].value_counts().reset_index()
                out.columns = ["Outcome", "Count"]
                st.plotly_chart(px.pie(out, names="Outcome", values="Count", hole=0.35), use_container_width=True)
            if "Training Program Name" in df_t_raw.columns and "Training Cost" in df_t_raw.columns:
                cost_by_prog = df_t_raw.groupby("Training Program Name")["Training Cost"].sum().reset_index().sort_values("Training Cost", ascending=False).head(20)
                st.plotly_chart(px.bar(cost_by_prog, x="Training Program Name", y="Training Cost"), use_container_width=True)

    else:
        st.info("Upload training CSV to see training insights.")

# -----------------------
# Recruitment Tab
# -----------------------
with tabs[4]:
    st.header("🧭 Recruitment (Applicants)")
    if recruit_df is None:
        st.info("Upload recruitment_data.csv to view recruitment analytics.")
    else:
        df_r = apply_global_filters(recruit_df)
        if df_is_empty_or_none(df_r):
            show_no_data_warning()
        else:
            a1, a2, a3, a4 = st.columns(4)
            applicants = int(df_r["Applicant ID"].nunique()) if "Applicant ID" in df_r.columns else df_r.shape[0]
            avg_des = safe_mean(df_r.get("Desired Salary", pd.Series(dtype=float)))
            avg_exp = safe_mean(df_r.get("Years of Experience", pd.Series(dtype=float)))
            hire_rate = df_r.get("Status", pd.Series(dtype=str)).astype(str).str.lower().eq("hired").mean() * 100 if "Status" in df_r.columns else np.nan
            a1.metric("Applicants", applicants)
            a2.metric("Avg Desired Salary", round(avg_des,2) if not pd.isna(avg_des) else "N/A")
            a3.metric("Avg Experience (yrs)", round(avg_exp,2) if not pd.isna(avg_exp) else "N/A")
            a4.metric("Hire Rate", pct(hire_rate) if not pd.isna(hire_rate) else "N/A")

            st.markdown("---")
            if "Status" in df_r.columns:
                st.subheader("Application status counts")
                status = df_r["Status"].value_counts().reset_index()
                status.columns = ["Status", "Count"]
                st.plotly_chart(px.bar(status, x="Status", y="Count"), use_container_width=True)

            if "Education Level" in df_r.columns:
                st.subheader("Education level distribution")
                edu = df_r["Education Level"].value_counts().reset_index()
                edu.columns = ["Education", "Count"]
                st.plotly_chart(px.bar(edu, x="Education", y="Count"), use_container_width=True)

            if "Years of Experience" in df_r.columns and "Desired Salary" in df_r.columns:
                st.subheader("Experience vs Desired Salary")
                st.plotly_chart(px.scatter(df_r, x="Years of Experience", y="Desired Salary", color="Status", hover_data=["Applicant ID", "Job Title"]), use_container_width=True)

            # simple funnel
            if "Status" in df_r.columns:
                stages = ["Applied", "Interviewing", "Offered", "Hired", "Rejected"]
                counts = df_r["Status"].value_counts()
                funnel_df = pd.DataFrame({
                    "Stage": [s for s in stages if s in counts.index],
                    "Count": [counts.get(s,0) for s in stages if s in counts.index]
                })
                if not funnel_df.empty:
                    st.subheader("Simple recruitment funnel")
                    st.plotly_chart(px.funnel(funnel_df, x="Count", y="Stage"), use_container_width=True)

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption("Built with ❤️ — Columns expected (Employee: Employee ID, DOB, StartDate, ExitDate, DepartmentType, Division, GenderCode, Performance Score, etc.). If charts don't appear, check CSV headers.")
