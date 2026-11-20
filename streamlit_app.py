import streamlit as st
import subprocess
import os
import yaml
import time

st.set_page_config(
    page_title="AutoAnalog v2.0",
    page_icon="🧬",
    layout="wide")

st.title("🧬 AutoAnalog v2.0")
st.subheader(
    "Automated Drug Discovery Pipeline — "
    "Disease → Ranked Lead Compounds")

st.sidebar.header("Pipeline Configuration")

disease = st.sidebar.text_input(
    "Disease", "endometriosis")
receptor = st.sidebar.text_input(
    "Receptor", "P2Y12")
pdb_id = st.sidebar.text_input(
    "PDB ID", "4NTJ")
sequence = st.sidebar.text_area(
    "OR Protein Sequence (if no PDB)", "")
inhibitor = st.sidebar.text_area(
    "Known Inhibitor SMILES",
    "CC1=CC=C(C=C1)S(=O)(=O)NC2=NC=C"
    "(C=N2)C3=CC=CC=N3")
ref_ic50 = st.sidebar.number_input(
    "Reference IC50 (nM)", 1, 10000, 89)
ref_name = st.sidebar.text_input(
    "Reference Name", "AZD1283")
ref_source = st.sidebar.text_input(
    "Reference Source",
    "Springthorpe et al. 2007")
exhaustiveness = st.sidebar.slider(
    "Docking Exhaustiveness", 1, 16, 8)
n_analogues = st.sidebar.slider(
    "Max Analogues", 10, 200, 50)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Disease", disease)
with col2:
    st.metric("Receptor", receptor)
with col3:
    st.metric("Reference IC50",
              f"{ref_ic50} nM")

st.divider()

if st.button("🚀 Run Pipeline",
             type="primary",
             use_container_width=True):

    config = {
        "disease":            disease,
        "receptor":           receptor,
        "pdb_id":             pdb_id,
        "sequence":           sequence,
        "known_inhibitor":    inhibitor,
        "reference_ic50_nm":  ref_ic50,
        "reference_name":     ref_name,
        "reference_source":   ref_source,
        "chembl_id":          "CHEMBL2563",
        "n_analogues":        n_analogues,
        "max_permutation_analogues": 100,
        "docking_exhaustiveness": exhaustiveness,
        "ml_cv_folds":        5,
        "md_steps":           25000,
        "md_temperature":     300,
        "md_timestep":        0.004,
        "active_threshold_nm": 1000,
        "inactive_threshold_nm": 10000,
        "use_admet_ai":       True,
        "admet_herg_threshold": 0.5,
        "admet_hepatotox_threshold": 0.5,
        "chembl_screen_n":    1000,
        "tanimoto_vs_threshold": 0.4,
        "top_hits_for_analogues": 5,
    }

    with open("config/config.yaml", "w") as f:
        yaml.dump(config, f,
                  default_flow_style=False)

    st.success("Config saved!")

    steps = [
        "Structure preparation",
        "Literature mining",
        "Analogue design",
        "ML activity + IC50",
        "ADMET Stage 1",
        "Virtual screening",
        "Molecular docking",
        "ADMET Stage 2",
        "MD simulation",
        "Final ranking",
        "PDF report generation",
    ]

    progress = st.progress(0)
    status   = st.empty()
    log_box  = st.empty()

    with st.spinner("Pipeline running..."):
        process = subprocess.Popen(
            ["snakemake", "--forceall",
             "--cores", "4"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd())

        output_lines = []
        step_num = 0

        for line in process.stdout:
            output_lines.append(line.strip())
            if len(output_lines) > 20:
                output_lines.pop(0)

            log_box.code(
                "\n".join(output_lines),
                language="bash")

            for i, step in enumerate(steps):
                kw = step.split()[0].lower()
                if kw in line.lower():
                    step_num = i + 1
                    progress.progress(
                        step_num / len(steps))
                    status.info(
                        f"Running: {step}")

        process.wait()

        if process.returncode == 0:
            progress.progress(1.0)
            status.success(
                "Pipeline complete!")
            st.balloons()
        else:
            status.error(
                "Pipeline failed — "
                "check logs")

st.divider()

report_path = "results/report/final_report.pdf"
rank_path   = "results/ranking/final_ranking.csv"

col1, col2 = st.columns(2)

with col1:
    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button(
                "📄 Download PDF Report",
                f,
                file_name=
                f"AutoAnalog_{receptor}_{disease}.pdf",
                mime="application/pdf",
                use_container_width=True)
    else:
        st.info(
            "Run pipeline to generate report")

with col2:
    if os.path.exists(rank_path):
        import pandas as pd
        r_df = pd.read_csv(rank_path)
        st.download_button(
            "📊 Download Rankings CSV",
            r_df.to_csv(index=False),
            file_name=
            f"Rankings_{receptor}.csv",
            mime="text/csv",
            use_container_width=True)

if os.path.exists(rank_path):
    import pandas as pd
    st.subheader("📊 Final Rankings")
    r_df = pd.read_csv(rank_path)

    display_cols = [
        c for c in
        ["rank","name","ml_score",
         "docking_score","composite",
         "predicted_ic50_nm",
         "better_than_ref_ic50"]
        if c in r_df.columns]

    st.dataframe(
        r_df[display_cols].head(10),
        use_container_width=True)

    plot_path = \
        "results/ranking/pareto_plot.png"
    if os.path.exists(plot_path):
        st.image(plot_path,
            caption="Final Ranking Plot",
            use_column_width=True)

st.divider()
st.caption(
    "AutoAnalog v2.0 | Snakemake | "
    "ChemBERTa + AutoDock Vina + OpenMM | "
    "Computational predictions only")
