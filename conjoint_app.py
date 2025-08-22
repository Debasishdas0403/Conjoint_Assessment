import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyDOE3 import fullfact
from itertools import product

# Pin page config
st.set_page_config(page_title="Conjoint D-Efficiency Analyzer", layout="wide")

class ConjointDesigner:
    def __init__(self):
        self.attributes = {}
    def calculate_parameters(self, attributes):
        return sum(len(levels)-1 for levels in attributes.values())
    def generate_full_factorial(self, attributes):
        levels = [list(range(len(l))) for l in attributes.values()]
        design = list(product(*levels))
        return pd.DataFrame(design, columns=list(attributes.keys()))
    def calculate_d_efficiency(self, df, p):
        X = pd.get_dummies(df, drop_first=True)
        M = X.T @ X + np.eye(X.shape[1])*1e-6
        det = np.linalg.det(M)
        if det<=0: return 0.0
        n = len(df)
        d_err = (det**(-1/p)) / n
        return min(1.0, 1/d_err)
    def optimize_design(self, attrs, n_resp, n_alt, tgt):
        p = self.calculate_parameters(attrs)
        full = self.generate_full_factorial(attrs)
        results = []
        for q in range(p+2, min(25, len(full)//2)+1):
            runs = n_resp*q*n_alt
            if runs>len(full):
                design = full.sample(runs, replace=True, random_state=42)
            else:
                design = full.sample(runs, replace=False, random_state=42)
            de = self.calculate_d_efficiency(design, p)
            results.append((q, de))
        return pd.DataFrame(results, columns=["Questions","D-Efficiency"])

def main():
    st.title("Conjoint D-Efficiency Analyzer")
    cd = ConjointDesigner()
    # Setup
    st.sidebar.header("Design Setup")
    if "attrs" not in st.session_state:
        st.session_state.attrs = {"Price":["Low","Med","High"],"Brand":["A","B"]}
    # Edit attributes...
    # Get respondents, alternatives, target
    n_resp = st.sidebar.number_input("Respondents",10,500,60)
    n_alt  = st.sidebar.number_input("Alts per question",2,5,2)
    tgt    = st.sidebar.slider("Target D-efficiency",0.5,1.0,0.8,0.05)
    if st.sidebar.button("Run Optimization"):
        df = cd.optimize_design(st.session_state.attrs, n_resp, n_alt, tgt)
        st.subheader("Results")
        st.line_chart(df.set_index("Questions")["D-Efficiency"])
        best = df[df["D-Efficiency"]>=tgt]
        if not best.empty:
            st.success(f"Min questions for ≥{tgt}: {best.iloc[0,0]}")
        else:
            st.warning("Target not reached")
        st.dataframe(df)

if __name__=="__main__":
    main()
