import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title=" Mushroom Safety Checker",
    page_icon="🍄",
    layout="centered"
)

# ── Load Model & Encoders ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model          = joblib.load('models/best_model.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        target_encoder = joblib.load('models/target_encoder.pkl')
        feature_cols   = joblib.load('models/feature_columns.pkl')
        return model, label_encoders, target_encoder, feature_cols
    except Exception as e:
        st.error(f" Could not load model files: {e}")
        st.info("Please run the notebook first to generate the model .pkl files, then place them in the models/ folder.")
        st.stop()

model, label_encoders, target_encoder, feature_columns = load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title(" ◆ About")
    st.markdown("""
    This app predicts whether a mushroom is **edible** or **poisonous**
    based on its physical characteristics.

    **Model:** Best of 7 ML classifiers  
    **Tuning:** GridSearchCV with 5-Fold CV  
    **Features:** 20+ physical traits  

    ---
    ▶ **IIT Madras BS Degree**  
    Machine Learning Practice  
    Kaggle Competition — Jan 2026  
    
    ---
    ⚠️ *For educational purposes only.
    Never eat a wild mushroom based
    on this prediction alone!*
    """)

# ── Title ─────────────────────────────────────────────────────────
st.title(" Mushroom Safety Checker")
st.markdown("#### Is this mushroom safe to eat or deadly poisonous?")
st.markdown("Fill in the mushroom's physical features below and click **Check Safety**.")
st.divider()

# ── Feature Options ───────────────────────────────────────────────
# Readable labels for each feature value
cap_shape_map    = {'b':'bell','c':'conical','f':'flat','k':'knobbed','s':'sunken','x':'convex'}
cap_surface_map  = {'f':'fibrous','g':'grooves','s':'smooth','y':'scaly'}
cap_color_map    = {'b':'buff','c':'cinnamon','e':'red','g':'gray','n':'brown','p':'pink','r':'green','u':'purple','w':'white','y':'yellow'}
bruises_map      = {'f':'no bruises','t':'bruises'}
odor_map         = {'a':'almond','c':'creosote','f':'foul','l':'anise','m':'musty','n':'none','p':'pungent','s':'spicy','y':'fishy'}
gill_attach_map  = {'a':'attached','d':'descending','f':'free','n':'notched'}
gill_spacing_map = {'c':'close','w':'crowded','d':'distant'}
gill_size_map    = {'b':'broad','n':'narrow'}
gill_color_map   = {'b':'buff','e':'red','g':'gray','h':'chocolate','k':'black','n':'brown','o':'orange','p':'pink','r':'green','u':'purple','w':'white','y':'yellow'}
stalk_shape_map  = {'e':'enlarging','t':'tapering'}
ring_type_map    = {'e':'evanescent','f':'flaring','l':'large','n':'none','p':'pendant'}
spore_color_map  = {'b':'buff','h':'chocolate','k':'black','n':'brown','o':'orange','r':'green','u':'purple','w':'white','y':'yellow'}
habitat_map      = {'d':'woods','g':'grasses','l':'leaves','m':'meadows','p':'paths','u':'urban','w':'waste'}

def fmt(d):
    return list(d.keys()), lambda x: f"{x}  —  {d[x]}"

# ── Input Form ────────────────────────────────────────────────────
st.subheader("🔍 Enter Mushroom Characteristics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Cap**")
    keys, fmt_fn = fmt(cap_shape_map)
    cap_shape = st.selectbox("Cap shape", keys, format_func=fmt_fn)

    keys, fmt_fn = fmt(cap_surface_map)
    cap_surface = st.selectbox("Cap surface", keys, format_func=fmt_fn)

    keys, fmt_fn = fmt(cap_color_map)
    cap_color = st.selectbox("Cap color", keys, format_func=fmt_fn)

    keys, fmt_fn = fmt(bruises_map)
    bruises = st.selectbox("Bruises?", keys, format_func=fmt_fn)

    st.markdown("**Odor**")
    keys, fmt_fn = fmt(odor_map)
    odor = st.selectbox("Odor", keys, format_func=fmt_fn)

    st.markdown("**Gill**")
    keys, fmt_fn = fmt(gill_attach_map)
    gill_attachment = st.selectbox("Gill attachment", keys, format_func=fmt_fn)

    keys, fmt_fn = fmt(gill_spacing_map)
    gill_spacing = st.selectbox("Gill spacing", keys, format_func=fmt_fn)

    keys, fmt_fn = fmt(gill_size_map)
    gill_size = st.selectbox("Gill size", keys, format_func=fmt_fn)

    keys, fmt_fn = fmt(gill_color_map)
    gill_color = st.selectbox("Gill color", keys, format_func=fmt_fn)

with col2:
    st.markdown("**Stalk**")
    keys, fmt_fn = fmt(stalk_shape_map)
    stalk_shape = st.selectbox("Stalk shape", keys, format_func=fmt_fn)

    stalk_root_opts = {'b':'bulbous','c':'club','e':'equal','r':'rooted','?':'missing'}
    keys, fmt_fn = fmt(stalk_root_opts)
    stalk_root = st.selectbox("Stalk root", keys, format_func=fmt_fn)

    stalk_surf_opts = {'f':'fibrous','k':'silky','s':'smooth','y':'scaly'}
    keys, fmt_fn = fmt(stalk_surf_opts)
    stalk_surface_above = st.selectbox("Stalk surface above ring", keys, format_func=fmt_fn)
    stalk_surface_below = st.selectbox("Stalk surface below ring", keys, format_func=fmt_fn)

    stalk_color_opts = {'b':'buff','c':'cinnamon','e':'red','g':'gray','n':'brown','o':'orange','p':'pink','w':'white','y':'yellow'}
    keys, fmt_fn = fmt(stalk_color_opts)
    stalk_color_above = st.selectbox("Stalk color above ring", keys, format_func=fmt_fn)
    stalk_color_below = st.selectbox("Stalk color below ring", keys, format_func=fmt_fn)

    st.markdown("**Other**")
    veil_color_opts = {'n':'brown','o':'orange','w':'white','y':'yellow'}
    keys, fmt_fn = fmt(veil_color_opts)
    veil_color = st.selectbox("Veil color", keys, format_func=fmt_fn)

    keys, fmt_fn = fmt(ring_type_map)
    ring_type = st.selectbox("Ring type", keys, format_func=fmt_fn)

    keys, fmt_fn = fmt(spore_color_map)
    spore_print_color = st.selectbox("Spore print color", keys, format_func=fmt_fn)

    population_opts = {'a':'abundant','c':'clustered','n':'numerous','s':'scattered','v':'several','y':'solitary'}
    keys, fmt_fn = fmt(population_opts)
    population = st.selectbox("Population", keys, format_func=fmt_fn)

    keys, fmt_fn = fmt(habitat_map)
    habitat = st.selectbox("Habitat", keys, format_func=fmt_fn)

st.divider()

# ── Predict ───────────────────────────────────────────────────────
if st.button("🔍 Check Safety", use_container_width=True, type="primary"):

    # Build raw input dict
    raw_input = {
        'cap-shape':               cap_shape,
        'cap-surface':             cap_surface,
        'cap-color':               cap_color,
        'bruises':                 bruises,
        'odor':                    odor,
        'gill-attachment':         gill_attachment,
        'gill-spacing':            gill_spacing,
        'gill-size':               gill_size,
        'gill-color':              gill_color,
        'stalk-shape':             stalk_shape,
        'stalk-root':              stalk_root,
        'stalk-surface-above-ring': stalk_surface_above,
        'stalk-surface-below-ring': stalk_surface_below,
        'stalk-color-above-ring':  stalk_color_above,
        'stalk-color-below-ring':  stalk_color_below,
        'veil-type':               'p',   # always 'p' in this dataset
        'veil-color':              veil_color,
        'ring-number':             'o',   # most common
        'ring-type':               ring_type,
        'spore-print-color':       spore_print_color,
        'population':              population,
        'habitat':                 habitat,
    }

    # Build dataframe using only columns the model knows
    input_dict = {}
    for col in feature_columns:
        val = str(raw_input.get(col, 'n'))
        input_dict[col] = val

    input_df = pd.DataFrame([input_dict])

    # Encode using saved label encoders
    input_encoded = input_df.copy()
    for col in feature_columns:
        if col in label_encoders:
            le  = label_encoders[col]
            val = str(input_df[col].iloc[0])
            if val in le.classes_:
                input_encoded[col] = le.transform([val])[0]
            else:
                input_encoded[col] = 0   # fallback for unseen value

    # Predict
    prediction_enc = model.predict(input_encoded[feature_columns])
    prediction     = target_encoder.inverse_transform(prediction_enc)[0]

    # Confidence
    confidence_text = ""
    if hasattr(model, 'predict_proba'):
        proba      = model.predict_proba(input_encoded[feature_columns])[0]
        confidence = max(proba) * 100
        confidence_text = f"  (Confidence: {confidence:.1f}%)"

    # Result
    st.divider()
    if prediction == 'e':
        st.success(f"## ✅ EDIBLE{confidence_text}")
        st.markdown("This mushroom appears to be **safe to eat** based on its physical features.")
        st.balloons()
    else:
        st.error(f"## ☠️ POISONOUS{confidence_text}")
        st.markdown("**⚠️ WARNING:** This mushroom appears to be **toxic and dangerous**. Do NOT eat it.")

    # Show what was entered
    with st.expander("See what you entered"):
        display_df = pd.DataFrame([raw_input]).T
        display_df.columns = ['Value']
        st.dataframe(display_df, use_container_width=True)

    st.divider()
    st.caption("⚠️ This prediction is for educational purposes only. Always consult a mycologist before consuming any wild mushroom.")

