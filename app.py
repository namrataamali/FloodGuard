from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='FloodGuard AI', page_icon='🌊', layout='wide')
ROOT = Path(__file__).parent
FEATURES = ['water_level','rainfall','Previous_Rainfall','Previous_Water_Level','Water_Level_Change','Rainfall_3Day','Warning_Distance','Danger_Distance']
THRESHOLD = 0.60
st.markdown('''<style>
.stApp {background: #0b1220; color:#e9f1fc;}
[data-testid="stSidebar"] {background:#101c2e;}
[data-testid="stMetric"] {background:#14233a;border:1px solid #263a55;border-radius:14px;padding:15px;}
h1,h2,h3 {color:#eaf5ff!important;} .subtle{color:#9cb0ca}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    paths = [ROOT/'models'/'risk_model.joblib', ROOT/'models'/'escalation_model.joblib']
    if not all(p.exists() for p in paths):
        return None, None
    return joblib.load(paths[0]), joblib.load(paths[1])

@st.cache_data
def read_uploaded(content, filename):
    if filename.lower().endswith('.parquet'):
        return pd.read_parquet(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content))

def prepare(data):
    df = data.copy()
    required = ['station_code','date_ist','water_level','rainfall','warning_level','danger_level']
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    df['date_ist'] = pd.to_datetime(df['date_ist'], errors='coerce')
    for c in ['water_level','rainfall','warning_level','danger_level']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=required).sort_values(['station_code','date_ist']).copy()
    df['Previous_Rainfall'] = df.groupby('station_code')['rainfall'].shift(1)
    df['Previous_Water_Level'] = df.groupby('station_code')['water_level'].shift(1)
    df['Water_Level_Change'] = df['water_level'] - df['Previous_Water_Level']
    df['Rainfall_3Day'] = df.groupby('station_code')['rainfall'].transform(lambda s: s.rolling(3).sum())
    df['Warning_Distance'] = df['warning_level'] - df['water_level']
    df['Danger_Distance'] = df['danger_level'] - df['water_level']
    prev = df.groupby('station_code')['date_ist'].shift(1)
    prev2 = df.groupby('station_code')['date_ist'].shift(2)
    consecutive = (df['date_ist']-prev).dt.days.eq(1) & (prev-prev2).dt.days.eq(1)
    df = df.loc[consecutive].dropna(subset=FEATURES).copy()
    df['Current_Risk'] = np.select([df['water_level'] >= df['danger_level'], df['water_level'] >= df['warning_level']], ['High','Medium'], default='Low')
    return df

st.title('🌊 FloodGuard AI')
st.caption('India River Monitoring • Historical CWC-derived observations • Next-day risk research prototype')
risk_model, escalation_model = load_models()
with st.sidebar:
    st.header('Control Center')
    page = st.radio('Workspace', ['Overview','Station Explorer','Batch Predictions','Model & Limitations'])
    st.divider()
    st.caption('Escalation alert-score threshold')
    st.metric('Threshold', f'{THRESHOLD:.2f}')
    st.caption('Chosen on validation data; not a calibrated flood probability.')
    st.divider()
    st.caption('Data source: CWC observations compiled by diagram-chasing/cwc-flood-forecasts. No live feed.')

if risk_model is None or escalation_model is None:
    st.warning('Trained model files are not installed. Export both models from Colab and place them in the models/ folder. Data exploration is available without them.')

st.info('Upload station-day observations (CSV or Parquet). Each station needs at least 3 consecutive daily records. Upload historical data only; this app does not retrieve live measurements.')
file = st.file_uploader('Upload station observations', type=['csv','parquet'])
if file is not None:
    try:
        raw = read_uploaded(file.getvalue(), file.name)
        data = prepare(raw)
        if data.empty:
            st.error('No valid 3-day consecutive station histories were found.')
            st.stop()
        if risk_model is not None:
            data['Predicted_Next_Day_Risk'] = risk_model.predict(data[FEATURES])
        if escalation_model is not None:
            eligible = data['Current_Risk'] != 'High'
            data['Escalation_Score'] = np.nan
            data['Escalation_Alert'] = 'Not applicable: already High'
            if eligible.any():
                scores = escalation_model.predict_proba(data.loc[eligible,FEATURES])[:,list(escalation_model.classes_).index(1)]
                data.loc[eligible,'Escalation_Score'] = scores
                data.loc[eligible,'Escalation_Alert'] = np.where(scores >= THRESHOLD,'Review needed','No alert')
        data['Station'] = data['name'].fillna(data['station_code'].astype(str)) if 'name' in data else data['station_code'].astype(str)
    except Exception as e:
        st.error(f'Could not process file: {e}')
        st.stop()
else:
    data = None

if page == 'Overview':
    st.subheader('Monitoring Overview')
    if data is None:
        st.write('Upload your station dataset to explore observations and predictions.')
    else:
        latest = data.sort_values('date_ist').groupby('station_code').tail(1)
        a,b,c,d = st.columns(4)
        a.metric('Stations', f'{len(latest):,}')
        b.metric('Observation rows', f'{len(data):,}')
        c.metric('Currently High', f'{(latest.Current_Risk=="High").sum():,}')
        d.metric('Escalation alerts', f'{(latest.get("Escalation_Alert",pd.Series(index=latest.index,dtype=str))=="Review needed").sum():,}')
        counts = latest['Current_Risk'].value_counts().reindex(['Low','Medium','High'],fill_value=0).reset_index()
        counts.columns = ['Risk','Stations']
        fig = px.bar(counts,x='Risk',y='Stations',color='Risk',color_discrete_map={'Low':'#27c4a8','Medium':'#e8b85b','High':'#f27b8b'},template='plotly_dark',title='Latest observed risk category')
        fig.update_layout(paper_bgcolor='#0b1220',plot_bgcolor='#0b1220',showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(latest[['Station','date_ist','water_level','rainfall','Current_Risk'] + ([ 'Predicted_Next_Day_Risk','Escalation_Score','Escalation_Alert'] if risk_model is not None and escalation_model is not None else [])],use_container_width=True)

elif page == 'Station Explorer':
    st.subheader('Station Explorer')
    if data is None:
        st.write('Upload data to explore individual monitoring stations.')
    else:
        options = data[['station_code','Station']].drop_duplicates('station_code').sort_values('Station')
        labels = {str(r.station_code):f'{r.Station} ({r.station_code})' for r in options.itertuples()}
        chosen = st.selectbox('Select station',list(labels),format_func=lambda v:labels[v])
        station = data[data['station_code'].astype(str)==chosen].sort_values('date_ist')
        last = station.iloc[-1]
        a,b,c = st.columns(3)
        a.metric('Latest water level',f'{last.water_level:.2f} m')
        b.metric('Current category',last.Current_Risk)
        c.metric('Next-day prediction',last.get('Predicted_Next_Day_Risk','Model unavailable'))
        if 'Escalation_Score' in station:
            st.write(f'Escalation alert score: **{last.Escalation_Score:.3f}**' if pd.notna(last.Escalation_Score) else 'Escalation model applies only when current risk is not High.')
            st.write(f'Alert status: **{last.Escalation_Alert}**')
        plot = station.tail(180)
        fig = px.line(plot,x='date_ist',y='water_level',template='plotly_dark',title='Observed water level (latest 180 records)')
        fig.add_hline(y=last.warning_level,line_dash='dash',line_color='#e8b85b',annotation_text='Warning level')
        fig.add_hline(y=last.danger_level,line_dash='dash',line_color='#f27b8b',annotation_text='Danger level')
        fig.update_layout(paper_bgcolor='#0b1220',plot_bgcolor='#0b1220')
        st.plotly_chart(fig,use_container_width=True)
        fig2 = px.bar(plot,x='date_ist',y='rainfall',template='plotly_dark',title='Daily rainfall field (verify source aggregation)')
        fig2.update_layout(paper_bgcolor='#0b1220',plot_bgcolor='#0b1220')
        st.plotly_chart(fig2,use_container_width=True)

elif page == 'Batch Predictions':
    st.subheader('Batch Predictions & Export')
    if data is None:
        st.write('Upload data to generate a station-level report.')
    elif risk_model is None or escalation_model is None:
        st.warning('Both trained models are required for predictions.')
    else:
        latest_only = st.checkbox('Only latest record per station',value=True)
        report = data.sort_values('date_ist').groupby('station_code').tail(1) if latest_only else data
        show = ['station_code','Station','date_ist','water_level','rainfall','Current_Risk','Predicted_Next_Day_Risk','Escalation_Score','Escalation_Alert']
        st.dataframe(report[show],use_container_width=True)
        st.download_button('Download predictions CSV',report[show].to_csv(index=False).encode('utf-8'),'floodguard_predictions.csv','text/csv')
        st.caption('Predictions are for the calendar day after each observation date, not necessarily tomorrow relative to today.')

else:
    st.subheader('Models, Evaluation & Limitations')
    st.write('**Risk classifier:** Random Forest trained to classify the next-day station-specific water-level threshold category.')
    st.write('**Escalation classifier:** Random Forest trained to identify transitions from non-High today to High tomorrow.')
    st.write('**Escalation cutoff:** 0.60, chosen from historical validation data (F1 ≈ 0.288; recall ≈ 34.3%; precision ≈ 24.9%).')
    st.write('**Important:** Scores are not calibrated probabilities; training used negative-class downsampling. The cutoff must be evaluated on the held-out test set before deployment.')
    st.write('**Limitations:** Historical observations, not live forecasts; no verified flood-event labels; thresholds may vary historically; rainfall aggregation needs source validation; missing dates are excluded; station selection and temporal shifts can affect generalization.')
    st.write('**Data attribution:** [CWC Flood Forecasts compilation](https://github.com/diagram-chasing/cwc-flood-forecasts). Review its provenance and terms before redistribution.')
