import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import altair as alt
import os
import json

# -----------------------------------
# CONFIGURATION PERSISTENCE
# -----------------------------------
CONFIG_FILE = "module_config.json"
ALL_MODULES = [
    "1. Anatomía diaria",
    "2. Reversiones intradía",
    "3. Cronoanálisis intradía",
    "4. Probabilidades de cierre",
    "5. Distribución de movimiento",
    "6. Comportamientos extremos",
    "7. Insights & señales",
    "8. Histograma retornos diarios",
    "9. Cambios de signo intradía"
]
if os.path.exists(CONFIG_FILE):
    try:
        saved_modules = json.load(open(CONFIG_FILE))
    except:
        saved_modules = ALL_MODULES.copy()
else:
    saved_modules = ALL_MODULES.copy()

# -----------------------------------
# DATA FETCH UTILITY
# -----------------------------------
def fetch_spy_data(interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    ticker = yf.Ticker('SPY')
    df = ticker.history(
        interval=interval,
        start=start_date.strftime('%Y-%m-%d'),
        end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
        prepost=False,
        auto_adjust=False
    )
    if df.empty and interval != '1d':
        limits = {'1m':7,'5m':59,'15m':59,'30m':59,'1h':730}
        fallback = limits.get(interval,59)
        st.warning(f"Intervalo {interval} sin datos; usando últimos {fallback} días.")
        df = ticker.history(
            period=f"{fallback}d",
            interval=interval,
            prepost=False,
            auto_adjust=False
        )
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df.tz_convert('America/New_York')

# -----------------------------------
# CSV LOADER UTILITY
# -----------------------------------
def load_csv_df(uploaded_file) -> pd.DataFrame:
    """
    Lee CSV de 15m con encabezado o sin él y normaliza nombres de columnas.
    """
    # Intento con encabezado existente
    try:
        df = pd.read_csv(
            uploaded_file,
            sep=';',
            header=0,
            index_col=0,
            parse_dates=True
        )
    except Exception:
        # Fallback: asignar nombres
        df = pd.read_csv(
            uploaded_file,
            sep=';',
            header=None,
            names=['Datetime','Open','High','Low','Close','Volume'],
            index_col=0,
            parse_dates=['Datetime']
        )
    # Normalizar nombres de columnas a capitalized
    df.columns = df.columns.str.strip().str.title()
    # Asegurar índice de tiempo con TZ correcto
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York')
    return df

# -----------------------------------
# MODULE IMPLEMENTATIONS
# -----------------------------------

def module_daily_anatomy(df: pd.DataFrame):
    st.subheader("1. Anatomía de la vela diaria")
    d = df.copy()
    d['Range'] = d['High'] - d['Low']
    d['Body']  = (d['Close'] - d['Open']).abs()
    d['Upper_Wick'] = d['High'] - d[['Open','Close']].max(axis=1)
    d['Lower_Wick'] = d[['Open','Close']].min(axis=1) - d['Low']
    stats = pd.DataFrame({
        'Mean':   [d[c].mean() for c in ['Range','Body','Upper_Wick','Lower_Wick']],
        'Median': [d[c].median() for c in ['Range','Body','Upper_Wick','Lower_Wick']],
        'Std':    [d[c].std() for c in ['Range','Body','Upper_Wick','Lower_Wick']]
    }, index=['Range','Body','Upper_Wick','Lower_Wick'])
    st.table(stats)
    st.write(f"% Alcista: {100*(d['Close']>d['Open']).mean():.2f}%")
    st.write(f"% Bajista: {100*(d['Close']<d['Open']).mean():.2f}%")


def module_intraday_reversion(intra: pd.DataFrame, daily: pd.DataFrame):
    st.subheader("2. Estadísticas de reversión intradía")
    if intra.empty:
        st.info("No hay datos intradía.")
        return
    df2 = intra.reset_index().rename(columns={intra.index.name or 'index':'Datetime'})
    df2['Day'] = df2['Datetime'].dt.date
    results=[]
    for day, grp in df2.groupby('Day'):
        prev_close = daily['Close'].shift(1).get(pd.Timestamp(day), np.nan)
        if pd.isna(prev_close): continue
        o, c = grp['Open'].iloc[0], grp['Close'].iloc[-1]
        rev = (o<prev_close and c>o) or (o>prev_close and c<o)
        d = {'Day':day, 'Reverted':rev, 'Gap%':100*(o/prev_close-1)}
        if c>o:
            d['Min%']=100*(grp['Low'].min()/o-1)
            d['T_Min']=grp.loc[grp['Low'].idxmin(),'Datetime'].time()
        else:
            d['Max%']=100*(grp['High'].max()/o-1)
            d['T_Max']=grp.loc[grp['High'].idxmax(),'Datetime'].time()
        results.append(d)
    res = pd.DataFrame(results)
    if res.empty:
        st.write("No se detectaron reversiones.")
        return
    st.write(f"Prob. reversión: {100*res['Reverted'].mean():.2f}%")


def module_chrono(intra: pd.DataFrame):
    st.subheader("3. Cronoanálisis intradía")
    if intra.empty:
        st.info("No hay datos intradía.")
        return
    df3 = intra.reset_index().rename(columns={intra.index.name or 'index':'Datetime'})
    df3['Block'] = df3['Datetime'].dt.strftime('%H:%M')
    df3['Ret%']  = 100*(df3['Close']/df3['Open']-1)
    summary = df3.groupby('Block')['Ret%'].agg(['mean','std','count'])
    summary['Up%'] = df3.groupby('Block').apply(lambda g:(g['Close']>g['Open']).mean()*100)
    st.line_chart(summary[['mean','Up%']])
    seq = (df3['Close']>df3['Open']).astype(int)
    trans = pd.crosstab(seq.shift(), seq, normalize='index')*100
    st.write("Matriz transiciones (%):")
    st.table(trans)

def module_close_prob(daily: pd.DataFrame):
    st.subheader("4. Probabilidades de cierre relativo")
    d = daily.copy()
    d['Prev_Close'] = d['Close'].shift(1)
    d['Prev_High']  = d['High'].shift(1)
    d['Prev_Low']   = d['Low'].shift(1)

    # Slider de offset ahora de -5% a +5%
    off = st.sidebar.slider(
        "Offset % (puede ser negativo)",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.1
    )

    # Precio objetivo con offset
    d['OffPrice'] = d['Prev_Close'] * (1 + off/100)

    # Métricas base
    st.write(f"% cierre > open:          {100*(d['Close']>d['Open']).mean():.2f}%")
    st.write(f"% cierre > prev close:    {100*(d['Close']>d['Prev_Close']).mean():.2f}%")
    st.write(f"% cierre > prev high:     {100*(d['Close']>d['Prev_High']).mean():.2f}%")
    st.write(f"% cierre < prev low:      {100*(d['Close']<d['Prev_Low']).mean():.2f}%")

    # Offset positive or negative
    st.write(f"% cierre > offset ({off:+.1f}%): {100*(d['Close']>d['OffPrice']).mean():.2f}%")
    st.write(f"% cierre < offset ({off:+.1f}%): {100*(d['Close']<d['OffPrice']).mean():.2f}%")



def module_distribution(daily: pd.DataFrame):
    st.subheader("5. Distribución de movimiento")
    # Cálculo del rango porcentual
    df = daily.copy()
    df['Range%'] = 100 * (df['High'] - df['Low']) / df['Open']
    r = df['Range%']

    # Determinar el máximo absoluto para el dominio (-max, +max)
    max_r = r.max()
    domain_max = round(max_r + 0.1, 1)
    domain = [0, domain_max]

    # Histograma con bins de 0.1% y dominio simétrico, interactivo para zoom
    chart = (
        alt.Chart(df)
           .mark_bar()
           .encode(
               alt.X('Range%:Q',
                     bin=alt.Bin(step=0.1, extent=domain),
                     title='Rango % (bins de 0.1%)'),
               y=alt.Y('count()', title='Frecuencia')
           )
           .properties(
               title='Distribución de amplitud intradía'
           )
           .interactive()   # <— Habilita zoom rectangular y paneo
    )
    st.altair_chart(chart, use_container_width=True)
	

def module_extremes(daily: pd.DataFrame):
    st.subheader("6. Comportamientos extremos")
    d = daily.copy()
    d['Gap%']=100*(d['Open']/d['Close'].shift(1)-1)
    st.write("Top 5 gaps %:")
    st.table(d['Gap%'].abs().nlargest(5))
    d['Size%']=100*(d['Close']/d['Open']-1)
    st.write("Top 5 velas %:")
    st.table(d['Size%'].abs().nlargest(5))


def module_insights(daily: pd.DataFrame):
    st.subheader("7. Insights & señales")
    d = daily.copy()
    d['Body']=abs(d['Close']-d['Open'])
    d['Low_Wick']=d[['Open','Close']].min(axis=1)-d['Low']
    pat = d[(d['Low_Wick']>2*d['Body'])&(d['Close']>d['Open'])]
    st.write(f"Patrones: {len(pat)} días")


def module_returns_histogram(daily: pd.DataFrame):
    st.subheader("8. Histograma retornos diarios")
    # Calcular retorno diario %
    df = daily.copy()
    df['Ret%'] = 100 * (df['Close'] / df['Open'] - 1)
    series = df['Ret%']

    # Dominio simétrico basado en el valor absoluto máximo
    max_r = series.abs().max()
    domain_max = round(max_r + 0.1, 1)
    extent = [-domain_max, domain_max]

    # Histograma con bins de 0.1% y zoom interactivo
    chart = (
        alt.Chart(df)
           .mark_bar()
           .encode(
               alt.X('Ret%:Q',
                     bin=alt.Bin(step=0.1, extent=extent),
                     title='Retorno diario (%)'),
               alt.Y('count()', title='Frecuencia')
           )
           .properties(title='Histograma retornos diarios')
           .interactive()   # habilita zoom rectangular y paneo
    )
    st.altair_chart(chart, use_container_width=True)



def module_sign_changes(intra: pd.DataFrame):
    st.subheader("9. Cambios de signo intradía & dispersión")
    if intra.empty:
        st.info("No hay datos intradía.")
        return
    # Cálculo de cierre diario a partir de intradía
    daily_close_series = intra['Close'].resample('D').last().dropna()
    # Mapear por fecha (date) sin zona horaria
    daily_close_by_date = {ts.date(): price for ts, price in daily_close_series.items()}

    # Preparar DataFrame base
    df2 = intra.reset_index().rename(columns={intra.index.name or 'index':'Datetime'})
    df2['Day'] = df2['Datetime'].dt.date
    records = []
    for day, grp in df2.groupby('Day'):
        prev_date = day - timedelta(days=1)
        pc = daily_close_by_date.get(prev_date, np.nan)
        if pd.isna(pc):
            continue
        # Serie de signos: 1 si Close > pc, 0 en caso contrario
        sig = (grp['Close'] > pc).astype(int)
        # Detectar cambios de signo (0->1 o 1->0)
        change_points = sig.diff().abs() == 1
        idxs = np.where(change_points)[0]
        cnt = len(idxs)
        if cnt == 0:
            continue
        if cnt > 1:
            diffs = np.diff(idxs)
            min_i = int(diffs.min())
            max_i = int(idxs[-1] - idxs[0])
        else:
            min_i = max_i = 0
        records.append({'Day': day, 'SignChanges': cnt, 'MinInterval': min_i, 'MaxInterval': max_i})
    if not records:
        st.write("Sin cambios de signo detectados.")
        return
    df_rec = pd.DataFrame(records).set_index('Day')
    st.dataframe(df_rec)
    st.write(f"Promedio cambios: {df_rec['SignChanges'].mean():.2f}")
    st.write(f"Intervalo mínimo promedio: {df_rec['MinInterval'].mean():.2f} velas")
    st.write(f"Intervalo máximo promedio: {df_rec['MaxInterval'].mean():.2f} velas")
    hist = (
        alt.Chart(df_rec.reset_index())
        .mark_bar()
        .encode(
            x=alt.X('SignChanges:Q', bin=alt.Bin(step=1), title='Nº cambios'),
            y=alt.Y('count():Q', title='Frecuencia')
        )
        .properties(title='Histograma de cambios de signo intradía')
    )
    st.altair_chart(hist, use_container_width=True)

# -----------------------------------
# MAIN APP
# -----------------------------------
st.title("Análisis SPY Avanzado")

# Sidebar: parámetros generales
start = st.sidebar.date_input("Inicio", datetime(2010,1,1))
end   = st.sidebar.date_input("Fin", datetime.now().date())
base_interval = st.sidebar.selectbox("Intervalo Yahoo", ['1d','1h','30m','15m','5m','1m'], index=0)
sign_interval = st.sidebar.selectbox("Intervalo señal", ['5m','15m','30m','1h'], index=0)
use_csv = st.sidebar.checkbox("Usar CSV histórico (15m)")
uploaded_file = None
if use_csv:
    uploaded_file = st.sidebar.file_uploader("Cargar CSV 15m", type='csv')
    if uploaded_file:
        st.sidebar.success("CSV cargado")
        base_interval = '15m'
modules = st.sidebar.multiselect("Módulos a ejecutar", ALL_MODULES, default=saved_modules)
run = st.sidebar.button("Analizar")

if 'run' not in st.session_state:
    st.session_state.run = False
if run:
    st.session_state.run = True

if not st.session_state.run:
    st.info("Configura y pulsa 'Analizar'.")
    st.stop()

# Carga de datos diarios
daily_df = fetch_spy_data(
    '1d',
    datetime.combine(start, datetime.min.time()),
    datetime.combine(end, datetime.min.time())
)
# Carga de datos intradía
if use_csv:
    if not uploaded_file:
        st.error("Debe subir un CSV.")
        st.stop()
    intra_df = load_csv_df(uploaded_file)
else:
    intra_df = fetch_spy_data(
        base_interval,
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.min.time())
    )
# Ejecución de módulos
for m in modules:
    if m.startswith('1'):
        module_daily_anatomy(daily_df)
    if m.startswith('2'):
        module_intraday_reversion(intra_df, daily_df)
    if m.startswith('3'):
        module_chrono(intra_df)
    if m.startswith('4'):
        module_close_prob(daily_df)
    if m.startswith('5'):
        module_distribution(daily_df)
    if m.startswith('6'):
        module_extremes(daily_df)
    if m.startswith('7'):
        module_insights(daily_df)
    if m.startswith('8'):
        module_returns_histogram(daily_df)
    if m.startswith('9'):
        module_sign_changes(intra_df)

# Exportar CSV
df_export = intra_df if base_interval != '1d' else daily_df
st.download_button("Exportar CSV", df_export.to_csv().encode('utf-8'), f"spy_{base_interval}.csv", "text/csv")


