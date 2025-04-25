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
# UTILITIES
# -----------------------------------
def get_symbol_from_filename(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0].upper()

def fetch_data(symbol: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
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
        st.warning(f"Intervalo {interval} sin datos; usando últimos {fallback} días de Yahoo.")
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

def load_csv_df(uploaded_file, dayfirst: bool = False) -> pd.DataFrame:
    """
    Lee un CSV (comma o semicolon), con o sin cabecera,
    y convierte el índice a DatetimeIndex (UTC→America/New_York).
    Usa `dayfirst=True` para formato dd/mm/yyyy.
    """
    # 1. Detectar delimitador
    delim = ','
    try:
        sample = uploaded_file.read(2048) if hasattr(uploaded_file, 'read') else open(uploaded_file, 'r').read(2048)
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
        sniff = csv.Sniffer().sniff(sample)
        delim = sniff.delimiter
    except Exception:
        pass

    # 2. Leer con cabecera o sin ella
    try:
        df = pd.read_csv(
            uploaded_file,
            sep=delim,
            engine='python',
            header=0,
            index_col=0
        )
    except Exception:
        df = pd.read_csv(
            uploaded_file,
            sep=delim,
            engine='python',
            header=None,
            names=['Datetime','Open','High','Low','Close','Volume'],
            index_col=0
        )

    # 3. Normalizar columnas y parsear índice
    df.columns = df.columns.str.strip().str.title()
    df.index = pd.to_datetime(df.index, utc=True, dayfirst=dayfirst)
    df.index = df.index.tz_convert('America/New_York')
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

    # Slider de offset entre -5% y +5%
    off = st.sidebar.slider(
        "Offset % (puede ser negativo)",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.1
    )
    d['OffPrice'] = d['Prev_Close'] * (1 + off / 100)

    # Validar que OffPrice esté dentro del rango del día
    valid = (d['OffPrice'] >= d['Low']) & (d['OffPrice'] <= d['High'])
    d_valid = d[valid].copy()

    # Métricas generales sobre días válidos
    if d_valid.empty:
        st.write("No hay días donde el precio offset esté dentro del rango diario.")
        return
    st.write(f"Días válidos (offset en rango): {len(d_valid)}/{len(d)}")
    st.write(f"% cierre > open:          {100*(d_valid['Close']>d_valid['Open']).mean():.2f}%")
    st.write(f"% cierre > prev_close:    {100*(d_valid['Close']>d_valid['Prev_Close']).mean():.2f}%")
    st.write(f"% cierre > prev_high:     {100*(d_valid['Close']>d_valid['Prev_High']).mean():.2f}%")
    st.write(f"% cierre < prev_low:      {100*(d_valid['Close']<d_valid['Prev_Low']).mean():.2f}%")
    st.write(f"% cierre > offset ({off:+.1f}%): {100*(d_valid['Close']>d_valid['OffPrice']).mean():.2f}%")
    st.write(f"% cierre < offset ({off:+.1f}%): {100*(d_valid['Close']<d_valid['OffPrice']).mean():.2f}%")

    # Filtrar las ocurrencias para descarga
    df_above = d_valid[d_valid['Close'] > d_valid['OffPrice']]
    df_below = d_valid[d_valid['Close'] < d_valid['OffPrice']]

    # Botones de descarga
    if not df_above.empty:
        csv_above = df_above.to_csv(index=True).encode('utf-8')
        st.download_button(
            "Descargar cierres > offset (válidos)",
            data=csv_above,
            file_name=f"cierres_above_offset_{off:+.1f}%.csv",
            mime="text/csv"
        )
    if not df_below.empty:
        csv_below = df_below.to_csv(index=True).encode('utf-8')
        st.download_button(
            "Descargar cierres < offset (válidos)",
            data=csv_below,
            file_name=f"cierres_below_offset_{off:+.1f}%.csv",
            mime="text/csv"
        )



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
st.title("Análisis Cuantitativo")

# Sidebar: ticker input (used if no CSV)
if 'ticker_input' not in st.session_state:
    st.session_state.ticker_input = "SPY"
ticker_input = st.sidebar.text_input(
    "Ticker (Yahoo)",
    value=st.session_state.ticker_input,
    key='ticker_input'
).upper()

# Sidebar: CSV upload options
use_csv_intra = st.sidebar.checkbox("Usar CSV intradía (15m)")
uploaded_intra = None
if use_csv_intra:
    # Listar archivos *_15m.csv en carpeta historicos
    import re
    intra_files = [f for f in os.listdir('historicos') if re.match(r'.+_15m\.csv$', f, re.IGNORECASE)]
    if intra_files:
        selected_intra = st.sidebar.selectbox("Selecciona CSV intradía", intra_files)
        uploaded_intra = os.path.join('historicos', selected_intra)
    else:
        st.sidebar.warning("No se encontraron CSV intradía en carpeta 'historicos'.")

use_csv_daily = st.sidebar.checkbox("Usar CSV diario (1d)")
uploaded_daily = None
if use_csv_daily:
    # Listar archivos *_1d.csv o *_daily.csv en carpeta historicos
    import re
    daily_files = [f for f in os.listdir('historicos') if re.match(r'.+_(1d|daily)\.csv$', f, re.IGNORECASE)]
    if daily_files:
        selected_daily = st.sidebar.selectbox("Selecciona CSV diario", daily_files)
        uploaded_daily = os.path.join('historicos', selected_daily)
    else:
        st.sidebar.warning("No se encontraron CSV diarios en carpeta 'historicos'.")

# Sidebar: Yahoo interval selector for intradía when not using CSV intradía for intradía when not using CSV intradía
yahoo_interval = st.sidebar.selectbox(
    "Intervalo intradía Yahoo", ['1h','30m','15m','5m','1m'], index=2
)

# Sidebar: module selection
modules = st.sidebar.multiselect(
    "Selecciona módulos a ejecutar",
    ALL_MODULES,
    default=st.session_state.get('modules', saved_modules),
    key='modules'
)

# Sidebar: run button
dirun = st.sidebar.button("Analizar", key="run_button")
if 'run' not in st.session_state:
    st.session_state.run = False
if dirun:
    st.session_state.run = True
    # Guardar selección de módulos para la próxima visita
    try:
        json.dump(st.session_state['modules'], open(CONFIG_FILE, 'w'))
    except Exception as e:
        st.error(f"Error guardando configuración de módulos: {e}")

# Block until Analyze
if not st.session_state.run:
    st.info("Configura y pulsa 'Analizar'.")
    st.stop()

# Helper to parse CSV filename: SYMBOL_INTERVAL.csv
import re

def parse_csv_filename(name: str):
    basename = os.path.splitext(os.path.basename(name))[0]
    parts = re.split(r'[_\-]', basename)
    if len(parts) >= 2:
        sym = parts[0].upper()
        intr = parts[-1].lower()
        return sym, intr
    return None, None

# Determine symbol and data source consistency
# Intradía CSV
symbol_intra = ticker_input
if use_csv_intra:
    if not uploaded_intra:
        st.error("Debe cargar el CSV intradía.")
        st.stop()
    sym_i, int_i = parse_csv_filename(uploaded_intra)  # corregido: usar archivo intradía
    if int_i != '15m':
        st.error(f"Intervalo inválido en nombre CSV intradía: {int_i}. Debe ser '15m'.")
        st.stop()
    symbol_intra = sym_i
# Diario CSV
symbol_daily = ticker_input
if use_csv_daily:
    if not uploaded_daily:
        st.error("Debe cargar el CSV diario.")
        st.stop()
    sym_d, int_d = parse_csv_filename(uploaded_daily)  # corregido: parsear ruta de CSV diario directamente
    if int_d not in ('1d','daily'):
        st.error(f"Intervalo inválido en nombre CSV diario: {int_d}. Debe ser '1d'.")
        st.stop()
    symbol_daily = sym_d
# Ensure same symbol
global_symbol = symbol_intra
if use_csv_daily:
    if symbol_daily != symbol_intra:
        st.error(f"Mismatch de símbolo entre intradía ({symbol_intra}) y diario ({symbol_daily}).")
        st.stop()
elif not use_csv_intra:
    global_symbol = ticker_input

# Validate consistency with ticker_input
if use_csv_intra and not use_csv_daily and ticker_input != symbol_intra:
    st.error(f"El ticker seleccionado ({ticker_input}) no coincide con el activo del CSV intradía ({symbol_intra}).")
    st.stop()
if use_csv_daily and not use_csv_intra and ticker_input != symbol_daily:
    st.error(f"El ticker seleccionado ({ticker_input}) no coincide con el activo del CSV diario ({symbol_daily}).")
    st.stop()
# Display asset
st.subheader(f"Activo: {global_symbol}")

# Date range inputs
start = st.sidebar.date_input("Fecha inicio", datetime(2010,1,1))
end   = st.sidebar.date_input("Fecha fin", datetime.now().date())

# Load daily data
if use_csv_daily:
    # Carga el CSV diario asumiendo fechas dd/mm/yyyy
    daily_df = load_csv_df(uploaded_daily, dayfirst=True)
else:
    daily_df = fetch_data(
        global_symbol,
        '1d',
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.min.time())
    )

# Load intraday data
intra_df = None
if use_csv_intra:
    intra_df = load_csv_df(uploaded_intra)
else:
    intra_df = fetch_data(global_symbol, yahoo_interval, datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time()))

# Optional raw display
if st.sidebar.checkbox("Mostrar datos brutos intradía"):
    st.subheader("Datos intradía brutos")
    st.write(intra_df.head())

# Execute selected modules
df_export = None
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
    df_export = intra_df if m.startswith(('2','3','9','5','8')) else daily_df

# Export CSV of last dataset used
if df_export is not None:
    st.download_button(
        "Exportar CSV",
        df_export.to_csv().encode('utf-8'),
        f"{global_symbol}_data.csv",
        "text/csv"
    )
