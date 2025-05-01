import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
import os
import json
import re
from datetime import datetime, timedelta
import altair as alt
import csv

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
    "8. Histograma multi-día",
    "9. Cambios de signo intradía",
    "10. Testeo rangos intradía"
]
if os.path.exists(CONFIG_FILE):
    try:
        saved_modules = json.load(open(CONFIG_FILE))
    except:
        saved_modules = ALL_MODULES.copy()
else:
    saved_modules = ALL_MODULES.copy()

# -----------------------------------
# UTILIDAD: Parsear nombre de CSV
# -----------------------------------
def parse_csv_filename(name: str):
    """
    Extrae símbolo e intervalo de archivos tipo SYMBOL_INTERVAL.csv
    """
    basename = os.path.splitext(os.path.basename(name))[0]
    parts = re.split(r'[_\-]', basename)
    if len(parts) >= 2:
        return parts[0].upper(), parts[-1].lower()
    return None, None

# -----------------------------------
# DATA & CSV UTILITIES
# -----------------------------------
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
        df = ticker.history(period=f"{fallback}d", interval=interval, prepost=False, auto_adjust=False)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df.tz_convert('America/New_York')

def load_csv_df(uploaded_file, dayfirst: bool = False) -> pd.DataFrame:
    # Detect delimiter via csv.Sniffer
    delim = ','
    try:
        sample = uploaded_file.read(2048) if hasattr(uploaded_file, 'read') else open(uploaded_file, 'r').read(2048)
        if hasattr(uploaded_file, 'seek'): uploaded_file.seek(0)
        sniff = csv.Sniffer().sniff(sample)
        delim = sniff.delimiter
    except:
        pass
    try:
        df = pd.read_csv(uploaded_file, sep=delim, engine='python', header=0, index_col=0)
    except:
        df = pd.read_csv(uploaded_file, sep=delim, engine='python', header=None,
                         names=['Datetime','Open','High','Low','Close','Volume'], index_col=0)
    df.columns = df.columns.str.strip().str.title()
    df.index = pd.to_datetime(df.index, utc=True, dayfirst=dayfirst)
    df.index = df.index.tz_convert('America/New_York')
    return df

# -----------------------------------
# MODULES 1-7,8,9
# -----------------------------------
def module_daily_anatomy(df: pd.DataFrame):
    st.subheader("1. Anatomía de la vela diaria")
    d = df.copy()
    d['Range'] = d['High'] - d['Low']
    d['Body']  = (d['Close'] - d['Open']).abs()
    d['Upper_Wick'] = d['High'] - d[['Open','Close']].max(axis=1)
    d['Lower_Wick'] = d[['Open','Close']].min(axis=1) - d['Low']
    stats = pd.DataFrame({
        'Mean':[d[c].mean() for c in ['Range','Body','Upper_Wick','Lower_Wick']],
        'Median':[d[c].median() for c in ['Range','Body','Upper_Wick','Lower_Wick']],
        'Std':[d[c].std() for c in ['Range','Body','Upper_Wick','Lower_Wick']]
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
        rec = {'Day':day, 'Reverted':rev, 'Gap%':100*(o/prev_close-1)}
        if c>o:
            rec['Min%']=100*(grp['Low'].min()/o-1)
            rec['T_Min']=grp.loc[grp['Low'].idxmin(),'Datetime'].time()
        else:
            rec['Max%']=100*(grp['High'].max()/o-1)
            rec['T_Max']=grp.loc[grp['High'].idxmax(),'Datetime'].time()
        results.append(rec)
    res=pd.DataFrame(results)
    if res.empty:
        st.write("No reversiones detectadas.")
        return
    st.write(f"Prob. reversión intradía: {100*res['Reverted'].mean():.2f}%")

def module_chrono(intra: pd.DataFrame):
    st.subheader("3. Cronoanálisis intradía")
    if intra.empty:
        st.info("No hay datos intradía.")
        return
    df3 = intra.reset_index().rename(columns={intra.index.name or 'index':'Datetime'})
    df3['Block'] = df3['Datetime'].dt.strftime('%H:%M')
    df3['Ret%']  = 100*(df3['Close']/df3['Open']-1)
    summary = df3.groupby('Block')['Ret%'].agg(['mean','std','count'])
    summary['Up%']=df3.groupby('Block').apply(lambda g:(g['Close']>g['Open']).mean()*100)
    st.line_chart(summary[['mean','Up%']])
    seq=(df3['Close']>df3['Open']).astype(int)
    trans=pd.crosstab(seq.shift(), seq, normalize='index')*100
    st.write("Matriz transiciones (%):")
    st.table(trans)

def module_close_prob(daily: pd.DataFrame):
    st.subheader("4. Probabilidades de cierre relativo")
    d=daily.copy()
    d['Prev_Close']=d['Close'].shift(1)
    d['Prev_High']=d['High'].shift(1)
    d['Prev_Low']=d['Low'].shift(1)
    off=st.sidebar.slider("Offset % (negativo a positivo)", -5.0,5.0,0.5,0.1)
    d['OffPrice']=d['Prev_Close']*(1+off/100)
    valid=(d['OffPrice']>=d['Low'])&(d['OffPrice']<=d['High'])
    d_valid=d[valid]
    if d_valid.empty:
        st.write("No hay días con offset en rango.")
        return
    st.write(f"Días válidos: {len(d_valid)} / {len(d)}")
    st.write(f"% cierre>offset ({off:+.1f}%): {100*(d_valid['Close']>d_valid['OffPrice']).mean():.2f}%")
    df_above=d_valid[d_valid['Close']>d_valid['OffPrice']]
    df_below=d_valid[d_valid['Close']<d_valid['OffPrice']]
    if not df_above.empty:
        st.download_button("Descargar cierres>offset", df_above.to_csv().encode(), f"above_{off}.csv","text/csv")
    if not df_below.empty:
        st.download_button("Descargar cierres<offset", df_below.to_csv().encode(), f"below_{off}.csv","text/csv")

def module_distribution(daily: pd.DataFrame):
    st.subheader("5. Distribución de movimiento")
    df=daily.copy()
    df['Range%']=100*(df['High']-df['Low'])/df['Open']
    r=df['Range%']
    max_r=r.max(); domain_max=round(max_r+0.1,1)
    domain=[-domain_max,domain_max]
    chart=(alt.Chart(df)
           .mark_bar()
           .encode(
               alt.X('Range%:Q', bin=alt.Bin(step=0.1, extent=domain), title='Range%'),
               y='count()')
           .properties(title='Distribución amplitud')
           .interactive())
    st.altair_chart(chart,use_container_width=True)

def module_extremes(daily: pd.DataFrame):
    st.subheader("6. Comportamientos extremos")
    d=daily.copy()
    d['Gap%']=100*(d['Open']/d['Close'].shift(1)-1)
    st.table(d['Gap%'].abs().nlargest(5))
    d['Size%']=100*(d['Close']/d['Open']-1)
    st.table(d['Size%'].abs().nlargest(5))

def module_insights(daily: pd.DataFrame):
    st.subheader("7. Insights & señales")
    st.markdown("Detecta patrones de velas clásicos y su retorno al día siguiente.")
    d=daily.copy()
    d['Body']=abs(d['Close']-d['Open'])
    d['Lower_Wick']=d[['Open','Close']].min(axis=1)-d['Low']
    d['Upper_Wick']=d['High']-d[['Open','Close']].max(axis=1)
    prev=d.shift(1)
    patterns={
        'Hammer':(d['Lower_Wick']>2*d['Body'])&(d['Close']>d['Open']),
        'Shooting Star':(d['Upper_Wick']>2*d['Body'])&(d['Close']<d['Open']),
        'Marubozu': d['Body']>=0.9*(d['High']-d['Low']),
        'Doji': d['Body']<=0.05*(d['High']-d['Low']),
        'Engulfing Alcista':(d['Close']>d['Open'])&(d['Open']<prev['Close'])&(d['Close']>prev['Open']),
        'Engulfing Bajista':(d['Close']<d['Open'])&(d['Open']>prev['Close'])&(d['Close']<prev['Open'])
    }
    next_ret=(d['Close'].shift(-1)/d['Open'].shift(-1)-1)*100
    rows=[]
    for name,mask in patterns.items():
        cnt=int(mask.sum())
        avg=next_ret[mask].mean() if cnt>0 else np.nan
        rows.append((name,cnt,round(avg,2)))
        st.markdown(f"- **{name}**: {cnt} días — Retorno 1D avg: {avg:.2f}%")
    summary=pd.DataFrame(rows,columns=['Patrón','Frecuencia','Avg Retorno 1D (%)']).set_index('Patrón')
    st.table(summary)


# -----------------------------------
# MÓDULO 8. Histograma multi-día
# -----------------------------------
def module_multi_day_histogram(daily: pd.DataFrame):
    """
    Muestra la distribución de retornos sobre ventanas de N días.
    El usuario elige N, y cada barra del histograma representa
    el retorno porcentual de ese bloque de N días.
    """
    st.subheader("8. Histograma de retornos multi-día")

    # 1) Selector de ventana
    window = st.slider(
        "Ventana (días)", 
        min_value=1, 
        max_value=60, 
        value=5, 
        step=1
    )

    # 2) Calcular retornos % de cada bloque de N días
    # pct_change con periodos=window: (C_t / C_{t-window} - 1)
    returns = daily['Close'].pct_change(periods=window) * 100

    # 3) Limpiar NaN iniciales
    df = returns.dropna().to_frame(name='Retorno%')
    df['Grupo'] = (np.arange(len(df)) // 1).astype(int)  # índice numérico

    # 4) Histograma con bins de 0.1% y zoom interactivo
    # Dominio simétrico
    max_abs = abs(df['Retorno%']).max()
    domain = [-round(max_abs+0.1,1), round(max_abs+0.1,1)]

    chart = (
        alt.Chart(df.reset_index())
            .mark_bar()
            .encode(
                alt.X('Retorno%:Q', 
                      bin=alt.Bin(step=0.1, extent=domain), 
                      title=f'Retorno {window} días (%)'),
                alt.Y('count()', title='Frecuencia')
            )
            .properties(
                title=f'Histograma de retornos en bloques de {window} días'
            )
            .interactive()  # permite zoom rectangular
    )

    st.altair_chart(chart, use_container_width=True)

    # 5) Botón de descarga de los retornos calculados
    csv = df['Retorno%'].to_csv().encode('utf-8')
    st.download_button(
        f"Descargar retornos {window}d (%)", 
        data=csv, 
        file_name=f"retornos_{window}d.csv", 
        mime="text/csv"
    )



def module_sign_changes(intra: pd.DataFrame):
    st.subheader("9. Cambios de signo intradía & dispersión")
    if intra.empty:
        st.info("No hay datos intradía.")
        return
    daily_close=intra['Close'].resample('D').last().dropna()
    close_map={ts.date():p for ts,p in daily_close.items()}
    df2=intra.reset_index().rename(columns={intra.index.name or 'index':'Datetime'})
    df2['Day']=df2['Datetime'].dt.date
    rec=[]
    for day,grp in df2.groupby('Day'):
        pc=close_map.get(day-timedelta(days=1),np.nan)
        if np.isnan(pc): continue
        sig=(grp['Close']>pc).astype(int)
        ch=(sig.diff().abs()==1)
        idxs=np.where(ch)[0]
        cnt=len(idxs)
        if cnt==0: continue
        if cnt>1:
            diffs=np.diff(idxs);mi=int(diffs.min());ma=int(idxs[-1]-idxs[0])
        else: mi=ma=0
        rec.append({'Day':day,'SignChanges':cnt,'MinInterval':mi,'MaxInterval':ma})
    if not rec:
        st.write("Sin cambios de signo detectados.")
        return
    df_rec=pd.DataFrame(rec).set_index('Day')
    st.dataframe(df_rec)
    st.write(f"Prom cambios: {df_rec['SignChanges'].mean():.2f}")
    st.write(f"Min int avg: {df_rec['MinInterval'].mean():.2f} velas")
    st.write(f"Max int avg: {df_rec['MaxInterval'].mean():.2f} velas")
    hist=(alt.Chart(df_rec.reset_index())
          .mark_bar()
          .encode(
              x=alt.X('SignChanges:Q',bin=alt.Bin(step=1),title='Cambios'),
              y=alt.Y('count():Q',title='Frecuencia'))
          .properties(title='Histograma cambios de signo')
          .interactive())
    st.altair_chart(hist,use_container_width=True)

# -----------------------------------
# MÓDULO 10: Testeo de rangos intradía
# -----------------------------------
def module_intraday_range_test(intra: pd.DataFrame, daily: pd.DataFrame):
    """
    Testea rangos intradía porcentuales respecto al cierre del día anterior
    sobre todo el histórico y muestra estadísticas de escapes y tiempos.
    """
    st.subheader("10. Testeo de Rangos Intradía vs. Cierre Anterior")
    # Parámetros de usuario
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        offset_min = st.number_input("Offset mínimo (%)", 0.0, 5.0, 0.2, 0.1)
    with c2:
        offset_max = st.number_input("Offset máximo (%)", offset_min, 5.0, 1.0, 0.1)
    with c3:
        step = st.number_input("Paso (%)", 0.01, 1.0, 0.1, 0.01)
    with c4:
        days_hist = st.number_input("Días históricos", 1, 365, 60, 1)

    # Preparar DataFrame con retornos vs cierre previo
    df = intra.copy()
    df['Date'] = df.index.date
    # Mapear cierre previo por fecha (date)
    prev_map = {ts.date(): price for ts, price in daily['Close'].shift(1).items()}
    df['Prev_Close'] = df['Date'].map(prev_map)
    df = df.dropna(subset=['Prev_Close'])
    if df.empty:
        st.write("No hay datos intradía válidos para el período seleccionado.")
        return
    df['Ret%'] = (df['Close'] - df['Prev_Close']) / df['Prev_Close'] * 100

    # Filtrar últimos N días
    unique_days = sorted(df['Date'].unique())
    if not unique_days:
        st.write("No hay días intradía válidos para el período seleccionado.")
        return
    recent_days = unique_days[-days_hist:]
    df = df[df['Date'].isin(recent_days)]
    total_days = len(recent_days)

    results = []
    offs = np.arange(offset_min, offset_max + 1e-9, step)
    for X in offs:
        lower, upper = -X, X
        clean_escapes = 0
        false_escapes = 0
        time_in_range = []
        time_to_escape = []

        for d, grp in df.groupby('Date'):
            ret = grp['Ret%'].values
            # Velas dentro del rango
            in_range = (ret >= lower) & (ret <= upper)
            time_in_range.append(in_range.sum())
            # Detectar primer escape
            escap = np.where((ret < lower) | (ret > upper))[0]
            if escap.size == 0:
                continue
            idx0 = int(escap[0])
            time_to_escape.append(idx0)
            after = ret[idx0+1:] if idx0 + 1 < len(ret) else np.array([])
            if np.any(np.isclose(after, 0.0, atol=1e-6)):
                false_escapes += 1
            else:
                clean_escapes += 1

        escapes = clean_escapes + false_escapes
        stats = {'Rango': f"±{X:.2f}%"}
        # Evitar división por cero
        stats['% Días que rompen'] = (escapes / total_days * 100) if total_days else np.nan
        stats['% Escapes Limpios'] = (clean_escapes / escapes * 100) if escapes else np.nan
        stats['% Falsos Escapes'] = (false_escapes / escapes * 100) if escapes else np.nan
        stats['Tiempo Medio en Rango'] = np.mean(time_in_range) if time_in_range else np.nan
        stats['Tiempo Medio a Escape'] = np.mean(time_to_escape) if time_to_escape else np.nan

        results.append(stats)

    df_res = pd.DataFrame(results).set_index('Rango')
    st.table(df_res)

    # Gráfico de escapes limpios
    chart1 = (
        alt.Chart(df_res.reset_index())
           .mark_bar()
           .encode(
               x='Rango:N',
               y='% Escapes Limpios:Q',
               tooltip=['% Días que rompen', '% Falsos Escapes']
           )
           .properties(title='% Escapes Limpios por Rango')
           .interactive()
    )
    st.altair_chart(chart1, use_container_width=True)

    # Gráfico de tiempo hasta escape
    chart2 = (
        alt.Chart(df_res.reset_index())
           .mark_line(point=True)
           .encode(
               x='Rango:N',
               y='Tiempo Medio a Escape:Q',
               tooltip=['Tiempo Medio en Rango']
           )
           .properties(title='Tiempo Medio hasta Escape por Rango')
           .interactive()
    )
    st.altair_chart(chart2, use_container_width=True)

    # Exportar resultados
    csv = df_res.to_csv().encode('utf-8')
    st.download_button("Exportar resultados rangos", csv, "rangos_intradia.csv", "text/csv")


# -----------------------------------
# MAIN APP
# -----------------------------------
st.title("Análisis Cuantitativo")
# Ticker input
if 'ticker_input' not in st.session_state: st.session_state.ticker_input='SPY'
ticker_input=st.sidebar.text_input("Ticker (Yahoo)",value=st.session_state.ticker_input,key='ticker_input').upper()
# CSV intradía
use_csv_intra=st.sidebar.checkbox("Usar CSV intradía (15m)")
uploaded_intra=None
if use_csv_intra:
    intra_files=[f for f in os.listdir('historicos') if re.match(r'.+_15m\.csv$',f, re.IGNORECASE)]
    if intra_files:
        sel_i=st.sidebar.selectbox("Selecciona CSV intradía", intra_files, key='sel_intra')
        uploaded_intra=os.path.join('historicos',sel_i)
    else:
        st.sidebar.warning("No hay CSV intradía en 'historicos'.")
# CSV diario
use_csv_daily=st.sidebar.checkbox("Usar CSV diario (1d)")
uploaded_daily=None
if use_csv_daily:
    daily_files=[f for f in os.listdir('historicos') if re.match(r'.+_(1d|daily)\.csv$',f,re.IGNORECASE)]
    if daily_files:
        sel_d=st.sidebar.selectbox("Selecciona CSV diario", daily_files, key='sel_daily')
        uploaded_daily=os.path.join('historicos',sel_d)
    else:
        st.sidebar.warning("No hay CSV diario en 'historicos'.")
# Yahoo intervalo intradía si no CSV
yahoo_interval=st.sidebar.selectbox("Intervalo intradía Yahoo",['1h','30m','15m','5m','1m'],index=2)
# Módulos
d_modules=st.sidebar.multiselect("Módulos a ejecutar",ALL_MODULES,default=saved_modules,key='modules')
# Botón analizar
run_btn=st.sidebar.button("Analizar",key='run_btn')
if 'run' not in st.session_state: st.session_state.run=False
if run_btn:
    st.session_state.run=True
    try: json.dump(st.session_state['modules'],open(CONFIG_FILE,'w'))
    except: pass
if not st.session_state.run:
    st.info("Configura y pulsa 'Analizar'.")
    st.stop()
# Validación símbolos
symbol_intra=ticker_input
if use_csv_intra:
    sym_i,interval_i=parse_csv_filename(uploaded_intra)
    if interval_i.lower()!='15m': st.error(f"CSV intradía inválido: {interval_i}"); st.stop()
    symbol_intra=sym_i
symbol_daily=ticker_input
if use_csv_daily:
    sym_d,interval_d=parse_csv_filename(uploaded_daily)
    if interval_d.lower() not in ('1d','daily'): st.error(f"CSV diario inválido: {interval_d}"); st.stop()
    symbol_daily=sym_d
if use_csv_intra and not use_csv_daily and ticker_input!=symbol_intra:
    st.error(f"Ticker ≠ CSV intradía: {ticker_input} vs {symbol_intra}"); st.stop()
if use_csv_daily and not use_csv_intra and ticker_input!=symbol_daily:
    st.error(f"Ticker ≠ CSV diario: {ticker_input} vs {symbol_daily}"); st.stop()
if use_csv_intra and use_csv_daily and symbol_intra!=symbol_daily:
    st.error("Símbolos intradía y diario difieren."); st.stop()
asset_symbol=symbol_intra if use_csv_intra else symbol_daily
st.subheader(f"Activo: {asset_symbol}")
# Fechas
start=st.sidebar.date_input("Fecha inicio",datetime(2010,1,1))
end=st.sidebar.date_input("Fecha fin",datetime.now().date())
# Carga diarios
if use_csv_daily:
    daily_df=load_csv_df(uploaded_daily,dayfirst=True)
else:
    daily_df=fetch_data(asset_symbol,'1d',datetime.combine(start,datetime.min.time()),datetime.combine(end,datetime.min.time()))
# Carga intradía
if use_csv_intra:
    intra_df=load_csv_df(uploaded_intra)
else:
    intra_df=fetch_data(asset_symbol,yahoo_interval,datetime.combine(start,datetime.min.time()),datetime.combine(end,datetime.min.time()))
# Mostrar brutos opcional
if st.sidebar.checkbox("Mostrar datos brutos intradía"):
    st.subheader("Datos intradía brutos")
    st.write(intra_df.head())
# Ejecutar módulos
for m in d_modules:
    if m.startswith('1'): module_daily_anatomy(daily_df)
    if m.startswith('2'): module_intraday_reversion(intra_df,daily_df)
    if m.startswith('3'): module_chrono(intra_df)
    if m.startswith('4'): module_close_prob(daily_df)
    if m.startswith('5'): module_distribution(daily_df)
    if m.startswith('6'): module_extremes(daily_df)
    if m.startswith('7'): module_insights(daily_df)
    if m.startswith('8'): module_multi_day_histogram(daily_df)
    if m.startswith('9'): module_sign_changes(intra_df)
    if m.startswith('10'): module_intraday_range_test(intra_df,daily_df)    

# Exportar último dataset

df_export = None
if any(str(m).startswith(tuple(['2','3','5','8','9','10'])) for m in d_modules):
    df_export = intra_df
elif any(str(m).startswith(tuple(['1','4','6','7'])) for m in d_modules):
    df_export = daily_df

if df_export is not None:
    st.download_button(
        "Exportar CSV",
        df_export.to_csv().encode('utf-8'),
        f"{asset_symbol}_data.csv",
        "text/csv"
    )