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
    "8. Histograma de retornos diarios",
    "9. Cambios de signo intradía"
]

# Load saved module selection
if os.path.exists(CONFIG_FILE):
    try:
        selected_modules = json.load(open(CONFIG_FILE))
    except Exception:
        selected_modules = ALL_MODULES.copy()
else:
    selected_modules = ALL_MODULES.copy()

# -----------------------------------
# UTILITIES
# -----------------------------------
def fetch_spy_data(interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    ticker = yf.Ticker('SPY')
    if interval == '1d':
        df = ticker.history(
            interval=interval,
            start=start_date.strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            prepost=False,
            auto_adjust=False
        )
    else:
        df = ticker.history(
            interval=interval,
            start=start_date.strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            prepost=False,
            auto_adjust=False
        )
        if df.empty:
            defaults = {'1m':7,'5m':59,'15m':59,'30m':59,'1h':730}
            fallback = defaults.get(interval,59)
            st.warning(f"Intervalo '{interval}' no tiene datos para el rango; mostrando últimos {fallback} días.")
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
# MODULE DEFINITIONS
# -----------------------------------
def module_daily_anatomy(df: pd.DataFrame):
    st.subheader("1. Anatomía diaria")
    df2 = df.copy()
    df2['Range'] = df2['High'] - df2['Low']
    df2['Body'] = (df2['Close'] - df2['Open']).abs()
    df2['Upper_Wick'] = df2['High'] - df2[['Open','Close']].max(axis=1)
    df2['Lower_Wick'] = df2[['Open','Close']].min(axis=1) - df2['Low']
    stats = pd.DataFrame({
        'Mean': [df2[c].mean() for c in ['Range','Body','Upper_Wick','Lower_Wick']],
        'Median': [df2[c].median() for c in ['Range','Body','Upper_Wick','Lower_Wick']],
        'Std': [df2[c].std() for c in ['Range','Body','Upper_Wick','Lower_Wick']]
    }, index=['Range','Body','Upper_Wick','Lower_Wick'])
    st.table(stats)
    st.write(f"% Alcista: {100*(df2['Close']>df2['Open']).mean():.2f}%")
    st.write(f"% Bajista: {100*(df2['Close']<df2['Open']).mean():.2f}%")
    st.write(f"% Alcistas w/ lower wick>cuerpo: {100*((df2['Close']>df2['Open'])&(df2['Lower_Wick']>df2['Body'])).mean():.2f}%")
    st.write(f"% Bajistas w/ upper wick>cuerpo: {100*((df2['Close']<df2['Open'])&(df2['Upper_Wick']>df2['Body'])).mean():.2f}%")
    df2['Prev_Close'] = df2['Close'].shift(1)
    df2['Gap'] = df2['Open'] - df2['Prev_Close']
    gaps = df2.dropna(subset=['Gap'])
    st.write(f"Gaps totales: {len(gaps)}, Alcista: {len(gaps[gaps['Gap']>0])}, Bajista: {len(gaps[gaps['Gap']<0])}")
    gaps['Filled'] = ((gaps['Gap']<0)&(gaps['Low']<=gaps['Prev_Close'])) | ((gaps['Gap']>0)&(gaps['High']>=gaps['Prev_Close']))
    st.write(f"% Gaps llenados: {100*gaps['Filled'].mean():.2f}%")


def module_intraday_reversion(intra: pd.DataFrame, daily: pd.DataFrame):
    st.subheader("2. Reversiones intradía")
    if intra.empty:
        st.info("No hay datos intradía.")
        return
    df2 = intra.reset_index().rename(columns={intra.index.name or 'index':'Datetime'})
    df2['Day']=df2['Datetime'].dt.date
    res=[]
    for day,grp in df2.groupby('Day'):
        prev = daily['Close'].shift(1).get(pd.Timestamp(day),np.nan)
        if pd.isna(prev): continue
        open_,close_=grp['Open'].iloc[0],grp['Close'].iloc[-1]
        d={'Day':day,'Gap%':100*(open_/prev-1),'Reverted':(open_<prev and close_>open_) or (open_>prev and close_<open_)}
        if close_>open_:
            d['Min%']=100*(grp['Low'].min()/open_-1)
            d['TimeMin']=grp.loc[grp['Low'].idxmin(),'Datetime'].time()
        else:
            d['Max%']=100*(grp['High'].max()/open_-1)
            d['TimeMax']=grp.loc[grp['High'].idxmax(),'Datetime'].time()
        res.append(d)
    rez=pd.DataFrame(res)
    if rez.empty:
        st.write("Sin reversiones.")
        return
    st.write(f"Prob. reversión: {100*rez['Reverted'].mean():.2f}%")
    if 'TimeMin' in rez:
        st.altair_chart(alt.Chart(rez).mark_bar().encode(x='TimeMin:T',y='Reverted:Q'),use_container_width=True)


def module_chrono(intra: pd.DataFrame):
    st.subheader("3. Cronoanálisis intradía")
    if intra.empty:
        st.info("No hay datos intradía.")
        return
    df3=intra.reset_index().rename(columns={intra.index.name or 'index':'Datetime'})
    df3['Block']=df3['Datetime'].dt.strftime('%H:%M')
    df3['Return%']=100*(df3['Close']/df3['Open']-1)
    summary=df3.groupby('Block')['Return%'].agg(['mean','std','count'])
    summary['Up%']=df3.groupby('Block').apply(lambda x:(x['Close']>x['Open']).mean()*100)
    st.line_chart(summary[['mean','Up%']])
    seq=(df3['Close']>df3['Open']).astype(int)
    trans=pd.crosstab(seq.shift(),seq,normalize='index')*100
    st.write("Transición (%):")
    st.table(trans)


def module_close_prob(daily: pd.DataFrame):
    st.subheader("4. Probabilidades de cierre")
    df=daily.copy()
    df['PrevClose']=df['Close'].shift(1)
    df['PrevHigh']=df['High'].shift(1)
    df['PrevLow']=df['Low'].shift(1)
    off=st.sidebar.slider("Offset%",0.0,5.0,0.5,0.1)
    df['OffPrice']=df['PrevClose']*(1+off/100)
    st.write(f">Open: {100*(df['Close']>df['Open']).mean():.2f}%")
    st.write(f">PrevClose: {100*(df['Close']>df['PrevClose']).mean():.2f}%")
    st.write(f">PrevHigh: {100*(df['Close']>df['PrevHigh']).mean():.2f}%")
    st.write(f"<PrevLow: {100*(df['Close']<df['PrevLow']).mean():.2f}%")
    st.write(f">Offset {off}%: {100*(df['Close']>df['OffPrice']).mean():.2f}%")


def module_distribution(daily: pd.DataFrame):
    st.subheader("5. Distribución de movimiento")
    df=daily.copy()
    df['Range%']=100*(df['High']-df['Low'])/df['Open']
    st.altair_chart(alt.Chart(df).mark_bar().encode(alt.X('Range%',bin=alt.Bin(maxbins=50)),y='count()'),use_container_width=True)


def module_extremes(daily: pd.DataFrame):
    st.subheader("6. Comportamientos extremos")
    df=daily.copy()
    df['Gap%']=100*(df['Open']/df['Close'].shift(1)-1)
    st.table(df['Gap%'].abs().nlargest(5))
    df['Candle%']=100*(df['Close']/df['Open']-1)
    st.table(df['Candle%'].abs().nlargest(5))


def module_insights(daily: pd.DataFrame):
    st.subheader("7. Insights & señales")
    df=daily.copy()
    df['Body']=abs(df['Close']-df['Open'])
    df['LowerW']=df[['Open','Close']].min(axis=1)-df['Low']
    pat=df[(df['LowerW']>2*df['Body'])&(df['Close']>df['Open'])]
    st.write(f"Velas mecha inf>2x cuerpo: {len(pat)}")
    if not pat.empty:
        st.write(f"Prob. alcista post patrón: {100*(pat['Close']>pat['Open']).mean():.2f}%")


def module_returns_histogram(daily: pd.DataFrame):
    st.subheader("8. Histograma retornos diarios")
    ret=100*(daily['Close']/daily['Open']-1)
    df=ret.to_frame('Ret%').reset_index()
    st.altair_chart(alt.Chart(df).mark_bar().encode(alt.X('Ret%:Q',bin=alt.Bin(maxbins=50)),y='count()'),use_container_width=True)


def module_sign_changes(sign_df: pd.DataFrame, daily: pd.DataFrame):
    st.subheader("9. Cambios de signo intradía")
    if sign_df.empty:
        st.info("No hay datos intradía.")
        return
    prev_map={dt.date():price for dt,price in zip(daily.index,daily['Close'].shift(1))}
    df=sign_df.copy()
    df['Date']=df.index.date
    res=[]
    for day,grp in df.groupby('Date'):
        prev=prev_map.get(day,np.nan)
        if pd.isna(prev): continue
        signs=(grp['Close']>prev).astype(int)
        ch=int(signs.diff().abs().sum())
        res.append({'Day':day,'SignChanges':ch})
    r=pd.DataFrame(res)
    if r.empty:
        st.write("No cambios.")
    else:
        st.table(r.set_index('Day'))
        st.write(f"Promedio: {r['SignChanges'].mean():.2f}")
        hist = (
        alt.Chart(r)
        .transform_aggregate(count='count()', groupby=['SignChanges'])
        .mark_bar()
        .encode(
            x=alt.X('SignChanges:Q', title='Número de cambios de signo'),
            y=alt.Y('count:Q', title='Frecuencia')
        )
        .properties(title='Histograma de cambios de signo intradía')
    )
    st.altair_chart(hist, use_container_width=True)

# -----------------------------------
# MAIN APP
# -----------------------------------
st.title("Análisis Estadístico SPY Avanzado")
# Sidebar inputs
st.sidebar.header("Configuración")
start=st.sidebar.date_input("Fecha inicio",datetime(2010,1,1))
end=st.sidebar.date_input("Fecha fin",datetime.now().date())
interval=st.sidebar.selectbox("Intervalo principal",['1d','1h','30m','15m','5m','1m'])
sign_interval=st.sidebar.selectbox("Intervalo signo",['5m','15m','30m','1h'])

# Module selection form
with st.sidebar.form(key='config'):
    sel=st.multiselect("Selecciona módulos",ALL_MODULES,default=selected_modules)
    btn=st.form_submit_button("Analizar")
    if btn:
        st.session_state.mods=sel
        json.dump(sel,open(CONFIG_FILE,'w'))
    elif 'mods' not in st.session_state:
        st.session_state.mods=selected_modules

# Load data
daily_df=fetch_spy_data('1d',datetime.combine(start,datetime.min.time()),datetime.combine(end,datetime.min.time()))
intra_df=fetch_spy_data(interval,datetime.combine(start,datetime.min.time()),datetime.combine(end,datetime.min.time()))
sign_df=fetch_spy_data(sign_interval,datetime.combine(start,datetime.min.time()),datetime.combine(end,datetime.min.time()))

# Raw data option
if st.sidebar.checkbox("Mostrar datos brutos"):
    st.subheader("Datos intradía brutos")
    st.write(intra_df.head())

# Execute selected modules
for m in st.session_state.mods:
    if m.startswith('1'): module_daily_anatomy(daily_df)
    if m.startswith('2'): module_intraday_reversion(intra_df,daily_df)
    if m.startswith('3'): module_chrono(intra_df)
    if m.startswith('4'): module_close_prob(daily_df)
    if m.startswith('5'): module_distribution(daily_df)
    if m.startswith('6'): module_extremes(daily_df)
    if m.startswith('7'): module_insights(daily_df)
    if m.startswith('8'): module_returns_histogram(daily_df)
    if m.startswith('9'): module_sign_changes(sign_df,daily_df)

# CSV export
df_export=intra_df if interval!='1d' else daily_df
st.download_button("Exportar CSV",df_export.to_csv().encode('utf-8'),"spy_data.csv","text/csv")
