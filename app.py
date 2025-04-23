import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import altair as alt

# -----------------------------------
# UTILITIES
# -----------------------------------
def fetch_spy_data(interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Descarga datos de SPY.
    Para velas diarias usa start/end.
    Para intradía intenta start/end; si no hay datos, recurre a periodo fallback.
    """
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
            defaults = {'1m':7, '5m':59, '15m':59, '30m':59, '1h':730}
            fallback = defaults.get(interval, 59)
            st.warning(f"Intervalo '{interval}' no tiene datos para el rango solicitado. Mostrando últimos {fallback} días.")
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
# MODULE 1: DAILY CANDLE ANATOMY
# -----------------------------------
def module_daily_anatomy(df: pd.DataFrame):
    st.subheader("1. Anatomía de la vela diaria")
    df2 = df.copy()
    df2['Range'] = df2['High'] - df2['Low']
    df2['Body'] = (df2['Close'] - df2['Open']).abs()
    df2['Upper_Wick'] = df2['High'] - df2[['Open','Close']].max(axis=1)
    df2['Lower_Wick'] = df2[['Open','Close']].min(axis=1) - df2['Low']

    stats = pd.DataFrame({
        'Mean': [df2[col].mean() for col in ['Range','Body','Upper_Wick','Lower_Wick']],
        'Median': [df2[col].median() for col in ['Range','Body','Upper_Wick','Lower_Wick']],
        'Std': [df2[col].std() for col in ['Range','Body','Upper_Wick','Lower_Wick']]
    }, index=['Range','Body','Upper_Wick','Lower_Wick'])
    st.table(stats)

    st.write(f"% Alcista: {100*(df2['Close']>df2['Open']).mean():.2f}%")
    st.write(f"% Bajista: {100*(df2['Close']<df2['Open']).mean():.2f}%")
    st.write(f"% Alcistas con lower wick > body: {100*((df2['Close']>df2['Open']) & (df2['Lower_Wick']>df2['Body'])).mean():.2f}%")
    st.write(f"% Bajistas con upper wick > body: {100*((df2['Close']<df2['Open']) & (df2['Upper_Wick']>df2['Body'])).mean():.2f}%")

    df2['Prev_Close'] = df2['Close'].shift(1)
    df2['Gap'] = df2['Open'] - df2['Prev_Close']
    gaps = df2.dropna(subset=['Gap'])
    st.write(f"Gaps totales: {len(gaps)}, Alcista: {len(gaps[gaps['Gap']>0])}, Bajista: {len(gaps[gaps['Gap']<0])}")
    gaps['Filled'] = ((gaps['Gap']<0)&(gaps['Low']<=gaps['Prev_Close'])) | ((gaps['Gap']>0)&(gaps['High']>=gaps['Prev_Close']))
    st.write(f"% Gaps llenados el mismo día: {100*gaps['Filled'].mean():.2f}%")

# -----------------------------------
# MODULE 2: INTRADAY REVERSIONS
# -----------------------------------
def module_intraday_reversion(intra: pd.DataFrame, daily: pd.DataFrame):
    st.subheader("2. Estadísticas de reversión intradía")
    if intra.empty:
        st.info("No hay datos intradía para calcular reversión.")
        return
    df2 = intra.reset_index().rename(columns={intra.index.name or 'index': 'Datetime'})
    df2['Day'] = df2['Datetime'].dt.date
    results = []
    for day, grp in df2.groupby('Day'):
        prev_close = daily['Close'].shift(1).get(pd.Timestamp(day), np.nan)
        if pd.isna(prev_close): continue
        open_ = grp['Open'].iloc[0]; close_ = grp['Close'].iloc[-1]
        d = {'Day': day, 'Gap%': 100*(open_/prev_close-1), 'Reverted': (open_<prev_close and close_>open_) or (open_>prev_close and close_<open_)}
        if close_>open_:
            d['Min_Intra%'] = 100*(grp['Low'].min()/open_-1)
            d['Time_of_Min'] = grp.loc[grp['Low'].idxmin(), 'Datetime'].time()
        else:
            d['Max_Intra%'] = 100*(grp['High'].max()/open_-1)
            d['Time_of_Max'] = grp.loc[grp['High'].idxmax(), 'Datetime'].time()
        results.append(d)
    res = pd.DataFrame(results)
    if res.empty:
        st.write("No hay días con reversión detectada.")
        return
    st.write(f"Prob. reversión intradía: {100*res['Reverted'].mean():.2f}%")
    if 'Time_of_Min' in res:
        chart = alt.Chart(res).mark_bar().encode(x='Time_of_Min:T', y='Reverted:Q')
        st.altair_chart(chart, use_container_width=True)

# -----------------------------------
# MODULE 3: INTRADAY CHRONOANALYSIS
# -----------------------------------
def module_chrono(intra: pd.DataFrame):
    st.subheader("3. Cronoanálisis intradía por intervalos")
    if intra.empty:
        st.info("No hay datos intradía para cronoanálisis.")
        return
    df3 = intra.reset_index().rename(columns={intra.index.name or 'index':'Datetime'})
    df3['Block'] = df3['Datetime'].dt.strftime('%H:%M')
    df3['Return%'] = 100*(df3['Close']/df3['Open']-1)
    summary = df3.groupby('Block')['Return%'].agg(['mean','std','count'])
    summary['Up%'] = df3.groupby('Block').apply(lambda x: (x['Close']>x['Open']).mean()*100)
    st.line_chart(summary[['mean','Up%']])
    seq = (df3['Close']>df3['Open']).astype(int)
    trans = pd.crosstab(seq.shift(), seq, normalize='index')*100
    st.write("Matriz de transición de bloques (%):")
    st.table(trans)

# -----------------------------------
# MODULE 4: CLOSURE PROBABILITIES
# -----------------------------------
def module_close_prob(daily: pd.DataFrame):
    st.subheader("4. Probabilidades de cierre relativo")
    df = daily.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    offset = st.sidebar.slider("Offset % desde Prev Close", 0.0, 5.0, 0.5, 0.1)
    df['Offset_Price'] = df['Prev_Close']*(1+offset/100)
    st.write(f"Prob. cierre>apertura: {100*(df['Close']>df['Open']).mean():.2f}%")
    st.write(f"Prob. cierre>prev close: {100*(df['Close']>df['Prev_Close']).mean():.2f}%")
    st.write(f"Prob. cierre>prev high: {100*(df['Close']>df['Prev_High']).mean():.2f}%")
    st.write(f"Prob. cierre<prev low: {100*(df['Close']<df['Prev_Low']).mean():.2f}%")
    st.write(f"Prob. cerrar>offset {offset}%: {100*(df['Close']>df['Offset_Price']).mean():.2f}%")

# -----------------------------------
# MODULE 5: MOVEMENT DISTRIBUTION
# -----------------------------------
def module_distribution(daily: pd.DataFrame):
    st.subheader("5. Distribución de amplitud de movimiento")
    df = daily.copy()
    df['Range%'] = 100*(df['High']-df['Low'])/df['Open']
    chart = alt.Chart(df).mark_bar().encode(
        alt.X('Range%', bin=alt.Bin(maxbins=50)),
        y='count()'
    )
    st.altair_chart(chart, use_container_width=True)

# -----------------------------------
# MODULE 6: EXTREMES
# -----------------------------------
def module_extremes(daily: pd.DataFrame):
    st.subheader("6. Comportamientos extremos")
    df = daily.copy()
    df['Gap'] = 100*(df['Open']/df['Close'].shift(1)-1)
    st.table(df['Gap'].abs().nlargest(5))
    df['Candle%'] = 100*(df['Close']/df['Open']-1)
    st.table(df['Candle%'].abs().nlargest(5))

# -----------------------------------
# MODULE 7: INSIGHTS & SIGNALS
# -----------------------------------
def module_insights(daily: pd.DataFrame):
    st.subheader("7. Insights y señales")
    df = daily.copy()
    df['Body'] = abs(df['Close']-df['Open'])
    df['Lower_Wick'] = df[['Open','Close']].min(axis=1)-df['Low']
    pattern = df[(df['Lower_Wick']>2*df['Body']) & (df['Close']>df['Open'])]
    st.write(f"Velas con mecha inferior >2x cuerpo: {len(pattern)} días")
    if not pattern.empty:
        st.write(f"Prob. cierre alcista tras patrón: {100*(pattern['Close']>pattern['Open']).mean():.2f}%")

# -----------------------------------
# MODULE 8: DAILY RETURNS HISTOGRAM
# -----------------------------------
def module_daily_returns_histogram(df: pd.DataFrame):
    st.subheader("8. Histograma de retornos diarios")
    returns = df['Close'].pct_change().dropna()*100
    ret_df = returns.to_frame('Return%').reset_index()
    chart = alt.Chart(ret_df).mark_bar().encode(
        alt.X('Return%', bin=alt.Bin(maxbins=50), title='Retorno diario (%)'),
        y='count()'
    )
    st.altair_chart(chart, use_container_width=True)

# -----------------------------------
# MODULE 9: SIGN CHANGES
# -----------------------------------
def module_sign_changes(intra: pd.DataFrame, daily: pd.DataFrame):
    st.subheader("9. Cambios de signo intradía")
    if intra.empty:
        st.info("No hay datos intradía para cambios de signo.")
        return
    # Mapa de cierre previo del día anterior
    prev = daily['Close'].shift(1)
    prev_map = {dt.date(): price for dt, price in zip(prev.index, prev.values)}
    df2 = intra.copy()
    df2['Date'] = df2.index.date
    results = []
    for day, grp in df2.groupby('Date'):
        prev_close = prev_map.get(day, np.nan)
        if pd.isna(prev_close):
            continue
        # Signo respecto al cierre previo
        signs = (grp['Close'] > prev_close).astype(int)
        # Cambios donde diff sea 1
        changes = int(signs.diff().abs().sum())
        results.append({'Day': day, 'Sign_Changes': changes})
    res = pd.DataFrame(results)
    if res.empty:
        st.write("No se detectaron cambios de signo.")
    else:
        st.write(res.set_index('Day'))
        st.write(f"Promedio diario de cambios de signo: {res['Sign_Changes'].mean():.2f}")

# -----------------------------------
# MAIN APP
# -----------------------------------
st.title("Análisis Estadístico SPY Avanzado")
start = st.sidebar.date_input("Fecha inicio", datetime(2010,1,1))
end = st.sidebar.date_input("Fecha fin", datetime.now().date())
interval = st.sidebar.selectbox("Intervalo", ['1d','1h','30m','15m','5m','1m'])
show_raw = st.sidebar.checkbox("Mostrar datos brutos")
# Selector para signo
sign_interval = st.sidebar.selectbox("Intervalo Cambios de Signo", ['5m','15m','30m','1h'])

# Carga de datos
daily_df = fetch_spy_data('1d', datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time()))
intra_df = fetch_spy_data(interval, datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time()))
sign_df  = fetch_spy_data(sign_interval, datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time()))

if show_raw:
    st.subheader("Datos brutos intradía")
    st.write(intra_df.head())

# Ejecución de módulos
module_daily_anatomy(daily_df)
module_intraday_reversion(intra_df, daily_df)
module_chrono(intra_df)
module_close_prob(daily_df)
module_distribution(daily_df)
module_extremes(daily_df)
module_insights(daily_df)
module_daily_returns_histogram(daily_df)
module_sign_changes(sign_df, daily_df)

# Exportar CSV
df_export = intra_df if interval!='1d' else daily_df
st.download_button("Exportar CSV", df_export.to_csv().encode('utf-8'), "spy_data.csv", "text/csv")
