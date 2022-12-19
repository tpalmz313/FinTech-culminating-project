#pip install yfinance
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import webbrowser
from PIL import Image


########################
# Technical Indicators #
########################

class IndicatorMixin:

    _fillna = False

    def _check_fillna(self, series: pd.Series, value: int = 0) -> pd.Series:

        if self._fillna:
            series_output = series.copy(deep=False)
            series_output = series_output.replace([np.inf, -np.inf], np.nan)
            if isinstance(value, int) and value == -1:
                series = series_output.fillna(method="ffill").fillna(value=-1)
            else:
                series = series_output.fillna(method="ffill").fillna(value)
        return series

    @staticmethod
    def _true_range(
        high: pd.Series, low: pd.Series, prev_close: pd.Series
    ) -> pd.Series:
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.DataFrame(data={"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        return true_range


def dropna(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with "Nans" values"""
    df = df.copy()
    number_cols = df.select_dtypes("number").columns.to_list()
    df[number_cols] = df[number_cols][df[number_cols] < math.exp(709)]  # big number
    df[number_cols] = df[number_cols][df[number_cols] != 0.0]
    df = df.dropna()
    return df


def _sma(series, periods: int, fillna: bool = False):
    min_periods = 0 if fillna else periods
    return series.rolling(window=periods, min_periods=min_periods).mean()


def _ema(series, periods, fillna=False):
    min_periods = 0 if fillna else periods
    return series.ewm(span=periods, min_periods=min_periods, adjust=False).mean()


def _get_min_max(series1: pd.Series, series2: pd.Series, function: str = "min"):
    """Find min or max value between two lists for each index"""
    series1 = np.array(series1)
    series2 = np.array(series2)
    if function == "min":
        output = np.amin([series1, series2], axis=0)
    elif function == "max":
        output = np.amax([series1, series2], axis=0)
    else:
        raise ValueError('"f" variable value should be "min" or "max"')

    return pd.Series(output)


class BollingerBands(IndicatorMixin):

    def __init__(
        self,
        close: pd.Series,
        window: int = 20,
        window_dev: int = 2,
        fillna: bool = False,
    ):
        self._close = close
        self._window = window
        self._window_dev = window_dev
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._window
        self._mavg = self._close.rolling(self._window, min_periods=min_periods).mean()
        self._mstd = self._close.rolling(self._window, min_periods=min_periods).std(
            ddof=0
        )
        self._hband = self._mavg + self._window_dev * self._mstd
        self._lband = self._mavg - self._window_dev * self._mstd

    def bollinger_mavg(self) -> pd.Series:
        mavg = self._check_fillna(self._mavg, value=-1)
        return pd.Series(mavg, name="mavg")

    def bollinger_hband(self) -> pd.Series:
        hband = self._check_fillna(self._hband, value=-1)
        return pd.Series(hband, name="hband")

    def bollinger_lband(self) -> pd.Series:
        lband = self._check_fillna(self._lband, value=-1)
        return pd.Series(lband, name="lband")

    def bollinger_wband(self) -> pd.Series:
        wband = ((self._hband - self._lband) / self._mavg) * 100
        wband = self._check_fillna(wband, value=0)
        return pd.Series(wband, name="bbiwband")

    def bollinger_pband(self) -> pd.Series:
        pband = (self._close - self._lband) / (self._hband - self._lband)
        pband = self._check_fillna(pband, value=0)
        return pd.Series(pband, name="bbipband")

    def bollinger_hband_indicator(self) -> pd.Series:
        hband = pd.Series(
            np.where(self._close > self._hband, 1.0, 0.0), index=self._close.index
        )
        hband = self._check_fillna(hband, value=0)
        return pd.Series(hband, index=self._close.index, name="bbihband")

    def bollinger_lband_indicator(self) -> pd.Series:
        lband = pd.Series(
            np.where(self._close < self._lband, 1.0, 0.0), index=self._close.index
        )
        lband = self._check_fillna(lband, value=0)
        return pd.Series(lband, name="bbilband")

class RSIIndicator(IndicatorMixin):

    def __init__(self, close: pd.Series, window: int = 14, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        diff = self._close.diff(1)
        up_direction = diff.where(diff > 0, 0.0)
        down_direction = -diff.where(diff < 0, 0.0)
        min_periods = 0 if self._fillna else self._window
        emaup = up_direction.ewm(
            alpha=1 / self._window, min_periods=min_periods, adjust=False
        ).mean()
        emadn = down_direction.ewm(
            alpha=1 / self._window, min_periods=min_periods, adjust=False
        ).mean()
        relative_strength = emaup / emadn
        self._rsi = pd.Series(
            np.where(emadn == 0, 100, 100 - (100 / (1 + relative_strength))),
            index=self._close.index,
        )

    def rsi(self) -> pd.Series:
        rsi_series = self._check_fillna(self._rsi, value=50)
        return pd.Series(rsi_series, name="rsi")
    
class MACD(IndicatorMixin):

    def __init__(
        self,
        close: pd.Series,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9,
        fillna: bool = False,
    ):
        self._close = close
        self._window_slow = window_slow
        self._window_fast = window_fast
        self._window_sign = window_sign
        self._fillna = fillna
        self._run()

    def _run(self):
        self._emafast = _ema(self._close, self._window_fast, self._fillna)
        self._emaslow = _ema(self._close, self._window_slow, self._fillna)
        self._macd = self._emafast - self._emaslow
        self._macd_signal = _ema(self._macd, self._window_sign, self._fillna)
        self._macd_diff = self._macd - self._macd_signal

    def macd(self) -> pd.Series:
        macd_series = self._check_fillna(self._macd, value=0)
        return pd.Series(
            macd_series, name=f"MACD_{self._window_fast}_{self._window_slow}"
        )

    def macd_signal(self) -> pd.Series:
        macd_signal_series = self._check_fillna(self._macd_signal, value=0)
        return pd.Series(
            macd_signal_series,
            name=f"MACD_sign_{self._window_fast}_{self._window_slow}",
        )

    def macd_diff(self) -> pd.Series:
        macd_diff_series = self._check_fillna(self._macd_diff, value=0)
        return pd.Series(
            macd_diff_series, name=f"MACD_diff_{self._window_fast}_{self._window_slow}"
        )

class ROCIndicator(IndicatorMixin):

    def __init__(self, close: pd.Series, window: int = 12, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        self._roc = (
            (self._close - self._close.shift(self._window))
            / self._close.shift(self._window)
        ) * 100

    def roc(self) -> pd.Series:
        # """Rate of Change (ROC)
        # Returns:
        #     pandas.Series: New feature generated.
        # """
        roc_series = self._check_fillna(self._roc)
        return pd.Series(roc_series, name="roc")

class TSIIndicator(IndicatorMixin):

    def __init__(
        self,
        close: pd.Series,
        window_slow: int = 25,
        window_fast: int = 13,
        fillna: bool = False,
    ):
        self._close = close
        self._window_slow = window_slow
        self._window_fast = window_fast
        self._fillna = fillna
        self._run()

    def _run(self):
        diff_close = self._close - self._close.shift(1)
        min_periods_r = 0 if self._fillna else self._window_slow
        min_periods_s = 0 if self._fillna else self._window_fast
        smoothed = (
            diff_close.ewm(
                span=self._window_slow, min_periods=min_periods_r, adjust=False
            )
            .mean()
            .ewm(span=self._window_fast, min_periods=min_periods_s, adjust=False)
            .mean()
        )
        smoothed_abs = (
            abs(diff_close)
            .ewm(span=self._window_slow, min_periods=min_periods_r, adjust=False)
            .mean()
            .ewm(span=self._window_fast, min_periods=min_periods_s, adjust=False)
            .mean()
        )
        self._tsi = smoothed / smoothed_abs
        self._tsi *= 100

    def tsi(self) -> pd.Series:
        tsi_series = self._check_fillna(self._tsi, value=0)
        return pd.Series(tsi_series, name="tsi")

    
##################
# Set up sidebar #
##################
st.sidebar.title('Crypto Dashboard :moneybag:')
url = 'https://finance.yahoo.com/crypto/?count=25&offset=0'

if st.sidebar.button('yahoo! finance'):
    webbrowser.open_new_tab(url)
    
# from PIL import Image 
image = Image.open('crypto_coins2.png')
st.sidebar.image(image)

option = st.sidebar.selectbox('Select a Cryptocurrency', ('BTC-USD','ETH-USD','USDT-USD','USDC-USD','BNB-USD','XRP-USD','BUSD-USD','DOGE-USD','ADA-USD','MATIC-USD','DOT-USD','DAI-USD','WTRX-USD','LTC-USD','SOL-USD','TRX-USD','SHIB-USD','HEX-USD','UNI7083-USD','STETH-USD','AVAX-USD','LEO-USD','LINK-USD','WBTC-USD','TON11419-USD'))

import datetime

today = datetime.date.today()
before = today - datetime.timedelta(days=730)
start_date = st.sidebar.date_input('Start date', before) 
end_date = st.sidebar.date_input('End date', today)
if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

st.sidebar.caption('Presented by Jeff, Thomas and Ray :hotsprings:')

##############
# Stock data #
##############
df = yf.download(option,start= start_date,end= end_date, progress=False)
st.title(option)
col1, col2 = st.columns(2)
tickerData = yf.Ticker(option)
with col1:
    tickerData.major_holders
with col2:
    tickerData.institutional_holders
st.caption('Provided by Yahoo! finance, results were generated a few mins ago. Pricing data is updated frequently. Currency in USD.')
progress_bar = st.progress(0)
st.subheader('_Technical Indicators_')
st.markdown('##### Bollinger Bands®')
indicator_bb = BollingerBands(df['Close'])

bb = df
bb['Bollinger_Band_High'] = indicator_bb.bollinger_hband()
bb['Bollinger_Band_Low'] = indicator_bb.bollinger_lband()
bb = bb[['Close','Bollinger_Band_High','Bollinger_Band_Low']]

macd = MACD(df['Close']).macd()

rsi = RSIIndicator(df['Close']).rsi()

tsi = TSIIndicator(df['Close']).tsi()

roc = ROCIndicator(df['Close']).roc()

###################
# Set up main app #
###################

st.line_chart(bb)

url = 'https://www.investopedia.com/articles/technical/102201.asp'

if st.button('Bollinger Bands® FAQs'):
    webbrowser.open_new_tab(url)

progress_bar = st.progress(0)
col1, col2 = st.columns(2)

with col1: 
    st.markdown('##### Moving Average Convergence Divergence (MACD)')
    st.area_chart(macd)

    url = 'https://www.investopedia.com/terms/m/macd.asp'

    if st.button('MACD FAQs'):
        webbrowser.open_new_tab(url)
    
with col2:
    st.markdown("##### Relative Strength Index (RSI)")
    st.line_chart(rsi)
    st.markdown(" ")
    url = 'https://www.investopedia.com/terms/r/rsi.asp'

    if st.button('Relative Strength Index (RSI) FAQs'):
        webbrowser.open_new_tab(url)

progress_bar = st.progress(0)
    
col1, col2 = st.columns(2)

with col1: 
    st.markdown("##### True Strength Index (TSI)")
    st.line_chart(tsi)
    
    url = 'https://www.investopedia.com/terms/t/tsi.asp'

    if st.button('True Strength Index (TSI) FAQs'):
        webbrowser.open_new_tab(url)

with col2:
    st.markdown("##### Rate of Change (ROC)")
    st.line_chart(roc)
    
    url = 'https://www.investopedia.com/terms/r/rateofchange.asp'

    if st.button('Rate of Change (ROC) FAQs'):
        webbrowser.open_new_tab(url)

progress_bar = st.progress(0)
        
st.markdown("##### 10 Day Snapshot :chart_with_upwards_trend:")
st.write(option)
st.dataframe(df.tail(10))
progress_bar = st.progress(0)
################
# Download csv #
################

import base64
from io import BytesIO

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    val = to_excel(df)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="download.xlsx">Download excel file</a>'

st.markdown(" ")
st.markdown("##### Create Crypto Report :pencil:")
st.markdown(get_table_download_link(df), unsafe_allow_html=True)

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='crypto.csv',
    mime='text/csv',
)

