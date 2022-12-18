
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf # https://pypi.org/project/yfinance/

##############################
# Technical Analysis Classes #
##############################

# https://github.com/bukosabino/ta/blob/master/ta/utils.py
class IndicatorMixin:
    """Util mixin indicator class"""

    _fillna = False

    def _check_fillna(self, series: pd.Series, value: int = 0) -> pd.Series:
        """Check if fillna flag is True.
        Args:
            series(pandas.Series): dataset 'Close' column.
            value(int): value to fill gaps; if -1 fill values using 'backfill' mode.
        Returns:
            pandas.Series: New feature generated.
        """
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


# https://github.com/bukosabino/ta/blob/master/ta/volatility.py
class BollingerBands(IndicatorMixin):
    # """Bollinger Bands
    # https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands
    # Args:
    #     close(pandas.Series): dataset 'Close' column.
    #     window(int): n period.
    #     window_dev(int): n factor standard deviation
    #     fillna(bool): if True, fill nan values.
    # """

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
        # """Bollinger Channel Middle Band
        # Returns:
        #     pandas.Series: New feature generated.
        # """
        mavg = self._check_fillna(self._mavg, value=-1)
        return pd.Series(mavg, name="mavg")

    def bollinger_hband(self) -> pd.Series:
        # """Bollinger Channel High Band
        # Returns:
        #     pandas.Series: New feature generated.
        # """
        hband = self._check_fillna(self._hband, value=-1)
        return pd.Series(hband, name="hband")

    def bollinger_lband(self) -> pd.Series:
        # """Bollinger Channel Low Band
        # Returns:
        #     pandas.Series: New feature generated.
        # """
        lband = self._check_fillna(self._lband, value=-1)
        return pd.Series(lband, name="lband")

    def bollinger_wband(self) -> pd.Series:
        # """Bollinger Channel Band Width
        # From: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_width
        # Returns:
        #     pandas.Series: New feature generated.
        # """
        wband = ((self._hband - self._lband) / self._mavg) * 100
        wband = self._check_fillna(wband, value=0)
        return pd.Series(wband, name="bbiwband")

    def bollinger_pband(self) -> pd.Series:
        # """Bollinger Channel Percentage Band
        # From: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_perce
        # Returns:
        #     pandas.Series: New feature generated.
        # """
        pband = (self._close - self._lband) / (self._hband - self._lband)
        pband = self._check_fillna(pband, value=0)
        return pd.Series(pband, name="bbipband")

    def bollinger_hband_indicator(self) -> pd.Series:
        # """Bollinger Channel Indicator Crossing High Band (binary).
        # It returns 1, if close is higher than bollinger_hband. Else, it returns 0.
        # Returns:
        #     pandas.Series: New feature generated.
        # """
        hband = pd.Series(
            np.where(self._close > self._hband, 1.0, 0.0), index=self._close.index
        )
        hband = self._check_fillna(hband, value=0)
        return pd.Series(hband, index=self._close.index, name="bbihband")

    def bollinger_lband_indicator(self) -> pd.Series:
        # """Bollinger Channel Indicator Crossing Low Band (binary).
        # It returns 1, if close is lower than bollinger_lband. Else, it returns 0.
        # Returns:
        #     pandas.Series: New feature generated.
        # """
        lband = pd.Series(
            np.where(self._close < self._lband, 1.0, 0.0), index=self._close.index
        )
        lband = self._check_fillna(lband, value=0)
        return pd.Series(lband, name="bbilband")

# https://github.com/bukosabino/ta/blob/master/ta/momentum.py
class RSIIndicator(IndicatorMixin):
    # """Relative Strength Index (RSI)
    # Compares the magnitude of recent gains and losses over a specified time
    # period to measure speed and change of price movements of a security. It is
    # primarily used to attempt to identify overbought or oversold conditions in
    # the trading of an asset.
    # https://www.investopedia.com/terms/r/rsi.asp
    # Args:
    #     close(pandas.Series): dataset 'Close' column.
    #     window(int): n period.
    #     fillna(bool): if True, fill nan values.
    # """

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
        """Relative Strength Index (RSI)
        Returns:
            pandas.Series: New feature generated.
        """
        rsi_series = self._check_fillna(self._rsi, value=50)
        return pd.Series(rsi_series, name="rsi")
    
class MACD(IndicatorMixin):
    # """Moving Average Convergence Divergence (MACD)
    # Is a trend-following momentum indicator that shows the relationship between
    # two moving averages of prices.
    # https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd
    # Args:
    #     close(pandas.Series): dataset 'Close' column.
    #     window_fast(int): n period short-term.
    #     window_slow(int): n period long-term.
    #     window_sign(int): n period to signal.
    #     fillna(bool): if True, fill nan values.
    # """

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
        """MACD Line
        Returns:
            pandas.Series: New feature generated.
        """
        macd_series = self._check_fillna(self._macd, value=0)
        return pd.Series(
            macd_series, name=f"MACD_{self._window_fast}_{self._window_slow}"
        )

    def macd_signal(self) -> pd.Series:
        """Signal Line
        Returns:
            pandas.Series: New feature generated.
        """

        macd_signal_series = self._check_fillna(self._macd_signal, value=0)
        return pd.Series(
            macd_signal_series,
            name=f"MACD_sign_{self._window_fast}_{self._window_slow}",
        )

    def macd_diff(self) -> pd.Series:
        """MACD Histogram
        Returns:
            pandas.Series: New feature generated.
        """
        macd_diff_series = self._check_fillna(self._macd_diff, value=0)
        return pd.Series(
            macd_diff_series, name=f"MACD_diff_{self._window_fast}_{self._window_slow}"
        )

####################
# Extra Indicators #
####################

class ROCIndicator(IndicatorMixin):
    # """Rate of Change (ROC)
    # The Rate-of-Change (ROC) indicator, which is also referred to as simply
    # Momentum, is a pure momentum oscillator that measures the percent change in
    # price from one period to the next. The ROC calculation compares the current
    # price with the price “n” periods ago. The plot forms an oscillator that
    # fluctuates above and below the zero line as the Rate-of-Change moves from
    # positive to negative. As a momentum oscillator, ROC signals include
    # centerline crossovers, divergences and overbought-oversold readings.
    # Divergences fail to foreshadow reversals more often than not, so this
    # article will forgo a detailed discussion on them. Even though centerline
    # crossovers are prone to whipsaw, especially short-term, these crossovers
    # can be used to identify the overall trend. Identifying overbought or
    # oversold extremes comes naturally to the Rate-of-Change oscillator.
    # https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum
    # Args:
    #     close(pandas.Series): dataset 'Close' column.
    #     window(int): n period.
    #     fillna(bool): if True, fill nan values.
    # """

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
    # """True strength index (TSI)
    # Shows both trend direction and overbought/oversold conditions.
    # https://school.stockcharts.com/doku.php?id=technical_indicators:true_strength_index
    # Args:
    #     close(pandas.Series): dataset 'Close' column.
    #     window_slow(int): high period.
    #     window_fast(int): low period.
    #     fillna(bool): if True, fill nan values.
    # """

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
        """True strength index (TSI)
        Returns:
            pandas.Series: New feature generated.
        """
        tsi_series = self._check_fillna(self._tsi, value=0)
        return pd.Series(tsi_series, name="tsi")

    
##################
# Set up sidebar #
##################
# video_file = open('crypto.mp4', 'rb')
# video_bytes = video_file.read()

# st.video(video_bytes)
st.sidebar.title('Crypto Dashboard')

from PIL import Image
image = Image.open('crypto_coins2.png')
st.sidebar.image(image)

# image = Image.open('banner.jpg')
# st.image(image)

# Add in location to select image.
# st.sidebar.title('Crypto Dashboard')


option = st.sidebar.selectbox('Select a Cryptocurrency', ('BTC-USD','ETH-USD','USDT-USD','USDC-USD','BNB-USD','XRP-USD','BUSD-USD','DOGE-USD','ADA-USD','MATIC-USD','DOT-USD','DAI-USD','WTRX-USD','LTC-USD','SOL-USD','TRX-USD','SHIB-USD','HEX-USD','UNI7083-USD','STETH-USD','AVAX-USD','LEO-USD','LINK-USD','WBTC-USD','TON11419-USD'))

st.title(option)
import datetime

today = datetime.date.today()
before = today - datetime.timedelta(days=730)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)
if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

st.sidebar.caption('Presented by Jeff, Thomas and Ray')

##############
# Stock data #
##############

# https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#momentum-indicators

df = yf.download(option,start= start_date,end= end_date, progress=False)

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

progress_bar = st.progress(0)

st.markdown('##### Moving Average Convergence Divergence (MACD)')
st.area_chart(macd)
st.markdown("MACD is used to identify changes in the direction or strength of a stock's price trend. MACD can seem complicated at first glance, because it relies on additional statistical concepts such as the exponential moving average (EMA). But fundamentally, MACD helps in detecting when the recent momentum in a stock's price may signal a change in its underlying trend. This can be useful when deciding when to enter, add to, or exit a position.")
st.markdown("##### Relative Strength Index (RSI)")
st.line_chart(rsi)
st.markdown("- Buying opportunities at oversold positions (when the RSI value is 30 and below)\n - Buying opportunities in a bullish trend (when the RSI is above 50 but below 70)\n - Selling opportunities at overbought positions (when the RSI value is 70 and above)\n - Selling opportunities in a bearish trend (when the RSI value is below 50 but above 30)")
st.markdown(" ")
st.markdown(" ")
st.markdown("##### True Strength Index (TSI)")
st.line_chart(tsi)
st.markdown("Shows both trend direction and overbought/oversold conditions.")
st.markdown(" ")
st.markdown(" ")
st.markdown("##### Rate of Change (ROC)")
st.line_chart(roc)
st.markdown("The Rate of Change (ROC) indicator, which is also referred to as simply Momentum, is a pure momentum oscillator that measures the percent change in price from one period to the next.")
st.markdown(" ")
st.markdown(" ")
st.markdown("##### 15 Day Snapshot")
st.write(option)
st.dataframe(df.tail(15))

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
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="download.xlsx">Download excel file</a>' # decode b'abc' => abc

st.markdown(" ")
st.markdown("##### Create Crypto Report :pencil:")
st.markdown(get_table_download_link(df), unsafe_allow_html=True)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='crypto.csv',
    mime='text/csv',
)
