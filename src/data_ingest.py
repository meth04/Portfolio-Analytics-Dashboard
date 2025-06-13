import yfinance as yf
import pandas as pd
import os
import argparse
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Get S&P500 list
SP500_WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'


def get_sp500_tickers() -> list:
    """
    Fetch S&P500 tickers by scraping Wikipedia using requests + BeautifulSoup.
    Avoid pandas.read_html dependency on lxml.
    """
    resp = requests.get(SP500_WIKI_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    # table with class 'wikitable sortable'
    table = soup.find('table', {'id': 'constituents'}) or soup.find('table', {'class': 'wikitable sortable'})
    symbols = []
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if len(cols) >= 1:
            sym = cols[0].text.strip()
            symbols.append(sym.replace('.', '-'))
    return symbols

def fetch_prices(tickers: list, start_date: str, end_date: str, interval: str = '1d', batch_size: int = 100) -> pd.DataFrame:
    """
    Batch download adjusted close prices in chunks to avoid timeouts.
    Returns DataFrame indexed by date, columns=tickers.
    """
    price_dfs = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            raw = yf.download(
                tickers=batch,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            # extract Close
            if isinstance(raw.columns, pd.MultiIndex) and 'Close' in raw.columns.levels[0]:
                adj = raw['Close']
            else:
                adj = raw
            price_dfs.append(adj)
            print(f"Batch {i//batch_size+1}: downloaded {len(adj.columns)} tickers.")
        except Exception as e:
            print(f"Failed batch {i//batch_size+1}: {e}")
    # combine on date index
    if price_dfs:
        df_all = pd.concat(price_dfs, axis=1)
        # drop columns all NaN
        df_all = df_all.dropna(axis=1, how='all')
        df_all.index = pd.to_datetime(df_all.index)
        df_all = df_all.sort_index()
        return df_all
    else:
        return pd.DataFrame()


if __name__ == '__main__':

    tickers = get_sp500_tickers()
    filename = 'sp500_prices.csv'

    # Tạo thư mục ghi data
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.abspath(os.path.join(base_dir, '..', 'data', 'raw'))
    os.makedirs(out_dir, exist_ok=True)

    # Xác định khoảng thời gian 5 năm
    today = datetime.today()
    end = today.strftime('%Y-%m-%d')
    start = (today - timedelta(days=5*365)).strftime('%Y-%m-%d')

    # Download prices
    print(f"Fetching {len(tickers)} tickers for S&P500 from {start} to {end}...")
    df = fetch_prices(tickers, start, end, batch_size=200)
    print(f"Downloaded {len(df.columns)} tickers, {len(df)} dates.")

    # Lưu CSV
    out_path = os.path.join(out_dir, filename)
    df.to_csv(out_path)
    print(f"Saved adjusted close prices to {out_path}")
