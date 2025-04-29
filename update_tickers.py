import requests
from bs4 import BeautifulSoup

def fetch_top_gainers(limit=30):
    url = "https://finviz.com/screener.ashx?v=111&s=ta_topgainers"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    tickers = []
    for row in soup.find_all("tr", class_="table-dark-row-cp") + soup.find_all("tr", class_="table-light-row-cp"):
        cols = row.find_all("td")
        if len(cols) > 1:
            ticker = cols[1].text.strip()
            tickers.append(ticker)

    return tickers[:limit]

def save_tickers(tickers, filename="tickers.txt"):
    with open(filename, "w") as f:
        for ticker in tickers:
            f.write(ticker + "\n")
    print(f"Saved {len(tickers)} tickers to {filename}")

if __name__ == "__main__":
    top_tickers = fetch_top_gainers()
    save_tickers(top_tickers)