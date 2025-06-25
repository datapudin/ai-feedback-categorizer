import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_google_play_reviews(app_id, max_reviews=50):
    url = f"https://play.google.com/store/apps/details?id={app_id}&hl=en&gl=US"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    reviews = []
    for span in soup.select("span[jsname='bN97Pc']"):
        review = span.get_text(strip=True)
        if review and len(reviews) < max_reviews:
            reviews.append(review)
    df = pd.DataFrame(reviews, columns=["review"])
    df["source"] = "Google Play"
    df["timestamp"] = pd.date_range(start='2025-01-01', periods=len(df), freq='D')
    return df

if __name__ == "__main__":
    app_id = "com.instagram.android"  # Replace with your desired app ID
    df = scrape_google_play_reviews(app_id, max_reviews=30)
    df.to_csv("google_play_reviews.csv", index=False)
    print("Scraped reviews saved to google_play_reviews.csv")
