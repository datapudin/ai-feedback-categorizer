from google_play_scraper import reviews, Sort
import pandas as pd
import os

# List of app IDs
app_ids = [
    "com.whatsapp", "com.instagram.android", "com.facebook.katana", "com.snapchat.android",
    "com.twitter.android", "com.google.android.youtube", "com.spotify.music", "com.netflix.mediaclient",
    "com.amazon.mShop.android.shopping", "com.google.android.gm", "com.zhiliaoapp.musically",
    "com.phonepe.app", "com.google.android.apps.maps", "com.ubercab", "com.swiggy.android"
]

# Output directory
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Limit to 100 reviews total
all_reviews = []
review_limit = 100
reviews_per_app = review_limit // len(app_ids)

for app_id in app_ids:
    try:
        app_reviews, _ = reviews(
            app_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=reviews_per_app
        ) 
        for r in app_reviews:
            all_reviews.append({
                "app_id": app_id,
                "review": r['content'],
                "score": r['score'],
                "at": r['at']
            })
    except Exception as e:
        print(f"Error fetching reviews for {app_id}: {e}")

# Save as CSV
df = pd.DataFrame(all_reviews)
csv_path = os.path.join(output_dir, "scraped_reviews.csv")
df.to_csv(csv_path, index=False)
print(f"\n✅ Scraped {len(df)} reviews saved to {csv_path}")
