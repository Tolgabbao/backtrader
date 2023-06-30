import praw
from textblob import TextBlob

# Read the NASDAQ tickers file
with open("nasdaqlisted.txt", "r") as file:
    lines = file.readlines()

# Create a dictionary to store the tickers
company_tickers = {}

# Iterate over the lines and extract the ticker and company name
for line in lines:
    parts = line.strip().split("|")
    if len(parts) >= 2:
        ticker = parts[0]
        company_name_unp = parts[1]
        company_name = company_name_unp.strip().split("-")[0].strip()

        # Add the company name and ticker to the dictionary
        company_tickers[company_name] = ticker

# Create a Reddit instance
reddit = praw.Reddit(
    client_id='LvHhbFoVTBc1J7G5xjJkfg',
    client_secret='YcIg7ikYxwRNrMCKTMWTKrU5MDpvlQ',
    user_agent='StockAnalysis/1.0',
)

# Specify the subreddit
subreddit = reddit.subreddit('stocks')

# Get the top 10 hot posts from the subreddit
posts = subreddit.hot(limit=10)

# Iterate over the posts and analyze sentiment
for post in posts:
    title_sentiment = TextBlob(post.title).sentiment.polarity
    body_sentiment = TextBlob(post.selftext).sentiment.polarity

    # Iterate over the company tickers and check if they are mentioned in the post
    for company_name, ticker in company_tickers.items():
        if company_name.lower() in (post.title + post.selftext).lower():
            # Print the post and its sentiment filtered by ticker
            print("Post:", post.title)
            print("Ticker:", ticker)
            print("Title Sentiment:", title_sentiment)
            print("Body Sentiment:", body_sentiment)
            print()
            break
