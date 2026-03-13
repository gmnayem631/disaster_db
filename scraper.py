from newspaper import Article

url = "https://www.thedailystar.net/news/the-parliament-watch/parliament/news/day-1-13th-parliament-walkout-speech-and-few-knowing-smiles-4127331"

article = Article(url)
article.download()
article.parse()

print("Title:", article.title)
print("Date:", article.publish_date)
print("Text preview:", article.text[:1500])