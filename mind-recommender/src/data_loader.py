import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# This is where your files actually live
data_path = r"C:\Users\solisrv\OneDrive - beloit.edu\Desktop\MIND\MINDsmall_train"

# Verify the path by listing files
print("Files in directory:", os.listdir(data_path))

# Load News Data
news_cols = ['news_id', 'category', 'subcategory', 'title',
             'abstract', 'url', 'title_entities', 'abstract_entities']

news_df = pd.read_csv(os.path.join(data_path, 'news.tsv'),
                       sep='\t', names=news_cols)
#Load Behaviors Data
beh_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
behaviors_df = pd.read_csv(os.path.join(data_path, 'behaviors.tsv'),
                       sep='\t', names=beh_cols)

print(f"Loaded {len(news_df)} news articles.")
print(news_df.head())
