import pandas as pd
from sklearn.preprocessing import LabelEncoder

date_columns = ['expiration_date', 'registration_init_time']

train_data = pd.read_csv('kkbox-music-data/train.csv')
test_data = pd.read_csv('kkbox-music-data/test.csv', index_col=0)
item_data = pd.read_csv('kkbox-music-data/songs.csv')

user_data = pd.read_csv('kkbox-music-data/members.csv', parse_dates=date_columns)

all_data = pd.concat([train_data, test_data])

all_data = all_data.merge(item_data, on='song_id', how='left')
all_data = all_data.merge(user_data, on='msno', how='left')

enc = LabelEncoder()

for col in [
    'msno', 'song_id', 'source_screen_name', 
    'source_system_tab', 'source_type', 'genre_ids', 
    'artist_name', 'composer', 'lyricist', 'gender'
]:
    all_data[col] = enc.fit_transform(all_data[col].fillna('nan'))
    
for col in ['language', 'city', 'registered_via']:
    all_data[col] = enc.fit_transform(all_data[col].fillna(-2)) 

n = len(train_data)
train_data = all_data[:n]
test_data = all_data[n:]

train_data.to_csv("datas/train.csv")
test_data.to_csv("datas/test.csv")