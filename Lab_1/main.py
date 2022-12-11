import pandas as pd
import numpy

pd.set_option('display.max_columns', None)
movies = pd.read_table("movies.dat", names=['MovieID','Title','Genres'], encoding='latin-1', sep='::', engine='python', nrows= 100000)
ratings = pd.read_table("ratings.dat", names=['UserID','MovieID','Rating','Timestamp'], encoding='latin-1', sep='::', engine='python', nrows= 100000)
users = pd.read_table("users.dat", names=['UserID','Gender','Age','Occupation','Zip-code'], encoding='latin-1', sep='::', engine='python', nrows= 100000)

df = users.merge(ratings).merge(movies)

param = df[(df['Gender'] == 'M')]
top10_male = pd.DataFrame(param.groupby(['Title'])['Rating'].mean())
top10_male= top10_male.sort_values(by='Rating', ascending=False)[:10]
print('Top 10 movies for male')
print(top10_male)
param = df[(df['Gender'] == 'F')]
top10_female = pd.DataFrame(param.groupby(['Title'])['Rating'].mean())
top10_female = top10_female.sort_values(by='Rating', ascending=False)[:10]
print('\nTop 10 movies for female')
print(top10_female)
arr_range = [1,18,25,35,45, 50, 56]
arr_data = []
for i in arr_range:
    if i == 56:
        pass
    else:
        index_i = arr_range.index(i)
        param = df[(df['Age'] > i - 1) & (df['Gender'] == 'M') & (df['Age'] < arr_range[index_i+1])]
        result_male = pd.DataFrame(param.groupby(['Title','Gender'])['Rating'].mean())
        result_male = result_male.sort_values(by='Rating', ascending=False)[:10]
        ranges = f'{i}-{arr_range[index_i+1]}'
        data = ' '.join(list(result_male.index.get_level_values('Title')))
        arr_data.append([ranges, 'M', data])

        index_i = arr_range.index(i)
        param = df[(df['Age'] > i - 1) & (df['Gender'] == 'F') & (df['Age'] < arr_range[index_i+1])]
        result_male = pd.DataFrame(param.groupby(['Title','Gender'])['Rating'].mean())
        result_male = result_male.sort_values(by='Rating', ascending=False)[:10]
        ranges = f'{i}-{arr_range[index_i+1]}'
        data = ' '.join(list(result_male.index.get_level_values('Title')))
        arr_data.append([ranges, 'F', data])

new_table = pd.DataFrame([*arr_data], columns=['ranges', 'Gender', 'Movies'])

table = new_table.pivot(index='ranges',columns='Gender', values='Movies')
print(table)

