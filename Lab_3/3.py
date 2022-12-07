import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv('Life Expectancy Data.csv')
print(df.head())

print(df.info())
print(df.isnull().sum())
df = df.fillna(0)
print(df.isnull().sum())

sns.histplot(df['Year'])
sns.rugplot(df['Year'], color='red')
plt.title('Year')
plt.show()

sns.histplot(df['Life expectancy '])
sns.rugplot(df['Life expectancy '], color='red')
plt.title('Life expectancy')
plt.show()

sns.boxplot(df['Adult Mortality'])
plt.title('Adult Mortality')
plt.show()

sns.boxplot(df['infant deaths'])
plt.title('infant deaths')
plt.show()

sns.boxplot(df['Alcohol'])
plt.title('Alcohol')
plt.show()

print('Дисперсія')
print(df.var())
print('Коваріація')
print(df.cov())
print('Кореляція')
corr = df.corr()
print(corr)

sns.heatmap(corr, annot = True, vmin=-1, vmax=1, center= 0)
plt.title('Кореляційна матриця')
plt.show()