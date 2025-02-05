import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 10

# 1. Veri Yükleme ve Temizleme
df = pd.read_csv('/Users/onurakyuz/Desktop/Onur Akyüz Fitness Analysis/data/ONUR FITNESS DATA.csv', sep=';')

# Boş satırları temizleme
df = df.dropna(how='all')

# '-' değerlerini NaN olarak değiştirme ve bu satırları kaldırma
df = df.replace('-', np.nan)
df = df.dropna()

# Tarih sütununu datetime formatına çevirme
df['Antrenman Tarihi'] = pd.to_datetime(df['Antrenman Tarihi'], format='%d.%m.%Y')

# Kalori değerlerini sayısal formata çevirme
df['Yakılan Kalori'] = df['Yakılan Kalori'].str.replace(' KCAL', '').astype(float)

# Kalp atış hızını sayısal formata çevirme
df['Ortalama Kalp Atış Hızı'] = df['Ortalama Kalp Atış Hızı'].str.replace(' v/dk', '').astype(float)

# Antrenman süresini dakikaya çevirme
def convert_time_to_minutes(time_str):
    if ':' not in time_str:
        return np.nan
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = map(int, parts)
    else:
        h, m = map(int, parts)
        s = 0
    return h * 60 + m + s/60

df['Antrenman Süresi (Dakika)'] = df['Antrenman Süresi'].apply(convert_time_to_minutes)

# 2. Temel İstatistikler
print("=== TEMEL İSTATİSTİKLER ===")
print(f"Toplam Antrenman Sayısı: {len(df)}")
print(f"Toplam Yakılan Kalori: {df['Yakılan Kalori'].sum():.0f} KCAL")
print(f"Ortalama Antrenman Süresi: {df['Antrenman Süresi (Dakika)'].mean():.1f} dakika")

print("\nAntrenör Bazlı Antrenman Sayıları:")
print(df['Antrenör'].value_counts())

print("\nDetaylı İstatistikler:")
stats = df[['Yakılan Kalori', 'Ortalama Kalp Atış Hızı', 'Antrenman Süresi (Dakika)']].describe()
print(stats)

# 3. Görselleştirmeler

# 3.1 Kalori Yakımı Trendi
plt.figure(figsize=(15, 6))
plt.plot(df['Antrenman Tarihi'], df['Yakılan Kalori'], marker='o', linestyle='-', linewidth=2, markersize=8)
plt.title('Antrenman Başına Yakılan Kalori Trendi', pad=20)
plt.xlabel('Tarih')
plt.ylabel('Yakılan Kalori (KCAL)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/kalori_trend.png', bbox_inches='tight', dpi=300)
plt.close()

# 3.2 Antrenör Bazlı Kalori Dağılımı
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Antrenör', y='Yakılan Kalori')
plt.title('Antrenör Bazlı Kalori Dağılımı', pad=20)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/antrenor_bazli_kalori.png', bbox_inches='tight', dpi=300)
plt.close()

# 3.3 Kalori ve Kalp Atış Hızı İlişkisi
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Ortalama Kalp Atış Hızı', y='Yakılan Kalori', hue='Antrenör', style='Antrenör')
plt.title('Kalori ve Kalp Atış Hızı İlişkisi', pad=20)
plt.tight_layout()
plt.savefig('visualizations/kalori_kalp_atis_iliski.png', bbox_inches='tight', dpi=300)
plt.close()

# 3.4 Aylık Performans Analizi
df['Ay'] = df['Antrenman Tarihi'].dt.strftime('%Y-%m')
monthly_stats = df.groupby('Ay').agg({
    'Yakılan Kalori': ['mean', 'count'],
    'Ortalama Kalp Atış Hızı': 'mean',
    'Antrenman Süresi (Dakika)': 'mean'
}).round(1)

monthly_stats.columns = ['Ort. Kalori', 'Antrenman Sayısı', 'Ort. Kalp Atış Hızı', 'Ort. Süre (dk)']
monthly_stats = monthly_stats.reset_index()

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Aylık Performans Metrikleri', fontsize=16, y=1.02)

# Aylık Ortalama Kalori
sns.barplot(data=monthly_stats, x='Ay', y='Ort. Kalori', ax=axes[0,0])
axes[0,0].set_title('Aylık Ortalama Yakılan Kalori')
axes[0,0].tick_params(axis='x', rotation=45)

# Aylık Antrenman Sayısı
sns.barplot(data=monthly_stats, x='Ay', y='Antrenman Sayısı', ax=axes[0,1])
axes[0,1].set_title('Aylık Antrenman Sayısı')
axes[0,1].tick_params(axis='x', rotation=45)

# Aylık Ortalama Kalp Atış Hızı
sns.barplot(data=monthly_stats, x='Ay', y='Ort. Kalp Atış Hızı', ax=axes[1,0])
axes[1,0].set_title('Aylık Ortalama Kalp Atış Hızı')
axes[1,0].tick_params(axis='x', rotation=45)

# Aylık Ortalama Antrenman Süresi
sns.barplot(data=monthly_stats, x='Ay', y='Ort. Süre (dk)', ax=axes[1,1])
axes[1,1].set_title('Aylık Ortalama Antrenman Süresi (dk)')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualizations/aylik_performans.png', bbox_inches='tight', dpi=300)
plt.close()

# 3.5 Antrenman Süresi ve Kalori İlişkisi
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Antrenman Süresi (Dakika)', y='Yakılan Kalori', hue='Antrenör')
plt.title('Antrenman Süresi ve Yakılan Kalori İlişkisi')
plt.tight_layout()
plt.savefig('visualizations/antrenman_sure_kalori.png', bbox_inches='tight', dpi=300)
plt.close()

# 3.6 Korelasyon Matrisi
correlation_matrix = df[['Yakılan Kalori', 'Ortalama Kalp Atış Hızı', 'Antrenman Süresi (Dakika)']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Performans Metrikleri Arasındaki Korelasyon')
plt.tight_layout()
plt.savefig('visualizations/korelasyon_matrisi.png', bbox_inches='tight', dpi=300)
plt.close()

# 4. Performans Analizleri
print("\n=== PERFORMANS ANALİZLERİ ===")

# En yüksek kalori yakılan antrenmanlar
print("\nEn Yüksek Kalori Yakılan 5 Antrenman:")
top_calories = df.nlargest(5, 'Yakılan Kalori')[['Antrenman Tarihi', 'Antrenör', 'Yakılan Kalori', 'Antrenman Süresi']]
print(top_calories)

# Antrenör bazlı ortalama performans
print("\nAntrenör Bazlı Ortalama Performans:")
trainer_stats = df.groupby('Antrenör').agg({
    'Yakılan Kalori': ['mean', 'std'],
    'Ortalama Kalp Atış Hızı': 'mean',
    'Antrenman Süresi (Dakika)': 'mean'
}).round(1)
trainer_stats.columns = ['Ort. Kalori', 'Std. Kalori', 'Ort. Kalp Atış Hızı', 'Ort. Süre (dk)']
print(trainer_stats)

# Haftalık trend analizi
df['Hafta'] = df['Antrenman Tarihi'].dt.isocalendar().week
weekly_stats = df.groupby('Hafta').agg({
    'Yakılan Kalori': ['mean', 'count'],
    'Antrenman Süresi (Dakika)': 'mean'
}).round(1)
weekly_stats.columns = ['Ort. Kalori', 'Antrenman Sayısı', 'Ort. Süre (dk)']

print("\nHaftalık Antrenman İstatistikleri:")
print(weekly_stats)
