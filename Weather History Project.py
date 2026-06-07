#Import Librarires 
import pandas as pd
import numpy as np 
import matplotlib. pyplot as plt
import seaborn as sns

#Load the dataset
df = pd.read_excel(r"C:\Users\Susmija\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\5004EC4E497AF9A8FBE2EA68B29A7C4F9BB768B1\transfers\2026-23\weatherHistory.csv.xlsx" ,nrows=5000)
df.head()
#Check for First and Last 10 rows of dataset
print(df.head(10))
print(df.tail(10))
#check size of the dataset 
print(df.shape)

#DATA CLEANING
#check for missing values 
print(df.isnull().sum())
#Fill the missing values
df = df.assign(
    **{
        'Precip Type': df['Precip Type'].fillna('No Precipitation'),
        'Wind Bearing (degrees)': df['Wind Bearing (degrees)'].fillna(
            df['Wind Bearing (degrees)'].mean()
        )
    }
)
#Check for duplicates
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
# Standardize Data Types
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
# Handle Outliers (IQR Method)
numerical_cols = ['Temperature (C)', 'Wind Speed (km/h)', 'Pressure (millibars)']
for col in numerical_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers to bounds
        df[col] = np.clip(df[col], lower_bound, upper_bound)

print("\n--- CLEANED DATA SUMMARY ---")
print(df.describe())

#DATA VISUALIZATION
# Set a clean visual theme
sns.set_theme(style="whitegrid")

# Enlarged canvas to give components breathing room
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Main dashboard title - made slightly smaller and cleaner
fig.suptitle('Weather Insights & Metrics Dashboard', fontsize=18, weight='bold', y=0.97)

# --- Plot 1: Top Left ---
sns.regplot(ax=axes[0, 0], data=df, x='Temperature (C)', y='Apparent Temperature (C)', 
            color='#FF6B6B', marker='o', scatter_kws={'alpha': 0.4, 's': 15}, line_kws={'color': 'black'})
# Reduced title to 11pt, labels to 9pt
axes[0, 0].set_title('Thermal Consistency (Actual vs Apparent)', fontsize=11, weight='bold', pad=12)
axes[0, 0].set_xlabel('Temperature (C)', fontsize=9)
axes[0, 0].set_ylabel('Apparent Temperature (C)', fontsize=9)
axes[0, 0].tick_params(labelsize=8) # Shrinks tick numbers

# --- Plot 2: Top Right ---
sns.histplot(ax=axes[0, 1], data=df, x='Humidity', kde=True, color='#4ECDC4', bins=20)
axes[0, 1].set_title('Atmospheric Humidity Distribution', fontsize=11, weight='bold', pad=12)
axes[0, 1].set_xlabel('Humidity Level (0 to 1)', fontsize=9)
axes[0, 1].set_ylabel('Frequency Count', fontsize=9)
axes[0, 1].tick_params(labelsize=8)

# --- Plot 3: Bottom Left ---
top_summaries = df['Summary'].value_counts().nlargest(5).index
filtered_df = df[df['Summary'].isin(top_summaries)]

sns.boxplot(ax=axes[1, 0], data=filtered_df, x='Summary', y='Wind Speed (km/h)', palette='Set2')
axes[1, 0].set_title('Wind Speed Dynamics by Major Weather Profiles', fontsize=11, weight='bold', pad=12)
axes[1, 0].set_xlabel('Weather Condition Summary', fontsize=9, labelpad=8)
axes[1, 0].set_ylabel('Wind Speed (km/h)', fontsize=9)
axes[1, 0].tick_params(axis='x', rotation=25, labelsize=8) # Shrunk categorical text sizes
axes[1, 1].tick_params(axis='y', labelsize=8)

# --- Plot 4: Bottom Right ---
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

# Shrunk internal heatmap font size to 8pt for better spacing
sns.heatmap(ax=axes[1, 1], data=correlation_matrix, annot=True, cmap='coolwarm', 
            fmt=".2f", linewidths=.5, cbar_kws={'shrink': 0.7}, annot_kws={"size": 8})
axes[1, 1].set_title('Core Meteorological Feature Correlations', fontsize=11, weight='bold', pad=12)
axes[1, 1].tick_params(labelsize=8) # Shrunk the row/column names on the heatmap edges

# Spacing allocation to keep plots neatly separated
plt.subplots_adjust(left=0.08, right=0.92, top=0.91, bottom=0.10, wspace=0.28, hspace=0.32)

# Save and show
plt.savefig('weather_insights_dashboard_small_text.png', dpi=300, bbox_inches='tight')
plt.show()
