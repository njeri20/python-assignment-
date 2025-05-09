# python-assignment-

# -*- coding: utf-8 -*-
"""
Assignment: Data Loading, Analysis, and Visualization with Pandas and Matplotlib/Seaborn

Objectives:
- Load and analyze a dataset using pandas.
- Create simple plots and charts with matplotlib/seaborn for visualizing the data.

Submission:
- This script contains the steps for data loading, exploration, analysis, visualizations,
  and findings, fulfilling the assignment requirements.
- Includes error handling and plot customizations as requested.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris # Using a standard dataset

# Set a consistent style for plots
sns.set_theme(style="whitegrid")

# ==============================================================================
# Task 1: Load and Explore the Dataset
# ==============================================================================

print("--- Task 1: Load and Explore the Dataset ---")

try:
    # Load the Iris dataset
    # load_iris returns a Bunch object, convert to DataFrame for easier handling
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # Add the target species as a column
    # Map numerical target to species names
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    # --- Adding a synthetic date column for the time-series plot example ---
    # The Iris dataset does not have a date/time component naturally.
    # We'll create a simple synthetic date sequence for demonstration purposes.
    # In a real scenario, you would ensure your dataset has a proper datetime column.
    num_rows = len(df)
    start_date = '2023-01-01'
    end_date = '2023-01-10' # Spread data over 10 days
    dates = pd.to_datetime(pd.date_range(start=start_date, end=end_date, periods=num_rows).date)
    df['date'] = dates.sort_values() # Add and sort dates


    print("\nDataset loaded successfully.")

    # Display the first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # Explore the structure of the dataset
    print("\nDataset Info:")
    df.info()

    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    # Observation: As expected for the Iris dataset loaded this way, there are no missing values.
    # If there were missing values, you could handle them here:
    # df.dropna(inplace=True) # To drop rows with any missing values
    # df['column_name'].fillna(df['column_name'].mean(), inplace=True) # To fill missing values with the mean

    # Display unique values and their counts for the 'species' column
    print("\nValue counts for Species:")
    print(df['species'].value_counts())


except FileNotFoundError:
    print("Error: Dataset file not found.")
    # You would add logic here to handle the missing file, maybe exit or load from a different source.
except Exception as e:
    print(f"An error occurred during data loading or initial exploration: {e}")
    # Catch any other unexpected errors during this phase


# ==============================================================================
# Task 2: Basic Data Analysis
# ==============================================================================

if 'df' in locals() and not df.empty: # Proceed only if DataFrame loaded successfully
    print("\n--- Task 2: Basic Data Analysis ---")

    try:
        # Compute the basic statistics of the numerical columns
        print("\nDescriptive Statistics of Numerical Columns:")
        print(df.describe())

        # Perform groupings on the 'species' column and compute the mean of numerical columns
        print("\nMean of numerical features grouped by Species:")
        mean_by_species = df.groupby('species')[iris.feature_names].mean()
        print(mean_by_species)

        # Identify any patterns or interesting findings from your analysis
        print("\nFindings and Observations from Basic Analysis:")
        print("- The `describe()` output gives a quick summary of the spread and central tendency for each measurement (sepal length, sepal width, petal length, petal width).")
        print(f"- For example, the average sepal length is around {df['sepal length (cm)'].mean():.2f} cm.")
        print("- Grouping by species clearly shows differences in average measurements.")
        print(f"- Petal dimensions (length and width) show the most significant differences between species, which is expected as these are key features for distinguishing Iris types.")
        print("- 'setosa' generally has the smallest petal dimensions, while 'virginica' generally has the largest.")
        print("- Sepal dimensions also vary, but with more overlap between species compared to petals.")

    except Exception as e:
        print(f"An error occurred during basic data analysis: {e}")


# ==============================================================================
# Task 3: Data Visualization
# ==============================================================================

if 'df' in locals() and not df.empty: # Proceed only if DataFrame loaded successfully
     print("\n--- Task 3: Data Visualization ---")

     try:
         # 3.1 Line chart showing trends over time
         # Using the synthetic date column and mean sepal length per day as an example
         # In a real scenario, you would plot a measure that genuinely changes over time.
         mean_sepal_length_over_time = df.groupby('date')['sepal length (cm)'].mean().reset_index()

         plt.figure(figsize=(10, 6))
         sns.lineplot(data=mean_sepal_length_over_time, x='date', y='sepal length (cm)')
         plt.title('Synthetic Daily Mean Sepal Length Trend')
         plt.xlabel('Date')
         plt.ylabel('Mean Sepal Length (cm)')
         plt.xticks(rotation=45) # Rotate labels for readability
         plt.tight_layout()
         plt.show()

         # 3.2 Bar chart showing the comparison of a numerical value across categories
         # Average Petal Length per Species
         plt.figure(figsize=(8, 6))
         sns.barplot(data=df, x='species', y='petal length (cm)', estimator='mean')
         plt.title('Average Petal Length per Species')
         plt.xlabel('Species')
         plt.ylabel('Average Petal Length (cm)')
         plt.show()

         # 3.3 Histogram of a numerical column to understand its distribution
         # Distribution of Petal Width
         plt.figure(figsize=(8, 6))
         sns.histplot(df['petal width (cm)'], bins=10, kde=True) # kde=True adds a density curve
         plt.title('Distribution of Petal Width')
         plt.xlabel('Petal Width (cm)')
         plt.ylabel('Frequency')
         plt.show()

         # 3.4 Scatter plot to visualize the relationship between two numerical columns
         # Sepal Length vs. Petal Length, colored by Species
         plt.figure(figsize=(10, 6))
         sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', s=80) # hue adds color based on species
         plt.title('Sepal Length vs. Petal Length by Species')
         plt.xlabel('Sepal Length (cm)')
         plt.ylabel('Petal Length (cm)')
         plt.legend(title='Species') # Add legend for species colors
         plt.show()

         print("\nVisualizations generated successfully.")


     except Exception as e:
         print(f"An error occurred during data visualization: {e}")

else:
    print("\nDataFrame was not loaded successfully. Skipping analysis and visualization tasks.")
=
