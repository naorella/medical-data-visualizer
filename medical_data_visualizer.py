#medical_data_visualizer
#code author: Nelson Orellana
#code template from freecodecamp.org
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
med_df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
#BMI calc is weight in kg/sq(height in meters)
#overweight if >25
med_df['overweight'] = med_df['weight'] / ((med_df['height'])/100)**2
#normalize the overweight column into a binary true false, 1 where if overweight
#using lambda as a transformation function to go through the whole data frame and apply a transformation to column 'overweight'
med_df['overweight'] = med_df['overweight'].apply(lambda x: 1 if x > 25 else 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
#lambda transform once again on cholesterol and gluc
med_df['cholesterol'] = med_df['cholesterol'].apply(lambda x: 1 if x >1 else 0)
med_df['gluc'] = med_df['gluc'].apply(lambda x: 1 if x >1 else 0)

# Draw Categorical Plot
#draw_cat_plot(): produces a png, a categorical bar graph based on the medical_examination.csv
#draw_cat_plot(): None -> fig
#requires: a csv filed named 'medical_examination.csv' be in the same folder
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    #melt the data with the coluns listed, and using ID and cardio as a combined primary key
    df_cat = med_df.melt(id_vars=['id', 'cardio'], 
                         value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active','overweight'])
 
    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    #count the values using vairable and cardio pairs. Reset the index, since the original index was lost in the transformation and we need
    #a new one to make this a dataframe again
    df_cat = df_cat.value_counts(['variable', 'cardio', 'value']).reset_index()
   
    #rename the newly indexed column total
    df_cat = df_cat.rename(columns={0:"total"})
    
    # Draw the catplot with 'sns.catplot()'
    #specify what type of categorical graph. Col is how we seperate the two graphs. X and Y are the varibles per graph
    #order it in the same way as the example
    graph = sns.catplot(data = df_cat, kind = 'bar', x = "variable", y="total", col = "cardio", hue='value',
                         order =['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']).fig #depricated, but tests wont pass unless used

    # Get the figure for the output
    fig = graph


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
#draw_heat_map(): produces a png, a diagnol correltaion matrix based on the medical_examination.csv
#draw_heat_map(): None -> fig
#requires: a csv filed named 'medical_examination.csv' be in the same folder
def draw_heat_map():
    # Clean the data
    #keep correct pressure measurements
    df_heat = med_df[(med_df['ap_lo'] <= med_df['ap_hi']) &
                    (med_df['height'] >= med_df['height'].quantile(0.025)) &
                    (med_df['height'] <= med_df['height'].quantile(0.975)) &
                    (med_df['weight'] >= med_df['weight'].quantile(0.025)) &
                    (med_df['weight'] <= med_df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    #taken from seaborn documentation
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize = (9,7))

    # Draw the heatmap with 'sns.heatmap()'
    sns.set_context("paper", font_scale=0.75) 


    sns.heatmap(corr, mask=mask, annot= True,
                    vmax=.3, center=0,
                    square=True, linewidths=.5,
                    fmt = '.1f',
                    cbar_kws={"shrink": .5,
                              "ticks": (-0.08, 0.00, 0.08, 0.16, 0.24)})

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
