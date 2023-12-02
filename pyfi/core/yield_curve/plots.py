import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_heatmap(df=None, title=''):
    cmap = sns.diverging_palette(20, 160, as_cmap=True)
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(20,5))
    plt.suptitle(title)
    sns.heatmap(df, cmap=cmap, vmax=.3, center=0,
                square=False, linewidths=.5, cbar_kws={"shrink": 1}, annot = True)
    
    
    
import plotly.graph_objects as go

def points_in_time(df):
    
    x = df.iloc[-252]
    y = df.iloc[-60]
    z = df.iloc[-20]
    a = df.iloc[-1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x.index, y=x.values, mode='lines', name='252-days ago'))
    fig.add_trace(go.Scatter(x=y.index, y=y.values, mode='lines', name='60-days ago'))
    fig.add_trace(go.Scatter(x=z.index, y=z.values, mode='lines', name='20-days ago'))
    fig.add_trace(go.Scatter(x=a.index, y=a.values, mode='lines', name='1-day ago'))

    fig.update_layout(title='Line Chart',
                      xaxis_title='X-axis',
                      yaxis_title='Y-axis',
                      plot_bgcolor='white',      
                      paper_bgcolor='white')     


    fig.show()
    
    
def periodic_change(df, title):
    
    dfX = df.diff().iloc[-5:].sum().to_frame(name = '5-days').T

    dfY = df.diff().iloc[-20:].sum().to_frame(name = '20-days').T

    dfZ = df.diff().iloc[-60:].sum().to_frame(name = '60-days').T

    dfA = df.diff().iloc[-252:].sum().to_frame(name = '252-days').T
    
    cat = pd.concat([dfA, dfZ, dfY, dfX], axis = 0)

    return plot_heatmap(df=cat, title=title)
