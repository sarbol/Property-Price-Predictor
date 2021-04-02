import plotly.graph_objects as go
import pickle
import plotly.express as px
from plotly.subplots import make_subplots

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dash
import pandas as pd
import numpy as np

app = dash.Dash(__name__, title = 'Lagos Housing')
server = app.server

df = pd.read_csv('model_app_data.csv')
rank = pd.read_csv('area_rank.csv')
location_rank = pd.read_csv('location_rank.csv')
infile = open('tree_model', 'rb')
model = pickle.load(infile)
infile.close()
areas = df['property_area'].unique()
med_price = pd.DataFrame(df.groupby('location')['price'].median())
med_price.reset_index(inplace = True)
import json
with open('lagos_Cities.geojson', 'r') as f:
    file = json.load(f)

def max_price(location, area, beds):
    val_loc = df[(df['location'] == location) & (df['bed'] == beds)]['price'].max()
    value = df[(df['property_area'] == area) & (df['bed'] == beds)]['price'].max()
    if value > 0:
        return f'#{int(value):,}'
    else:
        return f'#{int(val_loc):,}'

def median_price(location, area, beds):
    val_loc = df[(df['location'] == location) & (df['bed'] == beds)]['price'].median()
    value = df[(df['property_area'] == area) & (df['bed'] == beds)]['price'].median()
    if value > 0:
        return f'#{int(value):,}'
    else:
        return f'#{int(val_loc):,}'

def min_price(location, area, beds):
    val_loc = df[(df['location'] == location) & (df['bed'] == beds)]['price'].min()
    value = df[(df['property_area'] == area) & (df['bed'] == beds)]['price'].min()
    if value > 0:
        return f'#{int(value):,}'
    else:
        return f'#{int(val_loc):,}'

def min_location_price(location, beds):
    value = df[(df['location'] == location) & (df['bed'] == beds)]['price'].min()
    return f'{int(value):,}'

def med_location_price(location, beds):
    value = df[(df['location'] == location) & (df['bed'] == beds)]['price'].count()
    return f'{int(value):,}'

def max_location_price(location, beds):
    value = df[(df['location'] == location) & (df['bed'] == beds)]['price'].max()
    return f'#{int(value):,}'

def location_bar_graph(data = df, beds = 1):
    df = data
    df = df[df['bed'] == beds]
    df = df.groupby('location').agg(median_price = ('price', np.median))
    df.reset_index(inplace = True)
    fig = px.bar(df.sort_values(by = 'median_price', ascending = True), y="location", x="median_price", orientation = 'h', width = 500, height = 500, title="Median Price by Location")
    fig.update_layout(paper_bgcolor = "#f9f9f9",plot_bgcolor= "#f9f9f9")
    return fig

def area_bar_graph(data = df, location = 'ajah', beds = 2):
    df = data
    df = df[(df['location'] == location) & (df['bed'] == beds)]
    df_ = df.groupby('property_area').agg(median_price = ('price', np.median))
    df_.reset_index(inplace = True)
    fig = px.bar(df_.sort_values(by = 'median_price', ascending = True), y = "property_area", x = "median_price", width = 500, height = 500, orientation = 'h', title = "Median Price of a {}-Bedroom Apartment in {}".format(beds, location))
    fig.update_layout(paper_bgcolor = "#f9f9f9",plot_bgcolor= "#f9f9f9")
    return fig

def pred(location = 'ajah', area = 'ado', beds = 1):
    pos = rank[rank['property_area'] == area].index[0]
    idx = location_rank[location_rank['location'] == location].index[0]
    level = rank['Area_Rank'][pos]
    status = location_rank['Location_Rank'][idx]
    feature = np.array([beds, status, level]).reshape(1, 3)
    prediction = model.predict(feature)
    return f'#{int(prediction):,}'

def box_plot(data = df, location = 'ajah', area = 'abijo', beds = 2):
    df = data
    maxi = df[(df['property_area'] == area) & (df['bed'] == beds)].max()
    mini = df[(df['property_area'] == area) & (df['bed'] == beds)].min()
    pos = rank[rank['property_area'] == area].index[0]
    idx = location_rank[location_rank['location'] == location].index[0]
    level = rank['Area_Rank'][pos]
    status = location_rank['Location_Rank'][idx]
    feature = np.array([beds, status, level]).reshape(1, 3)
    pred = model.predict(feature)
    fig = px.box([pred, (0.8*pred), (1.2*pred)], height = 500, width = 500, title = 'Budget Range\n({}-Bedroom Apartment in {}, {})'.format(beds, area, location))
    fig.update_layout(paper_bgcolor = "#f9f9f9",plot_bgcolor= "#f9f9f9")
    return fig


app.layout = html.Main([
                html.Div([
                    html.H3(
                        'LAGOS STATE RENT BUDGET ESTIMATOR',
                        className = 'project-title'
                    ),
                    html.Form([
                        html.Div([
                            html.Label(
                                'Property Location',
                                className = 'location'
                            ),
                            dcc.Dropdown(
                                id = 'Locations',
                                options = [
                                    {'label':'Gbagada', 'value':'gbagada'},
                                    {'label':'Ajah', 'value':'ajah'},
                                    {'label':'Yaba', 'value':'yaba'},
                                    {'label':'Surulere', 'value':'surulere'},
                                    {'label':'Ikeja', 'value':'ikeja'},
                                    {'label':'Ikorodu', 'value':'ikorodu'},
                                    {'label':'Lekki-Phase-1', 'value':'lekki'}
                                ],
                                value = 'ajah'
                            )
                        ], className = 'form-group'),
                        html.Div([
                            html.Label(
                                'Property Specific Area',
                                className = 'area'
                            ),
                            dcc.Dropdown(
                                id = 'my area',
                            )
                        ], className = 'form-group'),
                        html.Div([
                            html.Label(
                                'Bedrooms',
                                className = 'beds'
                            ),
                            dcc.Dropdown(
                            id = 'bedrooms',
                            options = [
                                {'label':'1', 'value':1},
                                {'label': '2', 'value':2},
                                {'label': '3', 'value':3},
                                {'label': '4', 'value':4}
                            ],
                            value = 1
                            )
                        ], className = 'form-group')
                    ], className = 'my-form'),
                    html.Div([
                        html.Div([
                            html.H3(
                                children = '2000000',
                                id = 'median-price'
                            ),
                            html.P(
                                children = 'Median Price in Ajah',
                                id = 'median-description'
                            )
                        ], className = 'location-tab'),
                        html.Div([
                            html.H3(
                                children = '2000000',
                                id = 'maximum-price'
                            ),
                            html.P(
                                children = 'Maximum Price in Ajah',
                                id = 'maximum-description'
                            )
                        ], className = 'location-tab')
                    ], className = 'location-summary'),
                    html.Div([
                        html.Div([
                            html.H3(
                                children = '2000000',
                                id = 'med-price'
                            ),
                            html.P(
                                children = 'Median Price in Abijo',
                                id = 'med-description'
                            )
                        ], className = 'location-tab'),
                        html.Div([
                            html.H3(
                                children = '2000000',
                                id = 'max-price'
                            ),
                            html.P(
                                children = 'Maximum Price in Abijo',
                                id = 'max-description'
                            )
                        ], className = 'location-tab')
                    ], className = 'location-summary')
                ], id = 'side-bar'),
                html.Div([
                    html.Div([
                        html.Div([
                            dcc.Graph(id = 'bar-graph', figure = location_bar_graph(data = df, beds = 1))
                        ], className = 'left-top-container'),
                        html.Div([
                            html.H3('RENT PREDICTION (Average Price)', className = 'prediction'),
                            html.H1('3000000', id = 'price'),
                            html.P(id = 'descriptor')
                        ], className = 'left-bot-container')
                    ], className = 'left-graph'),
                    html.Div([
                        html.Div([
                            dcc.Graph(id = 'area-bargraph', figure = area_bar_graph(data = df, location = 'ajah', beds = 1))
                        ], className = 'right-graph-container'),
                        html.Div([
                            dcc.Graph(id = 'box-plot', figure = box_plot(data = df, location = 'ajah', area = 'abijo', beds = 2))
                        ], className = 'right-bot-graph-container')
                    ], className = 'right-graph')
                ], id = 'main-bar')
], className = 'overall-container')

@app.callback(
    Output(component_id = 'my area', component_property = 'options'),
    Input(component_id = 'Locations', component_property = 'value')
)
def update_area_options(location):
    df_area = df[df['location'] == location]['property_area'].unique()
    return [{'label': area, 'value': area} for area in df_area]

@app.callback(
    Output(component_id = 'my area', component_property = 'value'),
    Input(component_id = 'my area', component_property = 'options')
)
def update_area_value(available_options):
    return available_options[0]['value']

@app.callback(
    Output(component_id = 'median-price', component_property = 'children'),
    [Input(component_id = 'Locations', component_property = 'value'),
    Input(component_id = 'bedrooms', component_property = 'value')]
)
def update_med_location_price(location, beds):
    return med_location_price(location, beds)

@app.callback(
    Output(component_id = 'maximum-price', component_property = 'children'),
    [Input(component_id = 'Locations', component_property = 'value'),
    Input(component_id = 'my area', component_property = 'value'),
    Input(component_id = 'bedrooms', component_property = 'value')]
)
def update_med_area_price(location, area, beds):
    return median_price(location, area, beds)

@app.callback(
    Output(component_id = 'med-price', component_property = 'children'),
    [Input(component_id = 'Locations', component_property = 'value'),
    Input(component_id = 'my area', component_property = 'value'),
    Input(component_id = 'bedrooms', component_property = 'value')]
)
def update_med_area_price(location, area, beds):
    return max_price(location, area, beds)

@app.callback(
    Output(component_id = 'max-price', component_property = 'children'),
    [Input(component_id = 'Locations', component_property = 'value'),
    Input(component_id = 'my area', component_property = 'value'),
    Input(component_id = 'bedrooms', component_property = 'value')]
)
def update_max_area_price(location, area, beds):
    return min_price(location, area, beds)

@app.callback(
    Output(component_id = 'median-description', component_property = 'children'),
    Input(component_id = 'Locations', component_property = 'value')
)
def update_median_description(location):
    return 'Property Counts in {}'.format(location)

@app.callback(
    Output(component_id = 'maximum-description', component_property = 'children'),
    Input(component_id = 'my area', component_property = 'value')
)
def update_maximum_description(area):
    return 'Median price in {}'.format(area)

@app.callback(
    Output(component_id = 'med-description', component_property = 'children'),
    Input(component_id = 'my area', component_property = 'value')
)
def update_med_description(area):
    return 'Maximum price in {}'.format(area)

@app.callback(
    Output(component_id = 'max-description', component_property = 'children'),
    Input(component_id = 'my area', component_property = 'value')
)
def update_max_description(area):
    return 'Minimum price in {}'.format(area)

@app.callback(
    Output(component_id = 'bar-graph', component_property = 'figure'),
    Input(component_id = 'bedrooms', component_property = 'value')
)
def output_bar_graph(beds):
    return location_bar_graph(data = df, beds = beds)

@app.callback(
    Output(component_id = 'area-bargraph', component_property = 'figure'),
    [Input(component_id = 'Locations', component_property = 'value'),
    Input(component_id = 'bedrooms', component_property = 'value')]
)
def output_area_bar_graph(location, beds):
    return area_bar_graph(data = df, location = location, beds = beds)

@app.callback(
    Output(component_id = 'price', component_property = 'children'),
    [Input(component_id = 'Locations', component_property = 'value'),
    Input(component_id = 'my area', component_property = 'value'),
    Input(component_id = 'bedrooms', component_property = 'value')]
)
def output_pred(location, area, beds):
    return pred(location = location, area = area, beds = beds)

@app.callback(
    Output(component_id = 'descriptor', component_property = 'children'),
    [Input(component_id = 'Locations', component_property = 'value'),
    Input(component_id = 'my area', component_property = 'value'),
    Input(component_id = 'bedrooms', component_property = 'value')]
)
def output_descriptor(location, area, beds):
    return '{}-bedroom Apartment in {}, {}.'.format(beds, area, location)

@app.callback(
    Output(component_id = 'box-plot', component_property = 'figure'),
    [Input(component_id = 'Locations', component_property = 'value'),
    Input(component_id = 'my area', component_property = 'value'),
    Input(component_id = 'bedrooms', component_property = 'value')]
)
def update_boxplot(location, area, beds):
    return box_plot(data = df, location = location, area = area, beds = beds)



if __name__ == '__main__':
    app.run_server(debug=True)
