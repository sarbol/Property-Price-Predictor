import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dash
import pandas as pd

app = dash.Dash(__name__, title = 'Lagos Housing')

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

df = pd.read_csv('model_app_data.csv')

def median_price(area, beds):
    return df[(df['property_area'] == area) & (df['bed'] == beds)]['price'].median()

app.layout = html.Main([
                html.H1('LAGOS HOUSE PRICE'),
                html.Div([
                    dcc.Input(
                        id = 'input1',
                        type = 'text',
                        placeholder = 'Area',
                        value = 'abijo'
                    ),
                    dcc.Dropdown(
                        id = 'input2',
                        options = [
                            {'label': '1 bedroom', 'value': '1'},
                            {'label': '2 bedrooms', 'value': '2'},
                            {'label': '3 bedroom', 'value': '3'},
                            {'label': '4 bedroom', 'value': '4'}
            ],
            value = '1'
                    )
                ]),
                html.Br(),
                html.Div(id = 'output')
])

@app.callback(
    Output('output', 'children'),
    Input('input1', 'value'),
    Input('input2', 'value'),
)
def return_median_price(input1, input2):
    return median_price(input1, int(input2))

if __name__ == '__main__':
    app.run_server(debug=True)
