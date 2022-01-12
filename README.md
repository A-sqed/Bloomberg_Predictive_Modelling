## Installation:

Use pip install -r requirements.txt to install the required libraries.

## To Begin:

1. Use Bloomberg's Excel plugin to create an XLSX file and put into the \data folder
2. An Example file has been provided. It must contain dates and hard values
3. cd into the \Bloomberg\Predicative\Modelling Folder

## To Run the Web Interface:

1. Go to your terminal and move to the directory where the webapp.py code is
2. In the terminal type "streamlit run webapp.py"
3. Click on the top IP address link that is generated (should take you to the browser)
4. On the left, select an excel file, e.g., the default file included in data
5. Specify which category is your target for forecasting
5. Include a list of categories (separated by a comma (,) where you would like to include momentum features, e.g., rolling averages
6. Press the "Train Model" button

## Example Outputs:
Hisotrical Data & Forecasts:
![alt text](https://github.com/A-sqed/Bloomberg_Predictive_Modelling/blob/3c1415df764e103f68a542d6cbb434d1b9b71661/_img/example_forecast.PNG)

Feature Importance Over Time:
![alt text](https://github.com/A-sqed/Bloomberg_Predictive_Modelling/blob/3c1415df764e103f68a542d6cbb434d1b9b71661/_img/feats_importance_over_time.png)

Predictive Power:
![alt text](https://github.com/A-sqed/Bloomberg_Predictive_Modelling/blob/3c1415df764e103f68a542d6cbb434d1b9b71661/_img/predictive_power.png)

## Work-In-Progress Features:
1. Inclusion of additional Regression & Classification Models
2. Model Analytics for Classifiers
3. Ability to integrate database sources
4. Customize forecasting periods