# Flight-Delay-Prediction-predict
Created an website to predict a flight  delay

# Problem and Data Description

The unpleasant reality of travel is that there will always be
delays. Flight delays are progressively getting worse, which
causes airline firms to have greater financial problems and
unhappy passengers. But it turns out that they are rather
predictable if we know where to look. Additionally, being
informed in advance of a delay may help to lessen some of the
tension that results from not knowing the status of your flight
until the end moment. As more and more people choose to
travel by air, the number of flights that are delayed can result
in numerous financial issues, congestion at the airport, and
schedule disruptions for everyone, but it can also lead to a
reduction in efficiency, an increase in capital costs, the reallocation
of flight crews and aircraft, and several other additional
costs. It comes at a high price for both customers and airline
corporations. The Total Delay Impact Study estimated that in
2007, delays in air travel cost consumers and the airline sector
a total of 32.9 billion in the US, which resulted in a 4 billion
decline in GDP. Predicting delays can therefore enhance
airline operations and customer satisfaction.

This study’s objective is to investigate the approaches
used to create models that forecast airplane delays caused by
various reasons, including:
1. Air Carrier delay
2. Extreme Weather delay
3. National Aviation System delay
4. Security delay
5. Late Arriving Aircraft delay
6. Airlines
7. Specific time of the year
and other factors that affect the flight departures and arrivals.


# Data Preprocessing & Exploratory Data Analysis

Data mining is an essential technique for extracting useful
insights and knowledge from large datasets to improve the
accuracy and performance for deriving useful insights. It is
very important to preprocess the data to ensure that it is clean,
complete and in the right format. This may involve dealing
with missing data, correcting errors, and transforming the data
into a suitable format for analysis. Furthermore, selecting the
most relevant features from huge datasets is a crucial step in
pre-processing the data.
The dataset contains all flight information, including airlinespecific
cancellations and delays, for dates going back to January
2018. We have considered the years 2021 and 2022 for our analysis as we can relate to the most recent data to validate
our results.
We conducted data mining through the following sequential
steps:
The dataset we used for our analysis contains flight information
from January 2018 to 2022, with over 17 raw data
files for years 2021-2022. To prepare the data, we used the
pandas library to concatenate the raw data files into a single
dataframe. This concatenated dataframe contained over 63
million records and 120 columns, consisting of both numerical
and categorical columns with numerous NaN (missing)
values and columns that were not relevant for our analysis. To
mine the dataset for our study, we omitted irrelevant columns
such as Div5TailNum (aircraft tail number), Div5AirportID
(Diverted Airport 5 code), unique key IDs for each segment
of the airport, and information on detours, aerial time, and
wheels-off time, etc. that were not necessary for our analysis.
Additionally, we focused our analysis on the ORD airport,
which is one of the busiest airports in Chicago, and had a
significant amount of data available for growth rate analysis in
2021 and 2022, as reported by the Federal Aviation Administration.
Consequently, we reduced the dataset to 457,776 rows
and 48 columns. Furthermore, we handled missing values
from this final reduced dataset by excluding the missing data
from our analysis by ensuring that our approach to missing
data was appropriate and did not impact the integrity of our
analysis results.


# Algorithm and Methodology

1.) Importing the necessary packages: pandas, numpy, scikit
learn (for LabelEncoder, StandardScaler, and train test split,
xgboost, os, pickle and Flask.
2.) The load data function reads three CSV files containing
flight data from three airports ORD, DEN, and ATL
concatenates them into a single data frame creates a new column
FLIGHTSTATUS to represent whether the flight was
delayed or not, and selects a subset of columns that will be
used in the model.
3.) The preprocessing function handles missing values,
encodes categorical variables using LabelEncoder, standardizes
numerical variables using StandardScaler, and saves the
scaler and LabelEncoder objects as pickle files to be used for
new data.
4.) The model create function splits the data into features
and target, creates an XGBoost classifier, fits the model to the
data, and saves the trained model as a pickle file.
5.) The predict ans function loads the trained model,
scaler, and LabelEncoder from pickle files, applies the same
pre processing steps used for the training data to a new flight,
predicts the probability and the status of the flight, and returns
the results.
6.) The Flask application has two routes: home page and
prediction page . The home page is a simple HTML page that contains a form for user input. The prediction page takes the
input data, creates a new flight data frame, checks if a trained
model exists as a pickle file, if it doesn’t exist, the function
loads, preprocesses, and trains the data, and then predicts the
results. If the model exists, it just loads the model and makes
the prediction. Finally, it returns the result to the user via an
HTML template.
7.) The app.run command at the end of the script starts
the Flask application in debug mode.

# Summary and Conclusion

Based on our analysis of the flight delay dataset, we have
decided to use the XGBoost classifier model to predict flight
delays. Although the model’s accuracy was lower than other
models, it produced decent results when predicting flight delays
with a 50 percent threshold. The model’s proportional
predictions were in line with the historical data collected, accurately
predicting the chances of flight delay in terms of
percentage. The model classified flights with a delay time of
more than 2 hours as delayed and performed well in a real-life
scenario. We are confident that the XGBoost classifier model
will continue to accurately predict future flight delays based
on its performance with historical data.
After exploring various algorithms to predict flight delays,
we concluded that the bagging and random forest models provided
the best accuracy, but had longer run times, making
them less desirable. Although the decision tree algorithm initially
outperformed the XGBoost algorithm in terms of time
and accuracy, it was later found to overfit and produce inaccurate
predictions. Therefore, we decided to use the XGBoost
classifier model, which although had lower accuracy, still provided
decent results and was the best choice for predicting
flight delays.
The XGBoost model is a reliable tool that can be used by
airlines and airports to manage flight delays and improve operational
efficiency. It has the ability to handle large datasets
and learn complex patterns in the data, making it suitable
for predicting flight delays accurately. We believe that the
XGBoost classifier model has the potential to provide significant
benefits to the aviation industry by improving flight
scheduling, reducing waiting times, and ultimately enhancing
the overall travel experience for passengers.

