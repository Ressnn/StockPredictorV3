# StockPredictorV3

Before using the stock predictor make sure you have all the Dependencies installed, such as numpy, pandas, keras, pytrends, matplotlib,pandas_datareader, and scikit-learn

THIS STOCK PREDICTOR WAS TRAINED ON THE TICKERS AMD,INTC,QCOM,TXN, AND NVDA, SO IT IS ONLY USEFUL FOR MICROCHIP STOCKS, TRAIN IT ON DIFFRENT STOCKS IF YOU ARE GOING TO USE ANOTHER KIND. The predictor also always also predict that the general trend of the stock will go up, and thus for long term trends it is innaccurate however, it works well to predict fluctuations within the short term. For an example of this leave all the settings of Predict.py on default and then run it. You will see that NVDA's Stock went the complete opposite direction that the preictor thought it would go, but the predictor still accuratley predicted that a huge flutuation in stock price was going to occour and that could tell when.

To begin making predictions, go into the predict.py file and change the thicker, the desired start_date, the desired end_date, and set the target size to about 252 times the number of years. The value for splits is reccomened to be at two, but the only rule is that the target size must be divisible by the number of splits. Then, run the script. After it has finished, you should see a matplotlib window with a blue line, a blue line, a green line, and an orange line. The blue line is what actually happened. The orange line is a prediction beased on only the previous timestep, and the greern line is a future prediction.

To train the LSTM on your own possible models open the Main.py file, shift the variable unther the first comment accordingly, and trun the file
