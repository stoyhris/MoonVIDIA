# Forecasting Methods for Stock Price Prediction 

# Introduction 
The growing interest in stock price forecasting is driven by the increasing complexity and dynamism of financial markets, along with the potential for substantial financial gains. Advances in technology and data analytics have made it possible to process vast amounts of historical and real-time data, enabling more sophisticated modeling and prediction techniques. Investors, financial institutions, and individual traders are keen to leverage these capabilities to gain a competitive edge. The availability of high-frequency trading and algorithmic trading platforms has further fueled this interest, as these platforms rely heavily on accurate and timely forecasts to execute trades at optimal moments. Additionally, the development of machine learning and artificial intelligence has opened new avenues for more accurate and nuanced forecasting models, such as ARIMA-GARCH and other hybrid models, which can capture both linear trends and volatility patterns in stock prices.

The potential impact of improved stock price forecasting is significant, influencing both individual and institutional decision-making. Accurate forecasts can lead to better investment strategies, optimizing portfolio management and maximizing returns while minimizing risks. For individual investors, this could mean more informed decisions and greater financial security. For financial institutions, it could translate to enhanced trading algorithms and better risk management practices. Moreover, more accurate forecasting can contribute to the overall stability and efficiency of financial markets by reducing the likelihood of market anomalies and speculative bubbles. However, there is also the potential for negative impacts, such as increased market volatility due to the widespread use of similar forecasting models, which could lead to synchronized trading behaviors. Therefore, while the advancements in stock price forecasting offer substantial benefits, they also require careful consideration of the broader market implications and the need for robust risk management frameworks.

NVIDIA's business model centers on designing and manufacturing advanced graphics processing units (GPUs) and related technologies, serving a wide array of markets including gaming, professional visualization, data centers, and automotive. The company's GPUs are widely recognized for their high performance in gaming and graphics rendering, and have also become critical for AI and machine learning applications due to their parallel processing capabilities. NVIDIA's recent growth has been fueled by the surge in demand for AI and machine learning infrastructure, cloud computing, and data centers. NVIDIA's expansion into autonomous vehicle technology and its continued innovation in GPU technology have also contributed to its robust financial performance and market leadership. 

This growth has inevitably affected its stock price, with over 180% growth in the last year, as shown in the figure below. This study examines the sharp increase in NVIDIA's share price and builds to models to attempt to predict it. Experiments show that the ARIMA-GARCH model accurately captures the stock's movements in next-day pricing forecasts. However, both ARIMA-GARCH and LSTM models give poor out-of-sample performance.

![246185dc-c019-4c6b-bdcd-a382b6ce25d4](https://github.com/user-attachments/assets/17102e77-c074-4a55-a926-aad42337a866)

# Long Short-Term Memory 

A Long Short-Term Memory (LSTM) model is a type of recurrent neural network designed to handle and learn from sequential data. Unlike traditional RNNs, which struggle with long-term dependencies due to issues like vanishing gradients, LSTMs can capture and maintain information over long sequences. This is achieved through a complex architecture of gates (input, forget, and output gates) that control the flow of information, allowing the model to retain relevant information and discard unnecessary data. Consequently, LSTMs are particularly effective for tasks involving time series data, where understanding the context over extended sequences is crucial.

LSTM models can be applied to forecasting stock prices by leveraging their ability to learn from past data to predict future trends. In stock price forecasting, an LSTM model can be trained on historical price data, where each data point is part of a time series representing the stock's performance over time. By feeding sequences of past stock prices into the LSTM, the model learns patterns and temporal dependencies that can be used to predict future prices. For example, a model might be trained on sequences of the last few days' closing prices to forecast the next day's closing price. This approach can help investors and analysts make informed decisions by providing insights into potential future price movements based on historical trends.

### Data Pre-processing

While we have access to $NVDA's historical closing price from its IPO in 1999 to a recent date in 2024, considering the entire range would result in lots of overfitting. Given that the most recent year has had the largest volatility, which is hence more valuable to predict, we train the model using only the most recent six months. We then process the data into a dataframe which, for each day $t$, includes the past three days' price as well as the $t$-th day price. Finally, we split the data into training, validation, and testing using the common 80/10/10 split. 

![ddb6e5dd-0463-4e78-9ec4-4ec8a1b43b54](https://github.com/user-attachments/assets/53801759-158a-4388-8515-f4ddf82d4d08)

### Model Architecture 
While there are many ways to structure a sequential neural network, we employ a common one in the literature.
* Input layer which is the size of the number of historical days the model sees (3)
* LSTM layer
* 10% dropout layer to avoid overfitting
* Two linear layers with ReLU activation functions to model the complexity in stock price
* A final output layer that predicts the next day closing price
  
### Results 

#### Training Performance 
Ultimately, the LSTM model achieves strong results on training data. This is not particularly surprising given that the model has seen this data already.

![6e519412-5155-4a3c-ba3e-1ac1d8efbc4e](https://github.com/user-attachments/assets/5c4547c5-c513-4a1b-b36c-2467ed799cdc)

#### Validation Performance
We use the validation phase to tune hyperparameters. Despite lots of tuning, we failed to achieve adequate out of sample performance.

![ee212142-222b-462c-87ab-87cf62ac3679](https://github.com/user-attachments/assets/87ed6c39-1b81-497e-a8bf-71c5533bdba8)

#### Testing Performance 
Finally, the testing phase involves predicting data it has never seen. Since the model has not seen the price jump above around 95, it predicts that it will always be in that ballpark.

![dd2e84dd-0332-4096-b648-e8c2a1e53943](https://github.com/user-attachments/assets/daaf449d-2637-44ea-aade-3fc2e15811af)

#### Overall Performance
Overall, we see that the LSTM model fails to adapt to changes in volatility, leading to poor out-of-sample performance. 

![51b2f9e7-567e-4bc0-90c7-4b2999475994](https://github.com/user-attachments/assets/ac07147f-f3e4-48d2-a7e8-3894e65deb9d)


# ARIMA-GARCH 
While LSTM forecasts only the next day's closing price, ARIMA-GARCH is a forecasting technique which predicts the mean and volatility of time series data. The hypothesis is that, by considering the second moment, we will be able to derive greater insights about out-of-sample performance.

### ARIMA Overview 

AutoRegressive Integrated Moving Average (ARIMA) models are a class of statistical techniques used for analyzing and forecasting time series data. They work by combining three main components: autoregression (AR), which uses dependencies between an observation and a number of lagged observations; differencing (I), which makes the time series stationary by removing trends; and moving average (MA), which models the relationship between an observation and a residual error from a moving average model applied to lagged observations. In the context of predicting $NVDA, by analyzing past price movements and adjusting for noise and fluctuations, ARIMA models can provide insights into potential future price behavior, aiding investors and analysts in making informed decisions.

There are three hyperparameters in ARIMA models: 
* p: the number of lag observations included in the model
* d: the number of times that raw observations are differenced to make the time series stationary
* q: the size of the moving average window

### GARCH Overview 

Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models are used to estimate and forecast the volatility of time series data, particularly in financial markets. Unlike ARIMA models, which focus on the mean of the time series, GARCH models focus on the variability or volatility over time. They achieve this by modeling the variance of the current error term as a function of past error terms and past variances, capturing periods of high and low volatility. Given the recent volatility of $NVDA, this method may be particularly valuable to help investors better manage risk, set appropriate trading strategies, and make more informed decisions about potential price movements.

There are two hyperparameters in ARIMA models: 
* p: the number of lagged variances included in the model
* q: the number of lagged residuals included in the model

### ARIMA-GARCH Overview

ARIMA-GARCH models combine the strengths of both ARIMA and GARCH models to provide a comprehensive approach to time series forecasting. The ARIMA component focuses on modeling the mean or trend of the time series, capturing the linear relationships and dependencies in the data. Meanwhile, the GARCH component models the volatility or variance, accounting for periods of changing volatility and clustering of variance over time. This dual approach aids investors in making more informed decisions by understanding both expected price movements and the risk associated with those movements.

ARIMA-GARCH models use all the hyperparamters of ARIMA and GARCH models.

### Model Construction 

Unlike with LSTMs, ARIMA-GARCH models can read time series data as is, so no pre-processing was required. First, we fit an ARIMA model to forecast the mean NVDA price. Although hyperparamters can be tuned using partial autocorrelation plots, we opted to use a systematic grid search which searched for the lowest mean squared error. This experiment revealed that the ARIMA model should include 1 lag observation (consider only yesterday's price), 1 difference, an a 2 day moving average window. This resulted in an ARIMA model with all parameters being statistially significant at the 0.01 confidence level. Second, we fit a GARCH model on the residuals of the ARIMA model to predict their volatility. Once again, we used a grid search for hyperparamter tuning and found that the best model includes 2 lagged variances and 2 lagged residuals. Finally, we defined the ARIMA-GARCH prediction as the confidence interval centered around the ARIMA prediction and deviating by the square root of the GARCH prediction.  

### Results
Initial results once again showed poor out-of-sample performance. In this case, in the long run, the model is predicting each price to be the same as the day before, while the potential volatility grows. Obviously, this is not useful to investors.
![eefec9c7-f44a-4e8c-a30f-c26268dac79e](https://github.com/user-attachments/assets/7f26d0ab-3234-4667-bd50-f1326cd12148)

However, if we train the model on every observation before the day we want to predict, we achieve much better results. In this case, the out of sample performance is limited to one day, which results in a greater accuracy. Indeed, we see that the true price of NVDA is almost always in the band predicted by the ARIMA-GARCH model.
![c7ddeb0e-aff5-4f7d-952c-899d2498bea5](https://github.com/user-attachments/assets/9050a4b6-a1c7-4ded-917e-9ce7941c768f)

We can examine the performance more closely by analyzing the most recent two weeks' predictions. We see that, although there are times when NVDA is outside of the predicted range, the predictions are very close to the true values.
![fee1120c-3183-4a84-91bf-79e8bbf45d90](https://github.com/user-attachments/assets/1f846b6c-a888-47b9-af06-edb0aabc0a7b)

# Conclusion

This study builds two forecasting tools to predict the next day price of NVDA: an LSTM neural network and an ARIMA-GARCH model. Both models fail to perform well on out-of-sample data. The LSTM model naively predicts that the price will not exceed the highest price it has seen, while the ARIMA-GARCH model predicts the price will converge to one value on average, while its volatility will increase indefinitely. However, with enhanced data visibility, the ARIMA-GARCH model is able to produce next-day predictions that capture the true price of NVDA with excellent accuracy. A well-predicted mean price with the additional bands that GARCH offers create a baseline for traders to consider while investing.
