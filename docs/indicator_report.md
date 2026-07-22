# Technical Indicators Report

> Converted from a prior Word draft. Author bylines, course codes, and Drive links removed.

Indicator Evaluation
Abstract—This project evaluates different technical indicators and compares a theoretical optimal strategy that can view future data with a benchmark singular-buy and hold strategy.
Part 1: Indicators
In this project, the following technical indicators are evaluated: simple moving average, Bollinger Bands, momentum, exponential moving average, and volatility.
All the charts in this project have the adjusted close prices normalized to start at a value of 1.
Simple Moving Average
Simple moving average (SMA) is a moving arithmetic mean calculated by adding the price of several periods and then dividing by the count of periods. The formula and pseudocode for SMA are:
 
prices.rolling(n).mean()
where n is the number of periods.
SMA smooths price volatility and demonstrates the lagged characteristics of the price movement. Many traders watch for short-term averages to cross above longer-term averages to signal the beginning of a buy period and vice-versa for a sell. (Hayes, 2020 [3])
Figure 1: Multiple 20-Day and 50-Day SMA crosses happened in these 2 years
Figure 1: Multiple 20-Day and 50-Day SMA crosses happened in these 2 years
Bollinger Bands
Bollinger Bands® is another popular technical analysis technique. There are three main components to Bollinger Bands graphs – an upper band, a SMA, and a lower band. The upper and lower bands are plotted two standard deviations away from the simple moving average. This can be shown as:

 

Where n is the number of periods and width is the distance from the SMA.
In this project, N is 20 days and the width is 2. Thus, the bands are two standards deviations away from the SMA.
A Bollinger squeeze is when the bands come close together, thereby constricting the moving average. It indicates a low volatility period and is thought to be a possible sign of future increased volatility and trading opportunities. The wider the bands diverge, the likelier the possibility of a decrease in volatility and the greater the chance of exiting a trade. Nevertheless, these conditions are not trading signals as the bands do not tell when the change may take place or which direction price could move. (Woods, 2019)
Numerous investors judge that the closer the price is to the lower band, the more oversold the market, and the closer the price is to the upper band, the more overbought the market. While John Bollinger advised against using Bollinger Bands solely, an overbought market can imply that a trader might consider selling and vice versa. (Hayes, 2020 [1])
The Bollinger Band Value gives a relative position of the current price with regards to the bands and is calculated:
 
Figure 2: The upper and lower Bollinger bands get closer, in other words the standard deviation decreases, in the later half of 2009.
Figure 2: The upper and lower Bollinger bands get closer, in other words the standard deviation decreases, in the later half of 2009.
J. Bollinger suggested using Bollinger Bands with a few non-correlated indicators that provide more direct market signals. In summary, the indicator is designed to discover opportunities that give investors a higher probability of success. 
Momentum
Unlike SMA, momentum is a leading indicator measuring a security's rate-of-change. It compares the current price with the previous price from several periods prior and can be calculated as:
 
Where price is the current price,  is price at time n, and n is the number of periods.
In this project, n is 20 days.
Momentum helps investors determine the strength of a trend and is used by traders to buy stocks in an uptrend and selling shares in a downtrend. Since momentum can indicate a strong trend, investors must understand when they're investing with or against the momentum of a stock or the overall market.
If momentum is positive, it is bullish momentum. A trader could set a buy or sell signal based on if momentum is positive or negative and how it is trending.
When the momentum indicator goes below the zero line and then reverses in an upward direction, it doesn't undeniably mean that the downtrend is over but Figure 3: Momentum spiked after the downward trend reversal near 2009-04
Figure 3: Momentum spiked after the downward trend reversal near 2009-04
Exponential Moving Average
Like SMA, EMA is a technical indicator that produces buy and sell signals based on crossovers and divergences from the historical average. Here, 20 and 50-day EMA were used and a buy signal can be indicated when the 20 crosses above the 50 Day.
Different from SMA, EMA gives a larger weighting to recent data than older data. The steps to calculate the EMA are:
(1) Compute SMA as the initial EMA value
(2) Compute weighting multiplier = 2 / (N+1) where N is the number of days
(3) Compute Current EMA = Price * weighting multiplier + Previous EMA * (1- weighting multiplier)
In Python pseudocode, this can be represented as: 
prices.ewm(com=n).mean()
EMA is very applicable for traders who trade intraday. If an exponential moving average on a daily chart illustrates a strong upward trend, an intraday investor’s strategy might be to trade only on the long side. (Hayes, 2020 [2])Figure 4: The 20-Day EMA crossed over the 50-Day EMA on an uptrend at a later time than the 20-Day SMA crossed over the 5- Day SMA
Figure 4: The 20-Day EMA crossed over the 50-Day EMA on an uptrend at a later time than the 20-Day SMA crossed over the 5- Day SMA
Volatility
Volatility is a measure of the scattering of returns for a given security and can be measured using the standard deviation of the daily returns. Generally, the higher the volatility, it is believed the riskier the security. 
Volatility refers to the level of uncertainty about the size of changes in a security's value. A greater volatility means that a stock’s value could be spread out over a larger range of values. This means that it’s price can change severely over a short period in either direction. A smaller volatility means that a security's value does not oscillate intensely, but changes at a steady pace over time. 
For the project, volatility was calculated as the standard deviation from the daily returns over a 7 Day rolling window which was then multiplied by 2 to more clearly display the information on the graph.
Figure 5: High relative volatility occurred during the large downtrend at the beginning of 2009
Figure 5: High relative volatility occurred during the large downtrend at the beginning of 2009
Part 2: Theoretically Optimal Strategy
Strategy Creation
For the theoretically optimal strategy, the trader knows the exact price of the security in the future, thus he could buy or sell the security at present accordingly to maximize the return.
The strategy has three main phases. First, it calculates if there was a daily increase or decrease. If a day gets a 1, then the subsequent day is higher; if it is -1 then the subsequent day is lower. 
The normalized price difference at a given date will be compared with that of the previous day. If the sign of the normalized price difference changes, it indicates an order should be made. For instance, If the normalized price difference at a given date is -1 and the normalized price difference of the previous day is 1, a long order should be made and vice-versa.
Next, the strategy multiplies the -1 or 1 by 1000 since 1000 shares can be bought or sold at max. Lastly, it calculates the trades allowing for +2000 and -2000 as long as the net holdings are constrained to -1000, 0, and 1000.
This trading strategy is permissible and optimal because of the following assumptions: seeing the future is possible, there is no bankroll limit, and sales are instantaneous and only occur at adjusted closing price. Additionally, our strategy trades at most once per day with only three possible positions; and there is no transaction costs, commission, or impact.
For the project, all trades were made on JP Morgan Chase (JPM). The date range is 01-01-2008 to 12- 31-2009 and the starting value was $100,000.
Benchmark
The benchmark is defined as the performance of a portfolio starting with $100,000 cash, investing in 1000 shares of JPM and holding that position. It is normalized to 1.0 at the start. The goal of the benchmark is to set a checkpoint that should be attained at the least by the trading strategy. 
To implement this, a data frame with 1 row for the start date and end date of transactions was created. On the first day, 100 shares are bought, and since it can be held forever, the decision for the last day is to buy 0 shares of JPM. Figure 6 graphically illustrates the performance of the optimal strategy versus the benchFigure 6: The benchmark stayed relatively flat while the optimal strategy trended upward
Figure 6: The benchmark stayed relatively flat while the optimal strategy trended upward
The following table summarizes the performance of the theoretical optimal strategy portfolio and the benchmark.

Portfolio
Benchmark
Sharpe Ratio
13.3228
0.1569
Cumulative Return
5.7861
0.0123
Standard Deviation of Daily Returns
0.0045
0.0170
Average Daily Return
0.0038
0.0002
Final Value
$678,610
$101,230

References
1. Hayes, A. (2020, August 28). Bollinger Band®. Retrieved October 19, 2020, from https://www.investopedia.com/terms/b/bollingerbands.asp 
2. Hayes, A. (2020, September 10). Exponential Moving Average (EMA). Retrieved October 19, 2020, from https://www.investopedia.com/terms/e/ema.asp 
3. Hayes, A. (2020, September 22). Simple Moving Average (SMA) Definition. Retrieved October 19, 2020, from https://www.investopedia.com/terms/s/sma.asp
4. Staff, I. (2020, August 28). Momentum Indicates Stock Price Strength. Retrieved October 19, 2020, from https://www.investopedia.com/articles/technical/081501.asp
5. Woods, G. (2019, September 26). Trading with the Bollinger Band Squeeze. Retrieved October 19, 2020, from https://www.tradingsetupsreview.com/bollinger-squeeze/
