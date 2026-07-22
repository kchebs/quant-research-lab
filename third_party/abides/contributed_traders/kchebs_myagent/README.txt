kmc7_myagent.py is a trading agent designed to run in the Agent-Based Interactive Discrete Event Simulation environment. It is based on the framework provided in "Simple Trading Agent" and should not use leverage.

The agent process is as follows:

1. When the agent wakes, it cancels all orders and retrieves market prices.
2. It then compares the past mid-prices and places a buy action status if the first window mid-price exp mean is greater than the second window mid-price exp mean. It places a sell action status for the opposite condition and does nothing if equal.
3. Next, it calculates the mean price to buy and sell shares which is used to prevent being tricked by other agents placing single orders.
4. Then, the agent calculates the volume to purchase using this price and its available cash. It then recalculates the price based on this volume and on the standard deviation of bid-ask spreads for the current day. This new price is used in the final bid/ask. If short the agent tends to make higher bids and if long, lower asks. This is to reduce inventory exposure but to also maximize profit via high bid-ask spreads.

Yet, an order is only placed if the inventory exposure reduction strategy matches the momentum strategy. In other words, the bid volume from the inventory strategy must greater than 0 and the momentum strategy must give a buy status.

5. Then, when an order is placed the agent waits for acceptance from the broker to prevent too many orders being placed and the agent from being overleveraged. This is also where the count of trades occurs. The agent won't do anything on next wake up but will proceed the following wake up. This is so that the other side of the order has a chance to execute.
6. Lastly, the agent gets rid of shares if it is 5 minutes from the end of the day.

Ref:
https://arxiv.org/pdf/1904.12066.pdf
https://github.com/abides-sim/abides/wiki