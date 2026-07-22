# Martingale Report

> Converted from a prior Word draft. Author bylines, course codes, and Drive links removed.

Martingale in American Roulette
Abstract—A martingale is a stochastic process where the expectation of the next value is equal to the present value; in this project, the scenario is gambling. Thus, a gambler’s winning is a martingale if all the games are fair. The main idea behind martingales is that since statistically, you can’t lose every time, you should therefore increase investments in anticipation of a future win.
In Experiment 1, based ON the experiment results calculate the estimated probability of winning $80 within 1000 sequential bets.
The experiment results show within 1000 sequential bets, all of the simulations achieved $80. Thus, the results imply that the probability is 100% because all 1000 simulations of 1000 bets achieved $80; each 1000/1000 = 100%. Repeated martingale sequential bets eliminate the influences from extreme values and always bring winnings back to $80. This happens because each time the bet is lost, the bet amount doubles to cover the previous losses to get back to the previous high. It works because it operates via an “unlimited bank account”.
Figure 1: This graph portrays the first 300 bets of 10 simulations. This figure shows that out of these 10, all reached $80 by at least 200 sequential bets.
Figure 1: This graph portrays the first 300 bets of 10 simulations. This figure shows that out of these 10, all reached $80 by at least 200 sequential bets.
In Experiment 1, what is the estimated expected value of winnings after 1000 sequential bets?
The expected value is the weighted sum of all possible values multiplied by the probability of each value’s occurrence.
For an American roulette wheel with 0 and 00, there are 38 pockets with 18 black. Thus, the expected probability of winning by always betting on black pockets in a single spin is 18/38 (47.7%). Therefore, in a single bet, the expected value would be P(Black) * ($1 If Won) + P(Not Black) * (-$1 if Lost) = 18/28 * 1 + 20/38 * (-1) = -$0.05. The expected value of -$0.05 shows that we would lose over 5 cents on average each time we play.
Since this martingale strategy doubles bets until a win and no upper limit is specified, the resulted sequence of winnings will be composed of many geometric progressions each ending with a major win reversing all loss; and since we quit the game after winning $80, the expected value of winnings is $80. 
Another way to determine the expected value is to repeatedly calculate the current value and use the mean value as the predicted expected value. In this case, we take the mean value of the last bet for each of the 1000 simulations. This is always $80, so the mean value is $80; and, the estimated expected value of winnings after 1000 sequential bets is $80.
In Experiment 1, do the (mean + standard deviation) line and (mean – standard deviation) line reach a maximum value then stabilize? Do the lines converge as the number of sequential bets increases?
The mean + SD, mean, and mean – SD lines stabilize and converge. This occurs because once $80 is reached, the winnings freeze at $80, and the standard deviations approach 0. Thus, the standard deviation reached a maximum and then decreased. This also means that the mean – SD line and mean lines reach a max and that the variability on the mean + SD line decrease. The mean + SD line isn’t at its maximum or minimum compared to its historical values.

Figure 2: Graph of mean + SD (upper red), mean (black), and mean – SD (lower red) of Experiment 1
Figure 2: Graph of mean + SD (upper red), mean (black), and mean – SD (lower red) of Experiment 1

Figure 3: Graph of median + SD (upper red), median (black), and median – SD (lower red) of Experiment 1
Figure 3: Graph of median + SD (upper red), median (black), and median – SD (lower red) of Experiment 1

In Experiment 2, based ON the experiment results calculate the estimated probability of winning $80 within 1000 sequential bets.
Out of 1000 simulations with bankroll specified, only 666 succeeded in winning $80 within 1000 sequential bets. Thus, the corresponding estimated probability is 66.6%. The reduced probability of winning $80 is due to the capital cap which limited chances to reverse previous losses. I would expect the 66.6% to further decrease with more trials because the one bet percent chance of winning was 47.7%
In Experiment 2, what is the estimated expected value of our winnings after 1000 sequential bets?
Since the trials converge either on $80 with estimated probability 66.6% or on -$256 with estimated probability 33.3%, the expected value of winning is 66.6% * 80 + 34.3% * (-$256) = -$34.53. Since I expect the 66.6% chance of winning probability to decrease with more trials, I would also expect the -$34.53 to increase to a larger negative number with more trials.
In Experiment 2, do the (mean + standard deviation) line and (mean – standard deviation) line reach a maximum value then stabilize? Do the lines converge as the number of sequential bets increases?
Figure 4: Graph of mean + SD (upper red), mean (black), and mean – SD (lower red) of Experiment 2
Figure 4: Graph of mean + SD (upper red), mean (black), and mean – SD (lower red) of Experiment 2
 

Figure 5: Graph of median + SD (upper red), median (black), and median – SD (lower red) of Experiment 2
Figure 5: Graph of median + SD (upper red), median (black), and median – SD (lower red) of Experiment 2
