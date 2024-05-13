# Important Questions

## Explain the central limit theorem and give examples of when you can use it in a real-world problem?

The Central Limit Theorem (CLT) is a fundamental concept in statistics that describes the behavior of sample means from a population. It states that as the sample size increases, the distribution of sample means approaches a normal distribution, regardless of the shape of the original population distribution.

In simpler terms, imagine you have a large group of people, each with a different height. If you randomly select smaller groups of people from this population and calculate the average height of each group, you'll find that these average heights will tend to cluster around the true average height of the entire population. As you take larger and larger samples, the distribution of these sample means will become more and more like a bell-shaped curve, known as a normal distribution.

This theorem is incredibly useful in statistics because it allows us to make inferences about population parameters based on sample data. It's widely used in hypothesis testing, confidence interval estimation, and other statistical analyses.

Examples of real-world usage of CLT:

1. The CLT can be used at any company with a large amount of data. Consider companies like Uber/Lyft wants to test whether adding a new feature will increase the booked rides or not using hypothesis testing. So if we have a large number of individual ride X, which in this case is a Bernoulli random variable (since the rider will book a ride or not), we can estimate the statistical properties of the total number of bookings. Understanding and estimating these statistical properties play a significant role in applying hypothesis testing to your data and knowing whether adding a new feature will increase the number of booked riders or not.

2. Test Scores:
Imagine you're conducting a study on the average test scores of students in a particular subject. You collect multiple random samples of test scores from different classrooms. Even if the distribution of test scores in each classroom is not normal, the sampling distribution of the sample means (average test scores) across all classrooms will be approximately normal, according to the CLT. This enables you to estimate the population mean test score and assess the variability of the estimates.

3. Imagine you have a fair six-sided die, and you're interested in understanding the distribution of the sum of the numbers rolled when you roll the die multiple times. Each roll of the die is an independent random event, and the outcome can be any number from 1 to 6 with equal probability.

Now, if you were to roll the die only once, the possible outcomes and their probabilities would be:

Rolling a 1: Probability = 1/6
Rolling a 2: Probability = 1/6
Rolling a 3: Probability = 1/6
Rolling a 4: Probability = 1/6
Rolling a 5: Probability = 1/6
Rolling a 6: Probability = 1/6

Let's say you roll the die 100 times and calculate the average of the numbers rolled each time. According to the central limit theorem, as the number of rolls increases, the distribution of these sample means will tend to approach a normal distribution, regardless of the original distribution of individual outcomes.

In this case, since each roll has a uniform distribution with a mean of 3.5 (the average of the numbers 1 to 6), the distribution of the sample means will also have a mean of 3.5.

As you continue to roll the die and calculate the sample means, you'll find that the distribution of these means increasingly resembles a bell-shaped normal distribution, with the mean centered around 3.5.

This demonstrates the central limit theorem in action: even though the individual outcomes of rolling a die are not normally distributed, the distribution of sample means tends towards normality as the sample size increases.



## 2. What is A/B testing ? 
- A/B testing, also known as split testing, is a method used to compare two versions of a product or service to determine which one performs better.
- It involves dividing users into two groups, A and B, and exposing each group to a different version of the product or service (the original version for group A and a modified version for group B).
- The performance of each version is then measured and compared using predefined metrics, such as click-through rates, conversion rates, or revenue generated.
- A/B testing helps in making data-driven decisions by providing insights into which version yields better results, thereby optimizing user experience and maximizing desired outcomes.
- It is commonly used in marketing, web design, product development, and other fields to evaluate changes and improvements objectively.


A/B testing helps us to determine whether a change in something will cause a change in performance significantly or not. So in other words you aim to statistically estimate the impact of a given change within your digital product (for example).
Say you have identified a flaw in your game you aren't sure of how to fix it. We use a/b testing. Which is like testing different versions of the digital product at the same time. seeing whch version gets the best response from the players. It's called AB because the simplest test compared two versions. 

A - A version given to half of the users
B - B version given to other half of the users

These tests can help you repair any game content that your are unsure about. 

 ![Alt text](<images/image.png>)

![Alt text](<images/image2.png>)

![Alt text](<images/image3.png>)

![Alt text](<images/image4.png>)


## 3. Describe briefly the hypothesis testing and p-value in layman’s term? And give a practical application for them ?
In Layman's terms:

Hypothesis test is where you have a current state (null hypothesis) and an alternative state (alternative hypothesis). You assess the results of both of the states and see some differences. You want to decide whether the difference is due to the alternative approach or not.

You use the p-value to decide this, where the p-value is the likelihood of getting the same results the alternative approach achieved if you keep using the existing approach. It's the probability to find the result in the gaussian distribution of the results you may get from the existing approach.

The rule of thumb is to reject the null hypothesis if the p-value < 0.05, which means that the probability to get these results from the existing approach is <95%. But this % changes according to task and domain.

To explain the hypothesis testing in Layman's term with an example, suppose we have two drugs A and B, and we want to determine whether these two drugs are the same or different. This idea of trying to determine whether the drugs are the same or different is called hypothesis testing. The null hypothesis is that the drugs are the same, and the p-value helps us decide whether we should reject the null hypothesis or not.

p-values are numbers between 0 and 1, and in this particular case, it helps us to quantify how confident we should be to conclude that drug A is different from drug B. The closer the p-value is to 0, the more confident we are that the drugs A and B are different.

https://www.youtube.com/watch?v=0oc49DyA3hU&ab_channel=StatQuestwithJoshStarmer


## 4. Given a left-skewed distribution that has a median of 60, what conclusions can we draw about the mean and the mode of the data?

 ![Alt text](<images/distribution.jpg>)

 Left skewed distribution means the tail of the distribution is to the left and the tip is to the right. So the mean which tends to be near outliers (very large or small values) will be shifted towards the left or in other words, towards the tail.

While the mode (which represents the most repeated value) will be near the tip and the median is the middle element independent of the distribution skewness, therefore it will be smaller than the mode and more than the mean.

Mean < 60 Mode > 60


## 5. What is the meaning of selection bias and how to avoid it?

Sampling bias is the phenomenon that occurs when a research study design fails to collect a representative sample of a target population. This typically occurs because the selection criteria for respondents failed to capture a wide enough sampling frame to represent all viewpoints.

The cause of sampling bias almost always owes to one of two conditions.

Poor methodology: In most cases, non-representative samples pop up when researchers set improper parameters for survey research. The most accurate and repeatable sampling method is simple random sampling where a large number of respondents are chosen at random. When researchers stray from random sampling (also called probability sampling), they risk injecting their own selection bias into recruiting respondents.

Poor execution: Sometimes data researchers craft scientifically sound sampling methods, but their work is undermined when field workers cut corners. By reverting to convenience sampling (where the only people studied are those who are easy to reach) or giving up on reaching non-responders, a field worker can jeopardize the careful methodology set up by data scientists.

The best way to avoid sampling bias is to stick to probability-based sampling methods. These include simple random sampling, systematic sampling, cluster sampling, and stratified sampling. In these methodologies, respondents are only chosen through processes of random selection—even if they are sometimes sorted into demographic groups along the way.


Foe example : 

Imagine you're making a recipe and you want to know how much people like it. So, you decide to ask your family members if they enjoyed it. But here's the catch: you only ask the family members who are nearby, like your siblings who live with you. You're missing out on opinions from your cousins who live far away or your grandparents who don't visit often.

That's kinda like sampling bias. It's when you're not getting a fair mix of opinions because your method of asking people is flawed. Maybe you're only asking people who are easy to reach or who you know will say yes. This means you're not hearing from everyone who might have something to say about your recipe.

So, if you want to avoid this bias, you need to be more fair in how you choose who to ask. You could randomly select people from your whole family, or maybe ask every third person in a list. This way, you're more likely to get a mix of opinions from different kinds of people, and your results will be more accurate.


## 6. Explain the long-tailed distribution and provide three examples of relevant phenomena that have long tails. Why are they important in classification and regression problems?
Explanation of Long-Tailed Distribution: A long-tailed distribution is one where the frequencies of certain values decrease gradually as you move away from the center, rather than dropping off quickly. It's like a distribution with a long tail that stretches out towards the ends.

Examples of Phenomena with Long Tails:
* Frequencies of languages spoken: There are a few languages spoken by a large number of people (like English, Spanish, Mandarin), but there are also thousands of languages spoken by smaller populations.

* Population of cities: There are a few megacities with huge populations, but there are many more smaller cities with fewer inhabitants.

* Pageviews of articles: Some articles on the internet get millions of views, while many others get only a handful.

Importance in Classification and Regression Problems:

    * Outliers: In classification and regression, outliers (extreme values) can significantly impact the model's performance. Long-tailed distributions often have many outliers, so it's crucial to handle them appropriately.

    * Assumptions of Normality: Many machine learning algorithms assume that the data follows a normal distribution. Long-tailed distributions violate this assumption, so it's important to choose models and techniques that are robust to such distributions.

    * Data Preprocessing: Dealing with long-tailed data might require special preprocessing techniques to handle the imbalance between the frequent and infrequent values. For example, in classification, you might use techniques like oversampling or undersampling to balance the classes.

Understanding and accounting for long-tailed distributions is essential in classification and regression problems to ensure that the models are robust and accurate, especially in the presence of outliers and skewed data.



## 7. What is the meaning of KPI in statistics
KPI stands for key performance indicator, a quantifiable measure of performance over time for a specific objective. KPIs provide targets for teams to shoot for, milestones to gauge progress, and insights that help people across the organization make better decisions. From finance and HR to marketing and sales, key performance indicators help every area of the business move forward at the strategic level.

KPIs are an important way to ensure your teams are supporting the overall goals of the organization. Here are some of the biggest reasons why you need key performance indicators.

* Keep your teams aligned: Whether measuring project success or employee performance, KPIs keep teams moving in the same direction.
* Provide a health check: Key performance indicators give you a realistic look at the health of your organization, from risk factors to financial indicators.
* Make adjustments: KPIs help you clearly see your successes and failures so you can do more of what’s working, and less of what’s not.
* Hold your teams accountable: Make sure everyone provides value with key performance indicators that help employees track their progress and help managers move things along.

Types of KPIs Key performance indicators come in many flavors. While some are used to measure monthly progress against a goal, others have a longer-term focus. The one thing all KPIs have in common is that they’re tied to strategic goals. Here’s an overview of some of the most common types of KPIs.

* Strategic: These big-picture key performance indicators monitor organizational goals. Executives typically look to one or two strategic KPIs to find out how the organization is doing at any given time. Examples include return on investment, revenue and market share.
* Operational: These KPIs typically measure performance in a shorter time frame, and are focused on organizational processes and efficiencies. Some examples include sales by region, average monthly transportation costs and cost per acquisition (CPA).
* Functional Unit: Many key performance indicators are tied to specific functions, such finance or IT. While IT might track time to resolution or average uptime, finance KPIs track gross profit margin or return on assets. These functional KPIs can also be classified as strategic or operational.
* Leading vs Lagging: Regardless of the type of key performance indicator you define, you should know the difference between leading indicators and lagging indicators. While leading KPIs can help predict outcomes, lagging KPIs track what has already happened. Organizations use a mix of both to ensure they’re tracking what’s most important.

 ![Alt text](<images/image5.png>)


 # Statistics 
Statistics is a type of mathematical analysis that employs quantified models and representations to analyse a set of experimental data or real-world studies. The main benefit of statistics is that information is presented in an easy-to-understand format.

Data processing is the most important aspect of any Data Science plan. When we speak about gaining insights from data, we’re basically talking about exploring the chances. In Data Science, these possibilities are referred to as Statistical Analysis.

### Importance of Statistics

1) Using various statistical tests, determine the relevance of features.

2) To avoid the risk of duplicate features, find the relationship between features.

3) Putting the features into the proper format.

4) Data normalization and scaling This step also entails determining the distribution of data as well as the nature of data.

5) Taking the data for further processing and making the necessary modifications.

6) Determine the best mathematical approach/model after processing the data.

7) After the data are acquired, they are checked against the various accuracy measuring scales.





### Q8: Say you flip a coin 10 times and observe only one head. What would be the null hypothesis and p-value for testing whether the coin is fair or not?
Answer:

The null hypothesis is that the coin is fair, and the alternative hypothesis is that the coin is biased. The p-value is the probability of observing the results obtained given that the null hypothesis is true, in this case, the coin is fair.

In total for 10 flips of a coin, there are 2^10 = 1024 possible outcomes and in only 10 of them are there 9 tails and one head.

Hence, the exact probability of the given result is the p-value, which is 10/1024 = 0.0098. Therefore, with a significance level set, for example, at 0.05, we can reject the null hypothesis.


### Q9: You are testing hundreds of hypotheses, each with a t-test. What considerations would you take into account when doing this?

When conducting hundreds of hypothesis tests, such as t-tests, several considerations need to be taken into account to ensure the validity of the results. One crucial factor is the increased likelihood of obtaining a significant result purely by chance, which can lead to an inflated Type I error rate.

Type I error occurs when we reject the null hypothesis incorrectly, i.e., we conclude there is a significant difference when there isn't one. As the number of tests increases, the probability of observing at least one significant result due to chance alone rises, which can inflate the overall Type I error rate across all tests.

To address this issue, one common approach is to adjust the significance level for each individual test to maintain an appropriate overall Type I error rate. The Bonferroni correction is a widely used method for this purpose. It involves dividing the desired significance level (usually denoted as alpha, typically set at 0.05) by the number of tests conducted (K), resulting in a new, more stringent significance level known as alpha star.

For example, if conducting 100 tests and using a standard significance level of 0.05, the Bonferroni correction would yield an alpha star of 0.05/100 = 0.0005. This adjusted significance level ensures that the overall probability of committing a Type I error across all tests remains at an acceptable level.

By employing the Bonferroni correction or similar adjustments, researchers can mitigate the risk of spurious findings and maintain the integrity of their hypothesis testing procedure, especially when conducting multiple comparisons.


### Q10: What general conditions must be satisfied for the central limit theorem to hold?
In order to apply the central limit theorem, there are four conditions that must be met:

1.** Randomization:** The data must be sampled randomly such that every member in a population has an equal probability of being selected to be in the sample.

Independence: The sample values must be independent of each other.

The 10% Condition: When the sample is drawn without replacement, the sample size should be no larger than 10% of the population.

Large Sample Condition: The sample size needs to be sufficiently large.


### Q11: What is skewness discuss two methods to measure it?
Skewness refers to a distortion or asymmetry that deviates from the symmetrical bell curve, or normal distribution, in a set of data. If the curve is shifted to the left or to the right, it is said to be skewed.Skewness can be quantified as a representation of the extent to which a given distribution varies from a normal distribution. There are two main types of skewness negative skew which refers to a longer or fatter tail on the left side of the distribution, while positive skew refers to a longer or fatter tail on the right. These two skews refer to the direction or weight of the distribution.

The mean of positively skewed data will be greater than the median. In a negatively skewed distribution, the exact opposite is the case: the mean of negatively skewed data will be less than the median. If the data graphs symmetrically, the distribution has zero skewness, regardless of how long or fat the tails are.

There are several ways to measure skewness. Pearson’s first and second coefficients of skewness are two common methods. Pearson’s first coefficient of skewness, or Pearson mode skewness, subtracts the mode from the mean and divides the difference by the standard deviation. Pearson’s second coefficient of skewness, or Pearson median skewness, subtracts the median from the mean, multiplies the difference by three, and divides the product by the standard deviation.

 ![Alt text](<images/image18.png>)


1. Pearson's First Coefficient of Skewness (Pearson Mode Skewness):

* This measure calculates skewness by comparing the mode (the most frequent value) with the mean and standard deviation.
The formula for Pearson's first coefficient of skewness is:

 ![Alt text](<images/image20.png>)

* It measures the skewness based on the distance between the mean and mode, normalized by the standard deviation.
* If the mode is less than the mean, the distribution is negatively skewed (skewed to the left). If the mode is greater than the mean, the distribution is positively skewed (skewed to the right).

2. Pearson's Second Coefficient of Skewness (Pearson Median Skewness):
* This measure compares the median with the mean and standard deviation to assess skewness.
* The formula for Pearson's second coefficient of skewness is:
  
 ![Alt text](<images/image21.png>)
 
* It measures skewness by quantifying the distance between the mean and median, normalized by the standard deviation.
* Similar to the first coefficient, if the median is less than the mean, the distribution is negatively skewed; 
* if the median is greater than the mean, the distribution is positively skewed.


### Q12: You sample from a uniform distribution [0, d] n times. What is your best estimate of d?
Intuitively it is the maximum of the sample points. Here's the mathematical proof is in the figure below:
If you sample from a uniform distribution over the interval [0,d] n times, one way to estimate the maximum value d is by using the sample maximum.
The sample maximum is the largest value observed in the sample. In the case of a uniform distribution, the maximum observed value in a sufficiently large sample is a good estimator for the upper bound of the distribution, which is d.
So, your best estimate of d would be the maximum value observed in your sample.



### Q13: Discuss the Chi-square, ANOVA, and t-test
* Chi-square Test:

Purpose: It's used to assess the relationship between two categorical variables in a dataset.
Example: The food delivery company example you provided is suitable. For instance, it can determine if there's a significant relationship between a person's gender and their food preferences.
Interpretation: It helps identify whether the observed distribution of categorical variables differs significantly from the expected distribution.

* Analysis of Variance (ANOVA):

Purpose: ANOVA compares the means of two or more groups to assess if there are statistically significant differences.
Example: Imagine a study comparing the effectiveness of three different teaching methods on exam scores. ANOVA would determine if there's a significant difference in mean exam scores between the three groups.
Interpretation: ANOVA assesses whether the variation in scores between groups is larger than the variation within groups, indicating if at least one group mean significantly differs from the others.

* t-test:

Purpose: T-tests determine if there's a significant difference between the means of two groups.
Types:
One-sample t-test: It compares the mean of a sample to a known value or population mean.
Two-sample t-test: It compares the means of two independent samples.
Paired t-test: It compares means of two related samples (e.g., before and after measurements on the same group).
Example: Suppose you're testing a new drug and want to determine if there's a significant difference in effectiveness between patients who received the drug and those who received a placebo.
Interpretation: A significant result suggests that the difference in means observed in the sample is unlikely to have occurred by chance, indicating a true difference between the populations.
Each of these tests serves different purposes and is applicable in various scenarios, allowing researchers to gain insights into different aspects of their data.


### Q14: Say you have two subsets of a dataset for which you know their means and standard deviations. How do you calculate the blended mean and standard deviation of the total dataset? Can you extend it to K subsets?



### Q15: What is the relationship between the significance level and the confidence level in Statistics?
In statistics, the significance level (often denoted as α) and the confidence level are related concepts but represent different aspects of hypothesis testing and confidence intervals, respectively.

* Significance Level (α):
The significance level is the probability of rejecting the null hypothesis when it is actually true.
It represents the threshold for deciding whether an observed result is statistically significant.
Commonly used significance levels include 0.05, 0.01, or 0.10, indicating a 5%, 1%, or 10% chance of making a Type I error (incorrectly rejecting a true null hypothesis).

* Confidence Level:
The confidence level is the probability that a confidence interval contains the true population parameter.
It represents the reliability or certainty of the estimation process.
Commonly used confidence levels include 90%, 95%, or 99%, indicating the percentage of confidence intervals that would contain the true parameter if the estimation process were repeated many times.

* Relationship:
The significance level and confidence level are complementary to each other.
A 95% confidence level corresponds to a significance level of α = 0.05.
Similarly, a 99% confidence level corresponds to a significance level of α = 0.01.
The choice of significance level influences the critical value used in hypothesis testing, while the confidence level determines the width of the confidence interval.

In summary, while the significance level controls the risk of Type I errors in hypothesis testing, the confidence level quantifies the precision and reliability of estimation in constructing confidence intervals. They are both fundamental concepts in statistical inference and are used to make decisions and draw conclusions from data analysis.


### Q16: What is the Law of Large Numbers in statistics and how it can be used in data science ?
Answer: The law of large numbers states that as the number of trials in a random experiment increases, the average of the results obtained from the experiment approaches the expected value. In statistics, it's used to describe the relationship between sample size and the accuracy of statistical estimates.

In data science, the law of large numbers is used to understand the behavior of random variables over many trials. It's often applied in areas such as predictive modeling, risk assessment, and quality control to ensure that data-driven decisions are based on a robust and accurate representation of the underlying patterns in the data.

The law of large numbers helps to guarantee that the average of the results from a large number of independent and identically distributed trials will converge to the expected value, providing a foundation for statistical inference and hypothesis testing.

### Q17: What is the difference between a confidence interval and a prediction interval, and how do you calculate them?

A confidence interval is a range of values that is likely to contain the true value of a population parameter with a certain level of confidence. It is used to estimate the precision or accuracy of a sample statistic, such as a mean or a proportion, based on a sample from a larger population.

For example, if we want to estimate the average height of all adults in a certain region, we can take a random sample of individuals from that region and calculate the sample mean height. Then we can construct a confidence interval for the true population mean height, based on the sample mean and the sample size, with a certain level of confidence, such as 95%. This means that if we repeat the sampling process many times, 95% of the resulting intervals will contain the true population mean height.

The formula for a confidence interval is: confidence interval = sample statistic +/- margin of error

The margin of error depends on the sample size, the standard deviation of the population (or the sample, if the population standard deviation is unknown), and the desired level of confidence. For example, if the sample size is larger or the standard deviation is smaller, the margin of error will be smaller, resulting in a narrower confidence interval.

A prediction interval is a range of values that is likely to contain a future observation or outcome with a certain level of confidence. It is used to estimate the uncertainty or variability of a future value based on a statistical model and the observed data.

For example, if we have a regression model that predicts the sales of a product based on its price and advertising budget, we can use a prediction interval to estimate the range of possible sales for a new product with a certain price and advertising budget, with a certain level of confidence, such as 95%. This means that if we repeat the prediction process many times, 95% of the resulting intervals will contain the true sales value.

The formula for a prediction interval is: prediction interval = point estimate +/- margin of error

The point estimate is the predicted value of the outcome variable based on the model and the input variables. The margin of error depends on the residual standard deviation of the model, which measures the variability of the observed data around the predicted values, and the desired level of confidence. For example, if the residual standard deviation is larger or the level of confidence is higher, the margin of error will be larger, resulting in a wider prediction interval.

 ![Alt text](<images/image19.png>)



### Coursera Course 
The course have five modules 
1. Introduction and Descriptive statistics 
2. Data Visualization
3. Introduction to probability distribution
4. Hypothesis Testing 
5. Regression Analysis

 ![Alt text](<images/image6.png>)

 ![Alt text](<images/image7.png>)


 ![Alt text](<images/image8.png>)


 ### Measures of central tendency 
Certainly! Here are definitions of mean, mode, and median with examples:

* Mean: The mean, also known as the average, is a measure of central tendency that represents the typical value of a set of numbers. It is calculated by adding up all the numbers in a dataset and then dividing by the total number of values.Example: Consider the following set of test scores: 85, 90, 75, 80, and 95. To find the mean, add up all the scores and divide by the total number of scores:
Mean= 85+90+75+80+95/5
So, the mean test score is 85.

* Mode: The mode is the value that appears most frequently in a dataset. It represents the value that occurs with the highest frequency.Example: In the set of test scores from the previous example: 85, 90, 75, 80, 95, the mode is 85 because it appears more frequently than any other score.

* Median: The median is the middle value in a dataset when the values are arranged in ascending or descending order. If there is an even number of values, the median is the average of the two middle values.Example: Consider the set of test scores: 85, 90, 75, 80, 95. To find the median, arrange the scores in ascending order: 75, 80, 85, 90, 95. Since there are five scores, the median is the middle value, which is 85.
These measures of central tendency help summarize and understand the distribution of values in a dataset.


## Measure of Dispersion

* Standard deviation: Standard deviation is a measure of the amount of variation or dispersion in a set of values. It indicates how spread out the values are from the mean. A low standard deviation means that the values tend to be close to the mean, while a high standard deviation means that the values are spread out over a wider range.

Definition: Standard deviation is calculated by taking the square root of the average of the squared differences between each value and the mean of the dataset.

Example: Let's consider a set of exam scores: 70, 75, 80, 85, and 90. First, we find the mean of the scores:

Mean= 70+75+80+85+90/5= 400/5 = 80
​
Now, we calculate the squared differences between each score and the mean:
(70−80)2 =100

 ![Alt text](<images/image9.png>)


standard deviations are more useful than just simple average/mean values. 

* Measure of Variability: While the mean gives us a measure of central tendency, the standard deviation indicates the spread or variability of the data points around the mean. It helps us understand how much individual data points deviate from the average. For example, in finance, a higher standard deviation of returns indicates greater volatility in an investment, which might imply higher risk.

* Comparing Data Sets: Standard deviation allows us to compare the dispersion of data sets. For instance, if we're comparing the performance of two products based on customer ratings, knowing the standard deviation helps determine which product has more consistent ratings. A smaller standard deviation suggests that most customers rate the product similarly, while a larger standard deviation indicates more diverse opinions.

* Quality Control: In manufacturing and quality control processes, standard deviation is used to monitor consistency and detect deviations from desired standards. For example, in a production line, a high standard deviation in product dimensions might indicate a problem with the manufacturing process, prompting corrective action.

* Risk Assessment: Standard deviation is widely used in risk assessment and probability analysis. For instance, in insurance, actuaries use standard deviation to assess the variability of claims experience. Higher standard deviation implies higher risk, which needs to be accounted for in pricing policies.

* Experimental Design: In scientific research, standard deviation helps assess the reliability of experimental results. It indicates the consistency or precision of measurements. A smaller standard deviation suggests less variability and greater reliability in the results.


## Descriptive statistics
A descriptive statistic (in the count noun sense) is a summary statistic that quantitatively describes or summarizes features from a collection of information,[1] while descriptive statistics (in the mass noun sense) is the process of using and analysing those statistics. Descriptive statistics is distinguished from inferential statistics (or inductive statistics) by its aim to summarize a sample, rather than use the data to learn about the population that the sample of data is thought to represent.


### Visualization 

1. Comparing items with few categorise you can use bar charts or column charts
2. If you are comparing behaviors over time if you have time period running for several months you can use line charts
3. If the time periods are not many we can use 

 ![Alt text](<images/image10.png>)
![Alt text](<images/image11.png>)
![Alt text](<images/image12.png>)


## Random Numbers and Probability Distributions

**Probability**:
Probability is a measure of the likelihood or chance of an event occurring.
It quantifies uncertainty and helps us understand the likelihood of different outcomes.
Probability is typically represented as a number between 0 and 1, where 0 means the event is impossible and 1 means the event is certain.
For example, when flipping a fair coin, the probability of getting heads is 0.5 because there are two equally likely outcomes (heads or tails).

**Random Variables**:
A random variable is a variable that can take on different values as a result of a random process or experiment.
It represents the outcomes of a random phenomenon in numerical form.
Random variables can be discrete or continuous:
* Discrete random variables can only take on a finite or countably infinite number of distinct values. For example, the number of heads obtained when flipping a coin multiple times.
* Continuous random variables can take on any value within a certain range. For example, the height of a person.
Random variables are often denoted by letters such as 

**Probability Distribution**:
A probability distribution describes the likelihood of each possible outcome of a random variable.
It assigns probabilities to different values of the random variable.
Probability distributions can be discrete or continuous, depending on the type of random variable:
Discrete probability distributions are used for discrete random variables and are represented by probability mass functions (PMFs).
Continuous probability distributions are used for continuous random variables and are represented by probability density functions (PDFs).
Examples of probability distributions include the binomial distribution, normal distribution, and Poisson distribution.
Probability distributions provide a mathematical framework for analyzing uncertainty and making predictions about the outcomes of random processes.

**Sequence of Random Variables** 
Sequence: It's just a list of things that follow a particular order. In this case, it's a list of random variables.
Random Variables: These are variables whose values depend on outcomes of a random phenomenon. For instance, if you roll a fair six-sided die, the outcome (the number rolled) is a random variable.

So, when we say "sequence of random variables," we're talking about a series of these unpredictable outcomes. Each random variable in the sequence could represent something different: maybe it's the outcome of rolling a die multiple times, or the result of flipping a coin repeatedly, or any other random process you can think of.

![Alt text](<images/Screenshot 2024-05-09 at 11.09.42.png>)

**Convergence of random Variables**
Convergence of random variables refers to how the behavior of a sequence of random numbers changes as we collect more and more of them. It's like watching a pattern emerge as we gather more data points.

![Alt text](<images/Screenshot 2024-05-09 at 11.49.27.png>)

## state of hypothesis 

### Statistical Hypothesis Testing:

Statistical hypothesis testing is a method used to make decisions or inferences about a population parameter based on sample data. It involves comparing the observed data to what would be expected under a certain hypothesis or assumption about the population.

Here's how it generally works:

**Formulating Hypotheses**:
In hypothesis testing, we start by stating two competing hypotheses: the null hypothesis (H0) and the alternative hypothesis (H1).
The null hypothesis typically represents a default assumption or a statement of no effect or no difference.
The alternative hypothesis represents what we want to test or what we believe to be true.

**Collecting Data**:
Next, we collect sample data from the population of interest.
Analyzing Data:We then analyze the sample data to determine whether there is enough evidence to reject the null hypothesis in favor of the alternative hypothesis.

**Making a Decision**:
Based on the analysis, we make a decision to either reject the null hypothesis or fail to reject it.
If the evidence is strong enough, we reject the null hypothesis in favor of the alternative hypothesis. If not, we fail to reject the null hypothesis.

**Interpreting Results**:
Finally, we interpret the results of the hypothesis test and draw conclusions about the population parameter of interest.
Overall, hypothesis testing is a systematic way to evaluate claims or hypotheses about population parameters using sample data. It helps researchers make informed decisions based on evidence and data analysis.

**Interpretation**:
If p-value < alpha: Reject the null hypothesis.
If p-value >= alpha: Fail to reject the null hypothesis.

**Steps**:
Formulate null and alternative hypotheses.
Choose a significance level (alpha).
Select an appropriate test statistic and distribution.
Collect data and calculate the test statistic.
Determine the p-value.
Make a decision: reject or fail to reject the null hypothesis based on the p-value and significance level.

### Z test or T test

If the population standard deviation is known then we should perform z test,
when the standard deviation is not known then we should use t-test

![Alt text](<images/image14.png>)

![Alt text](<images/image15.png>)


### Levene Test 

Levene test is an inferential statistic to assess the equality of variances.
![Alt text](<images/image16.png>)



### ANOVA 
Comparing means of more than two groups

ANOVA stands for Analysis of Variance. It's a statistical method used to compare means among two or more groups to determine whether there are statistically significant differences between them. ANOVA tests the null hypothesis that there are no differences in means across the groups against the alternative hypothesis that at least one group mean is different from the others.

ANOVA works by partitioning the total variance observed in the data into different sources: the variance due to differences between groups and the variance due to differences within groups. It then compares the ratio of these two variances to assess whether the differences between groups are larger than what would be expected by chance.

There are different types of ANOVA, including:

One-Way ANOVA: This is used when there is only one categorical independent variable (factor) with two or more levels, and the dependent variable is continuous. It compares the means across the levels of the independent variable.
Two-Way ANOVA: This is used when there are two independent variables (factors) and one dependent variable. It examines the main effects of each independent variable as well as the interaction effect between them.
Repeated Measures ANOVA: This is used when measurements are taken on the same subjects under different conditions or at different times. It compares the means of within-subject factors while accounting for the correlation between repeated measures.
ANOVA provides an F-statistic and associated p-value to determine whether the observed differences between groups are statistically significant. If the p-value is below a chosen significance level (often 0.05), the null hypothesis is rejected, indicating that there are significant differences between at least two of the group means.

ANOVA is widely used in various fields such as experimental research, clinical trials, social sciences, and quality control to compare means across multiple groups and understand the sources of variation in data.

### correlation Tests





## Importance of Statistics in Machine Learning
Statistics plays a crucial role in machine learning. The key points highlighting the importance of statistics in machine learning are:

Data Modeling
Inference and Estimation
Evaluation and Validation
Feature Selection and Dimensionality Reduction
Regularization and Model Selection
Experimental Design
Causal Inference
Uncertainty Quantification
Fairness and Bias Detection
Theoretical Foundations

* Data Modeling: Statistical techniques like probability distributions, hypothesis testing, and regression analysis are used to model data patterns and relationships.
* Inference and Estimation: Many machine learning algorithms are based on statistical inference and estimation methods, such as maximum likelihood estimation and Bayesian inference.
* Evaluation and Validation: Statistical methods like cross-validation, hypothesis testing, and confidence intervals are used to evaluate model performance and generalization ability.
* Feature Selection and Dimensionality Reduction: Techniques like correlation analysis, principal component analysis (PCA), and factor analysis are used for feature selection and dimensionality reduction.
* Regularization and Model Selection: Regularization methods like L1 (Lasso) and L2 (Ridge) regularization, derived from statistical principles, help in model selection and preventing overfitting.
* Experimental Design: Concepts like randomization, blocking, and factorial designs from statistics are used to design and analyze experiments for machine learning.
* Causal Inference: Statistical methods like propensity score matching, instrumental variables, and structural equation modeling are used for understanding cause-and-effect relationships in data.
* Uncertainty Quantification: Techniques like bootstrapping, Bayesian methods, and conformal prediction are used to quantify the uncertainty associated with machine learning models and their predictions.
* Fairness and Bias Detection: Statistical methods are employed to detect and mitigate biases in machine learning models, ensuring fairness and preventing discrimination.
* Theoretical Foundations: Statistics provides theoretical foundations for machine learning algorithms, ensuring principled and reliable models with interpretable behavior.

