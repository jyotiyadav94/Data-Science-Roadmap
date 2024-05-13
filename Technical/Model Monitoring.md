# How to Monitor Your Models in Production

Link: [How to Monitor Your Models in Production - Neptune AI Blog](https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide)

## Machine Learning Project Lifecycle

### Key Points Covered:
- Why deployment is not the final step
- Importance of owning and monitoring models in production
- What to monitor in production and how
- Different monitoring and observability platforms
- Logging and alerting
- Challenges and best practices for monitoring models in production

---

| Production Challenge       | Key Questions | Solutions                                                                                                                |
|----------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------|
| Data Drift / Data distribution changes | Why are there sudden changes in the values of my features? | Data drift occurs if the statistical properties of the input data change over time, despite the relationship between input features and churn remaining the same. Detecting data drift involves monitoring changes in the statistical properties of the input data (e.g., income, call duration) over time and comparing them to a baseline or reference distribution. Significant deviations from the baseline distributions could indicate data drift. Solutions include using statistical checks to detect data drift and periodically updating the baseline dataset to reflect changes in the underlying data distribution, especially if the data generating process is expected to evolve over time. |
| Model ownership in production | Who owns the model in production? The DevOps team? Engineers? Data Scientists? | Model ownership in production is a shared responsibility.                                                                     |
| Training-serving skew       | Why is the model giving poor results in production despite our rigorous testing and validation attempts during development? | Training-serving skew refers to a situation where there is a mismatch between the conditions under which a model is trained and the conditions under which it is used in production. Ensure your production data is not vastly different from your training data, and your production and training data are processed the same way. |
| Concept drift               | Why was my model performing well in production and suddenly the performance dipped over time? | Concept drift occurs if the factors influencing customer churn change over time. Detecting concept drift involves monitoring the performance of the churn prediction model over time. If the model's accuracy or other performance metrics start to degrade, despite no changes in the model or its input data, it could indicate concept drift. Solutions include retraining the model on new data or developing another model on new data. |
| Black box models            | How can I interpret and explain my model’s predictions in line with the business objective and to relevant stakeholders? | View segments of model predictions for explainability.                                                                     |
| Concerted adversaries (security attack) | How can I ensure the security of my model? Is my model being attacked? | Use unsupervised learning methods for outlier detection, including statistical checks, to protect your system from security threats. |
| Model readiness             | How will I compare results from newer version(s) of my model against the in-production version(s)? | Use shadow testing for testing challenger (newly trained) model vs champion model (model currently in production).               |
| Pipeline health issues      | Why does my training pipeline fail when executed? Why does a retraining job take so long to run? | Use logs to audit errors and alerts to inform the service owner.                                                           |
| Underperforming system     | Why is the latency of my predictive service very high? Why am I getting vastly varying latencies for my different models? | Use logs to audit the various services for those that are not meeting required SLAs.                                        |
| Cases of extreme events (outliers) | How will I be able to track the effect and performance of my model in extreme and unplanned situations? | Understand it is an instantaneous or temporary drift before taking action.                                                 |
| Data quality issues        | How can I ensure the production data is being processed in the same way as the training data was? | Write a data integrity test and perform data quality checks.                                                                |


---

## Goal of Monitoring Your Models in Production:
- To detect problems with your model and the system serving your model in production before they start to generate negative business value,
- To take action by triaging and troubleshooting models in production or the inputs and systems that enable them,
- To ensure their predictions and results can be explained and reported,
- To ensure the model’s prediction process is transparent to relevant stakeholders for proper governance,
- Finally, to provide a path for maintaining and improving the model in production.

---

## Functional Monitoring

![Alt text](<Images/Functional-Monitoring.webp>)

## Operational Monitoring
![Alt text](<Images/Operational-Monitoring.webp>)

## Open Source Tools

### Evidently AI

- Evidently AI is an open-source tool that helps detect data drift, concept drift, and target drift.

#### Data Drift:
- Data drift is a change in the statistical properties and characteristics of the input data.
- Evidently AI helps compare distributions of key features over time to detect data drift.

Data drift in this scenario would occur if the statistical properties of the input data change over time, even though the relationship between input features and churn remains the same. For example:

Example: Predicting Customer Churn

Imagine you're working for a telecommunications company, and your task is to develop a machine learning model to predict customer churn (i.e., whether a customer will cancel their subscription).


The telecommunications company might expand its customer base to new geographic regions with different demographics and usage patterns.


As a result, the distribution of features like income levels, average call duration, and types of devices used might change over time, even though these features still influence churn in the same way.

Detecting data drift involves monitoring changes in the statistical properties of the input data. For instance, you might track the distributions of key features (e.g., income, call duration) over time and compare them to a baseline or reference distribution. Significant deviations from the baseline distributions could indicate data drift.



#### Concept Drift:
- Concept drift is a change in the relationship between the input data and the model target.
- Evidently AI helps evaluate model performance over time to detect concept drift.


Concept drift in this scenario would occur if the factors influencing customer churn change over time.

Example: Predicting Customer Churn

Imagine you're working for a telecommunications company, and your task is to develop a machine learning model to predict customer churn (i.e., whether a customer will cancel their subscription).

Initially, customer churn might be strongly correlated with factors like call duration, monthly charges, and customer service interactions.


However, over time, consumer preferences may shift, and factors like internet speed, streaming service availability, and contract length may become more important predictors of churn.


Detecting concept drift involves monitoring the performance of the churn prediction model over time. If the model's accuracy or other performance metrics start to degrade, despite no changes in the model or its input data, it could indicate concept drift.



#### Target Drift:
- Target drift refers to a change in the definition or meaning of the target variable over time.

---

## Detecting Drift with Evidently:
- Evidently is an open-source Python library that helps implement testing and monitoring for production machine learning models.
- It offers various checks for datasets and provides interactive visual reports to analyze the results, including data drift.

---

