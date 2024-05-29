# Project Report

## 1. Task
* Task Description: Examine the quality of the attached dataset. Use ML (Python) to find insights, hidden patterns, and forecast trends; anything related to financial and risk forecasting, as well as ideal customer profile (ICP).

* Instructions: You may invest as much time and effort as you wish. You can complete the task at your discretion, using whichever methods, libraries, and tools you think are most effective.

## Dataset Description
The dataset consists of insurance data samples, containing various attributes related to customers and their insurance policies. The dataset is stored in a CSV file.

These are the below attributes of the dataset. 
![Alt text](<images/image.png>)

### 2. Data Understanding
* Shape of the Dataset: The dataset contains 23,906 rows and 18 columns.

* Top 5 Records: 

|    | Car_id       | Date       | Customer_Name   | Gender   |   Annual_Income | Dealer_Name                         | Company   | Model      | Engine                      | Transmission   | Color      |   Price_($) | Dealer_No   | Body_Style   |   Phone |   Amount_paid_for_insurance |   Claim_amount | City    |
|----|--------------|------------|-----------------|----------|-----------------|-------------------------------------|-----------|------------|-----------------------------|----------------|------------|-------------|-------------|--------------|---------|-----------------------------|----------------|---------|
|  0 | C_CND_000001 | 01/02/2022 | Geraldine       | Male     |           13500 | Buddy Storbeck's Diesel Service Inc | Ford      | Expedition | DoubleÃ‚Â Overhead Camshaft | Auto           | Black      |       26000 | 06457-3834  | SUV          | 8264678 |                        1665 |              0 | Riga    |
|  1 | C_CND_000002 | 01/02/2022 | Gia             | Male     |         1480000 | C & M Motors Inc                    | Dodge     | Durango    | DoubleÃ‚Â Overhead Camshaft | Auto           | Black      |       19000 | 60504-7114  | SUV          | 6848189 |                        1332 |           1900 | Liepaja |
|  2 | C_CND_000003 | 01/02/2022 | Gianna          | Male     |         1035000 | Capitol KIA                         | Cadillac  | Eldorado   | Overhead Camshaft           | Manual         | Red        |       31500 | 38701-8047  | Passenger    | 7298798 |                        1897 |              0 | Riga    |
|  3 | C_CND_000004 | 01/02/2022 | Giselle         | Male     |           13500 | Chrysler of Tri-Cities              | Toyota    | Celica     | Overhead Camshaft           | Manual         | Pale White |       14000 | 99301-3882  | SUV          | 6257557 |                        1176 |              0 | Jelgava |
|  4 | C_CND_000005 | 01/02/2022 | Grace           | Male     |         1465000 | Chrysler Plymouth                   | Acura     | TL         | DoubleÃ‚Â Overhead Camshaft | Auto           | Red        |       24500 | 53546-9427  | Hatchback    | 7081483 |                        1323 |           2450 | Liepaja |


* Data Types:The dataset contains a mix of data types: object (string), int64 (integer), and float64 (floating-point numbers).
![Alt text](<images/image copy.png>)

* Summary Statistics:
![Alt text](<images/image copy 2.png>)
- The describe() function provides summary statistics for numerical columns, including count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum values.
- Key statistics for numeric columns such as Annual_Income, Price_($), Phone, Amount_paid_for_insurance, and Claim_amount are provided.

* Dublicate Rows : There is No dublicate rows present in our dataset. 

* Missing Values:There is one missing value in the Customer_Name column.All other columns have no missing values.
![Alt text](<images/imagecopy.png>)

- So we decided to impute the missing values with the string value "Missing". Now we don't have any Missing Values in the Customer_Name column. 
![Alt text](<images/image copy 3.png>)

Alternatively we could have used other technicques to fill the missing values But the column was string column we do not want to mix up with the other values. 
 - Forward Fill (ffill) and Backward Fill (bfill): Use the value of the previous non-missing data point (ffill) or the next non-missing data point (bfill) to fill missing values, respectively. This method is suitable for time series data with numerical column.
 - Mean/Median/Mode Imputation: Replace missing values with the mean, median, or mode of the column. This method is suitable for numerical and categorical data.


## Exploratory Data Analysis
We further generate a complete and exhaustive report for the dataset using the following
    - Pandas profiling 
    - Autoviz
    - Sweet

Let's look at each of the columns one by one. 

* **Car_id**: Each car ID in the dataset is unique, indicating that there are no duplicate entries for vehicle identifiers.
The "Car_id" column should be removed because it contains unique values for each entry in the dataset. Since each car ID is unique, this column essentially serves as an identifier or key for individual records and does not provide any meaningful information for analysis or modeling purposes. Including such a column in predictive modeling can introduce noise and computational overhead without contributing to the model's predictive power. Therefore, removing the "Car_id" column simplifies the dataset and improves the efficiency and interpretability of machine learning algorithms.

* **Date**: Date Extraction for Temporal Features: We start by converting the 'Date' column to a datetime data type using the pd.to_datetime() function. This conversion ensures that the 'Date' column is recognized and treated as datetime objects, enabling subsequent manipulation and extraction of temporal information.
  
Initial Dataset Information:
After the conversion, we verify the data types and information of the DataFrame using the df.info() function. This step is crucial to confirm the successful conversion of the 'Date' column to datetime format and to ensure the integrity of the dataset.
![Alt text](<images/image copy 4.png>)

Feature Extraction:

- **Year Extraction**: We extract the year component from the 'Date' column using the .dt.year accessor and create a new column named 'Year'. This allows us to analyze trends and patterns at a yearly granularity.
- **Month Extraction**: Similarly, we extract the month component from the 'Date' column using the .dt.strftime('%b') method, which formats the month as abbreviated names (e.g., Jan, Feb) and create a new column named 'Month'. This facilitates analysis based on monthly variations and seasonality.
- **Day Extraction**: Additionally, we extract the day component from the 'Date' column using the .dt.day accessor and create a new column named 'Day'. This captures the day of the month for each record, enabling insights related to cyclic behaviors or specific temporal events within a month.

**Customer Names**: 

  ![Alt text](<images/image copy 7.png>)
- We are using word cloud or text visualization composed of various names.The names are arranged and sized differently, with some names appearing larger and more prominently than others. Some key observations:
- The central and largest names in the image are Emma, Lucas, Thomas, and Nathan, suggesting these may be popular or common names represented in this visualization.-
- The names are predominantly of English or Western origin, with a few names like Alexis, Antoine, and Nicolas that could be of French or European origin.
- The names seem to be a mix of both male and female names, indicating a representation of both genders.
![Alt text](<images/image copy 8.png>)

Gender:

  ![Alt text](<images/image copy 5.png>)
- The image presents gender distribution data visualized through a pie chart and a bar chart.
- The pie chart shows that 78.6% i.e (18798) of the individuals in the dataset are male, while 21.4% (5108) are female.
- The bar chart further reinforces this observation by displaying the count or frequency of males and females in the dataset. The blue bar representing males is significantly higher than the bar for females, indicating a higher number of males in the dataset.


Transmission : 

  ![Alt text](<images/plot1.png>)
- The histogram shows the counts or frequencies of two transmission types: "Auto" and "Manual".
- The "Auto" transmission type has a significantly higher count (52.99%) compared to the "Manual" transmission type (47.41%).
- This suggests that automatic transmissions are more common or prevalent in the dataset represented by this histogram.

Engine: 

  ![Alt text](<images/plot2.png>)
- The histogram displays the counts or frequencies of two engine types, but the specific engine types are not labeled.
- The distribution of counts between the two engine types is identical to the distribution of transmission types in Image 1.
- One engine type has a count of 52.99%, while the other has a count of 47.41%.
- This similarity in distribution suggests a potential correlation or relationship between the transmission type and engine type in the dataset.

Dealer_Name: 

  ![Alt text](<images/plot3.png>)
- This histogram shows the counts or frequencies of different dealer names.
- The dealer names are displayed along the x-axis, but they are not explicitly mentioned.
- The histogram reveals a wide range of counts or frequencies for different dealer names, with one dealer having a significantly higher count (51.49%) than the others.
- Most dealer names have relatively low counts, typically less than 5%.
- This histogram provides insights into the distribution of data across different dealers or dealerships in the dataset.

Company:

  ![Alt text](<images/plot4.png>)
- The histogram displays the counts or frequencies of different companies or manufacturers.
- The companies or manufacturers are listed along the x-axis.
- The company with the highest count is "Chevrolet" (61%), indicating that it is the most prevalent or dominant company in the dataset.
- Other companies like "Dodge" (9.99%), "Ford" (7.75%), and "Volkswagen" (5.58%) have relatively lower but notable counts.
- Several companies have very low counts, suggesting they have a smaller presence or representation in the dataset.
- This histogram provides insights into the distribution of data across different vehicle manufacturers or companies. 


Engine 

  ![Alt text](<images/plot5.png>)
- There are two engine types represented in the data: "Double A, Overhead Camshaft" and "Overhead Camshaft".
- The "Double A, Overhead Camshaft" engine type has a significantly higher count, represented by the taller blue bar, with 52.59% of the total counts.
- The "Overhead Camshaft" engine type has a lower count, represented by the shorter green bar, with 47.41% of the total counts.
- While the "Double A, Overhead Camshaft" engine type has a higher count, the difference between the two engine types is not extremely large, indicating that both types are relatively common in the data set.
- The histogram provides a clear visual representation of the distribution and relative frequencies of the two engine types, allowing for easy comparison and analysis.

* Phone : This contains This was information which was not relevant to us. 

**Observations w.r.t to City**

![Alt text](<images/image copy 10.png>)
- Riga has the highest count for both "Auto" and "Manual" transmissions across all cities.
- Daugavpils and Tukums have relatively lower counts compared to other cities for both transmission types.

![Alt text](<images/image copy 11.png>)
- The distribution pattern across cities is similar for both "Double-A, Overhead Camshaft" and "Overhead Camshaft" engine types.
- Riga again has the highest counts for both engine types, followed by Liepaja and Jelgava.
- Ventspils, Daugavpils, and Tukums have lower counts compared to the larger cities.

![Alt text](<images/image copy 12.png>)
- There is a wide range of companies represented across the cities.
- Riga has the highest counts for most companies, with some notable peaks for companies like "Ford," "Audi," and "Mercedes-Benz."
- Other cities like Liepaja, Jelgava, and Ventspils also have significant counts for certain companies.

![Alt text](<images/image copy 13.png>)
- There is a large number of individual dealer names represented across the cities.
- The distribution pattern is similar across cities, with a mix of higher and lower counts for different dealer names.
- Riga, being the largest city, tends to have higher counts for most dealer names compared to other cities.

![Alt text](<images/image copy 14.png>)
- This bar chart shows the distribution of dealer names across different cities.
- The cities included are Riga, Liepaja, Jelgava, Ventspils, Daugavpils, and Tukums.
- Some dealer names like "Pilot Cars" and "Gerts Ratte" appear to have a significant presence across multiple cities.
- The chart highlights the diversity of dealer names operating within each city.

![Alt text](<images/image copy 15.png>) 
- This stacked bar chart displays the gender distribution (male and female) across the same set of cities.
- The male population is significantly larger than the female population in all cities.
- Riga has the highest overall population for both genders, followed by Liepaja and Jelgava.
- The chart allows for easy comparison of gender ratios between different cities.

![Alt text](<images/image copy 16.png>)
- These stacked bar charts show the distribution of different body styles (SUV, Passenger, Hatchback, Hatchback, and Sedan) across the cities.
- The SUV and Passenger body styles appear to be the most prevalent in all cities.
- Riga and Liepaja have the highest counts for each body style compared to other cities.
- The charts provide insights into the popularity of various vehicle body styles in different regions.

* **Correlation Analysis**:
![Alt text](<images/image copy 6.png>)
 - Amount_paid_for_insurance and Price_($): There is a high overall correlation between the amount paid for insurance and the price of the vehicles. This correlation suggests that as the price of the vehicle increases, the amount paid for insurance also tends to increase.

- Dealer_Name and Dealer_No: There is a high overall correlation between dealer names and dealer numbers. This correlation indicates a strong association between specific dealerships and their unique identification numbers.

- Engine and Transmission: There is a high overall correlation between the type of engine and transmission used in vehicles. This correlation suggests that certain engine types are commonly paired with specific transmission types.


The highly correlated features contains the redundant information. So we decided to keep only the one features and remove the redundant features. 
Observations: 
    - Highly correlated features can make it challenging to interpret the importance of individual features. Feature importance becomes ambiguous when two or more features convey similar information.
    - Models trained on datasets with highly correlated features may perform well on the training data but struggle to generalize to new, unseen data.
    - Correlated features can lead to a phenomenon known as the “curse of dimensionality,” where the model’s performance degrades as the number of features increases.

   - Interpretable models are essential for understanding the factors influencing predictions. Removing correlated features aids in clearer interpretation and better decision-making.
   - Removing correlated features helps improve a model’s ability to generalize to new data, reducing the risk of overfitting and making the model more robust.
   - Removing redundant features can lead to simpler models that are less prone to overfitting and perform better on unseen data.

* **Claim_amount**: A significant portion of claim amounts (90.0%) in the dataset are zeros. This observation suggests that the majority of insurance claims associated with the vehicles in the dataset do not involve monetary compensation.

After this observation we decided to remove the below columns. 
* Car_id 
* Customer_Name
* Amount_paid_for_insurance
* Dealer_No
* Transmission
* Customer_Name
* Phone


Now let's look at the Numerical Variables below 

![Alt text](<images/image copy 9.png>)

* **Annual Income Chart**:The distribution of annual income displays a right-skewed or positively skewed pattern. This means that the majority of the data points are concentrated on the lower end of the income range, with a long tail extending towards the higher income values. This type of distribution is common for income data, as a relatively small proportion of the population tends to have very high incomes, while the majority fall within lower to middle-income ranges. The sharp peak near zero suggests a significant number of individuals with little to no income.

* **Price Chart**:The price distribution appears to be bimodal or potentially multimodal, indicating the presence of two or more distinct peaks or modes. This pattern could arise in scenarios where there are different tiers or categories of pricing for products or services. For example, one peak might represent lower-priced items, while another peak corresponds to higher-end or premium offerings. The multiple peaks suggest that prices tend to cluster around certain values, rather than being evenly distributed across the range.

* **Claim Amount Chart**:The claim amount distribution exhibits a strong right-skewed or positive skew. The distribution is heavily concentrated towards lower claim amounts, with a steep peak near zero, implying that a large number of claims are relatively small in value. However, the long tail extending towards higher claim amounts indicates the presence of a smaller number of claims with significantly larger monetary values. This pattern is typical in insurance or healthcare contexts, where most claims are for routine or minor expenses, while a few claims involve substantial costs, potentially due to major medical events or accidents.


**Observations w.r.t to Claim Amount**
![Alt text](<images/image copy 17.png>)

Based on the multi-panel visualization, I can make the following observations related to financial and risk forecasting for car insurance claims:

- More claims come from customers with lower incomes, so insurers may need to adjust pricing or risk models for this customer segment.
- Most claims are for smaller amounts around $1,500, but there are also some larger claims up to $7,500 or more. Insurers need to plan for both frequent small claims and occasional large claims.
- There is a pattern of more claims happening around the middle of the year, likely due to factors like more driving or weather impacts. Understanding these seasonal patterns can help predict claim volumes.So the maximum peaks or highest volumes of claims occur around the middle of the year, likely during the summer months of June, July and August.
- The distribution of claims across different vehicle prices/policy types provides insights into potential losses for each pricing tier.
- Sudden spikes in daily claims, or outlier events with many claims at once, indicate a need to account for volatility and extreme scenarios when calculating risks and financial reserves.

In simple terms, this data can help insurers better estimate claim costs, set appropriate pricing, and have sufficient funds reserved to cover claims while accounting for patterns, different customer groups, and potential volatility in the claims process.

**Outliers in Annual_income , Claim_amount & Price**
Based on the pie charts provided, I can make the following observations:

![Alt text](<images/image copy 18.png>)


![Alt text](<images/image copy 19.png>)





**Observations based Maximum Value of Amount claimed**
![Alt text](<images/image copy 20.png>)
- This chart shows the distribution of claims amount by city. Tukums has claimed the highest amount by a percentage at 17.7%, followed by Liepaja at 17%, Daugavpils at 16.7% ,entspils at 16.7%&  Riga at 15.2%.

![Alt text](<images/image copy 21.png>)
- This chart displays the distribution of claims by car company. The top companies with the claimed of highest amount by a percentages are Chevrolet (7.2%) ,Dodge (6.7%) ,Ford (6.2%) Oldsmobile (5.8%),Cadillac (4.4%) ,Mercedes-B (5.1%), Mitsubishi (5%), Mercury (4%).

![Alt text](<images/image copy 22.png>)
- This chart illustrates the distribution of claims by car color. Pale White accounts for the highest percentage at 45.4% for Pale White, followed by Black at 32.3%, and Red at 22.3%.

![Alt text](<images/image copy 23.png>)
- This chart presents the distribution of claims by car body style. SUVs have the highest percentage at 25.3% for Hatchback, followed by SUV at 24.5%,  Sedans at 20.1%,Passenger vehicles at 16.9%, and Hardtops at 13.2%.

![Alt text](<images/image copy 24.png>)
- This chart shows the distribution of claims by month. The month with the highest percentage is December (14.8%), followed by November (13.9%), April (14%), October (10%), and September (9%).

![Alt text](<images/image copy 25.png>)
This chart shows the distribution of claims by Year. The year with the claimed the highest amount by percentage is 2023 (55.5%), followed by 2022 (44.5%).

These pie charts provide insights into the distribution of car insurance claims based on various factors such as city, car company, color, body style, and the month of the year. This information can be valuable for insurance companies to analyze risk patterns and make informed decisions regarding pricing, underwriting, and risk management strategies.




**Observations based TimeSeries Claim_amount , Price & Annual Income**


Given that the data is related to car insurance, with the price representing the car's price, annual income representing the customer's annual income, and claim amount representing the claims made by the customer, the following observations can be made from a financial and risk analysis perspective:

Claim Amount:

![Alt text](<images/image copy 28.png>)
- The highly volatile and fluctuating nature of claim amounts suggests that there are periods with higher risk of claims, possibly due to factors like accidents, natural disasters, or other events leading to increased claims activity.
Insurance companies need to maintain adequate reserves and implement effective risk management strategies to handle these claim spikes and ensure financial stability.
Analysis of claim patterns and underlying causes can help identify potential risk factors and develop targeted risk mitigation strategies.

Price (Image 2):

![Alt text](<images/image copy 29.png>)
- The volatility in car prices can impact the insurance premium calculations and the potential claim payouts for insured vehicles.
Highly fluctuating car prices may indicate market uncertainties or changing demand and supply dynamics, which can affect the insurer's risk exposure and profitability.
Insurers may need to adjust their pricing models and underwriting criteria to account for the volatility in car prices and associated risks.

Annual Income (Image 3):

![Alt text](<images/image copy 30.png>)
- The relatively stable pattern of annual income data suggests a more predictable cash flow for customers, which can positively impact their ability to pay insurance premiums consistently.
However, fluctuations in annual income can affect customers' purchasing power and their willingness or ability to maintain adequate insurance coverage.
Insurers may consider offering flexible payment options or adjusting premiums based on income levels to ensure affordability and maintain customer retention.

Overall, the high volatility in claim amounts and car prices highlights the importance of robust risk management practices for car insurance companies. Effective underwriting, pricing models, and risk mitigation strategies are crucial to address these fluctuations and maintain profitability while providing adequate coverage to customers. Monitoring and analyzing these time series data can help insurers identify emerging risks, adjust their strategies, and ensure long-term financial stability in the car insurance industry.


