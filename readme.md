# Project Report

## 1. Task
* Task Description: Examine the quality of the attached dataset. Use ML (Python) to find insights, hidden patterns, and forecast trends; anything related to financial and risk forecasting, as well as ideal customer profile (ICP).

* Instructions: You may invest as much time and effort as you wish. You can complete the task at your discretion, using whichever methods, libraries, and tools you think are most effective.

## Project Structure
In this Notebook we will examine the Exploratory Data Analysis for the Machine learning. 


``` bash
InsuranceClaims/
│
├── Notebooks/
│   ├── MachineLearningEDA+Modelling.ipynb
│   └── LLM_fine_tuningipynb.ipynb
│
├── dataset/
│   ├── datasetqa.csv
│   ├── features.csv
│   └── formatted.json
│
├── images/
│   └── (all images used in README.md)
│
├── Reports/
│   ├── ReportPandasProfiling.html
│   ├── sweetviz_report.html
│
├── .gitignore
├── LICENSE
└── README.md
```

Notebooks/
This directory contains all the Jupyter notebooks used for analysis, EDA, and model training.

- **MachineLearningEDA+Modelling.ipynb**  
The problem is approached using machine learning, encompassing both Exploratory Data Analysis (EDA) and modeling. The objective is to forecast the Claim_amount. This is tackled as both a classification and a regression problem:

* Classification: All non-zero values are converted to 1, indicating that a claim has been made, while 0 indicates no claim.
* Regression: Predict the actual claim amount for the instances where claims are made.

- **LLM_fine_tuningipynb.ipynb**  
  The Modelling is also done using the LLM model FLANT-5 Model. The demo for this is created on HuggingFace.  
  The Dataset is prepared in a supervised way where the model is asked to actually claim amount for the instances where claims are made.
  Please find the space here: [HuggingFace Space](https://huggingface.co/spaces/Jyotiyadav/InsuranceClaim)  
  Model: [FLANT-5 Model](https://huggingface.co/Jyotiyadav/InsuranceModel1.0)

    ![Alt text](<images/image copy 44.png>)

dataset/
This directory contains all the dataset files.

- **datasetqa.csv**  
  This dataset is converted into the format for training an LLM Model.

- **features.csv**  
  The final features stored using machine learning for fine-tuning the LLM.

- **formatted.json**  
  The dataset converted into the JSON format for training the LLM model.

images/
This directory contains all the images used in the `README.md` file. The images are stored here for organizational purposes and to keep the repository structure clean.

Reports/
In this section we can see a basic Intermediatory report in the format .html. Can be directly loaded in the Browser. 

README.md
The main README file for the repository, which typically contains an overview of the project, instructions on how to set it up, usage examples, and more.

.gitignore
(Optional) A file specifying which files and directories should be ignored by Git. This can include temporary files, build artifacts, etc.


## Dataset Description
The dataset consists of insurance data samples, containing various attributes related to customers and their insurance policies. The dataset is stored in a CSV file.

These are the below attributes of the dataset. 
![Alt text](<images/image.png>)

## 2. Quality of the dataset
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



## Exploratory Data Analysis
We further generate a complete and exhaustive report for the dataset using the following
    - Pandas profiling 
    - Autoviz
    - Sweet

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


* **Car_id**: Each car ID in the dataset is unique, indicating that there are no duplicate entries for vehicle identifiers.
The "Car_id" column should be removed because it contains unique values for each entry in the dataset. Since each car ID is unique, this column essentially serves as an identifier or key for individual records and does not provide any meaningful information for analysis or modeling purposes. Including such a column in predictive modeling can introduce noise and computational overhead without contributing to the model's predictive power. Therefore, removing the "Car_id" column simplifies the dataset and improves the efficiency and interpretability of machine learning algorithms.

* **Date Feature Extraction**: Converte the 'Date' column to a datetime data type using the pd.to_datetime() function. This conversion ensures that the 'Date' column is recognized and treated as datetime objects, enabling subsequent manipulation and extraction of temporal information.
  
Initial Dataset Information:
After the conversion, we verify the data types and information of the DataFrame using the df.info() function. This step is crucial to confirm the successful conversion of the 'Date' column to datetime format and to ensure the integrity of the dataset.
![Alt text](<images/image copy 4.png>)

Feature Extraction:

- **Year Extraction**: Extract the year component from the 'Date' column using the .dt.year accessor and create a new column named 'Year'. This allows us to analyze trends and patterns at a yearly granularity.
- **Month Extraction**: Similarly, we extract the month component from the 'Date' column using the .dt.strftime('%b') method, which formats the month as abbreviated names (e.g., Jan, Feb) and create a new column named 'Month'. This facilitates analysis based on monthly variations and seasonality.
- **Day Extraction**: Additionally, we extract the day component from the 'Date' column using the .dt.day accessor and create a new column named 'Day'. This captures the day of the month for each record, enabling insights related to cyclic behaviors or specific temporal events within a month.



**Customer Names**: 

  ![Alt text](<images/image copy 7.png>)
- We are using word cloud or text visualization composed of various names.Some key observations:
- The central and largest names in the image are Emma, Lucas, Thomas, and Nathan, suggesting these may be popular or common names represented in this visualization.
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
- The histogram displays the counts or frequencies of two engine types.
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

Phone 

 ![Alt text](<images/image copy 50.png>)
Each Phone No in the dataset is unique, indicating that there are 99.9% distinct values in the dataset. This dataset can be removed as it contains almost unique values for each entry in the dataset.

**Observations Total claim amount over the months**

 ![Alt text](<images/plot6.png>)

- The highest claim amounts occur during the later months of the year, peaking in November and December of both 2022 and 2023.
- There's a consistent dip in claim amounts during the early months of the year, particularly in January and February.
- The months of May, August, September, and October show relatively high claim amounts compared to the early months of the year.
- This suggests a cyclical pattern where claims increase towards the end of the year and decrease at the beginning. This could be due to various factors such as weather conditions, holiday travel, or other seasonal trends that influence driving patterns and accident rates.


**Observations w.r.t to Average Claim Amount**

![Alt text](<images/image copy 42.png>)

* Dealer Name: Significant variation in average claim amounts exists across dealers. Some dealers consistently experience higher average claims, suggesting potential differences in the types of cars sold, customer demographics, or claim handling practices.

* Company Name: Similar to dealer names, company names (possibly insurers) also show notable differences in average claim amounts. This could indicate varying underwriting standards, risk assessment, or claim settlement processes.

* City: Certain cities exhibit much higher average claim amounts than others. This could be due to factors like traffic density, accident rates, road conditions, or the prevalence of fraudulent claims.

* Annual Income: As annual income increases, the average claim amount also tends to increase, although the relationship is not strictly linear. Higher-income individuals might own more expensive cars, leading to higher repair costs in case of accidents.

* Price: The average claim amount shows a clear upward trend with the car's price. This is expected since more expensive cars generally have higher repair costs.

* Month: There's a slight seasonal variation in average claim amounts, with higher claims observed during certain months. This could be related to weather conditions, holidays, or travel patterns.

* Day:  The variation in average claim amounts across days of the week is relatively small. There might be a slight increase in claims during weekends, possibly due to increased leisure driving.

* Engine: The type of engine seems to have a minor impact on average claim amounts. However, some engine types consistently show higher claims, suggesting potential differences in their risk profiles.

* Body Style: Different body styles have varying average claim amounts. Some body styles like SUVs might be more prone to accidents or have higher repair costs due to their size and complexity.

* Gender: The average claim amount is slightly higher for one gender compared to the other. This could be due to differences in driving behavior, mileage, or the types of cars owned.

**Observations for Insurancers**

* Risk Assessment: The observed patterns can be valuable for risk assessment and pricing. Insurers could use this information to adjust premiums based on the dealer, company, city, car price, engine type, body style etc.

* Fraud Detection: Identifying cities or dealers with unusually high average claim amounts could help in detecting potential fraud or areas where claims handling processes need improvement.

**Observations w.r.t to Normal Distribution**

![Alt text](<images/image copy 41.png>)

Observations:

- Gender: The average claim amount is slightly higher for one gender (most probably, males) compared to the other. This could be due to differences in driving behavior, mileage, or the types of cars owned.

- Year: There's a slight difference in average claim amounts across the years. There could be a slight increase in claims in 2021, possibly due to an increase in the number of cars or miles driven in this year.

- City: Certain cities exhibit much higher average claim amounts than others. This could be due to factors like traffic density, accident rates, road conditions, or the prevalence of fraudulent claims.

- Dealer Name: Significant variation in average claim amounts exists across dealers. Some dealers consistently experience higher average claims, suggesting potential differences in the types of cars sold, customer demographics, or claim handling practices.

- Day: The variation in average claim amounts across days of the week is relatively small. There might be a slight increase in claims during weekends, possibly due to increased leisure driving.

- Body Style: Different body styles have varying average claim amounts. Some body styles like SUVs might be more prone to accidents or have higher repair costs due to their size and complexity.

- Annual Income: As annual income increases, the average claim amount also tends to increase, although the relationship is not strictly linear. Higher-income individuals might own more expensive cars, leading to higher repair costs in case of accidents.

- Month: There's a slight seasonal variation in average claim amounts, with higher claims observed during certain months (Summer and Fall). This could be related to weather conditions, holidays, or travel patterns.



**Observations w.r.t to Model**

![Alt text](<images/image copy 43.png>)
- The central and largest names in the image are Coupe, Ram, grand, and Diamente, suggesting these may be popular or common names represented in this visualization.


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


**After Removing Outliers using Cappping**

![Alt text](<images/image copy 40.png>)


![Alt text](<images/image copy 45.png>)


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


![Alt text](<images/image copy 46.png>)
This chart shows the distribution of claims by Gender. The Male have claimed the highest amount by percentage is Male (78.5%%), followed by female (21.5%).

![Alt text](<images/image copy 47.png>)
This chart shows the distribution of claims by Engine. The Engine with the claimed the highest amount by percentage is Double A (51.8%), followed by Overheard camshaft (48.2%).


![Alt text](<images/image copy 48.png>)



![Alt text](<images/image copy 49.png>)

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


correlation Matrix 
![Alt text](<images/image copy 36.png>)


This dataset appears to contain various features related to car insurance claims. Here are the observations from each feature:
![Alt text](<images/image copy 31.png>)

* Annual_Income: The distribution shows a skewed pattern, with a large number of claims coming from individuals with lower annual income levels.
* Price_($): The distribution of claim amounts related to the price of the vehicle seems to peak around certain values, suggesting that certain price ranges are more prone to claims.
* Claim_amount: The distribution of claim amounts is right-skewed, with most claims being relatively small, but a few large claims contributing to the long tail.
* Gender: The distribution shows a higher frequency for one gender over the other, indicating a potential difference in claim patterns between genders.
* Month: There seems to be a seasonal pattern in the claims, with some months having higher claim frequencies than others.
* Company: The distribution suggests that certain companies may have higher or lower claim frequencies compared to others.
* City: The pattern indicates that there may be variations in claim frequencies across different cities or locations.
* Model: The distribution shows that certain car models tend to have higher claim frequencies compared to others.
* Color: While the distribution is not as pronounced as some other features, it suggests that the color of the vehicle may be associated with different claim patterns.
* Engine: The distribution shows a potential relationship between engine size or type and claim frequency.
* Body_Style: The pattern indicates that certain body styles (e.g., sedan, SUV, coupe) may have varying claim frequencies.

Overall, this dataset provides insights into various factors that could influence the likelihood and magnitude of car insurance claims, which can be valuable for insurance companies in risk assessment and pricing.


**Ideal Customer Profiles Information**


For More Modelling refer to Notebooks: 
