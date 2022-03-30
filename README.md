# Supervised_Regression1
## Philadelphia Property Value Prediction
### By: Muhammad Rivaldi Prabowo along with Billy Witanto and Vinsensia Fresian Meiliana as member of AlphaEngineer Group in Job Connector Data Science Program Batch 15 Purwadhika Startup and Coding School.

<p align="center">
   
![ReadMe Header](https://user-images.githubusercontent.com/99151517/160825562-6bb39e41-b52d-48a9-9fac-adcf3ddc5f91.jpg)
</p>

<p align='justify' style="font-weight: bold;">
These notebooks serve as the final project of Job Connector-Data Science and Machine Learning program at Purwadhika Start-up and Coding School.
</p>


## Background
<p align='justify' style="font-weight: bold;">
Philadelphia city is one of the hottest real estate market in the US. With high increment every year, its property price is suprisingly lower compared to other cities in the US. That said, with lowered home demand index in 2022 compared to 2021, this is the best time for property agent to actively promote its remaining properties listed. 
One of the success factor for property transaction is the estimation of the property's market value. This estimation always done by professional property appraisal which has 300 USD fee per appraisal <a href="https://www.homeadvisor.com/cost/inspectors-and-appraisers/hire-a-property-appraiser/">(Home Advisor, 2022)</a>. Our team as Data Scientist who work in one of the biggest property agent in Philadelphia have an idea to build machine learning which can predict/estimate property market_value in Philadelphia. By doing so, the property agent can give a rough justified estimated price for the promotion without any help from appraisal professional. Thus, reducing the cost. Also, by getting the justified estimated price which are not under- or over-valued, hopefully, will benefited both seller and buyer, thus increasing the amount of success transactions of the property agent.
</p>

## Dataset Source
<p align='justify' style="font-weight: bold;">
This is a real-world data of properties in Philadelphia city. You can download it from <a href="https://www.kaggle.com/datasets/adebayo/philadelphia-buildings-database">Kaggle</a> or <a href="https://www.kaggle.com/datasets/adebayo/philadelphia-buildings-database">City of Philadelphia: Metadata Catalog</a>.
</p>

## Data Understanding
<p align='justify' style="font-weight: bold;">
With such an enormous dataset with a lot of columns and rows, we first need to know what information the dataset has. From a thorough investigation, we summarize them in <a href="https://docs.google.com/spreadsheets/d/1WapgNftGZMUBt6H2SkDbNedN6vAwY56xvdr9P1My-30/edit#gid=781668512">Spreadsheet</a>. 
</p>

## Data Cleaning
<p align='justify' style="font-weight: bold;">
First, we pick columns which informative enough to help us fill the missing value and anomalies in the dataset based on its description. Thus, reducing the columns. Then, initial exploratory data analysis (EDA) was performed to match the column description with the data, also to detect and correcting missing values and anomalies. Furthermore, after correcting missing values, this section resulting in clean dataset, free from missing values. <a href="https://github.com/PurwadhikaDev/AlphaEngineer_JC_DS_FT_BSD_JKT_15_FinalProject/blob/main/1.%20Background%20and%20Data%20Cleaning.ipynb">Jupyter</a>
</p>

## Detailed EDA
<p align='justify' style="font-weight: bold;">
Detailed EDA to further understand the characteristics and correlations of the features and label to determine the proper preprocessing of the data. This section will also help us in gaining insights about feature engineering. <a href="https://github.com/PurwadhikaDev/AlphaEngineer_JC_DS_FT_BSD_JKT_15_FinalProject/blob/main/2.%20Detailed%20EDA.ipynb">Jupyter</a>
<a href="https://public.tableau.com/views/FinalProjectPurwadhika/MarketValuebyYear?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link">Tableau Vizualization</a>
</p>

## Feature Selection
<p align='justify' style="font-weight: bold;">
We select the features that will be used in the model building based on evidence in detailed EDA and also on the basis of domain knowledge. We decide to build two models, so the feature selection of those models will slightly different.
   
![FS1](https://user-images.githubusercontent.com/99151517/160816200-1eaf4573-c540-423d-b59e-a8701b302fa7.JPG)
   
![FS2](https://user-images.githubusercontent.com/99151517/160816208-3418fc39-9b21-4f11-a48c-bf74fc0b10e8.JPG)
</p>

## Modeling
<p align='justify' style="font-weight: bold;">
   
1. **Model 1**
    * Selected algorithm models for cross-validation process are: Linear Regression (Parametric), Random Forest Regressor (Non-Parametric), dan XGBoost Classifier (Non-Parametric).
    * Evaluation metric that suite for this bussiness case is mean absolute percentage error (MAPE), also we calculate the others evaluation metrics, such as MSE, RMSE, MAE, and R-Squared.
   
    * ![cv1](https://user-images.githubusercontent.com/99151517/160820504-0d24f4fb-0352-4540-8ac7-ac4c95402b64.JPG)
    * Selected model from cross-validation process is Random Forest Regressor with MAPE score 14.47%.
   
    * ![model performance1](https://user-images.githubusercontent.com/99151517/160821354-1ab4d00a-f3b5-43a5-94ed-568442e2a7ed.JPG)
   
    * ![model1](https://user-images.githubusercontent.com/99151517/160822661-1685f34e-7672-4508-8c35-b871be582b4f.jpg)
    * Evaluation this model using dataset test give us MAPE score 13.18%.
    * This model don't use hyperparameter tunning, due to incapability of our personal laptop to run the process with enormous data.
    * Selected algorithm for this model is Random Forest Regressor with Recall score 13.18% (Dataset test).


2. **Model 2**
    * Selected algorithm models for cross-validation process are: Linear Regression (Parametric), Random Forest Regressor (Non-Parametric), dan XGBoost Classifier (Non-Parametric).
    * This Model use extra feature engineering in order to boost model performace (grouping zip_codes into several simplified category based on distance from city center, simplify fireplaces feature, extracting words and grouping it based on building code description along with creation of new feature.)
    * Evaluation metric that suite for this bussiness case is mean absolute percentage error (MAPE), also we calculate the others evaluation metrics, such as MSE, RMSE, MAE, and R-Squared.
   
    * ![cv2](https://user-images.githubusercontent.com/99151517/160820982-e10654f1-ff01-4084-8ccc-2ad0a0094959.JPG)
    * Selected model from cross-validation process is Random Forest Regressor with MAPE score 13.32%.
   
    * ![model2](https://user-images.githubusercontent.com/99151517/160822667-83d24a55-2b5e-46c5-8643-b22df15a6fde.jpg)
   
    * ![model performance2](https://user-images.githubusercontent.com/99151517/160821495-e7268693-dacc-4644-bdd6-abe870c3b848.JPG)
    * Evaluation this model using dataset test give us MAPE score 12.25%.
    * This model don't use hyperparameter tunning, due to incapability of our personal laptop to run the process with enormous data.
    * Selected algorithm for this model is Random Forest Regressor with Recall score 12.25% (Dataset test).
</p>

## Model Analysis
<p align='justify' style="font-weight: bold;">
1. General analysis
  
From the result before, I choose Model 2 over Model 1 because it shows higher MAPE score on dataset test (12.25%). Based on reference <a href="https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119199885.app1">Reference</a> MAPE below 10% indicating an excellent accuracy of the prediction model, while 10-20% indicating a good accuracy, so our model is a Good Model.
  
This model has its own limitation, this model can only use inside these criteria:
* 5500<= market_value <=150.000.000
* 0<= number_of_bedrooms <=93
* 0<= number_of_rooms <=154
* 0<= number_stories <=40
* 0<= property_age <=368
* 600<= total_area <=100.000
* 600<= total_livable_area <=798.189
  
2. Confidence area
Our model can work better in property which has market_value below 20.000.000, we call this confidence area. Our model also can be use to predict market_value above 20.000.000 with lower confidence level, because there are 65% chance of our prediction can classified as not a good prediction (MAPE score above 20%).
  
3. There are unrelevant feature value
![model analysis 3](https://user-images.githubusercontent.com/99151517/160823785-11c51782-d5bf-4732-88be-1feb8a8bb7b5.JPG)
As we can see in the data frame, Vacant Land should be an empty land without number stories,  but there are value in those features, so this unwanted occurrence value cause MAPE mean score for this category has high value (180.482%).
  
4. Lower actual market_value
![model analysis 5](https://user-images.githubusercontent.com/99151517/160824170-1c93afbc-f4b7-4a55-8109-1ef94acac793.JPG)
As we can see in the Dataframe (dataset test), dataframe which grouped by categorical code Industrial, the highest livable_total_area has MAPE score 1231,71% and many of the top 12 highest livable_total_area has MAPE score above 20% (MAPE score for good model) (9 of 12 top data). We analyze that such high MAPE was caused by the actual market_value was set too low from the majority of data with the same specifications. This could be external factor that model cannot predict (of course it’s called outliers).
  
5. Worldwide External Factor
![model analysis 7](https://user-images.githubusercontent.com/99151517/160824484-c6f289c2-ec29-4ad2-aa2e-de66e1e780ff.JPG)
![model analysis 9](https://user-images.githubusercontent.com/99151517/160824689-7b348ad0-2961-4dcb-b768-3861c84c06d3.JPG)
From dataframe we use selection indexing to filter sale_year where worldwide crisis happen (economy crisis and covid pandemic) with threshold of  MAPE score above 20%. We can get 380 and 1917 data in it, it means 15,9% of the data above good MAPE model score (dataset test) affected by those external crisis.
</p>

# Summary and Recommendation
<p align='justify' style="font-weight: bold;">
   
Model conclusion:
   * There are 24 features (8 numerical and 16 categorical) and 484.058 rows data for modeling purposes.
   * Based on Cross Validation we choose Random Forest Regressor model, because it has the lowest MAPE score (14.47% in Model 1 and 13.32% in Model 2) and the most stable (lowest standard deviation).
   * From comparison between Model 1 and Model 2 we choose Model 2 over Model 1 because extra feature engineering can boost test score (dataset test) from MAPE score 13,18% in Model 1 to 12,25% in Model 2. This extra feature engineering also boost others metric evaluation score.
   * `total_livable_area` is the most importance feature in Model 2 and then followed by `number_stories`, `zip_codes`, `property_age`, `sale_year` ,`total_area`, and `overall_condition`, respectively.

Our group can make model to predict market value property with MAPE score (dataset test) around 12,12% which categorized as good model prediction. This model can answer the problem statement, so the property agent has an option not to use professional appraisal to asses market value of property in Philadelphia, just use this model instead to reduce operational cost. But there are some limitation for this model.

Model Recommendation:
* Read more literature to know more about the domain knowledge, this will reduce the assumption with fact.
* Doing another extra feature engineering by exploring more about current features (deepen the feature analysis and take a look the relation between feature-feature and feature-label).
* Improvement by doing hyperparameter tuning.
* Gather another data to increase confidence level while predict high market value properties
* Need for another extra data/feature, like data for every property sold in Philadelphia in 2020 until 2022.
</p>
