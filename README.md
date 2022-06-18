# Machine-Learning - Classification and Regression Trees


A data mining routine has been applied to the transaction dataset and has classified 88 records as fraudulent (30 correctly so) and
952 as non-fraudulent (920 correctly so), so constructed the confusion matrix and calculated the overall error rate.

Interpreted the Decile chart and explained its usefulness between two metrics of model performance (Error Rate and Lift)

Computed error rates, sensitivity, and specificity using cutoffs of 0.25, 0.5, and 0.75 and created a decile lift chart.


**Competion Auctions - Ebay**

Data Preprocessing. Converted variable Duration into a categorical variable. Split the data into training (60%) and validation (40%) datasets.

Fitted a classification tree using all predictors. To avoid overfitting, set the minimum number of records in a terminal node to 50 and the maximum tree depth to 7 to determine which variable to choose.

Fitted another classification tree (using a tree with a minimum number of records per terminal node = 50 and maximum depth = 7), this time only with predictors that can be used for predicting the outcome of a new auction.


**Predicting Prices of Used Cars (Regression Trees)**

Ran a full-grown regression tree (RT) with outcome variable Price and predictors Age_08_04,
KM, Fuel_Type (first convert to dummies), HP, Automatic, Doors, Quarterly_Tax,
Mfr_Guarantee, Guarantee_Period, Airco, Automatic_airco, CD_Player, Powered_Windows,
Sport_Model, and Tow_Bar. Set random_state=1

Compared the prediction errors of the training and validation sets by examining their RMS
error and by plotting the two boxplots.

Created a smaller tree by using GridSearchCV() with cv = 5 to find a fine-tuned tree and then compared it to the full-grown tree to determine predictive performance on the validation set.
