# Machine Learning Assignment 2

## a. Problem statement

The task is to predict the price range of a mobile phone from its hardware specs. The target is not the actual selling price. It is one of four ranges: 0 (low cost), 1 (medium), 2 (high) and 3 (very high). Five classifiers were trained on the same train/test split and compared on Accuracy, AUC, Precision, Recall, F1 and MCC. The saved models are loaded in a Streamlit app and scored on uploaded test data without retraining.

## b. Dataset description

Dataset: Mobile Price Classification, published on Kaggle by Abhishek Sharma.
Link: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

The labeled file (`train.csv` on Kaggle, saved here as `data/mobile_price_full.csv`) has 2000 rows and 20 numeric features. Kaggle's `test.csv` has no labels, so it was not used. There are no missing values. The four price ranges are balanced, 500 phones each.

Features: battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi. Target column: price_range.

RAM is by far the strongest feature (ANOVA F-score about 2800 on the training rows). Battery power, px_height and px_width come next (F around 16 to 26). Everything else, including bluetooth, wifi, 4G and the cameras, is close to zero.

Data cleaning check: no missing values and no duplicate rows. 180 rows have sc_w = 0 and 2 rows have px_height = 0, which look like missing values stored as zero. Replacing those zeros with the training median moved 5-fold CV accuracy by between -0.12 and +0.06 percentage points across the five models (about 2 phones out of 1600 at most), so the values were left as they are.

Split: 80/20, stratified on price_range, random_state = 7. That gives 1600 training rows (`data/train_data.csv`) and 400 test rows (`test_data.csv`, 100 phones in each price range).

Because there are 4 classes, Precision, Recall and F1 are weighted averages, and AUC is one-vs-rest, also weighted.

## c. Github Repository Link

Link: https://github.com/KanishkaGhosh21/ML-Assignment-2

## d. Models used

All five models were trained on the same 1600 training rows and scored on the same 400-row test set. Each model was tuned with a grid search using 5-fold stratified cross-validation on the training rows only. The test set was not used for any choice. Feature selection (`SelectKBest` with the ANOVA F-score) is a step inside each pipeline, with the number of kept features (4, 6, 10 or all 20) tuned like any other setting. Scaling is also inside the pipeline. So the saved models in `model/` take all 20 raw columns from `test_data.csv` and are not retrained when the file is scored.

Final metrics on the test set:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9994 | 0.9826 | 0.9825 | 0.9825 | 0.9767 |
| Decision Tree | 0.8675 | 0.9117 | 0.8673 | 0.8675 | 0.8660 | 0.8242 |
| KNN | 0.9150 | 0.9920 | 0.9163 | 0.9150 | 0.9150 | 0.8871 |
| Naive Bayes | 0.8075 | 0.9573 | 0.8060 | 0.8075 | 0.8059 | 0.7439 |
| Support Vector Machine | 0.9775 | 0.9994 | 0.9776 | 0.9775 | 0.9775 | 0.9700 |

Settings chosen by cross-validation:

| ML Model Name | Features kept | Chosen settings |
|---|---|---|
| Logistic Regression | all 20 | scaled, C = 1000 |
| Decision Tree | ram, battery_power, px_height, px_width | entropy, max_depth = 12, min_samples_leaf = 1 |
| KNN | ram, battery_power, px_height, px_width | scaled, k = 51, distance weighting, Manhattan distance |
| Naive Bayes | all 20 | GaussianNB, no settings to tune |
| Support Vector Machine | top 10 by F-score | scaled, linear kernel, C = 100, probability estimates on for AUC |

Baseline vs tuned test accuracy (baseline = default settings, all 20 features, scaling where needed):

| ML Model Name | Baseline | Tuned | Change |
|---|---|---|---|
| Logistic Regression | 0.9725 | 0.9825 | +1.00 |
| Decision Tree | 0.8275 | 0.8675 | +4.00 |
| KNN | 0.5150 | 0.9150 | +40.00 |
| Naive Bayes | 0.8075 | 0.8075 | 0.00 |
| Support Vector Machine | 0.9025 | 0.9775 | +7.50 |

The full baseline table is in `model/metrics_baseline.csv` and the final one in `model/metrics_summary.csv`. With 400 test rows, one phone is 0.25 percentage points, so differences below about 1 point should be treated as noise.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best model, 98.25% (393 of 400). The price ranges are mostly ordered by RAM, with battery and screen resolution adding a little, so the four classes are close to linearly separable. The default C = 1 regularised the weights too much (97.25%). Raising C to 1000 lets the model draw sharper boundaries. CV kept all 20 features: the weak ones get near-zero weights, so they do no harm here. The 7 mistakes are phones on the boundary between two neighboring ranges. |
| Decision Tree | 86.75% (347 of 400), up from 82.75% with defaults. A tree splits one feature at a time, so it approximates the price boundaries with steps, and those steps need many splits. Keeping only the 4 strong features stops it from wasting splits on noise columns, which gave the gain. It is still well below the linear models because a straight boundary that mixes RAM and battery is hard to build from single-feature cuts. AUC (0.9117) is the lowest because leaf probabilities are coarse. |
| KNN | Biggest improvement: 51.50% with defaults to 91.50% (366 of 400). With all 20 scaled features, the 16 weak columns dominate the distance, so neighbors are often phones with very different RAM. Keeping only RAM, battery and the two pixel columns makes "near" mean similar specs that matter. A large k (51) with distance weighting and Manhattan distance smooths out noise. It is still behind LR and SVM because a local vote is less precise than a learned boundary near the class edges. |
| Naive Bayes | 80.75% (323 of 400), unchanged by tuning. Gaussian NB has no real settings, and CV found that dropping features did not help. Its limit is its assumptions: features independent and bell-shaped inside each class. Neither holds here: the camera columns move together, every 4G phone also has 3G, and the binary columns are not Gaussian. AUC (0.9573) is much higher than accuracy because the class probabilities are ranked well, but the final label often lands in the neighboring range. |
| Support Vector Machine | Second best, 97.75% (391 of 400), up from 90.25%. The default RBF kernel with C = 1 smoothed the boundaries too much. CV picked a linear kernel with C = 100 on the top 10 features, which agrees with Logistic Regression: the classes are close to linearly separable. It is 2 phones behind Logistic Regression, which is within noise on a 400-row test set. |
| Overall, Winner for your dataset? | Logistic Regression. It has the best Accuracy, Precision, Recall, F1 and MCC, and ties with SVM on AUC (0.9994). Linear SVM is effectively equal, which makes sense since both learn linear boundaries. Tuning mattered more than data cleaning: feature selection fixed KNN (+40 points), and choosing the right C and kernel lifted SVM by 7.5 points. Naive Bayes is last after tuning because its independence assumption does not fit this data. |

## Streamlit app

Link: https://2026ak05122-ml-assignment-2-vcw8yg8upjucuwxmnytgeu.streamlit.app/

`app.py` loads the five saved models from `model/` and scores an uploaded CSV. Nothing is retrained in the app.

- Upload a test CSV in the sidebar. It needs the 20 feature columns and `price_range`; extra columns are ignored. The bundled `test_data.csv` can be downloaded from the sidebar or used directly with a toggle.
- Pick a model from the dropdown.
- **Scorecard** tab: Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model.
- **Confusion matrix** and **Classification report** tabs for the chosen model.
- **All models** tab: the same file scored by all five models, with the best value in each column highlighted.
- **Predictions** tab: row-by-row predictions, a filter for the misses, and a CSV download.

Run locally:

```
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
app.py                  Streamlit app
model_training.ipynb    data checks, training, tuning, evaluation, saving the models
requirements.txt
test_data.csv           400-row test split (features + price_range)
data/                   full Kaggle file and the 1600-row training split
model/                  five saved models (.joblib) and the baseline/final metric tables
.streamlit/config.toml  app theme
```
