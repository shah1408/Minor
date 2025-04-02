# Minor
Project overview

This script performs audio genre classification using supervised machine learning techniques. It starts with feature extraction and preprocessing, followed by data visualization to understand the dataset. Finally, it trains four ML models and evaluates their performance. Feature Extraction & Preprocessing.

Dataset information

The dataset contains extracted audio features from music files, primarily used for genre classification. Features include spectral, rhythmic, and energy-related attributes. Key columns: filename: Name of the audio file (removed during processing). label: Genre of the audio file (used as the target variable). length: Duration of the audio file (removed as it is not used in modeling). spectral_centroid_mean: Indicates the center of mass of the spectrum, related to the brightness of the sound. rms_mean: Root Mean Square energy, representing the loudness of the signal. zero_crossing_rate_mean: Measures the rate at which the signal changes from positive to negative, indicating noisiness. spectral_bandwidth_mean: Measures the width of the spectrum, related to timbre. Additional MFCC (Mel-Frequency Cepstral Coefficients) features that capture important frequency characteristics.

Feature Extraction & Preprocessing:
Removal of non-feature columns (filename, label, length).
Encoding categorical target labels using LabelEncoder.
Standardization of numerical features using StandardScaler.
Splitting the dataset into training and testing sets (80/20 split).

Data Visualization:
Histogram: Distribution of spectral centroid values to analyze frequency balance.
Boxplot: RMS mean across different genres to assess dynamic range.
Heatmap: Feature correlation matrix to identify multicollinearity.
Scatter Plot: Relationship between zero-crossing rate and spectral bandwidth to understand texture.

Supervised Machine Learning Models:
Random Forest Classifier: An ensemble-based decision tree model.
Support Vector Machine (SVM): A linear classifier with margin maximization.
K-Nearest Neighbors (KNN): A distance-based classification approach.
Logistic Regression: A probabilistic model for multi-class classification.

Machine Learning Models Used

1. Random Forest Classifier An ensemble learning method based on multiple decision trees. Aggregates predictions from many trees to improve accuracy and reduce overfitting. Advantages: Handles complex data well, robust to noise.
2. Support Vector Machine (SVM) A linear classifier that finds the optimal hyperplane separating different classes. Works well for high-dimensional data. Advantages: Effective for small datasets, strong generalization.
3. K-Nearest Neighbors (KNN) A distance-based algorithm where a sample is classified based on the majority class of its nearest neighbors. Advantages: Simple, no need for model training, effective for well-separated classes.
4. Logistic Regression A probabilistic model that predicts class probabilities based on a linear decision boundary. Advantages: Efficient, interpretable, and performs well on linearly separable data.
Each model is trained on preprocessed audio features and evaluated using accuracy scores and classification reports

Model Evaluation

Models are assessed using accuracy score and classification reports.
-The accuracy score represents the overall correctness of predictions.
-The classification report includes:
-Precision: The proportion correct positive predictions.
-Recall: The ability of the model to detect all positive instances.
-F1-score: The harmonic mean of precision and recall, balancing both metrics.
These metrics provide a detailed performance evaluation for each genre classification.

Exploratory Data Analysis (EDA)

1. histograms - count VS tempo
2. boxplots - rms_mean VS label
3. scatter plots - mfcc1 VS mfcc2
4. correlation heatmaps - Feature Correlation

Results

The Random Forest Classifier achieved the highest accuracy, demonstrating its effectiveness in capturing complex relationships in the data.
SVM performed well for linearly separable data, though it may struggle with highly nonlinear patterns.
KNN had moderate performance, with accuracy dependent on the choice of k value.
Logistic Regression worked reasonably well but was outperformed by tree-based models.
The classification reports revealed that certain genres were easier to classify than others, depending on feature separability.
