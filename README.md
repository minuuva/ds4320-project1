# DS 4320 Project 1: Predicting Movie Ratings Using Collaborative Filtering

## Executive Summary
[To be completed]

**Author:** [Minu Choi]  
**NetID:** [qce2dp]  
**DOI:** [To be created]  
**License:** [MIT License](LICENSE)

### Quick Links
- **Press Release:** [Press Release](PRESS_RELEASE.md)
- **Data:** [OneDrive Data Folder](https://myuva-my.sharepoint.com/:f:/g/personal/qce2dp_virginia_edu/IgBsbp8bikwgRJCWBQ1DHrxQAdNoYJJKxI8g9_wMz_vkTgw?e=olv76k)
- **Pipeline:** [Analysis Pipeline](pipeline/)

---

## Problem Definition

### General Problem
Recommending content (e.g., Netflix) - General Problem #14 from rubric

### Specific Problem
Predicting personalized movie ratings on a 0.5-5.0 star scale for individual users by leveraging collaborative filtering patterns, temporal rating dynamics, and genre preferences to improve recommendation accuracy beyond baseline popularity-based and demographic approaches.

### Rationale for Refinement
The general problem of "recommending content" is too broad and encompasses multiple solution approaches (collaborative filtering, content-based filtering, hybrid methods, ranking systems, etc.). We refined this to focus specifically on **rating prediction** rather than binary like/dislike or ranking because:

1. **Granular Feedback:** Star ratings provide more nuanced user preference signals (a 5-star vs 2-star rating reveals intensity of preference) compared to implicit feedback (clicks, views) or binary indicators, enabling more precise personalization.

2. **Standardized Evaluation:** Rating prediction has well-established evaluation metrics (RMSE, MAE) that allow direct comparison with decades of research and industry baselines, particularly from the Netflix Prize competition which established RMSE benchmarks.

3. **Temporal Component:** By incorporating temporal dynamics (how user preferences evolve over time), we address a critical real-world challenge that simpler popularity-based approaches ignore—users change their tastes, and recent ratings should inform predictions more than decade-old ratings.

4. **Cold-Start Distinction:** Explicitly focusing on rating prediction allows us to measure and communicate performance separately for warm-start (established users/movies) versus cold-start (new users/movies) scenarios, which is crucial for deployment planning in production recommendation systems.

### Motivation
Streaming platforms like Netflix, Amazon Prime Video, and Disney+ collectively serve over 1.5 billion subscribers globally, with recommendation systems driving 70-80% of content consumption according to industry reports. A 1% improvement in recommendation accuracy translates to millions of dollars in retained subscribers and reduced churn. However, current systems struggle with three key challenges:

**Personalization at Scale:** With hundreds of thousands of titles and millions of users, generic popularity-based recommendations create filter bubbles and fail to surface niche content that individual users would love. Effective collaborative filtering can help users discover hidden gems outside mainstream trends.

**The Cold-Start Problem:** New users with few ratings and new releases with limited viewing history receive poor recommendations, creating a critical onboarding challenge. Understanding model performance degradation in cold-start scenarios enables better user experience design and hybrid recommendation strategies.

**Temporal Drift:** User preferences evolve—someone who loved action movies in college may prefer documentaries a decade later. Static models trained on all historical data equally fail to adapt, while temporal weighting and recency-aware models capture these shifts.

By predicting ratings with collaborative filtering on the MovieLens 25M benchmark dataset (the gold standard with 25 million ratings from 162,000 users), we can develop and evaluate recommendation algorithms that address these challenges, contributing to the broader goal of helping users discover content they'll genuinely enjoy rather than just popular mainstream titles.

**Press Release:** [Building a Smarter Movie Recommendation Engine](PRESS_RELEASE.md)

---

## Domain Exposition

### Terminology
| **Term** | **Definition** |
|----------|----------------|
| **Collaborative Filtering (CF)** | Recommendation technique that predicts user preferences by learning from patterns of similar users or items |
| **Content-Based Filtering** | Recommendation approach that suggests items similar to those a user previously liked, based on item features |
| **RMSE** | Root Mean Square Error - Primary evaluation metric measuring average magnitude of prediction errors; lower is better |
| **MAE** | Mean Absolute Error - Alternative evaluation metric measuring average absolute difference between predicted and actual ratings |
| **Cold Start Problem** | Challenge of making accurate recommendations for new users (few ratings) or new items (limited interaction history) |
| **Matrix Factorization** | Technique decomposing user-item rating matrix into lower-dimensional latent factor matrices to discover hidden patterns |
| **SVD** | Singular Value Decomposition - Mathematical matrix factorization method widely used in collaborative filtering |
| **ALS** | Alternating Least Squares - Optimization algorithm for matrix factorization that alternates between fixing user/item factors |
| **User-Based CF** | Collaborative filtering approach that recommends items liked by similar users |
| **Item-Based CF** | Collaborative filtering approach that recommends items similar to those the user previously liked |
| **Latent Factors** | Hidden features learned by matrix factorization (e.g., movie "genres" or user "preferences") not explicitly provided in data |
| **Explicit Feedback** | Direct user ratings (e.g., 1-5 stars) that clearly indicate preference strength |
| **Implicit Feedback** | Indirect signals of preference (e.g., clicks, views, watch time) without explicit rating |
| **Precision@K** | Percentage of top-K recommendations that are relevant (user would actually like) |
| **Recall@K** | Percentage of all relevant items that appear in top-K recommendations |
| **NDCG** | Normalized Discounted Cumulative Gain - Ranking metric that rewards relevant items appearing higher in recommendation list |
| **Netflix Prize** | 2006-2009 competition offering $1M prize for 10% RMSE improvement; established RMSE 0.8567 as gold standard |
| **Sparsity** | Challenge where most users rate very few items, creating a mostly empty user-item matrix |
| **Long Tail** | Distribution where few popular items receive most ratings/views, while many niche items have minimal data |
| **Filter Bubble** | Phenomenon where recommendation systems trap users in narrow preference zones, limiting content discovery |
| **Warm Start** | Scenario where sufficient rating history exists for users/items, enabling accurate predictions |
| **Baseline Model** | Simple prediction approach (e.g., user average, item average) used as performance benchmark |
| **Overfitting** | When model learns training data patterns too specifically, performing poorly on new unseen data |
| **Cross-Validation** | Technique splitting data into training/test sets multiple times to evaluate model generalization |

### Domain Overview
This project operates within the domain of recommender systems and collaborative filtering, a critical subfield of machine learning that powers content discovery for billions of users across digital platforms. Recommender systems address the fundamental challenge of information overload: with Netflix offering 15,000+ titles, Spotify hosting 100 million songs, and Amazon listing hundreds of millions of products, users need intelligent guidance to discover content matching their unique preferences. The domain gained prominence during the 2006-2009 Netflix Prize competition, which challenged researchers worldwide to improve rating prediction accuracy by 10% for a $1 million prize, ultimately advancing the field through innovations in matrix factorization, ensemble methods, and temporal dynamics modeling. Collaborative filtering represents the dominant approach in this domain, making predictions by identifying patterns across users and items rather than relying on explicit content features—if User A and User B both loved Movies X, Y, and Z, the system infers they have similar tastes and recommends Movie W (loved by User B) to User A. The MovieLens dataset from GroupLens Research serves as the benchmark standard for academic and industry research, providing a clean, well-documented testbed with explicit ratings that enable rigorous evaluation using established metrics like RMSE (Root Mean Square Error) and MAE (Mean Absolute Error). This domain sits at the intersection of data science, user experience design, and business strategy, where even small improvements in recommendation accuracy translate to measurable impacts on user engagement, subscription retention, and platform revenue, making it one of the most commercially valuable applications of machine learning in the modern digital economy.

### Background Reading
All background reading materials are available in this folder: [Background Readings Folder](https://drive.google.com/drive/folders/1gbzGKXDH2UGdPDgIPnDbU24Pla-HGTWc?usp=drive_link). 

The folder contains 5 research readings related to collobarative filtering, deep learning approaches, and recommender system evaluation. 

### Summary Table of Background Readings
| **Title** | **Brief Description** | **Link to File** |
|-----------|----------------------|------------------|
| A Survey on Deep Neural Networks in Collaborative Filtering Recommendation Systems | Comprehensive 2024 survey examining how Deep Neural Networks (MLPs, CNNs, RNNs, GNNs, Autoencoders, GANs) overcome limitations of traditional CF by modeling complex non-linear relationships. Covers modern architectures including graph neural networks and transformer-based approaches. | [Link 1](https://drive.google.com/file/d/1PevLWSt3iLJZvBU_oF8QhwKqZuX2HUaC/view?usp=sharing) |
| Our Model Achieves Excellent Performance on MovieLens: What Does It Mean? | Critical 2024 analysis revealing that MovieLens records user-platform interactions rather than true user-movie preferences, questioning whether excellent MovieLens performance generalizes to real-world scenarios. Essential reading for understanding dataset limitations and evaluation validity. | [Link 2](https://drive.google.com/file/d/1z6jNgQoR8OO1Fy8iuxj-WVbScBrOmUfT/view?usp=drive_link) |
| Graph Neural Networks in Recommender Systems: A Survey | Comprehensive 2023 survey by Gao et al. reviewing GNN-based recommender systems, covering graph construction, embedding propagation, model optimization, and computational efficiency. Discusses spectral and spatial GNN models for capturing collaborative signals in user-item interaction graphs. | [Link 3](https://drive.google.com/file/d/1olG56ltS1uaVt0m3LBCiei29Lx601xF3/view?usp=drive_link) |
| Cold-Start Recommendation towards the Era of Large Language Models: A Comprehensive Survey and Roadmap | January 2025 survey examining how cold-start solutions evolved from content features and graph relations to leveraging large language models' world knowledge. Provides roadmap for addressing new user/item challenges with modern deep learning approaches. | [Link 4](https://drive.google.com/file/d/1ba4KQp59hC4THRgsBrE6XquwZu2IBi9w/view?usp=drive_link) |
| A Survey of Graph Neural Networks for Social Recommender Systems | December 2022 survey by Sharma et al. examining GNN-based social recommender systems leveraging both user-item interactions and user-user social relations. Analyzes 80 papers with detailed taxonomy of input types, GNN encoders, and evaluation metrics for social recommendation tasks. | [Link 5](https://drive.google.com/file/d/1jNTWOTxueeASSgiNxLW-49gbWGuCuGYo/view?usp=drive_link) |
---

## Data Creation

### Data Provenance

This dataset was obtained from GroupLens Research at the University of Minnesota, a well-established research laboratory specializing in recommender systems and collaborative filtering. The data was downloaded directly from their official repository at https://grouplens.org/datasets/movielens/25m/ on March 10, 2026.

The MovieLens 25M dataset contains 25 million ratings and 1 million tag applications applied to 62,000 movies by 162,541 users between January 9, 1995 and November 21, 2019. All ratings were collected through the MovieLens website (movielens.org), a non-commercial movie recommendation service operated by GroupLens Research. Users voluntarily provided ratings on a 0.5-5.0 star scale in half-star increments, with timestamps recorded in Unix epoch format for each rating event.

The dataset was downloaded as a 265 MB ZIP archive (`ml-25m.zip`), which expands to approximately 1.1 GB of CSV files. No preprocessing, filtering, or modification was applied to the raw data—this represents the complete, unmodified dataset as published by GroupLens. The only derived table is `users.csv`, which aggregates statistics from the ratings table to create an explicit user entity for improved relational modeling.

### Code Documentation

| File | Description | Location |
|------|-------------|----------|
| data_acquisition.ipynb | Complete data acquisition pipeline that downloads MovieLens 25M dataset from GroupLens, extracts CSV files, creates derived users table from ratings, converts to Parquet format, and verifies all rubric requirements. Includes comprehensive error handling, logging, and data validation. | [code/data_acquisition.ipynb](code/data_acquisition.ipynb) |

### Bias Identification

Several sources of systematic bias exist in this dataset that must be acknowledged:

**Selection Bias:** Users self-selected to participate in the MovieLens platform and actively chose to rate movies, representing a subset of the population with above-average interest in films and willingness to provide ratings. This creates a bias toward movie enthusiasts rather than casual viewers, potentially skewing rating distributions higher than the general population.

**Rating Bias:** Users demonstrate a tendency to rate movies they enjoyed or strongly disliked, creating a bimodal distribution. Movies with few ratings may not be representative of broader opinion, as neutral experiences are less likely to generate ratings. This "voluntary response bias" means the dataset overrepresents strong opinions.

**Temporal Bias:** The 24-year data collection period (1995-2019) reflects changing movie consumption patterns, rating behaviors, and platform evolution. Earlier users may have systematically different rating patterns than recent users due to platform maturity, user interface changes, and shifting cultural norms around online reviews.

**Popularity Bias:** Popular mainstream movies receive disproportionately more ratings than niche or independent films, creating a highly skewed long-tail distribution. The top 1% of movies account for a significant portion of all ratings, while thousands of films have minimal coverage.

**Survivorship Bias:** Only movies that users chose to watch and subsequently rate appear in the dataset. Films that users deliberately avoided or never heard of are completely missing, creating an incomplete picture of true preferences and potentially overstating quality assessments.

**Genre Bias:** Certain genres (Drama, Comedy, Action) dominate the dataset while others (Documentary, Foreign Language, Silent Films) are significantly underrepresented, reflecting both platform demographics and broader Hollywood production biases.

### Bias Mitigation

To address the identified biases and ensure robust analysis, we implement the following mitigation strategies:

**Minimum Rating Threshold:** Apply a minimum threshold of 10 ratings per movie to filter out noise from the long tail and reduce the impact of individual outliers. This threshold can be adjusted based on analysis requirements and will be documented in the analysis pipeline.

**Temporal Segmentation:** Segment data by time periods (e.g., 1995-2005, 2006-2015, 2016-2019) to identify and account for temporal trends in rating behavior. Model training will focus on recent years (2015-2019) to ensure predictions reflect current user preferences while maintaining awareness of historical patterns.

**Stratified Sampling:** Use stratified sampling across genres, release years, and popularity tiers when creating training/test splits to ensure underrepresented categories receive adequate representation in model evaluation. This prevents the model from overfitting to popular mainstream content.

**Weighted Rating Systems:** Implement inverse frequency weighting to give more influence to ratings of less popular movies during model training, partially counteracting popularity bias and improving recommendations for niche content.

**User Normalization:** Normalize ratings by each user's mean and standard deviation to account for individual rating tendencies. Some users consistently rate high (grade inflation) while others rate low (harsh critics), and normalization helps isolate true preference signals from rating style.

**Cold-Start Awareness:** Develop separate evaluation metrics and handling strategies for new users (fewer than 10 ratings) and new movies (fewer than 10 ratings) to explicitly measure and communicate model performance limitations in cold-start scenarios rather than masking this weakness.

**Transparency in Limitations:** All bias mitigation strategies and their parameters will be documented in the analysis pipeline with explicit acknowledgment of remaining limitations. Model performance will be evaluated separately across different movie popularity tiers and genres to identify where biases persist.

### Rationale for Critical Decisions

**Dataset Selection:** We chose MovieLens 25M over custom data collection to ensure reproducibility, leverage well-documented provenance, and access sufficient scale (1+ GB) with clean relational structure. Custom web scraping would introduce ethical concerns around terms of service violations, create data quality issues, and reduce reproducibility. MovieLens provides the gold standard benchmark for recommender system research with extensive academic validation.

**User Table Derivation:** While users are implicit in the original dataset, we explicitly created a `users` table by aggregating rating statistics (`num_ratings`, `avg_rating`, `first_rating_date`, `last_rating_date`). This decision improves the relational model's clarity and enables user-centric analysis. The derived statistics are deterministic calculations from the ratings table with no introduced uncertainty beyond the underlying rating data.

**Timestamp Preservation:** Unix epoch timestamps were retained in their original format rather than converting to datetime during data acquisition. This preserves data fidelity for reproducibility, enables efficient storage and computation, and supports temporal analysis without precision loss. Conversion to human-readable formats occurs in the analysis pipeline where needed.

**No Filtering During Acquisition:** We deliberately preserved all movies and ratings without applying minimum threshold filters during data acquisition. This maintains dataset integrity and allows analysts to apply their own filtering criteria. Long-tail movies may be valuable for specific use cases (e.g., niche recommendation engines) even if excluded from general model training.

**Genome Data Inclusion:** We included the optional `genome-scores` and `genome-tags` tables despite adding 415 MB of data. These tables provide machine-generated tag relevance scores that enable content-based filtering approaches beyond pure collaborative filtering. This enriches the dataset for hybrid recommendation models while remaining optional for basic implementations.

**Parquet Alongside CSV:** Both CSV and Parquet formats were generated (rubric requirement) despite 81.8% redundancy. CSV ensures broad compatibility and human readability for validation, while Parquet provides efficient compression and faster read performance for large-scale analysis. This dual format approach optimizes for both accessibility and performance.

**Uncertainty Sources:** Primary sources of uncertainty include: (1) user rating subjectivity and variance across repeat viewings, (2) temporal drift in user preferences over the 24-year collection period, (3) incomplete coverage of the movie universe (only ~62K of millions of films), and (4) demographic skew of the MovieLens user base versus general population. These uncertainties are inherent to observational user behavior data and cannot be fully eliminated, only acknowledged and accounted for in modeling decisions.

---

## Metadata

### Schema
[To be completed]

### Data Tables
[To be completed]

### Data Dictionary
[To be completed]

---

## Repository Structure