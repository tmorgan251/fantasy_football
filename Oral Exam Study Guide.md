**Oral Exam Study Guide**  
SIADS 501: Being a Data Scientist

1. What is the four-stage pipeline and how does it apply to your project?   

**Answer:** The four-stage pipeline is:
1. **Problem Formulation**: We formulated the problem as "How do draft, waiver wire, and start/sit decisions contribute to fantasy football success?" We defined baselines (autodraft, Q25 waiver rate, projected-optimal lineups) to measure decision value.

2. **Data Collection and Cleaning**: 
   - Collected data from ESPN API (drafts, lineups, transactions) and nfl_data_py (injuries)
   - Cleaned by removing duplicates, normalizing player names/slots, handling missing values, filtering to standard leagues

3. **Data Analysis and Modeling**:
   - Computed expected values using autodrafted teams as baseline
   - Analyzed injury impact using point differentials
   - Calculated waiver wire and start/sit baselines
   - Used robust statistics (trimmed mean, MAD-based z-scores) and polynomial regression

4. **Presenting and Integration**:
   - Created visualizations (heatmaps, distributions, cumulative plots)
   - Integrated findings into actionable insights for fantasy managers
   - Documented methodology and limitations in README

2. Explain how the law of small numbers applies to the work you did for your project.

**Answer:** The law of small numbers refers to overinterpreting patterns from small samples, assuming they represent the population. In our project:
- **Trade analysis**: If we analyzed only a small subset of trades, we might incorrectly infer that trades are always beneficial or always harmful. We avoided this by not completing trade analysis (recognizing insufficient data).
- **Injury type analysis**: We set `min_population=30` in heatmaps to avoid drawing conclusions from rare injury combinations (e.g., "Ankle injury for Kickers" with only 5 observations).
- **League filtering**: We aggregated across many leagues (hundreds) rather than analyzing individual leagues, avoiding overinterpreting patterns from single leagues.
- **Position-specific analysis**: We ensure sufficient sample sizes before making position-specific claims (e.g., "RB injuries have X impact" requires enough RB injury cases).

3. What sources of bias did you identify in your project?

**Answer:** Multiple sources of bias:
- **Baseline bias**: Our autodraft baseline represents minimal competency, not optimal play. This biases our "points added" metrics - we're measuring against a low bar, not true optimal value.
- **Noisy data**: ESPN data may have errors (incorrect autodraft flags, missing transactions, lineup errors). Injury data from nfl_data_py may have reporting inconsistencies across teams.
- **Default algorithms**: ESPN's autodraft algorithm may have systematic biases (e.g., overvaluing certain positions). Our polynomial regression assumes a specific curve shape.
- **ESPN projections**: Projections may be systematically biased (e.g., always overestimate RB performance). Our start/sit analysis depends entirely on projection quality.
- **Selection bias**: We only analyzed public leagues matching specific criteria (10 teams, 1.0 PPR). This may not represent all fantasy leagues.
- **Temporal bias**: Data from 2021-2024 may not represent future seasons (rule changes, player trends). 

4. To what degree did you follow the "10 Rules for Creating Reproducible Results in Data Science" in your project work?

**Answer:** We followed most of the rules:
- **Version Control**: Code stored in git with clear commit history
- **Documentation**: Comprehensive docstrings, README, and inline comments explaining every line
- **Reproducible Environment**: Requirements documented (pandas, numpy, scipy, matplotlib, seaborn, espn-api, nfl-data-py)
- **Modular Code**: Separated into classes (DraftValueAnalyzer, InjuryAnalyzer, NotebookWorkflow) with clear responsibilities
- **Data Versioning**: Raw data stored in year-specific directories, preprocessed data saved with clear naming
- **Deterministic Processing**: Used random seeds where applicable (e.g., in league harvesting)
- **Clear Workflow**: NotebookWorkflow class provides one-line execution of major pipeline steps
- **Error Handling**: Try-except blocks and validation checks throughout
- **Output Documentation**: All CSV outputs include metadata about processing steps
- **Code Review**: Code reviewed for clarity and efficiency before final submission

Areas for improvement: Could have used more explicit random seed documentation and environment.yml for exact package versions.

5. How did you apply data cleaning principles in your project?

**Answer:** We applied comprehensive data cleaning:
- **Duplicate Removal**: `clean_raw_data()` method drops duplicates from CSV files before processing
- **Name Normalization**: `normalize_player_name()` uses regex (`re.sub(r"\s+", " ", s)`) to standardize whitespace, removes "Jr."/"Sr." suffixes, and applies synonym dictionaries
- **Slot Normalization**: `normalize_slot()` standardizes position designations (DST/DEF/D/ST → D/ST, BE/BENCH → BE)
- **Type Conversion**: Explicit type casting (e.g., `pd.to_numeric()`, `.astype(int)`) with error handling
- **Missing Value Handling**: Used `.fillna()` with appropriate defaults (0 for numeric, "UNKNOWN" for categorical)
- **Filtering Outliers**: Removed leagues with non-standard scoring rules using robust z-scores (MAD-based)
- **Data Validation**: Required column checks before processing to catch data quality issues early
- **Deduplication After Merges**: Dropped duplicates on key columns (League_ID, Year, Overall) after enrichment merges 

6. Are there places where overfitting may have played a role in your analyses?

**Answer:** We avoided overfitting through several strategies:
- **Polynomial regression**: We use degree-4 polynomial instead of fitting every data point. The polynomial captures general trend without overfitting to noise in individual picks.
- **Pooled denoised baseline**: We pool data across all years and use robust estimators (trimmed mean) to reduce impact of outliers, preventing overfitting to year-specific anomalies.
- **Smoothing**: Rolling average smoothing (window=5) reduces noise before polynomial fitting.
- **Cross-validation concept**: While we didn't formally cross-validate, we test on multiple years (2021-2024) to ensure patterns hold across seasons.
- **Simple models**: We use simple aggregations (mean, median) rather than complex machine learning models that could overfit.

**Potential overfitting risks we avoided**: (1) Year-specific expected curves (we pool across years), (2) Fitting to every pick individually (we use polynomial), (3) Complex injury models (we use simple aggregations). 

7. What is cross-validation?

**Answer:** Cross-validation splits data into folds (typically k=5 or k=10), trains on k-1 folds, tests on remaining fold, repeats for all folds. Purpose: Test model generalization, avoid overfitting to training data.

**In our project**: We didn't use formal cross-validation because:
- We're not building predictive models (we're doing descriptive analysis)
- We test across multiple years (2021-2024) which serves similar purpose
- Our "test" is whether patterns hold across different seasons

**If we did use CV**: We could split leagues into folds, compute expected values on training folds, test on held-out fold to see if draft value patterns generalize.

8. What is p-hacking and does it apply to your project work?

**Answer:** P-hacking is repeatedly testing different hypotheses or data subsets until finding statistical significance (p < 0.05), then reporting only that result. It inflates false positive rates and produces non-reproducible results.

**In our project**: We avoided p-hacking by:
- **Pre-defined analysis plan**: We decided upfront to analyze draft value, injury impact, and waiver baselines - didn't test many hypotheses and cherry-pick
- **No p-values**: We focus on effect sizes (points added) rather than statistical significance, avoiding the temptation to p-hack
- **Transparency**: We report all analyses, not just "significant" ones
- **Effect sizes over significance**: We care more about "how much value" than "is it significant"

**If we had used p-values**: We would need to adjust for multiple comparisons (Bonferroni correction) since we test multiple positions, rounds, injury types, etc. 

9. What is the relationship between correlation and causation?

	Correlation does not equal causation. 

10. How did you engage in the process of storytelling when producing your project report?

**Answer:** We structured the narrative around a fantasy manager's decision-making journey:
- **Opening**: Problem statement - "How do I make better fantasy decisions?"
- **Act 1 - Draft**: "Does my draft matter?" → Show human advantage over autodraft, especially early rounds
- **Act 2 - Waiver Wire**: "Can I find value on waivers?" → Show waiver wire contributes meaningful points
- **Act 3 - Start/Sit**: "Am I setting good lineups?" → Show managers generally make good decisions
- **Climax**: Cumulative impact - all three decision points matter
- **Resolution**: Actionable insights - "Focus on early draft rounds, be active on waivers, trust your start/sit instincts"

**Storytelling elements**:
- **Character**: The fantasy manager (reader)
- **Conflict**: Uncertainty about decision value
- **Journey**: Through draft → waiver → start/sit analysis
- **Revelation**: Quantified value of each decision type
- **Transformation**: From uncertainty to data-driven strategy 

11. Discuss the role of uncertainty in your project.

**Answer:** Uncertainty appears at multiple levels:
- **Injury randomness**: Injuries are unpredictable - a player's injury status, recovery time, and performance impact vary widely. We quantify average impact but acknowledge high variance (shown in distributions with wide confidence intervals).
- **Draft variance**: Even at the same pick, player performance varies significantly (shown in expected value distributions with P10-P90 ranges). A pick's value is uncertain until the season plays out.
- **Projection uncertainty**: ESPN projections are estimates with error. Our start/sit analysis depends on projection accuracy - if projections are systematically wrong, our metrics are biased.
- **Data quality uncertainty**: We don't know if ESPN data is 100% accurate (autodraft flags, transaction logs, lineup data). Errors introduce uncertainty in our baselines.
- **Model uncertainty**: Our polynomial regression assumes a specific curve shape. Different degrees or methods might produce different expected values.
- **Sampling uncertainty**: We analyze a subset of leagues. Results might differ if we analyzed all ESPN leagues.

**How we address uncertainty**: (1) Show distributions not just means, (2) Use robust statistics (trimmed mean, MAD), (3) Acknowledge limitations, (4) Show variance in visualizations (confidence intervals, percentile ranges).

SIADS 503: Data Science Ethics

1. List 5 misconceptions about data science ethics and explain how each of them might apply to your project.

**Answer:**
1. **"Ethics is optional"**: We had a responsibility to handle ESPN API data ethically, respecting rate limits and not scraping private leagues without permission. We only used public leagues and implemented delays/retries to avoid overwhelming servers.

2. **"Data is neutral"**: Our baselines (autodraft, Q25 waiver rate) introduce bias - they represent minimal competency, not optimal play. This bias affects all our conclusions about draft value and waiver wire decisions.

3. **"More data is always better"**: We filtered to standard leagues (10 teams, 1.0 PPR) rather than including all leagues, recognizing that including non-standard formats would introduce confounding variables that make results less meaningful.

4. **"If it's public, it's fair game"**: Even though ESPN leagues were public, we anonymized manager names in our analysis and didn't publish identifying information. We also included a disclaimer that this is for educational purposes only, not gambling advice.

5. **"Technical correctness equals ethical correctness"**: Our injury analysis could be misused for gambling. We explicitly stated in our README that this is educational only and should NOT be used for gambling or betting purposes. 

2. Comment on the importance of data privacy in your project work.

	Data was protected by ESPN, but not perfectly. For example, public 		teams occasionally had manager names

3. Describe how your project could be used as a case study about bias.

**Answer:** Our project demonstrates multiple types of bias:
- **Baseline bias**: We use autodraft as "minimal competency" baseline, but this assumes autodraft represents true minimal skill. If autodraft is worse than actual minimal skill, we overestimate human advantage. If autodraft is better, we underestimate it. This is a form of selection bias in baseline choice.

- **Algorithmic bias**: ESPN's autodraft algorithm may have systematic biases (e.g., overvaluing certain positions based on default rankings). Our analysis inherits these biases.

- **Measurement bias**: We measure "points added" but this assumes our expected value calculations are unbiased. If our polynomial regression is wrong, all our measurements are biased.

- **Sampling bias**: We only analyze leagues matching our criteria (10 teams, 1.0 PPR). This may not represent all fantasy leagues - we're biased toward standard formats.

- **Temporal bias**: Analyzing 2021-2024 may not represent future seasons. Fantasy football evolves (rule changes, player trends), so our findings may be time-bound.

**Case study value**: Shows how bias can propagate through an analysis pipeline - from data collection (league selection) → baseline definition (autodraft) → measurement (expected values) → interpretation (points added). Each stage introduces potential bias that compounds. 

4. With respect to Data Provenance, Aggregation, and Trust, what does your project "leave out"?

**Answer:** Our project leaves out several important elements:
- **Trade Analysis**: We identified trades but didn't complete the trade baseline analysis (only Draft, Waiver, Start/Sit were fully analyzed)
- **Individual League Context**: We aggregated across all leagues, losing league-specific context (draft strategies, manager skill levels, league competitiveness)
- **Temporal Dynamics**: We treat all weeks equally, missing how strategies evolve during the season or how early-season performance affects later decisions
- **Player Context**: We don't account for matchups, weather, or other contextual factors that affect performance
- **Projection Methodology**: We use ESPN projections but don't know how they're calculated, creating a "black box" dependency
- **Injury Severity Grading**: We categorize injuries by type/status but don't have detailed severity metrics
- **Waiver Priority/FAAB**: We track adds but don't know waiver priority or FAAB bid amounts, missing strategic context

5. Discuss issues around data provenance with respect to the data you used for your project.

**Answer:** Several data provenance issues:
- **ESPN API Data**: We don't control the source - ESPN could change their API structure, scoring rules, or data availability at any time. We documented the API endpoints used but can't guarantee future access.
- **Injury Data (nfl_data_py)**: We rely on a third-party package that scrapes/aggregates NFL injury reports. The original source (NFL injury reports) may have inconsistencies in reporting across teams.
- **Projection Data**: ESPN projections are proprietary - we don't know the methodology, update frequency, or how they account for injuries/matchups. This creates uncertainty in our start/sit baseline.
- **Data Collection Timing**: Data was collected at different times, potentially missing mid-season corrections ESPN might make to historical data.
- **League Selection Bias**: We used seeded sampling (near known good league IDs) which may introduce geographic or demographic biases in league selection.
- **Missing Metadata**: We don't track when data was collected, API version used, or if ESPN made retroactive corrections to historical data.

6. What role does trust play in the analyses that you did for your project?

**Answer:** Trust is critical at multiple levels:
- **Trust in ESPN Data**: We must trust that ESPN's draft data correctly identifies autodrafted picks (via `autoDraftTypeId`), that lineup data accurately reflects who started, and that transaction logs correctly capture waiver adds/drops. If ESPN's data is wrong, all our baselines are wrong.

- **Trust in Injury Data**: We trust that nfl_data_py accurately represents NFL injury reports and that injury statuses (Questionable/Doubtful/Out) are correctly recorded and timed relative to games.

- **Trust in Projections**: Our start/sit analysis depends entirely on ESPN projections being reasonable. If projections are systematically biased (e.g., always overestimate RB performance), our start/sit metrics are meaningless.

- **Trust in Our Code**: We must trust that our name normalization correctly matches players across datasets, that our optimal lineup algorithm correctly selects best players, and that our filtering doesn't introduce systematic biases.

- **Trust in Statistical Methods**: We use robust estimators (trimmed mean, MAD-based z-scores) assuming they handle outliers appropriately, and polynomial regression assuming it captures the true expected value curve.

- **Trust for Reproducibility**: Future researchers must trust that our documented process will produce the same results, which requires trust in our code, data sources, and methodology.

7. Comment on issues arising from the publication of results stemming from your project.

**Answer:** Several publication concerns:
- **Gambling Misuse**: Our findings could be misused for sports betting or daily fantasy sports gambling. We included explicit disclaimers, but publication still risks enabling harmful gambling behavior.

- **Overconfidence in Results**: Publishing "draft value" findings might lead managers to over-optimize on draft day, potentially making worse decisions by following our patterns too rigidly. Past performance doesn't guarantee future results.

- **Privacy Concerns**: Even with public leagues, publishing aggregated patterns might reveal information about individual managers or leagues that they didn't intend to share.

- **Methodological Limitations**: Our baselines (autodraft, Q25) are arbitrary choices. Publishing without emphasizing these limitations could mislead readers about the "correctness" of our approach.

- **Reproducibility**: If ESPN changes their API or data structure, our results become non-reproducible, potentially misleading future readers.

- **Educational vs. Commercial Use**: We must be clear this is educational research, not a commercial product. Publication in academic contexts is appropriate, but commercial use would require different considerations.

- **Bias Amplification**: Publishing findings about "optimal" strategies might amplify existing biases in fantasy football (e.g., overvaluing certain positions) if not carefully contextualized.

SIADS 505: Data Manipulation

1. Describe how regular expressions can be used to analyze text-based data.  

**Answer:** In our project, we used regex in `normalize_player_name()`: `re.sub(r"\s+", " ", s)` replaces multiple whitespace characters with a single space. Regex is also used implicitly in pandas string methods like `.str.contains('Not injury related', case=False, na=False)` for filtering injury data. Regex helps standardize inconsistent text formatting (e.g., "Josh  Allen" → "Josh Allen") which is critical for matching player names across datasets.

2. Describe some of the functionality that NumPy provides and explain how it is related to the pandas library.  

**Answer:** NumPy provides array operations, mathematical functions, and statistical computations. Pandas is built on NumPy - DataFrames use NumPy arrays internally. NumPy functions we used: `np.nanmedian()` for robust statistics, `np.polyfit()` for polynomial regression, `np.where()` for conditional logic, `np.isfinite()` for validation. Pandas provides the DataFrame interface, but NumPy does the heavy computational lifting.

3. Provide an example of how you can use the NumPy library in the analysis of your project data.  

**Answer:** In `_fit_polynomial_baseline()`, we use `np.polyfit(x, y, deg=degree, w=w)` to fit a weighted polynomial to expected draft values. We also use `np.nanmedian()` in robust z-score calculations to handle outliers, and `np.where()` for conditional assignments like `np.where(osel["SelectedOptimal"], osel["WeekPoints"], 0.0)` to compute valid points.

4. What are some of the challenges associated with using NumPy  

**Answer:** Challenges: (1) Type handling - NumPy arrays require homogeneous types, requiring explicit casting; (2) NaN handling - operations like `np.mean()` return NaN if any NaN present, requiring `np.nanmean()`; (3) Memory - large arrays can be memory-intensive; (4) Broadcasting rules can be confusing; (5) Integration - converting between pandas and NumPy requires `.to_numpy()` calls and can lose index information.

5. Describe the pandas Series and DataFrame objects and explain how they are related to each other.  

**Answer:** A Series is a 1D labeled array (like a column), a DataFrame is a 2D labeled structure (like a table) made of Series. Each DataFrame column is a Series. In our project, `df["Player_norm"]` returns a Series, while `df[["League_ID", "Year", "Player"]]` returns a DataFrame. We use Series for single-column operations (e.g., `.value_counts()`) and DataFrames for multi-column operations (e.g., `.groupby().agg()`).

6. How does indexing work with pandas DataFrames and Series? Provide examples of indexing that you used in your project.  

**Answer:** We use multiple indexing methods:
- **Column indexing**: `df["Player"]`, `df[["League_ID", "Year"]]` (single vs multiple columns)
- **Boolean indexing**: `df[df["Is_Autodrafted"] == 0]` filters rows
- **Label-based**: `df.loc[df["has_injury"], "injury_status"] = "No Injury"` sets values conditionally
- **Position-based**: `df.iloc[0:10]` for first 10 rows
- **Multi-index**: After `groupby().agg()`, we flatten with `.reset_index()`
- **Index operations**: `df.set_index(["League_ID", "Year"])` for hierarchical indexing

7. List different ways to deal with missing values and explain when each of those is appropriate. Provide examples from your project of how you either had to, or, if you didn't have missing values, how you would have, dealt with missing values.  

**Answer:** We used multiple strategies:
- **`.fillna(0.0)`**: For numeric columns where 0 is meaningful (e.g., `Season_Total_Points_Valid.fillna(0.0)` - player didn't score)
- **`.fillna("UNKNOWN")`**: For categorical (e.g., `Position.fillna("UNKNOWN")` - position couldn't be inferred)
- **`.fillna(False)`**: For boolean (e.g., `has_injury.fillna(False)` - no injury data = healthy)
- **`.dropna()`**: When rows with missing values are invalid (e.g., after merges, we drop duplicates)
- **`.notna()` filtering**: `df[df["Player"].notna()]` to keep only rows with valid player names
- **Conditional logic**: `np.where(condition, value_if_true, value_if_false)` for conditional filling

8. What are some common techniques that you use to manipulate pandas data structures.  

**Answer:** Common techniques in our project:
- **`.copy()`**: Always copy before modifying to avoid SettingWithCopyWarning
- **`.merge()`**: Join datasets on keys (e.g., `draft.merge(lineups, on=["League_ID", "Year"])`)
- **`.groupby().agg()`**: Aggregate by groups (e.g., `groupby("Overall").agg(mean=("Points", "mean"))`)
- **`.apply()` / `.map()`**: Transform values (e.g., `df["Player_norm"] = df["Player"].map(normalize_player_name)`)
- **`.pivot_table()`**: Reshape data (e.g., in injury heatmaps: `pivot(index='injury_type', columns='position')`)
- **`.sort_values()`**: Order data (e.g., `sort_values(["Year", "Overall"])`)
- **`.drop_duplicates()`**: Remove duplicates
- **String methods**: `.str.upper()`, `.str.strip()`, `.str.contains()` for text processing

9. Describe how you used merging or joining in your project. Suggest ways in which the merging could have been made more efficient.  

**Answer:** We used merges extensively:
- `draft.merge(lineups, on=["League_ID", "Year", "Player_norm"])` to add position/points
- `waiver_stints.merge(optimal_selected, on=["League_ID", "Year", "Team", "Player_norm"])` for waiver points
- `injury_df.merge(lineup_df, left_on=['season', 'week', 'player_norm'], right_on=['Year', 'Week', 'player_norm'])` for injury analysis

**Efficiency improvements:**
- **Index before merge**: `df.set_index(["League_ID", "Year"])` then merge on index (faster than column merge)
- **Filter before merge**: Reduce DataFrame size first (e.g., `lineups[lineups["Is_Starter"]==1]` before merging)
- **Specify `suffixes`**: Avoid column name conflicts upfront
- **Use `pd.concat()` for same-structure DataFrames**: Faster than iterative merges
- **Sort keys before merge**: Can enable faster merge algorithms
- **Use `how="inner"` when possible**: Smaller result set than outer joins

10. How do/did you decide to use group by when performing exploratory data analysis. What are some characteristics of variables that you can use to group by?  

**Answer:** We group by variables that:
- **Define natural categories**: `groupby("Position")` to analyze by position, `groupby("Year")` for temporal trends
- **Create meaningful aggregations**: `groupby(["League_ID", "Year", "Team"])` to get team-level totals
- **Enable comparison**: `groupby("Is_Autodrafted")` to compare manual vs autodraft
- **Represent hierarchical structure**: `groupby(["Year", "Overall"])` for pick-level analysis within years

**Characteristics of good grouping variables:**
- **Categorical or discretized**: Position, Year, Round (not continuous like Points)
- **Low to moderate cardinality**: Not too many unique values (e.g., 10 positions, not 1000+ player names)
- **Meaningful for analysis**: Groups should answer a question (e.g., "How does draft value vary by round?")
- **Stable**: Values shouldn't change during analysis (e.g., League_ID, not Week which changes per row)

11. Pivot tables are often used to summarize data. Describe the process of taking data in "long" form and using a pivot table to convert it to "wide" form.  

**Answer:** In our injury heatmaps, we convert long → wide:
- **Long form**: Each row is (injury_type, position, avg_point_differential)
- **Pivot**: `heatmap_data.pivot(index='injury_type', columns='position', values='point_differential')`
- **Result**: Wide form with injury_type as rows, position as columns, values in cells

**Process:**
1. Identify index (rows): `injury_type`
2. Identify columns: `position` 
3. Identify values: `point_differential`
4. Aggregate if needed: `groupby().mean()` first if multiple values per cell
5. Handle missing: `.fillna()` or leave as NaN for heatmap visualization

We also use `pivot_table()` with `aggfunc="sum"` when we need aggregation during pivoting.

12. Time series functionality in pandas allows you to "upscale" and "downscale" time-based data. Explain how these functions work and indicate when it's appropriate to use them.

**Answer:** 
- **Upscaling (downsampling)**: Aggregate higher frequency → lower (e.g., daily → weekly). Use `resample("W").sum()` or `.mean()`. We could use this to aggregate weekly lineup data to season totals.
- **Downscaling (upsampling)**: Lower frequency → higher (e.g., weekly → daily). Use `resample("D").ffill()` or interpolation. Less common in our project.

**When appropriate:**
- **Upscaling**: When you need season-level aggregates from weekly data, or when granularity is too fine for analysis
- **Downscaling**: When you need to align datasets with different frequencies, or fill in missing time points

**Our project**: We work with weekly data (Week 1-17) and aggregate to season level using `groupby(["League_ID", "Year", "Team"]).sum()`, which is conceptually similar to upscaling but done manually rather than using time series resampling (since we don't have datetime indexes, just week numbers).  
13. Describe the process of generating hypotheses, including how you come up with a null hypothesis. Why are null hypotheses used?  

**Answer:** 
**Hypothesis generation process:**
1. **Observation**: Human drafters seem to outperform autodraft in early rounds
2. **Research question**: "Do human drafters add value over autodraft baseline?"
3. **Alternative hypothesis (H1)**: Human drafters score more points than autodraft baseline
4. **Null hypothesis (H0)**: Human drafters score the same as autodraft baseline (no difference)

**Why null hypotheses:**
- Provides a baseline for statistical testing
- Allows us to quantify evidence against the null
- Prevents confirmation bias by assuming no effect until proven otherwise
- Enables p-value calculation and significance testing

**In our project**: We didn't use formal hypothesis testing, but our analysis implicitly tests H0: "Manual picks = Autodraft expected value" vs H1: "Manual picks ≠ Autodraft expected value" by computing points added over expected.

14. A t-test is often used to look for differences between two groups. What are some of the assumptions about the data that need to be met in order to conduct a statistically valid t-test.  

**Answer:** Assumptions:
1. **Independence**: Observations must be independent (violated if same player appears multiple times)
2. **Normality**: Data should be approximately normally distributed (we use robust methods instead)
3. **Equal variances**: Two groups should have similar variance (homoscedasticity)
4. **Random sampling**: Data should be randomly sampled (our leagues are convenience sample)
5. **Continuous data**: Dependent variable should be continuous (Points is continuous, but may have outliers)

**Why we didn't use t-tests**: Our data violates assumptions (non-normal distributions, outliers, non-independent observations - same players across leagues). Instead, we used robust statistics (trimmed mean, MAD-based z-scores) and visual comparisons.

15. Explain what p-hacking is and why it's a bad thing.  

**Answer:** P-hacking is repeatedly testing different hypotheses or data subsets until you find a statistically significant result (p < 0.05), then reporting only that result. It's bad because:
- **False discoveries**: By chance, 5% of tests will be significant even if null is true
- **Inflated false positive rate**: Multiple tests increase chance of false significance
- **Non-reproducible**: Results won't replicate in new data
- **Misleading conclusions**: Appears to find effects that don't actually exist

**In our project**: We avoided p-hacking by:
- Pre-defining our analysis plan (draft value, injury impact, waiver baselines)
- Not running multiple tests and cherry-picking significant ones
- Using effect sizes (points added) rather than p-values
- Being transparent about all analyses, not just "significant" ones

16. The scipy package, while providing some useful statistical functionality, isn't as widely used one might think. Suggest reasons why that's the case.

**Answer:** Reasons scipy is less used:
1. **Pandas/NumPy cover most needs**: Basic stats (mean, std, median) are in pandas/NumPy
2. **Specialized packages**: For specific tasks, specialized packages exist (statsmodels for regression, scikit-learn for ML)
3. **API complexity**: scipy has many submodules with different APIs (scipy.stats, scipy.optimize, etc.)
4. **Documentation**: Can be harder to find the right function compared to pandas
5. **Overkill for simple tasks**: Many data scientists don't need advanced statistical functions

**In our project**: We used scipy sparingly:
- `scipy.stats.trim_mean` for robust mean estimation
- `scipy.stats.mstats.winsorize` for outlier handling
- `scipy.stats` for confidence intervals in injury analysis
- But most statistics came from pandas `.agg()` methods

SIADS 511: SQL and Databases: 

1. How would you set up a database to store the data that you used in your project?

**Answer:** We would create normalized tables:
- **leagues** (league_id PK, year, num_teams, ppr_points)
- **drafts** (draft_id PK AUTO_INCREMENT, league_id FK, year, team, player, round, pick, overall, is_autodrafted)
- **lineups** (lineup_id PK AUTO_INCREMENT, league_id FK, year, week, team, player, slot, points, projected_points, is_starter)
- **transactions** (transaction_id PK AUTO_INCREMENT, league_id FK, year, week, team, player, action)
- **injuries** (injury_id PK AUTO_INCREMENT, season, week, player, position, injury_type, injury_status)
- **players** (player_id PK AUTO_INCREMENT, player_name, player_norm UNIQUE) - normalized player names

Indexes on: league_id+year (composite), player_norm, overall (for draft analysis), week (for time-based queries).  
2. Why do we use indexes with databases?  
3. Provide an example of an AUTO\_INCREMENT field that you might use with the data from your project.  
4. What are some common data types in SQL databases?  
5. What’s the difference between a primary key and a logical key?  
6. What does database normalization mean and why is it important?  
7. Describe how you would use a JOIN statement to combine data from two tables.  
8. How do you model a many-to-many relationship in a SQL database?  
9. Give an example of how transactions are useful for mitigating problem associated with concurrency.  
10. What are stored procedures? Give an example of a stored procedure that you might have found useful in your project if you had used a SQL database to store your data.  
11. What are subqueries and when should they be used?  
12. Provide an example of a GROUP BY statement that could be used in your project if your data was stored in a SQL database.  
13. Describe some ways to store text data in SQL databases.  
14. What are some common functions that can be applied to text data in SQL databases?  
15. What is a b-tree index?  
16. Show how regular expressions can be used with SQL databases.

SIADS 515: Efficient Data Processing:

1. How would you explain what the linux CLI is to someone new to data science?  
2. What are some of the more common CLI commands?  
3. What are some of the difficulties associated with using the CLI?  
4. Describe ways in which you could have (or did) use the linux command line in your project.  
5. Describe some of the common Jupyter magic commands and identify those which have proved to be useful to you in your work to date.  
6. Look at the code that you used for your project and see if you can identify at least one place where you either used or could have used a generator.  
7. Recall the use of decorators and suggest at least one place in your code where you could use this approach.  
8. Caching involves storing the results of a deterministic function (i.e. given an input, the output is exactly the same each time the function is called). Identify one place in your code where you used a function and indicate why you did or did not use caching.  
9. While developing your code for the project there were likely times that your code didn’t work, or didn’t work as expected. Describe the process you followed to fix the problem(s) with your code.  
10. Suggest some reasons why print debugging, while common, may not be the best approach.  
11. JupyterLab provides access to a debugger but it isn’t commonly used. Propose reasons why that’s the case and suggest ways in which the debugger could be changed to make it more accessible.  
12. One of the more common errors is the syntax error. Sometimes, syntax errors are accompanied by a statement like “missing : “. Explain why, if python knows what’s missing, it can’t simply fix the problem for you.  
13. Why do we care about code complexity?  
14. Identify places in your code where the complexity of your approach resulted in suboptimal efficiency. Propose ways in which you could have improved your code.  
15. Jupyter notebooks promote a fragmented approach to coding, with each cell being executed independently of others. Explain why this promotes inefficient code design and suggest ways to promote the creation of efficient code.

SIADS 516: Big Data: Scalable Data Processing:

1. Explain how source code for underlying libraries such as pandas, numpy, and scipy can be examined to understand how the underlying operations are coded.  
2. What are the characteristics of Big Data?  
3. What is map reduce and how can it be used to help analyze Big Data?  
4. What are some advantages and limitations of using distributed computing?  
5. Provide a description of Hadoop and explain why it’s a reasonable choice for analyzing Big Data.  
6. What is Apache Spark?  
7. How do Resilient Distributed Datasets facilitate the use of distributed computing?  
8. How do Pair RDDs differ from plain RDDs, and why is that important?  
9. What are some limitations of using RDDs?  
10. What are Spark DataFrames?  
11. Describe how Spark DataFrames can be used for the manipulation and analysis of structured data.  
12. What are user-defined functions and when is it appropriate to use them?  
13. What are some limitations of using Spark DataFrames?  
14. Describe how Spark SQL works and explain why it is sometimes advantageous to use it.  
15. Describe, using an example, of how Spark DataFrames can be merged or joined using Spark SQL.  
16. How do user-defined functions (UDFs) work in Spark SQL and explain how they differ from UDFs from the previous Module.  
17. Describe a scenario where you would interchange data between Spark and Pandas.  
18. What is matplotlib?  
19. Describe how you could use matplotlib to create a histogram of some data that you used in your project.  
20. Create a boxplot of one of the variables from your data.  
21. Create a univariate plot of your choice and annotate something noteworthy in your plot.  
22. What is meant by “the computational narrative”? How did you create a computational narrative in your project work?  
23. Demonstrate the effect of changing the number of bins in a histogram of one of your variables.  
24. What are the advantages and drawbacks of using violin plots?  
25. Show how a probability plot can be used to gain insights about any of the variables you used in your project.  
26. What sorts of data are suitable for use in heat maps?  
27. What are tree maps used for?  
28. Demonstrate the use of a SPLOM with data from your project.  
29. What are some advantages and drawbacks of using 3D plots?  
30. What is autocorrelation and how can visualization be used to detect it?  
31. Describe some of the challenges associated with using geographic maps as visualizations.

SIADS 522: Information Visualization I: 

1. What is Anscombe’s Quartet and why is it important in visualization?  
2. Who is Edward Tufte and why is he important in visualization?  
3. What is the block model and how might it apply to the work you did in your project?  
4. What are nominal, ordinal and quantitative data types? Provide examples from your project.  
5. In visualization, what does “encoding” mean?  
6. In visualization, what do “expressiveness” and “effectiveness” mean?  
7. What is the grammar of graphics?  
8. Describe how the limits of perceptual systems affect your choices of visualizations.  
9. What is preattentive processing and describe how it applies to visualizations you generated for your project.  
10. What does Gestalt psychology have to offer the field of visualization?  
11. What is change blindness?  
12. Comment on several design principles that you used when generating visualizations for your project.  
13. Provide a hypothetical example of how you could lie about your project data using visualizations.  
14. What is chart junk? Comment on the degree to which you have chart junk in your visualizations.  
15. Discuss the role of ethics when creating visualizations.  

---

## Partner's Oral Exam Questions

### Coding Question: `pd.to_numeric()` with `errors="coerce"`

**Question:** What does `joined["Week"] = pd.to_numeric(joined["Week"], errors="coerce").astype(int)` do?

**Answer:** This line converts the "Week" column to numeric integers, handling invalid values gracefully.

**Step-by-step breakdown:**
1. **`pd.to_numeric(joined["Week"], errors="coerce")`**: 
   - Attempts to convert each value in the "Week" column to a numeric type (float)
   - `errors="coerce"` means: if a value cannot be converted to a number, replace it with `NaN` (Not a Number) instead of raising an error
   - Returns a pandas Series with numeric values (or NaN for unconvertible values)

2. **`.astype(int)`**: 
   - Converts the numeric Series to integer type
   - Note: This will convert NaN values to -9223372036854775808 (or raise an error in newer pandas versions if NaN exists)

**What format does the data end up in?**
- **Final format**: Integer (int64) - whole numbers like 1, 2, 3, ..., 17
- **Data type**: `pandas.Series` with `dtype: int64` (or `Int64` if nullable integers are used)
- **Example values**: `[1, 2, 3, 4, 5, ..., 17]` representing weeks of the NFL season

**What exactly does "coerce" do?**
- **Without `errors="coerce"`**: If pandas encounters a non-numeric value (like "two" or "t w o"), it would raise a `ValueError` and stop execution
- **With `errors="coerce"`**: Non-numeric values are converted to `NaN` (missing value), allowing the operation to continue
- **Other options**: 
  - `errors="raise"` (default): Raise error on invalid values
  - `errors="ignore"`: Return original Series unchanged if conversion fails

**What if a user put in "t w o"?**
- **Result**: The value "t w o" cannot be converted to a number, so `pd.to_numeric()` with `errors="coerce"` converts it to `NaN`
- **After `.astype(int)`**: 
  - In older pandas: `NaN` becomes a very large negative integer (-9223372036854775808) or 0
  - In newer pandas (with nullable integers): May raise an error or become `<NA>`
  - **Best practice**: Filter out NaN values first:
    ```python
    joined["Week"] = pd.to_numeric(joined["Week"], errors="coerce")
    joined = joined[joined["Week"].notna()]  # Remove rows with NaN
    joined["Week"] = joined["Week"].astype(int)
    ```

**Why we use this in our project:**
- Week numbers come from CSV files where they might be stored as strings ("1", "2", "3")
- Some values might be missing or malformed
- We need integers for comparisons (`joined["Week"] >= 1`) and filtering
- `errors="coerce"` prevents crashes from bad data, allowing us to handle errors gracefully

**Example from our code:**
```python
# Line 2131 in draft_value_analyzer.py
joined["Week"] = pd.to_numeric(joined["Week"], errors="coerce").astype(int)
joined = joined[(joined["Week"] >= 1) & (joined["Week"] <= season_end_week)].copy()
```
This ensures Week is an integer for the subsequent filtering operations.

---

### What is a Linear Relationship?

**Answer:** A linear relationship is one where two variables have a constant rate of change - when one variable increases by a fixed amount, the other increases (or decreases) by a proportional fixed amount. Mathematically: `y = mx + b`, where `m` is the slope and `b` is the y-intercept.

**Characteristics:**
- **Constant slope**: The relationship forms a straight line when plotted
- **Proportional change**: If x doubles, y doubles (if positive relationship)
- **No curvature**: The relationship doesn't curve or bend

**In our project:**
- **Draft value vs. Pick number**: We expect a **non-linear** relationship - early picks are worth much more than late picks, but the difference between picks 1 and 2 is larger than between picks 159 and 160. This is why we use **polynomial regression** (degree 4) rather than linear regression - the relationship curves downward.
- **If it were linear**: We would expect each pick to be worth a constant amount less than the previous pick (e.g., pick 1 = 200 points, pick 2 = 195 points, pick 3 = 190 points, etc.). But in reality, the curve is steeper early and flatter later.
- **Why polynomial not linear**: A linear model would underestimate early pick value and overestimate late pick value. The polynomial captures the non-linear decay in expected value.

**Visual example:**
- **Linear**: A straight diagonal line
- **Our data (non-linear)**: A curved line that drops steeply at first, then levels off (exponential decay pattern)

---

### Questions About the Charts in the Report

**Answer:** Our project includes several types of visualizations:

**1. Heatmaps:**
- **Injury Type × Position**: Shows how different injury types affect different positions (e.g., hamstring injuries hurt RBs more than QBs)
- **Injury Type × Status**: Shows how injury status (Questionable/Doubtful/Out) affects performance by injury type
- **Position × Status**: Shows how injury status affects different positions
- **Color encoding**: Red (negative impact), Green (positive impact), with color intensity showing magnitude
- **Sample sizes**: Displayed in each cell (e.g., "n=45") to show statistical reliability

**2. Distribution Plots:**
- **Point differential distributions**: Histograms comparing injured vs. healthy players
- **By injury status**: Separate distributions for Questionable, Doubtful, Out
- **By position**: Distributions for top positions (RB, WR, QB, TE)
- **Box plots**: Comparing distributions across injury statuses
- **Includes**: Mean lines, confidence intervals, density curves

**3. Draft Value Visualizations:**
- **Expected value by pick**: Line plots showing how expected points decrease with draft pick
- **Human advantage by round/position/year**: Bar charts showing where human drafters outperform autodraft
- **Cumulative plots**: Line plots showing cumulative points added over the season (Draft + Waiver + Start/Sit)

**4. Waiver Wire Visualizations:**
- **Baseline exploration**: Histograms showing distribution of waiver rates
- **Yearly trends**: Line plots showing how waiver activity changes over years
- **Cumulative waiver points**: Time series showing waiver value accumulation

**5. Start/Sit Visualizations:**
- **By year**: Bar charts showing average start/sit decision quality by season
- **Cumulative impact**: Integrated with draft and waiver in combined plots

**Key design choices:**
- **Color scheme**: Consistent across plots (red=negative, green=positive)
- **Sample sizes**: Always displayed for statistical transparency
- **Minimal decoration**: Focus on data, minimal chart junk
- **Accessibility**: Colorblind-friendly palettes, text annotations for exact values

---

### What Would Tufte Say About the Charts?

**Answer:** Edward Tufte would likely have both praise and criticism for our visualizations:

**What Tufte would APPROVE:**

1. **High data-ink ratio**: Our charts minimize decorative elements (minimal grid lines, clean axes, no unnecessary borders). Most of the "ink" on the page represents actual data.

2. **Clear encoding**: We use position (most effective) as primary encoding (x/y axes), color as secondary. This follows Tufte's principle of using the most effective visual channel for the most important information.

3. **Sample sizes displayed**: We show "n=X" in heatmap cells and plot annotations, providing context about data reliability. Tufte emphasizes showing all relevant data.

4. **No chart junk**: We avoid 3D effects, decorative images, unnecessary gradients, or other non-data elements that distract from the information.

5. **Small multiples concept**: Our injury heatmaps show multiple dimensions (type × position × status) in a single view, allowing comparison across categories - similar to Tufte's "small multiples" approach.

**What Tufte would CRITICIZE:**

1. **Color dependency**: Our heatmaps rely heavily on color to encode information. Tufte would prefer we also use other encodings (size, shape, position) since color can be ambiguous and colorblind users may struggle. We partially address this with text annotations, but could do more.

2. **Grid lines**: While minimal (alpha=0.2-0.3), Tufte might argue some grid lines are unnecessary - the data points themselves provide reference. However, our grid lines are subtle enough to be acceptable.

3. **Legend placement**: Some legends could be integrated into the plot area rather than separate, reducing "non-data ink" further.

4. **Multiple plots per figure**: Our distribution plots have 4 subplots (2×2 grid). Tufte might prefer showing them as small multiples with consistent scales for easier comparison.

5. **Missing sparklines**: For time series data (cumulative plots), Tufte would love to see sparklines - small, word-sized graphics showing trends inline with text.

**Tufte's core principles we follow:**
- ✅ **Show the data**: We display actual values, not just summaries
- ✅ **Induce the viewer to think**: Our heatmaps reveal patterns that require interpretation
- ✅ **Avoid distorting what the data says**: We use appropriate scales, don't truncate axes misleadingly
- ✅ **Present many numbers in a small space**: Heatmaps efficiently show many data points
- ✅ **Make large data sets coherent**: Our aggregations (by type, position, status) organize complex injury data

**Overall assessment:** Tufte would likely rate our visualizations as **good but improvable** - we follow many of his principles (data-ink ratio, clarity, no chart junk) but could enhance them with better color alternatives, sparklines for time series, and more integrated legends. Our focus on showing sample sizes and avoiding misleading scales would earn particular praise.

**Specific improvements Tufte might suggest:**
1. Add pattern/texture to heatmap cells in addition to color (for colorblind accessibility)
2. Use sparklines for weekly trends instead of full line plots
3. Integrate legends into plot areas where possible
4. Consider small multiples for distribution comparisons
5. Add more direct value labels (we do this in some plots, but could be more consistent)
    

