# Study Guide Answers Supplement

This document provides comprehensive answers to remaining questions in the Oral Exam Study Guide, organized by course.

## SIADS 511: SQL and Databases (Remaining Answers)

**2. Why do we use indexes with databases?**
Indexes speed up queries by creating sorted data structures (B-trees) that allow fast lookups without scanning entire tables. In our project, we'd index `(league_id, year)` for filtering leagues, `player_norm` for name matching, `overall` for draft pick analysis, and `week` for time-series queries. Without indexes, JOINs and WHERE clauses require full table scans (O(n)); with indexes, lookups are O(log n).

**3. Provide an example of an AUTO_INCREMENT field**
`draft_id INT AUTO_INCREMENT PRIMARY KEY` in the drafts table. Each draft pick gets a unique ID automatically, providing a stable primary key independent of business logic.

**4. What are some common data types in SQL databases?**
INT/BIGINT (IDs, counts), VARCHAR(n) (text), DECIMAL(p,s) (precise numeric like points), FLOAT/DOUBLE (approximate numeric), BOOLEAN/TINYINT (flags like is_autodrafted), DATE/DATETIME (timestamps), TEXT (longer text fields).

**5. What's the difference between a primary key and a logical key?**
Primary key: Unique identifier for each row, used for indexing (e.g., `draft_id AUTO_INCREMENT`). Logical key: Business-meaningful identifier (e.g., `(league_id, year, overall)` uniquely identifies a draft pick). Primary keys are stable and simple; logical keys reflect business rules.

**6. What does database normalization mean and why is it important?**
Normalization reduces data redundancy by organizing data into related tables. We'd normalize to 3NF: separate `players` table to avoid storing player names in every draft/lineup row. Prevents update anomalies, reduces storage, ensures data consistency.

**7. Describe how you would use a JOIN statement**
```sql
SELECT d.*, l.points, l.projected_points
FROM drafts d
INNER JOIN lineups l 
  ON d.league_id = l.league_id 
  AND d.year = l.year 
  AND d.player_norm = l.player_norm
WHERE d.year = 2024;
```
This matches draft picks to their lineup performance. INNER JOIN gets only players in both tables.

**8. How do you model a many-to-many relationship?**
Use a junction table. Example: `player_teams` table with `player_id FK, team_id FK, league_id, year, add_week, drop_week`. In our project, waiver stints are essentially many-to-many: players ↔ teams with temporal attributes.

**9. Give an example of how transactions are useful for concurrency**
```sql
BEGIN TRANSACTION;
INSERT INTO drafts (league_id, year, ...) VALUES (...);
UPDATE leagues SET last_updated = NOW() WHERE league_id = ?;
COMMIT;
```
If UPDATE fails, INSERT is rolled back, preventing partial data. In our parallel data collection, transactions prevent race conditions.

**10. What are stored procedures?**
Precompiled SQL code stored in database. Example:
```sql
CREATE PROCEDURE GetDraftValue(IN p_year INT, IN p_round INT)
BEGIN
  SELECT AVG(points_added) as avg_value
  FROM draft_scored
  WHERE year = p_year AND round = p_round AND is_autodrafted = 0;
END;
```
Useful for encapsulating complex queries, performance (precompiled), security, reusability.

**11. What are subqueries and when should they be used?**
Queries nested inside other queries. Example:
```sql
SELECT player, points FROM lineups
WHERE points > (SELECT AVG(points) FROM lineups WHERE year = 2024);
```
Use for filtering based on aggregated values, when JOIN would be inefficient, or for correlated subqueries.

**12. Provide an example of a GROUP BY statement**
```sql
SELECT year, round, AVG(points_added) as avg_points_added, COUNT(*) as num_picks
FROM draft_scored
WHERE is_autodrafted = 0
GROUP BY year, round
ORDER BY year, round;
```
Equivalent to our pandas `groupby(["Year", "Round"]).agg()` operations.

**13. Describe ways to store text data**
VARCHAR(n) for fixed max length (player names), TEXT for variable length (injury descriptions), CHAR(n) for fixed length codes (position abbreviations), JSON/JSONB for structured text (PostgreSQL).

**14. Common functions for text data**
UPPER()/LOWER() (case conversion), TRIM() (whitespace removal), SUBSTRING()/LEFT()/RIGHT() (extract portions), REPLACE() (substring replacement), CONCAT() (combine strings), LIKE/REGEXP (pattern matching).

**15. What is a b-tree index?**
B-tree (balanced tree) is a self-balancing tree that keeps data sorted. Used for indexes because: fast lookups (O(log n)), efficient range queries, maintains sort order. In our project, indexing `overall` enables fast queries like "all picks in rounds 1-3".

**16. Show how regular expressions can be used with SQL**
```sql
-- MySQL/PostgreSQL
SELECT * FROM drafts WHERE player REGEXP '^[A-Z][a-z]+ [A-Z][a-z]+$';
-- Matches "First Last" format, useful for validating/cleaning player names
```

## SIADS 515: Efficient Data Processing

**1. How would you explain the linux CLI to someone new?**
Command-line interface (CLI) is a text-based way to interact with your computer. Instead of clicking icons, you type commands. Essential for data science because: (1) Automation - write scripts to process data, (2) Remote servers - most data science happens on Linux servers, (3) Version control - git is CLI-based, (4) Efficiency - faster than GUI for repetitive tasks.

**2. Common CLI commands**
- `ls` (list files), `cd` (change directory), `pwd` (print working directory)
- `mkdir` (make directory), `rm` (remove), `cp` (copy), `mv` (move)
- `grep` (search text), `find` (find files), `cat` (display file)
- `chmod` (change permissions), `ps` (process status), `top` (system monitor)

**3. Difficulties with CLI**
- Steep learning curve (memorizing commands)
- No visual feedback (must remember file structure)
- Easy to make mistakes (deleting wrong files)
- Different syntax across systems (Linux vs macOS vs Windows)
- Error messages can be cryptic

**4. Ways we used CLI in project**
- File management: Organizing data files into year directories
- Git operations: Version control for code
- Running Python scripts: `python src/data_fetchers/data_collector.py`
- Data inspection: `head`, `tail`, `wc -l` to check CSV files
- Environment setup: `pip install` for dependencies

**5. Common Jupyter magic commands**
- `%time` / `%%time`: Time execution (used to profile slow operations)
- `%matplotlib inline`: Display plots in notebook
- `%load_ext`: Load extensions (e.g., `%load_ext autoreload`)
- `%autoreload 2`: Auto-reload modules during development
- `%who`: List variables in namespace
- `%reset`: Clear all variables

**6. Places we could use generators**
In `load_multi_season_data()`, instead of loading all years into memory with `pd.concat()`, we could use a generator:
```python
def load_drafts_generator():
    for year_dir in sorted_years:
        yield pd.read_csv(year_dir / "draft_data.csv")
```
This would process one year at a time, reducing memory usage for large datasets.

**7. Places to use decorators**
We could add a `@timing` decorator to log execution time:
```python
def timing(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

@timing
def compute_optimal_startable_points(self, lineups):
    # existing code
```
Useful for profiling without modifying each method.

**8. Places we didn't use caching (and why)**
We didn't cache `normalize_player_name()` because: (1) It's very fast (regex operations), (2) Input space is huge (all possible player names), (3) Memory overhead would exceed benefit. We DID implicitly cache by storing results in DataFrames (e.g., `draft_enriched`), avoiding recomputation.

**9. Process for fixing code problems**
1. **Reproduce**: Run code to see exact error
2. **Read error message**: Python tracebacks show line number and error type
3. **Add print statements**: Debug by printing intermediate values
4. **Check data**: Verify input data format (e.g., `df.head()`, `df.info()`)
5. **Simplify**: Comment out parts to isolate the issue
6. **Search**: Look for similar errors online (Stack Overflow)
7. **Test incrementally**: Fix one thing at a time, test after each change
8. **Use debugger**: For complex issues, step through code with pdb

**10. Why print debugging isn't best**
- **Clutters code**: Must remove prints later
- **Not systematic**: Easy to miss important values
- **Slow**: Printing large DataFrames is slow
- **No breakpoints**: Can't pause execution to inspect state
- **Hard to trace**: No call stack information
Better: Use proper debugger (pdb, IDE debugger) or logging framework.

**11. Why JupyterLab debugger isn't commonly used**
- **Learning curve**: Different from traditional debuggers
- **Not intuitive**: UI can be confusing
- **Cell-based execution**: Hard to set breakpoints across cells
- **Variable inspection**: Less convenient than IDE debuggers
- **Print is "good enough"**: For simple issues, print statements work

**Improvements**: Better integration with notebook cells, clearer UI, better variable inspection, step-through execution across cells.

**12. Why Python can't auto-fix syntax errors**
Python's parser can identify WHERE the error is, but not WHAT the programmer intended. Example: `if x = 5` - Python knows `=` is wrong in `if`, but doesn't know if programmer meant `==` (comparison) or `if x = 5:` (assignment then check). Context matters - auto-fixing could introduce bugs.

**13. Why we care about code complexity**
- **Maintainability**: Complex code is hard to understand and modify
- **Bugs**: More complexity = more places for bugs
- **Performance**: Complex algorithms may be inefficient
- **Testing**: Hard to test complex code thoroughly
- **Collaboration**: Others can't understand overly complex code

**14. Places with suboptimal efficiency**
- **Multiple merges**: We do sequential merges (`draft.merge(pos).merge(pts)`) - could combine into one merge
- **Repeated groupby**: We groupby the same keys multiple times - could cache grouped data
- **String operations in loops**: `normalize_player_name()` called repeatedly - could vectorize with `.apply()` (we do this, but could optimize further)
- **Large DataFrames in memory**: Loading all years at once - could process incrementally

**Improvements**: Use `pd.merge()` with multiple DataFrames, cache groupby results, use vectorized string operations, process data in chunks.

**15. Why Jupyter promotes inefficient code**
- **Cell independence**: Each cell runs independently, encouraging repeated computations
- **No function structure**: Code stays in cells instead of reusable functions
- **State management**: Variables persist between cells, making dependencies unclear
- **Hard to refactor**: Moving code between cells is awkward

**Solutions**: (1) Extract code to functions/classes (we did this with `DraftValueAnalyzer`), (2) Use `NotebookWorkflow` to manage state, (3) Write tests for functions, (4) Use `%autoreload` to develop in modules, (5) Document cell dependencies clearly.

## SIADS 516: Big Data & Visualization

**1. How to examine source code of libraries**
Libraries are open source - can view on GitHub (pandas, numpy, scipy all on GitHub). Can also use `inspect` module: `import inspect; print(inspect.getsource(pd.DataFrame.merge))` to see implementation. Useful for understanding performance characteristics and edge cases.

**2. Characteristics of Big Data**
Volume (large scale), Velocity (fast generation), Variety (different formats), Veracity (data quality issues). Our project: Moderate volume (millions of rows), low velocity (historical data), high variety (drafts, lineups, transactions, injuries), moderate veracity (ESPN data quality varies).

**3. What is map reduce**
Programming model: (1) Map: Apply function to each element, (2) Shuffle: Group by key, (3) Reduce: Aggregate grouped values. Example: Count picks by round - Map each pick to (round, 1), Reduce by summing counts. Our `groupby().agg()` operations are conceptually similar.

**4. Advantages and limitations of distributed computing**
**Advantages**: Handles large datasets, parallel processing, fault tolerance, scalability. **Limitations**: Network overhead, data locality issues, complexity, debugging distributed systems is hard. Our project doesn't need it (fits in memory), but if we scaled to all ESPN leagues, would benefit.

**5. What is Hadoop**
Distributed file system (HDFS) + MapReduce framework. Reasonable for Big Data because: (1) Handles petabytes, (2) Fault tolerant (replicates data), (3) Works on commodity hardware, (4) Batch processing for large-scale analytics. Our project: Overkill for our data size.

**6. What is Apache Spark**
In-memory distributed computing framework. Faster than Hadoop because data stays in memory. Provides: RDDs (Resilient Distributed Datasets), DataFrames, SQL interface, MLlib. Our project: Could use Spark if we scaled up, but pandas is sufficient.

**7. How RDDs facilitate distributed computing**
RDDs are immutable, distributed collections. They track lineage (how data was derived), enabling fault tolerance - if a node fails, RDD can be recomputed from lineage. Lazy evaluation allows optimization. Our project: Don't use RDDs, but our DataFrames are conceptually similar (immutable operations, lazy evaluation in some cases).

**8. How Pair RDDs differ from plain RDDs**
Pair RDDs have key-value structure `(key, value)`, enabling operations like `reduceByKey()`, `groupByKey()`. Plain RDDs are just collections. Important because many operations (joins, aggregations) need key-value structure. Our project: Our groupby operations create key-value pairs conceptually.

**9. Limitations of RDDs**
- Low-level API (must write map/reduce functions)
- No schema (harder to optimize)
- Less user-friendly than DataFrames
- Performance: Slower than DataFrames for structured data
- Type safety: Runtime errors instead of compile-time

**10. What are Spark DataFrames**
Distributed DataFrames with schema, similar to pandas but distributed. Benefits: (1) Catalyst optimizer, (2) Schema enforcement, (3) SQL interface, (4) Better performance than RDDs for structured data. Our project: Could use Spark DataFrames if we scaled up.

**11. How Spark DataFrames manipulate structured data**
Similar to pandas: `select()`, `filter()`, `groupBy()`, `agg()`, `join()`. But operations are lazy (build execution plan) and distributed. Example: `df.filter(df.year == 2024).groupBy("round").agg(avg("points"))` - same syntax, distributed execution.

**12. What are user-defined functions (UDFs)**
Custom functions applied to DataFrame columns. In Spark, UDFs are less efficient than built-in functions (can't optimize, serialization overhead). Use when: (1) Complex logic not available in Spark SQL, (2) Custom transformations. Our project: `normalize_player_name()` could be a UDF in Spark.

**13. Limitations of Spark DataFrames**
- UDF performance: Slower than built-in functions
- Small data: Overhead not worth it for small datasets
- Learning curve: Different from pandas
- Debugging: Harder to debug distributed execution
- Data types: Some pandas features not available

**14. How Spark SQL works**
SQL interface on top of Spark DataFrames. Can write SQL queries instead of DataFrame API. Advantages: (1) Familiar syntax for SQL users, (2) Easier for complex queries, (3) Same optimizer as DataFrame API. Example: `spark.sql("SELECT year, AVG(points) FROM drafts GROUP BY year")`.

**15. Merging Spark DataFrames with Spark SQL**
```sql
SELECT d.*, l.points
FROM drafts d
INNER JOIN lineups l 
  ON d.league_id = l.league_id 
  AND d.year = l.year;
```
Same as pandas merge, but distributed. Can also use DataFrame API: `drafts.join(lineups, on=["league_id", "year"], how="inner")`.

**16. UDFs in Spark SQL vs previous module**
Spark SQL UDFs: Registered functions usable in SQL queries. Example:
```python
spark.udf.register("normalize_name", normalize_player_name)
spark.sql("SELECT normalize_name(player) FROM drafts")
```
Previous module UDFs: Applied with `.apply()``. Spark SQL UDFs: More integrated with SQL queries, but same performance limitations.

**17. Interchanging Spark and Pandas**
Convert Spark DataFrame to Pandas: `spark_df.toPandas()` (collects to driver - only for small data). Convert Pandas to Spark: `spark.createDataFrame(pandas_df)`. Use case: (1) Small results from Spark → Pandas for visualization, (2) Local development in Pandas → scale up with Spark. Our project: If we scaled up, would use Spark for processing, Pandas for final analysis/visualization.

**18. What is matplotlib**
Python plotting library, foundation for many other visualization libraries (seaborn, pandas plotting). Provides low-level control over plots. We use it directly for custom plots and indirectly through seaborn.

**19. How to create histogram with matplotlib**
```python
import matplotlib.pyplot as plt
plt.hist(draft_scored['Points_Added_Poly'], bins=30, alpha=0.7)
plt.xlabel('Points Added Over Expected')
plt.ylabel('Frequency')
plt.title('Distribution of Draft Value Added')
plt.show()
```
We use this in `plot_team_total_valid_points_distribution()`.

**20. Create a boxplot**
```python
plt.boxplot([healthy_points, injured_points], labels=['Healthy', 'Injured'])
plt.ylabel('Point Differential')
plt.title('Point Differential: Healthy vs Injured Players')
plt.show()
```
We create boxplots in `plot_point_differential_distribution()` for injury analysis.

**21. Univariate plot with annotation**
```python
plt.plot(expected_by_pick['Overall'], expected_by_pick['Expected_Smoothed'])
plt.axvline(x=25, color='r', linestyle='--', label='Round 2-3 boundary')
plt.annotate('Early round advantage', xy=(10, 150), 
             xytext=(5, 200), arrowprops=dict(arrowstyle='->'))
plt.show()
```
We annotate plots to highlight key findings (e.g., early round draft advantage).

**22. What is "the computational narrative"**
The story told through code execution - how data flows through transformations, how results are computed, how visualizations are generated. In our project: `NotebookWorkflow` creates a narrative - load data → filter → enrich → score → visualize. Each step builds on previous, telling the story of how we analyze draft value.

**23. Effect of changing bins in histogram**
Fewer bins: Smoother, less detail, may hide patterns. More bins: More detail, may show noise. We use `bins=30` for injury distributions, `bins=50` for point differentials. Can experiment: `plt.hist(data, bins=[10, 20, 30, 50, 100])` to see effect.

**24. Advantages and drawbacks of violin plots**
**Advantages**: Shows distribution shape (not just quartiles), compares multiple groups well, shows density. **Drawbacks**: Can be hard to read, requires larger sample sizes, less familiar to general audience. We use boxplots instead for clarity, but violin plots would show injury impact distributions better.

**25. Probability plot for insights**
```python
from scipy import stats
stats.probplot(draft_scored['Points_Added_Poly'], dist="norm", plot=plt)
plt.show()
```
Shows if data is normally distributed. If points fall on line, data is normal. Our draft value data is likely non-normal (outliers), which is why we use robust statistics.

**26. Data suitable for heat maps**
Two categorical variables + one quantitative variable. In our project: `plot_heatmap_type_position()` - injury_type (rows) × position (columns) × avg_point_differential (color). Also suitable: correlation matrices, time × category matrices, geographic data.

**27. What are tree maps used for**
Hierarchical data visualization - nested rectangles sized by value. Could use for: (1) Points by position (QB, RB, WR, etc. as rectangles), (2) Draft value by round (each round as rectangle, sized by total value). We don't use tree maps, but they could show position/round importance.

**28. SPLOM (Scatter Plot Matrix)**
Grid of scatter plots showing pairwise relationships. Could create for: Points_Added_Poly, Points_Added_Pooled, Points_Added_Year, Overall, Round. Would show correlations between different baseline measures. We don't use SPLOMs, but they'd be useful for exploring baseline relationships.

**29. Advantages and drawbacks of 3D plots**
**Advantages**: Show three variables simultaneously, can reveal patterns not visible in 2D. **Drawbacks**: Hard to read (perspective distortion), can't print well, interaction required, can mislead. We avoid 3D plots - use 2D with color/faceting instead (e.g., heatmaps with injury_type × position × status).

**30. Autocorrelation and visualization**
Autocorrelation: Correlation of a variable with itself at different time lags. Could visualize with: (1) Lag plot (x_t vs x_{t+1}), (2) ACF plot (autocorrelation function). In our project: Weekly lineup decisions might have autocorrelation (good managers stay good). Could use `pandas.plotting.autocorrelation_plot()`.

**31. Challenges with geographic maps**
- **Projection distortion**: Different map projections distort areas/distances
- **Data availability**: Need geographic coordinates for each data point
- **Visual clutter**: Overlapping points hard to distinguish
- **Color interpretation**: Color scales can be misleading
- **Missing context**: Need to know geography to interpret

Our project: No geographic component, but if we analyzed by team location or manager location, would face these challenges.

## SIADS 522: Information Visualization I

**1. What is Anscombe's Quartet**
Four datasets with identical summary statistics (mean, variance, correlation) but completely different distributions. Importance: Shows that summary statistics alone don't tell the full story - must visualize data. In our project: Our aggregated stats (mean points added) could hide important patterns - that's why we create visualizations (heatmaps, distributions).

**2. Who is Edward Tufte**
Pioneer in data visualization, author of "The Visual Display of Quantitative Information". Principles: (1) Maximize data-ink ratio, (2) Avoid chart junk, (3) Show data variation, not design variation. Influenced our visualizations - we use clean plots, minimal decoration, focus on data.

**3. What is the block model**
Framework: Data → Visual Form → User. Applies to our project: (1) Data: Draft/lineup/injury data, (2) Visual Form: Heatmaps, distributions, line plots, (3) User: Fantasy football managers making decisions. Each visualization transforms raw data into actionable insights.

**4. Nominal, ordinal, quantitative data types**
- **Nominal**: Categories without order (Position: QB, RB, WR, TE - no inherent order)
- **Ordinal**: Categories with order (Injury Status: Out > Doubtful > Questionable > No Injury)
- **Quantitative**: Numeric values (Points, Points_Added, Overall pick number)

**5. What does "encoding" mean in visualization**
Mapping data attributes to visual properties. In our heatmaps: injury_type → rows, position → columns, point_differential → color. Other encodings: size (bar height), position (x/y axes), shape (marker type), texture (pattern).

**6. Expressiveness and effectiveness**
- **Expressiveness**: Visualization shows all and only the data (no false patterns, no missing information)
- **Effectiveness**: Visualization uses most effective encoding for the data type (position for quantitative, color for categorical)

Our heatmaps: Expressive (shows all combinations), Effective (color effectively encodes point differential).

**7. What is the grammar of graphics**
Systematic framework for building visualizations (like ggplot2). Components: Data → Aesthetics (mapping) → Geometries (plot type) → Scales → Facets → Statistics → Coordinates. Our matplotlib approach is less systematic, but we follow similar principles.

**8. Limits of perceptual systems**
Human vision has limitations: (1) Color blindness (we use color + patterns), (2) Limited color discrimination (we use color scales carefully), (3) Position > length > angle > area > volume (we prioritize position/length encodings), (4) Small number estimation (we use exact numbers in annotations). Our visualizations account for these - use position primarily, color as secondary.

**9. Preattentive processing**
Visual properties processed automatically, before conscious attention: color, size, orientation, position. In our visualizations: Red/green in heatmaps (negative/positive) is preattentive - viewers immediately see patterns without reading numbers. We leverage this for quick pattern recognition.

**10. Gestalt psychology and visualization**
Principles: (1) Proximity (nearby elements grouped), (2) Similarity (similar elements grouped), (3) Closure (complete shapes), (4) Continuity (smooth lines). In our plots: Group related elements (e.g., injury types together), use consistent colors for similar concepts, complete visual groupings.

**11. What is change blindness**
Failure to notice changes in visual scene. In our project: If we update visualizations, viewers might miss changes. We address this by: (1) Clear titles, (2) Annotations highlighting changes, (3) Consistent formatting across plots.

**12. Design principles we used**
- **Data-ink ratio**: Maximize ink showing data, minimize decoration (minimal grid lines, clean axes)
- **Consistency**: Same color scheme across plots (red=negative, green=positive)
- **Clarity**: Clear labels, titles, legends
- **Hierarchy**: Most important information emphasized (e.g., mean lines in distributions)
- **Accessibility**: Colorblind-friendly palettes, text annotations for exact values

**13. How we could lie with visualizations**
- **Truncated y-axis**: Start y-axis at 50 instead of 0 to exaggerate differences
- **Cherry-picked time range**: Show only years with favorable results
- **Misleading scales**: Use area instead of length (area scales as square, exaggerates differences)
- **Selective data**: Only show certain positions/leagues that support our narrative

We avoid these by: Full axis ranges, showing all years, using length not area, transparent about data selection.

**14. Chart junk in our visualizations**
Minimal chart junk - we use:
- Simple grid lines (alpha=0.2-0.3) for reference
- Clean axes without excessive decoration
- Minimal borders
- No unnecessary 3D effects
- No decorative images/icons

Could reduce further: Remove some grid lines, simplify legends, but current level is acceptable.

**15. Role of ethics in visualizations**
- **Honest representation**: Don't mislead with scales/truncation
- **Accessibility**: Colorblind-friendly, clear labels
- **Context**: Provide enough context (sample sizes, confidence intervals)
- **Transparency**: Show limitations, don't hide unfavorable results
- **Purpose**: Use visualizations to inform, not manipulate

In our project: We show full data ranges, include sample sizes, acknowledge limitations, use visualizations for education not manipulation.

