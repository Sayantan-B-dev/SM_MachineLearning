# Model (simple linear regression)

Equation (predicted value with a “hat”):
[
\hat{y}=\beta_0+\beta_1 x
]

* (x): input (predictor, independent variable).
* (\hat y): predicted output (estimate of the true (y)).
* (\beta_0): intercept (predicted (\hat y) when (x=0)).
* (\beta_1): slope (change in (\hat y) per unit change in (x)).

# Error (residual) and objective

* Residual (error) for observation (i): (e_i = y_i - \hat y_i = y_i - (\beta_0 + \beta_1 x_i)).
* Ordinary Least Squares (OLS) fits (\beta_0,\beta_1) by minimizing the Sum of Squared Errors (SSE):
  [
  \text{SSE}(\beta_0,\beta_1)=\sum_{i=1}^n (y_i-\beta_0-\beta_1 x_i)^2
  ]

# Closed-form OLS solution (derivation sketch)

Minimize SSE by setting partial derivatives to zero:
[
\frac{\partial \text{SSE}}{\partial \beta_0}=-2\sum_{i}(y_i-\beta_0-\beta_1 x_i)=0
]
[
\frac{\partial \text{SSE}}{\partial \beta_1}=-2\sum_{i}x_i(y_i-\beta_0-\beta_1 x_i)=0
]
Solve the two normal equations → express in terms of sample means (\bar x,\bar y):
[
\beta_1=\dfrac{\sum_{i}(x_i-\bar x)(y_i-\bar y)}{\sum_{i}(x_i-\bar x)^2} \quad\text{(covariance/variance)}
]
[
\beta_0=\bar y-\beta_1\bar x
]
where (\bar x=\frac{1}{n}\sum x_i,\ \bar y=\frac{1}{n}\sum y_i).

# Key derived quantities and formulas

* Residuals: (e_i=y_i-\hat y_i).
* SSE (=\sum e_i^2).
* SSR (regression sum of squares) (=\sum(\hat y_i-\bar y)^2).
* SST (total sum of squares) (=\sum(y_i-\bar y)^2).
* (R^2 = \dfrac{\text{SSR}}{\text{SST}} = 1 - \dfrac{\text{SSE}}{\text{SST}}).
* Mean Squared Error (estimate of residual variance): (\displaystyle \text{MSE}=\frac{\text{SSE}}{n-2}) (two parameters estimated).
* Root MSE (RMSE): (\sqrt{\text{MSE}}).
* Standard error of slope: (\displaystyle \operatorname{SE}(\beta_1)=\sqrt{\frac{\text{MSE}}{\sum (x_i-\bar x)^2}}).
* Standard error of intercept: (\displaystyle \operatorname{SE}(\beta_0)=\sqrt{\text{MSE}\left(\frac{1}{n}+\frac{\bar x^2}{\sum(x_i-\bar x)^2}\right)}).
* (t)-statistic for (\beta_j): (t=\beta_j/\operatorname{SE}(\beta_j)) with (df=n-2).
* 95% CI for (\beta_j): (\beta_j \pm t_{0.975,n-2}\cdot\operatorname{SE}(\beta_j)).
* Prediction for new input (x^*): (\hat y^*=\beta_0+\beta_1 x^*).

  * SE of mean response: (\sqrt{\text{MSE}\left(\frac{1}{n}+\frac{(x^*-\bar x)^2}{\sum(x_i-\bar x)^2}\right)}).
  * SE of individual prediction: (\sqrt{\text{MSE}\left(1+\frac{1}{n}+\frac{(x^*-\bar x)^2}{\sum(x_i-\bar x)^2}\right)}).
  * 95% prediction interval: (\hat y^* \pm t_{0.975,n-2}\cdot\text{SE}_{\text{pred}}).

# Real numerical example — step-by-step (digit-by-digit arithmetic)

Dataset (hours studied (x) → exam score (y)), (n=6):

|  i | (x_i) | (y_i) |
| -: | :---: | :---: |
|  1 |   1   |   6   |
|  2 |   2   |   8   |
|  3 |   3   |   11  |
|  4 |   4   |   13  |
|  5 |   5   |   15  |
|  6 |   6   |   18  |

1. Means:
   [
   \bar x=\frac{1+2+3+4+5+6}{6}=\frac{21}{6}=3.5
   ]
   [
   \bar y=\frac{6+8+11+13+15+18}{6}=\frac{71}{6}\approx 11.833333333333334
   ]

2. Build table of deviations, squares, cross-products:

|  i | (x_i) | (y_i) | (x_i-\bar x) |           (y_i-\bar y)          | ((x_i-\bar x)^2) | ((x_i-\bar x)(y_i-\bar y)) |
| -: | :---: | :---: | :----------: | :-----------------------------: | :--------------: | :------------------------: |
|  1 |   1   |   6   |  1−3.5=−2.5  | 6−11.8333333=−5.833333333333334 |       6.25       |     14.583333333333334     |
|  2 |   2   |   8   |     −1.5     |        −3.833333333333334       |       2.25       |            5.75            |
|  3 |   3   |   11  |     −0.5     |        −0.833333333333334       |       0.25       |     0.4166666666666667     |
|  4 |   4   |   13  |      0.5     |        1.166666666666666        |       0.25       |      0.583333333333333     |
|  5 |   5   |   15  |      1.5     |        3.1666666666666665       |       2.25       |            4.75            |
|  6 |   6   |   18  |      2.5     |        6.166666666666666        |       6.25       |     15.416666666666666     |

Now sums:
[
\sum (x_i-\bar x)^2 = 6.25+2.25+0.25+0.25+2.25+6.25=17.5
]
[
\sum (x_i-\bar x)(y_i-\bar y)=14.583333333333334+5.75+0.4166666666666667+0.583333333333333+4.75+15.416666666666666=41.5
]

3. Slope and intercept:
   [
   \beta_1=\frac{41.5}{17.5}=\frac{83/2}{35/2}=\frac{83}{35}\approx 2.3714285714285714
   ]
   [
   \beta_0=\bar y-\beta_1\bar x=11.833333333333334-2.3714285714285714\times 3.5\approx 3.533333333333333
   ]

4. Predicted (\hat y_i) and residuals (e_i):
   [
   \hat y_i=\beta_0+\beta_1 x_i
   ]
   Calculated values (rounded display):

|  i | (x_i) |     (\hat y_i)     |  (e_i=y_i-\hat y_i)  |
| -: | :---: | :----------------: | :------------------: |
|  1 |   1   |  5.904761904761905 |  0.0952380952380949  |
|  2 |   2   |  8.276190476190475 |  -0.2761904761904752 |
|  3 |   3   | 10.647619047619047 |  0.3523809523809529  |
|  4 |   4   | 13.019047619047619 | -0.01904761904761898 |
|  5 |   5   |  15.39047619047619 | -0.39047619047619087 |
|  6 |   6   |  17.76190476190476 |  0.2380952380952408  |

5. SSE, SSR, SST:

* Squared residuals: (e_i^2) ≈ [0.009070294785, 0.076281179138, 0.124172335601, 0.000362811791, 0.152471655329, 0.056689342404]
* (\text{SSE}=\sum e_i^2 \approx 0.4190476190476204).

Compute SST and SSR:
[
\text{SST}=\sum(y_i-\bar y)^2 \approx 98.83333333333333
]
[
\text{SSR}=\sum(\hat y_i-\bar y)^2 = \text{SST}-\text{SSE}\approx 98.41428571428568
]

6. (R^2):
   [
   R^2=1-\frac{\text{SSE}}{\text{SST}}=1-\frac{0.4190476190476204}{98.83333333333333}\approx 0.99576
   ]
   (about 99.576% of variance explained)

7. Residual variance, RMSE:
   [
   \text{MSE}=\frac{\text{SSE}}{n-2}=\frac{0.4190476190476204}{6-2}=\frac{0.4190476190476204}{4}\approx 0.1047619047619051
   ]
   [
   \text{RMSE}=\sqrt{\text{MSE}}\approx 0.3236694374850754
   ]

8. Standard errors of coefficients:
   [
   \operatorname{SE}(\beta_1)=\sqrt{\frac{\text{MSE}}{\sum(x_i-\bar x)^2}}=\sqrt{\frac{0.1047619047619051}{17.5}}\approx 0.07737179432986642
   ]
   [
   \operatorname{SE}(\beta_0)=\sqrt{\text{MSE}\left(\frac{1}{n}+\frac{\bar x^2}{\sum(x_i-\bar x)^2}\right)}\approx 0.3013198479915505
   ]

9. (t)-statistic for (\beta_1) and 95% CI (df = (n-2=4), (t_{0.975,4}\approx 2.776)):
   [
   t=\frac{\beta_1}{\operatorname{SE}(\beta_1)}\approx\frac{2.3714285714285714}{0.07737179432986642}\approx 30.65
   ]
   Very large (t) → coefficient highly significant (p-value ≪ 0.01).
   95% CI for (\beta_1):
   [
   \beta_1 \pm t_{0.975,4}\cdot \operatorname{SE}(\beta_1)\approx 2.37143 \pm 2.776\times0.0773718
   ]
   [
   \text{CI}_{95%}(\beta_1)\approx(2.15661,;2.58625)
   ]

10. Prediction example for new input (x^*=7):
    [
    \hat y^*=\beta_0+\beta_1\times7\approx 3.5333333+2.3714285714\times7\approx 20.133333333333333
    ]
    SE of mean response at (x^*=7):
    [
    \text{SE}*{\text{mean}}=\sqrt{\text{MSE}\left(\frac{1}{n}+\frac{(7-\bar x)^2}{\sum (x_i-\bar x)^2}\right)}\approx 0.3013198479915505
    ]
    95% CI for mean response:
    [
    (19.296735316278248,;20.969931350388418)
    ]
    SE of individual prediction:
    [
    \text{SE}*{\text{pred}}=\sqrt{\text{MSE}\left(1+\frac{1}{n}+\frac{(7-\bar x)^2}{\sum (x_i-\bar x)^2}\right)}\approx 0.44221663871405403
    ]
    95% prediction interval (wider; includes individual variability):
    [
    (18.905543111338673,;21.361123555327993)
    ]

# Interpretation of the example

* Estimated slope (\beta_1\approx2.3714): each additional unit of (x) (e.g., one more hour studied) increases predicted (y) (score) by ≈2.37 points on average.
* Intercept (\beta_0\approx3.5333): predicted score when (x=0) ≈3.53 (may be extrapolation if (x=0) is outside observed range).
* Very small SSE and RMSE (~0.3237) and (R^2\approx0.9958) indicate an excellent fit for this dataset (almost all variation explained).
* Large (t)-statistic for slope indicates the slope is statistically significant (effect is not due to chance in this sample).
* Prediction interval is wider than CI for mean response because it includes observation-level noise.

# Assumptions to check (error analysis diagnostics)

1. **Linearity**: relationship between (x) and (y) is linear. Check residuals vs (x) plot (should show no pattern).
2. **Independence**: residuals independent (no autocorrelation). Check Durbin–Watson for time series.
3. **Homoscedasticity**: residual variance constant across (x). Check residuals vs fitted; formal tests: Breusch–Pagan.
4. **Normality of residuals**: for inference (t/CIs) approximate normality of residuals; check Q–Q plot or Shapiro–Wilk.
5. **No leverage/outliers**: extreme (x) or (y) can distort fit; check Cook’s distance, leverage, studentized residuals.

# Practical checklist / exam-style notes

* Given ((x_i,y_i)): compute (\bar x,\bar y).
* Compute (S_{xx}=\sum (x_i-\bar x)^2,\ S_{xy}=\sum (x_i-\bar x)(y_i-\bar y)).
* Compute (\beta_1=S_{xy}/S_{xx},\ \beta_0=\bar y-\beta_1\bar x).
* Compute residuals (e_i), SSE, MSE (=) SSE/(n−2), RMSE.
* Compute (\operatorname{SE}(\beta_1)=\sqrt{\text{MSE}/S_{xx}}); compute t-stat and CI.
* For prediction at (x^*), compute (\hat y^*) and the prediction interval using the formula above.

# Simple explanation (for a 10-year-old)

Think of a straight line that best goes through a scatter of points. The line has two numbers: one says where it starts (intercept), the other says how steep it is (slope). The errors are how far each dot is from the line (up or down). The method picks the line so the sum of the squares of those up-and-down distances is as small as possible. Then the line can be used to guess a new dot and say how sure the guess is.

---

Equation:
[
\hat{y} = \beta_0 + \beta_1 x
]

**Think like this:**

You are trying to draw a *straight line* through a bunch of points on a graph.

* (x) → the input (example: hours of study)
* (y) → the actual result (example: exam score)
* (\hat{y}) → the *predicted result* from your line
* (\beta_0) → where the line starts (the “y-intercept”)
* (\beta_1) → how steep the line is (the “slope”)

---

### Real-life Example

| Hours studied (x) | Exam score (y) |
| :---------------: | :------------: |
|         1         |        6       |
|         2         |        8       |
|         3         |       11       |
|         4         |       13       |
|         5         |       15       |
|         6         |       18       |

If we draw a line that best fits these points, we find:
[
\hat{y} = 3.5 + 2.37x
]

That means:

* When no hours are studied ((x=0)), predicted score is about **3.5** marks.
* For each extra hour of study, the score increases by **2.37** marks.

---

### What “error” means

For each student, the prediction (\hat{y}) is usually not *exactly* the real (y).
The **error** (or “residual”) is:
[
\text{error} = y - \hat{y}
]

Example for (x=2):
[
\hat{y} = 3.5 + 2.37\times2 = 8.24
]
Real score (y = 8).
Error = (8 - 8.24 = -0.24).
So the line guessed a little higher than reality.

---

### Why we care about errors

If you make one line, and the dots are close to it → small errors → *good model*.
If dots are far away → big errors → *bad model*.
We calculate **Sum of Squared Errors (SSE)** to measure this:
[
\text{SSE} = \sum (y - \hat{y})^2
]
We pick the line that makes SSE as small as possible. That’s called **Least Squares**.

---

### How we measure goodness

We use (R^2) (read as “R-squared”).
It tells how much of the change in (y) is explained by (x).

[
R^2 = 1 - \frac{\text{SSE}}{\text{SST}}
]

If (R^2 = 1): perfect line (no error).
If (R^2 = 0.5): only half of the variation is explained.
If (R^2 = 0): line is useless.

In our example, (R^2 \approx 0.996).
That means the line explains **99.6%** of the pattern. Great fit!

---

### What the slope really means

(\beta_1 = 2.37) → for every 1 extra hour studied, predicted score increases by 2.37 marks.
So if you study 7 hours:
[
\hat{y} = 3.5 + 2.37\times7 = 20.09
]

Predicted score ≈ **20 marks**.

---

### In simple English

* You have data → draw the best straight line through it.
* That line tells you how one thing affects another.
* “Error” is how far the real point is from the line.
* Smaller errors → better prediction.
* The equation (\hat{y}=\beta_0+\beta_1x) is just “line = start + slope×input.”
* We use it to **predict**, **analyze**, and **understand** relationships.

---

**Precision and Recall — Easy Beginner Explanation**

---

### 💡 Imagine a real example

You build a program that detects **spam emails**.
For each email, your model says either:

* **Spam** (positive)
* **Not spam** (negative)

But your model can make mistakes. So we check its results against the real truth.

---

### 📊 Confusion Matrix (the result table)

|                             | **Actually Spam (Yes)**                         | **Actually Not Spam (No)**                      |
| --------------------------- | ----------------------------------------------- | ----------------------------------------------- |
| **Predicted Spam (Yes)**    | ✅ True Positive (TP) – correctly predicted spam | ❌ False Positive (FP) – said spam, but it’s not |
| **Predicted Not Spam (No)** | ❌ False Negative (FN) – missed spam             | ✅ True Negative (TN) – correctly said not spam  |

---

### ⚙️ Precision

**Precision = How many predicted spams were actually spam?**

[
\text{Precision} = \frac{TP}{TP + FP}
]

If your model says “spam” 100 times, and 90 are truly spam,
then precision = 90 / 100 = **0.9** (or **90%**).
→ It means your model is **accurate** when it predicts positive.

---

### ⚙️ Recall

**Recall = How many of all real spams did the model find?**

[
\text{Recall} = \frac{TP}{TP + FN}
]

If there were 120 real spam emails, and your model found 90 of them,
then recall = 90 / 120 = **0.75** (or **75%**).
→ It means your model **caught 75%** of the real spam.

---

### ⚖️ Summary Comparison

| Concept       | Formula        | Meaning                                                    | High value means… |
| ------------- | -------------- | ---------------------------------------------------------- | ----------------- |
| **Precision** | TP / (TP + FP) | Out of what model called “positive”, how many were correct | Few false alarms  |
| **Recall**    | TP / (TP + FN) | Out of all actual “positives”, how many were found         | Few misses        |

---

### 🧠 Simple Analogy

* **Precision** = “Of all the people I said have a disease, how many really do?”
* **Recall** = “Of all the people who actually have the disease, how many did I catch?”

---

### 🧾 Example Numbers

Suppose:

* TP = 70
* FP = 30
* FN = 20
* TN = 80

Then:

* Precision = 70 / (70 + 30) = **0.7 (70%)**
* Recall = 70 / (70 + 20) = **0.777 (77.7%)**

---

### 🎯 Bonus: F1-Score

If you want one number that balances both:
[
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
]
F1 is high only when both precision **and** recall are good.

---

### 🧩 In one line:

* **Precision** = “How right were my positive guesses?”
* **Recall** = “How many real positives did I find?”

---

# **F1 Score — Easy Beginner Explanation**

---

### 💡 Meaning

F1 Score is a single number that **combines both Precision and Recall**.
It tells you how *balanced* your model is between:

* **Catching positives** (Recall)
* **Being accurate when it says positive** (Precision)

---

### ⚙️ Formula

[
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
]

It’s called the **harmonic mean** of precision and recall — not just an average.
It gives more weight to the *smaller* of the two numbers, so both must be good.

---

### 📊 Example

Suppose your spam detector has:

* Precision = 0.80 (80%)
* Recall = 0.60 (60%)

Then:
[
F1 = 2 \times \frac{0.8 \times 0.6}{0.8 + 0.6} = 2 \times \frac{0.48}{1.4} = 0.6857
]
So, **F1 = 0.69 (69%)**

---

### 🔍 Why not just use accuracy?

Accuracy can fool you when data is unbalanced.
Example: if only 1 out of 100 emails is spam, predicting “Not Spam” always gives 99% accuracy — but finds **no spam at all** (Recall = 0).
That’s why F1 is better — it focuses only on the positive class.

---

### ⚖️ Quick Comparison Table

| Metric        | Formula         | Focus                    | High value means                    |
| ------------- | --------------- | ------------------------ | ----------------------------------- |
| **Precision** | TP / (TP + FP)  | Correctness of positives | Few false alarms                    |
| **Recall**    | TP / (TP + FN)  | Finding all positives    | Few misses                          |
| **F1 Score**  | 2 × (P×R)/(P+R) | Balance of both          | Good mix of accuracy + completeness |

---

### 🧠 Simple Analogy

Imagine a doctor diagnosing a disease:

* **Precision**: Of all patients the doctor said “sick,” how many really are?
* **Recall**: Of all actually sick patients, how many did the doctor find?
* **F1 Score**: How well the doctor balances both — not missing sick people and not mislabeling healthy ones.

---

### 📘 **Advanced Understanding of Precision, Recall & F1-Score**

These are core metrics in **classification performance analysis**, especially in **imbalanced datasets** where simple accuracy fails.

---

## 🧮 1. Confusion Matrix (Base for All Metrics)

|                            | **Actual Positive (1)** | **Actual Negative (0)** |
| -------------------------- | ----------------------- | ----------------------- |
| **Predicted Positive (1)** | True Positive (**TP**)  | False Positive (**FP**) |
| **Predicted Negative (0)** | False Negative (**FN**) | True Negative (**TN**)  |

---

## ⚙️ 2. Mathematical Formulas

[
\text{Precision} = \frac{TP}{TP + FP}
]

[
\text{Recall} = \frac{TP}{TP + FN}
]

[
\text{F1 Score} = 2 \times \frac{Precision \times Recall}{Precision + Recall}
]

---

## 📊 3. Geometric & Statistical Interpretation

| Metric        | What it measures                     | Geometric meaning                                                                                      | Statistical role                              |
| ------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------ | --------------------------------------------- |
| **Precision** | Purity of predicted positives        | Ratio of correctly classified “positive” points to all points inside the **predicted positive region** | *Positive Predictive Value (PPV)*             |
| **Recall**    | Coverage of actual positives         | Ratio of “captured” positive points among all **true positives in the real space**                     | *Sensitivity / True Positive Rate (TPR)*      |
| **F1 Score**  | Balance between precision and recall | Harmonic mean emphasizes lowest metric                                                                 | *Harmonic mean for stability under imbalance* |

---

## 🧠 4. Harmonic Mean Reason (Why Not Average?)

Arithmetic mean gives equal weight, but harmonic mean punishes imbalance.

Example:

* Precision = 0.9
* Recall = 0.1

Arithmetic mean = 0.5 → looks average
Harmonic mean (= 2×(0.9×0.1)/(0.9+0.1) = 0.18) → clearly bad balance

Hence, F1 = **low if one metric is weak**, even if the other is high.

---

## ⚖️ 5. Weighted Fβ-Score (Generalization of F1)

Sometimes you value Recall more (e.g., medical cases) or Precision more (e.g., fraud detection).

[
F_\beta = (1 + \beta^2) \times \frac{Precision \times Recall}{(\beta^2 \times Precision) + Recall}
]

* β > 1 → more weight on Recall
* β < 1 → more weight on Precision
* β = 1 → normal **F1 Score**

| Example | Focus           | β   | Formula Effect           |
| ------- | --------------- | --- | ------------------------ |
| F₁      | balanced        | 1   | equal weight             |
| F₂      | recall-heavy    | 2   | recall more important    |
| F₀․₅    | precision-heavy | 0.5 | precision more important |

---

## 🧾 6. Multi-Class Classification Case

In multi-class problems, compute Precision, Recall, F1 **per class**, then average:

| Type                 | Formula                                              | Meaning                     |
| -------------------- | ---------------------------------------------------- | --------------------------- |
| **Macro average**    | Average metric for all classes equally               | Equal weight per class      |
| **Micro average**    | Compute global TP, FP, FN first, then apply formulas | Weighted by class frequency |
| **Weighted average** | Weighted by number of true samples per class         | Adjusts for imbalance       |

Formally:
[
\text{Precision}*{macro} = \frac{1}{k}\sum*{i=1}^k P_i
\quad ; \quad
\text{Precision}_{micro} = \frac{\sum_i TP_i}{\sum_i (TP_i + FP_i)}
]
(similarly for Recall and F1)

---

## 📉 7. Relation to Other Metrics

| Related Metric                   | Formula                                      | Link                           |
| -------------------------------- | -------------------------------------------- | ------------------------------ |
| **Specificity**                  | TN / (TN + FP)                               | True Negative Rate             |
| **Accuracy**                     | (TP + TN) / (TP + TN + FP + FN)              | Overall correctness            |
| **Balanced Accuracy**            | (Recall + Specificity) / 2                   | Adjusted for imbalance         |
| **Precision–Recall Curve**       | Precision vs Recall for different thresholds | Helps choose optimal threshold |
| **AUC–PR (Area Under PR Curve)** | Area under precision–recall curve            | Measures average performance   |

---

## 🔍 8. Threshold & Trade-Off Behavior

When you adjust classification threshold (e.g., logistic regression output ≥ 0.5 → positive):

| Threshold       | Effect on Precision | Effect on Recall |
| --------------- | ------------------- | ---------------- |
| Higher (strict) | ↑ Precision         | ↓ Recall         |
| Lower (lenient) | ↓ Precision         | ↑ Recall         |

So, models must choose **thresholds** that balance both (maximize F1 or tune for domain).

---

## 🧩 9. Practical Example

Spam email detector outputs probabilities.
You set threshold = 0.5
Confusion matrix after test:

|                    | Actual Spam | Actual Not Spam |
| ------------------ | ----------- | --------------- |
| Predicted Spam     | 80          | 20              |
| Predicted Not Spam | 10          | 90              |

Then:

* TP = 80, FP = 20, FN = 10, TN = 90
* Precision = 80 / (80 + 20) = 0.8
* Recall = 80 / (80 + 10) = 0.8889
* F1 = 2 × (0.8×0.8889) / (0.8 + 0.8889) = 0.8421

If threshold raised to 0.7:
TP ↓ 70, FP ↓ 10, FN ↑ 20
Precision = 70 / 80 = 0.875, Recall = 70 / 90 = 0.777 → F1 = 0.823

So, trade-off visible — higher threshold increases precision but lowers recall.

---

## 🧠 10. Intuitive Analogy (Advanced)

| Scenario                            | Metric that matters most             |
| ----------------------------------- | ------------------------------------ |
| **Medical Diagnosis** (Cancer test) | Recall (don’t miss sick patients)    |
| **Email Spam Detection**            | Precision (don’t block real mails)   |
| **Security System (fraud)**         | F1 (balanced)                        |
| **Search Engine Results**           | Precision (show relevant ones first) |
| **Intrusion Detection**             | Recall (catch every possible threat) |

---

## 🔬 11. Mathematical Properties

1. **Range**: all three metrics in [0, 1]
2. **Symmetry**: F1 symmetric in Precision & Recall
3. **Monotonicity**: increasing TP increases all metrics; increasing FP decreases Precision; increasing FN decreases Recall
4. **F1 maximization** occurs where Precision = Recall
5. **Not differentiable** for threshold optimization → use surrogate losses (like Fβ in differentiable form)

---

## 🧭 12. Summary Table

| Metric    | Formula            | Range | Best when      | Notes                              |
| --------- | ------------------ | ----- | -------------- | ---------------------------------- |
| Precision | TP / (TP + FP)     | 0–1   | FP costly      | High = fewer false alarms          |
| Recall    | TP / (TP + FN)     | 0–1   | FN costly      | High = fewer misses                |
| F1 Score  | 2PR / (P+R)        | 0–1   | balance needed | Harmonic mean; penalizes imbalance |
| Fβ        | (1+β²)PR / (β²P+R) | 0–1   | domain-tunable | β controls trade-off               |

---

### 🧩 **In short:**

* **Precision** — How *right* the positives are.
* **Recall** — How *complete* the positives are.
* **F1 / Fβ** — The *balance* between being right and being complete.

---

