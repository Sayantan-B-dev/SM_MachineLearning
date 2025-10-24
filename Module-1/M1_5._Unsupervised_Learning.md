## **Unsupervised Learning**

Unsupervised learning is a type of machine learning where **we do not have labeled data**.
The algorithm tries to **find patterns, structure, or relationships** within the dataset **without knowing the output beforehand**.

---

### **Key Characteristics**

* **No labels** → No predefined "right answer"
* **No feedback** → The model is not corrected during training
* **Goal** → Discover hidden structure (clusters, patterns, lower-dimensional representations)

---

### **Major Types**

### **1. Clustering**

The algorithm groups data points based on similarity.
Objective: Points in the same group are more similar to each other than points in other groups.

**Example Algorithms:**

* **K-Means Clustering** → Partitions data into *k* groups
* **Hierarchical Clustering** → Builds a tree of clusters
* **DBSCAN** → Finds dense regions & treats outliers as noise

```md
# Resembling an Image (Clusters)
⬤ ⬤ ⬤        ⬤ ⬤        ⬤⬤⬤⬤⬤
⬤ ⬤           ⬤⬤⬤       ⬤
⬤⬤⬤           ⬤⬤⬤⬤⬤    ⬤⬤⬤⬤
(Imagine: Left cluster, middle cluster, right cluster)
```

---

### **2. Dimensionality Reduction**

Reduces the number of features/variables while keeping as much information as possible.

**Why we do this:**

* Make data easier to visualize (e.g., project 3D → 2D)
* Reduce computational cost
* Remove noise & redundant features

**Common Techniques:**

* **PCA (Principal Component Analysis)** → Finds directions of maximum variance
* **t-SNE, UMAP** → Better for visualization of complex data

```md
# Resembling an Image (Dimensionality Reduction)
3D Cube → 2D Plane → Line

[■■■]  →  [■■]  →  [■]
(Imagine: Reducing number of dimensions step by step)
```

---

### **Applications**

#### **1. Anomaly Detection**

* Finds data points that do not belong to any cluster (outliers)
* **Use Cases:**

  * Credit card fraud detection
  * Network intrusion detection
  * Machine failure prediction

```md
# Resembling an Image (Anomaly Detection)
⬤⬤⬤⬤⬤⬤⬤⬤⬤      ⬤ (← Anomaly)
⬤⬤⬤⬤⬤⬤⬤⬤⬤
```

---

#### **2. Zone/Community Detection (Clustering Example)**

* **Use Case:** Dividing a city map into zones based on population density
* Each point (house, shop) looks for its neighbors → forms a community

```md
# Resembling an Image (Zones/Regions)
─────────────── (Decision Boundary)
Cluster A       | Cluster B
⬤⬤⬤⬤⬤⬤⬤⬤⬤ | ⬤⬤⬤⬤⬤⬤⬤⬤⬤
─────────────── (Decision Boundary)
Cluster C
⬤⬤⬤⬤⬤⬤⬤⬤⬤⬤⬤⬤
```

---

### **KNN (K-Nearest Neighbors) Note**

Even though **KNN** is mostly used for classification, you can think of it as:

* Each new point checks **its K nearest neighbors**
* Decides which cluster/class it should belong to based on majority vote

```md
# Resembling an Image (KNN)
⬤⬤⬤(Neighbor 1)⬤⬤
   ⬤ (New Point → Joins cluster of nearest neighbors)
```

---

## **📌 K-Means Clustering — Algorithm**

```md
# K-Means Clustering Pseudo-Code

1. Choose number of clusters K
2. Randomly select K points as initial centroids
3. Repeat until convergence:
   a) Assign each data point to the nearest centroid
   b) Update centroids (average of all points in cluster)
4. When centroids stop changing significantly → DONE
```

---

### **Flowchart-like Representation**

```md
          ┌──────────────────────┐
          │  Choose K Clusters   │
          └─────────┬────────────┘
                    ↓
          ┌──────────────────────┐
          │  Pick K Centroids    │
          └─────────┬────────────┘
                    ↓
          ┌──────────────────────┐
          │ Assign Points to     │
          │ Nearest Centroid     │
          └─────────┬────────────┘
                    ↓
          ┌──────────────────────┐
          │ Recompute Centroids  │
          └─────────┬────────────┘
                    ↓
          ┌──────────────────────┐
          │  Centroids Change?   │
          └─────┬─────┬─────────┘
                │Yes   │No
                │      │
                ↓      ↓
      ┌──────────────────────┐
      │ Repeat Assignment    │
      └──────────────────────┘
                    ↓
          ┌──────────────────────┐
          │ Final Clusters Formed│
          └──────────────────────┘
```

---

## **📌 Anomaly Detection — Algorithm**

```md
# Anomaly Detection (Unsupervised) Pseudo-Code

1. Collect normal data points (no anomalies)
2. Learn distribution/pattern of normal data
3. For a new data point:
   a) Calculate probability of belonging to normal distribution
   b) If probability < threshold → FLAG as anomaly
   c) Else → Mark as normal
```

---

### **Flowchart-like Representation**

```md
          ┌──────────────────────┐
          │ Train on Normal Data │
          └─────────┬────────────┘
                    ↓
          ┌──────────────────────┐
          │ New Data Point Comes │
          └─────────┬────────────┘
                    ↓
          ┌──────────────────────┐
          │ Compute Probability  │
          │ (Belongs to Normal?) │
          └─────────┬────────────┘
                    ↓
          ┌──────────────────────┐
          │ Probability < Thresh?│
          └─────┬─────┬─────────┘
                │Yes   │No
                │      │
                ↓      ↓
      ┌──────────────────────┐
      │  FLAG AS ANOMALY     │
      └──────────────────────┘
                │
                │
                └───────────────→ NORMAL
```

---

These flowcharts look quite nice in Markdown because they give a **visual hierarchy** of steps — easy for internship notes and interviews.

---