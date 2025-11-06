# 🛒 Supermart Grocery Sales – Retail Analytics Dataset

## 📖 Overview
The **Supermart Grocery Sales – Retail Analytics** project focuses on analyzing and modeling retail sales data from a supermarket chain to understand performance trends and optimize business decisions.

Using **Python (Pandas, NumPy, Seaborn, Scikit-learn)**, this project performs:
- Sales performance tracking
- Profit margin evaluation
- Customer behavior analysis
- Predictive modeling to forecast future sales

A pre-trained `.pkl` file (model) is included and can be accessed through Google Drive for easy loading and testing.

---

## 🎯 Objectives
- Perform **Exploratory Data Analysis (EDA)** on grocery sales data.
- Identify **high-performing product categories** and **profitable regions**.
- Detect **sales trends** and **seasonal demand**.
- Build a **Machine Learning model** to predict future sales or profits.
- Provide an interactive and reusable `.pkl` model file for further experimentation.

---

## 📊 Dataset
- **Source:** [Kaggle – Supermarket Sales Data](https://www.kaggle.com/)
- **Format:** CSV file (`supermart_sales.csv`)
- **Key Columns:**
  - `Order ID`
  - `Customer Name`
  - `Region`
  - `Category`
  - `Sub-Category`
  - `Sales`
  - `Quantity`
  - `Discount`
  - `Profit`
  - `Order Date`

---

## 🧰 Technologies Used
| Category | Tools / Libraries |
|-----------|-------------------|
| **Programming Language** | Python 3.8+ |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Model Saving** | Pickle (`.pkl` format`) |
| **Environment** | Jupyter Notebook / Google Colab |

---

## 🧾 Project Workflow
### 1️⃣ Data Preprocessing
- Load dataset using Pandas
- Handle missing values and remove duplicates
- Convert date columns to datetime format
- Create new derived metrics such as `Profit Margin` and `Revenue per Product`

### 2️⃣ Exploratory Data Analysis (EDA)
- Analyze top-selling categories and sub-categories
- Study discount impact on profit
- Identify regional and segment-based performance
- Visualize sales and profit trends using charts

### 3️⃣ Predictive Modeling
- Train regression/classification models to predict future sales
- Split data into training and testing sets
- Evaluate model performance using metrics like RMSE and R² score
- Save the trained model as `supermart_sales_model.pkl`

### 4️⃣ Visualization
- Sales vs Profit scatter plots
- Category-wise sales distribution
- Region-wise revenue heatmaps
- Time-series sales trends

---

## 📦 Model File (PKL Download Section)
The pre-trained `.pkl` model file is hosted on Google Drive for easy access.

📁 **Download Pre-trained Model:**  
👉 [Click here to download `supermart_sales_model.pkl`](https://drive.google.com/uc?id=YOUR_FILE_ID&export=download)

> ⚠️ Replace `YOUR_FILE_ID` with your actual Google Drive file ID.  
> Example:
> ```
> https://drive.google.com/uc?id=1aBcDeFgHiJkLmNoPqRsTuVwXyZ123456&export=download
> ```

**To load the model in Python:**
```python
import pickle
import requests

url = "https://drive.google.com/uc?id=YOUR_FILE_ID&export=download"
response = requests.get(url)
open("supermart_sales_model.pkl", "wb").write(response.content)

with open("supermart_sales_model.pkl", "rb") as file:
    model = pickle.load(file)
