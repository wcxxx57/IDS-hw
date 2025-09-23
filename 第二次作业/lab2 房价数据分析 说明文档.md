# Lab2 房价数据分析说明文档

## 1.实验概述

本实验借助提供的**房价数据**，完整展示了数据分析的基本流程，包括**数据读取、缺失值处理、异常值检测、特征相关性分析、数据标准化与离散化**等步骤。

## 2.实验流程

### 0.数据读取与初步查看

首先通过 `pandas` 读取原始数据，并用 `head()` 方法预览数据结构，了解数据的基本情况。

```python
df = pd.read_csv('train.csv')
df.head()
```

### 1.缺失值检测与处理

- 统计每一列的缺失值数量，按降序排列，直观展示数据质量。
- 将特征分为**“分类列”和“数值列”**，对分类列缺失值统一用 `'None'` 填充，对数值列采用 **KNN（K近邻）方法**进行插补。
- 注意到**部分数值型特征实际表达的是分类含义**（如 `MSSubClass`房屋建筑类型编码、`OverallQual` 整体材料与装修质量评分等），将其归入分类列，避免错误处理。

```python
cat_cols = df.select_dtypes(include=['object']).columns.tolist() # 分类列
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()# 数值列

# 移除 SalePrice（不对目标变量插补）
if 'SalePrice' in num_cols:
    num_cols.remove('SalePrice')

# 手动修正：把一些应该是分类的数字也加入分类列
special_categorical = ['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YrSold']
for col in special_categorical:
    if col in num_cols:
        num_cols.remove(col)
        cat_cols.append(col)

# 分类列的缺失值处理
for col in cat_cols:
    df[col] = df[col].fillna('None')
# 数值列的缺失值处理
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])
```

### 2. 异常值检测

- 选取关键数值特征（如房价、面积等），通过**箱线图（Boxplot）和散点图（Scatterplot）进行可视化**，直观发现异常点。
- 利用 **IQR（四分位距）方法量化检测异常值**，统计每个特征的异常样本数。
- 关注单变量异常的同时，也关注**“双变量联合异常”（如面积大但价格低的房屋）**。

**可视化示例：**

- 箱线图展示各特征的分布及异常值：

  ![](picture\yi_xiang.png)

- 利用散点图揭示面积与价格的关系及联合异常，可以看出，如下图所示，右下角有两个数据点**地上居住面积很大，价格却很低**，我猜测可能是：

  - 数据录入错误（价格少写了一个零？）
  - 特殊交易（如家庭内部转让、法院拍卖）
  - 房屋存在严重结构性问题（未在 `Functional` 中体现）
  
  ![散点图示例](picture\yi_san.png)
  
  **IQR异常值统计结果示例**
  
  ![](picture\yi_shu.png)

### 3. 特征相关性分析

- 仅保留**数值型变量**，计算**相关性矩阵**，分析各特征间的线性关系。

  ```python
  # 只保留数值型变量
  numeric_df = df.select_dtypes(include=[np.number])
  # 计算相关性矩阵
  corr_matrix = numeric_df.corr()
  # 打印相关性矩阵
  print(corr_matrix)
  ```

- 通过**热力图（Heatmap）可视化**相关性矩阵，快速定位与房价高度相关的特征。热力图如下所示：

  ![热力图](picture\hot.png)

- 通过热力图，我们可以大致看出，`SalePrice`（房价）和`OverallQual`（整体质量）相关性较大，由于价格决定品质；`GarageYrBlt`（车库建造时间）和`YearBuilt`（房屋建造时间）相关性较大，由于车库和房屋一般是差不多一起建造的；`TotRmsAbvGrd`（地面以上总房间数）和`GrLivArea`（房屋面积）相关性较大，由于房屋面积越大，一般房间数也会更多等信息，大多也都符合我们现有常识。

### 4. 对`price`数据的标准化与离散化

- 利用`StandardScaler`对房价（`SalePrice`）进行标准化处理，使其**均值为0、方差为1**，便于后续的建模和特征比较。
- 使用 **KBinsDiscretizer** 将房价分为**5个等宽的等级**，实现连续变量的离散化。
- 相关代码如下：

```python
# 初始化标准化器
scaler = StandardScaler()
# 进行标准化
df['SalePrice_scaled'] = scaler.fit_transform(df[['SalePrice']])

# 初始化离散化器（分成五个区间，等宽）
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
# 进行离散化
df['SalePrice_bin'] = discretizer.fit_transform(df[['SalePrice']])
```

### 5. 与`price`相关性最高特征的解释

- 通过相关性排序，找出**与房价最相关的三个特征：`OverallQual`（整体质量）、`GrLivArea`（地上生活面积）、`GarageCars`（车库容量）。**
- 对三个相关性最高的特征的合理解释：

  - **`OverallQual`（整体质量）**:  

    房屋的建材、工艺和装修水平等整体质量直接决定其市场价值。高质量房屋更受欢迎，价格更高。

  - **`GrLivArea`（地上生活面积）**:  

    **居住空间**是购房者最关注的因素之一。面积越大，使用功能越强，价格自然越高。

  - **`GarageCars`（车库容量）**:  

    **车位数量反映房屋配套完整性**，而且大车库通常出现在高档社区，也意味着更大的土地和更高的建造成本。因此车库容量大的房屋往往价格更改

## 3.实验相关的思考

1. **关于异常值处理**

   异常值处理是数据分析很重要的一步，若是不去除异常值处理可能会影响后续相关性分析等进一步操作。然而，**从问题实际意义来看，这些看似异常的数据其实也具有分析的价值**，比如为什么有些面积大的房子价格反而低？这可能与房屋年代、地理位置、装修状况等因素有关。如果单纯的把这些数据忽略，可能就无法得出更丰富客观的结论

2. **特征间的相关与因果关系**  

   相关性高的特征是否存在因果关系？如经过数据分析得出 `OverallQual` 与房价高度相关，但是否意味着提升质量就能提升价格？并不，**相关不等于因果**，具体情况还需结合市场和实际调研。

3. **其他隐藏变量**

   除了数据集所提供的变量外，房价的决定因素还有很多其他的**隐藏变量**？如**家庭收入、学区、交通便利性**等，可能也会对房价有更深远的影响。

