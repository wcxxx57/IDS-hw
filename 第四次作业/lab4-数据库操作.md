# lab4-数据库操作

由于无法登录网站，本次实践作业使用的是**胡老师提供的数据**。

## 1.导入到关系型数据库系统

我使用的关系型数据库系统是`Mysql`，并使用`Mysql Workbench`的GUI进行操作。

连接到本地数据库后，首先**新建了一个名为`idshw_esi`的schema**，并且在`Mysql Workbench`中将不同学科的csv格式的数据**手动通过`Table Data Import Wizar`导入为该schema中的table**，导入后的数据库形式如下图所示：

![image-20251013234054436](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20251013234054436.png)

## 2.优化关系型数据，并整理一个合理的schema

导入后数据库中有**22个学科表**，每个表结构相同但数据不同，会导致：**查询复杂**（需要动态SQL），**管理困难**（要维护22个表），且**无法统一分析**等问题，因此需要先优化整理一个合理的schema结构。

我采用的优化方案是将现有的22个学科表**合并为1个数据表**，在综合表中新增一个`subject_name`（学科名）字段，并增加`id`字段作为每条数据的**主键**，使查询和维护都更加简单。具体步骤及sql语句如下：

1. **新建**一个包含学科名字段和id主键的统一的数据表`unified_data`

   ```sql
   CREATE TABLE unified_data (
       id INT PRIMARY KEY AUTO_INCREMENT,   -- id字段（主键）
       subject_name VARCHAR(100) NOT NULL,  -- 学科名字段
       institution_name VARCHAR(255),
       country_region VARCHAR(100),
       world_rank INT,                      -- 排名字段（对应原‘Top’字段）
       web_of_science_documents INT,
       cites INT,
       cites_per_paper DECIMAL(10,2),
       top_papers INT
   );
   ```

2. 将22个学科表的数据**插入**到新的统一的数据表中，插入的sql语句如下所示：

   ```sql
   -- 插入化学学科表的数据
   INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
   SELECT 
       'chemistry',  -- 直接写死学科名称
       `Institutions`,
       `Countries/Regions`, 
       `Top`,
       `Web of Science Documents`,
       `Cites`,
       `Cites/Paper`,
       `Top Papers`
   FROM `chemistry`;
   
   -- 插入工程学科表的数据
   INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
   SELECT 
       'engineering',
       `Institutions`,
       `Countries/Regions`,
       `Top`,
       `Web of Science Documents`,
       `Cites`,
       `Cites/Paper`,
       `Top Papers`
   FROM `engineering`;
   
   -- 继续插入其他20个表...
   ```

最终得到如下所示，共包含34121条数据的**统一数据表**：

![image-20251014094125835](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20251014094125835.png)

## 3.获取华东师范大学在各个学科中的排名

在统一的数据表`unified_data`中利用**SELECT**和**WHERE**查找`institution_name`=`'EAST CHINA NORMAL UNIVERSITY'`的数据中的学科和排名字段，并按照排名升序展示。SQL语句如下：

```sql
SELECT 
    subject_name as 学科,
    world_rank as 全球排名
FROM unified_data 
WHERE institution_name = 'EAST CHINA NORMAL UNIVERSITY'
ORDER BY world_rank ASC;
```

查询结果显示华东师范大学共有17个学科上榜，结果如下：![image-20251014190719510](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20251014190719510.png)

可见华东师范大学在**化学、数学、环境**等学科的全球学术排名较为靠前。

## 4.获取中国（大陆地区）大学在各个学科中的表现

### 4.1 获取中国（大陆地区）大学在各个学科中的数据

类似上面3中的操作，在统一的数据表`unified_data`中利用**SELECT**和**WHERE**查找`country_region = 'CHINA MAINLAND'`的数据，并按照一级排序学科和二级排序排名升序展示。SQL语句如下：

```sql
SELECT 
    subject_name as 学科,
    institution_name as 大学名称,
    world_rank as 全球排名,
    web_of_science_documents as 论文数,
    cites as 引用数,
    cites_per_paper as 篇均引用,
    top_papers as 高水平论文数
FROM unified_data 
WHERE country_region = 'CHINA MAINLAND'
ORDER BY subject_name, world_rank ASC;
```

查询结果共**4061条**中国（大陆地区）大学的数据，部分数据如下图所示：

![image-20251014134212488](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20251014134212488.png)

### 4.2 将各个学科中的中国大学表现简要分析并展示

为了更清晰地展示中国大陆地区大学在各个学科中的表现，针对每个学科，查询并计算**上榜大学数量，上榜大学的最好排名，最好排名的大学，最差排名，平均排名，总论文数总引用数，平均篇均引用，总高水平论文数**等信息，并按学科进行展示。其中**最好排名大学**利用了**子查询**来获取。对应的SQL语句如下：

```sql
SELECT 
    subject_name as 学科,
    COUNT(*) as 上榜大学数量,
    MIN(world_rank) as 最好排名,
    (SELECT institution_name     -- 利用子查询来获取最好排名的大学
     FROM unified_data u2 
     WHERE u2.subject_name = u1.subject_name 
       AND u2.country_region = 'CHINA MAINLAND' 
       AND u2.world_rank = MIN(u1.world_rank)
     LIMIT 1) as 最好排名大学,
    MAX(world_rank) as 最差排名,
    ROUND(AVG(world_rank), 1) as 平均排名,
    SUM(web_of_science_documents) as 总论文数,
    SUM(cites) as 总引用数,
    ROUND(AVG(cites_per_paper), 2) as 平均篇均引用,
    SUM(top_papers) as 总高水平论文数
FROM unified_data u1
WHERE country_region = 'CHINA MAINLAND'
GROUP BY subject_name
ORDER BY 上榜大学数量 DESC;
```

共22个学科的查询结果如下图所示：

![image-20251014135107366](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20251014135107366.png)

由查询到的结果可以大致看出，**CHINESE ACADEMY OF SCIENCE（中国科学院大学）**在工程、化学、材料科学等在内的很多学科上都是中国大陆大学中排名最高的，中国大陆大学的**计算机（平均排名378.1）、数学（平均排名190.5）等学科**在全球排名整体较高。

## 5.分析全球不同区域在各个学科中的表现

首先通过观察榜上的区域数据，将不同的国家分为**北美洲**（包含美国、加拿大），**欧洲**（包含德国、英国、法国、意大利等多个国家），**东南亚**（包含中国大陆、中国台湾、日本、韩国、印度、新加坡等），**大洋洲**（包括澳大利亚、新西兰），**南美洲**（包括阿根廷、巴西等国家），**中东**（包括土耳其、埃及等国家）和**非洲**，另外还有一些其他的较为小众的国家都归为**其他地区**。然后考察各个学科中不同地区的学术水平，用类似4中的方法获取其**上榜大学数量，学科内占比（占上榜总机构的百分比），上榜大学的最好排名，最差排名，平均排名，总论文数总引用数，平均篇均引用，总高水平论文数**等信息。

对应的SQL语句如下：

```sql
SELECT 
    subject_name as 学科,
    CASE 
        -- 北美洲
        WHEN country_region = 'USA' OR country_region = 'CANADA' THEN 'North America'
        -- 欧洲
        WHEN country_region LIKE '%GERMANY%' OR 
             country_region = 'UNITED KINGDOM' OR country_region = 'UK' OR 
             country_region = 'ENGLAND' OR country_region = 'SCOTLAND' OR 
             country_region = 'WALES' OR country_region = 'FRANCE' OR 
             country_region = 'ITALY' OR country_region = 'SPAIN' OR 
             country_region = 'RUSSIA' OR country_region = 'NETHERLANDS' OR 
             country_region = 'SWITZERLAND' OR country_region = 'SWEDEN' OR 
             country_region = 'BELGIUM' OR country_region = 'POLAND' OR 
             country_region = 'PORTUGAL' OR country_region = 'FINLAND' OR 
             country_region = 'DENMARK' OR country_region = 'NORWAY' OR 
             country_region = 'AUSTRIA' OR country_region = 'IRELAND' THEN 'Europe'
        -- 东南亚
        WHEN country_region = 'CHINA MAINLAND' OR country_region = 'JAPAN' OR 
             country_region = 'SOUTH KOREA' OR country_region = 'KOREA' OR 
             country_region = 'TAIWAN' OR country_region = 'INDIA' OR 
             country_region = 'SINGAPORE' OR country_region = 'MALAYSIA' OR 
             country_region = 'THAILAND' THEN 'Southeast Asia'
        -- 大洋洲
        WHEN country_region = 'AUSTRALIA' OR country_region = 'NEW ZEALAND' THEN 'Oceania'
        -- 南美洲
        WHEN country_region = 'BRAZIL' OR country_region = 'ARGENTINA' OR 
             country_region = 'CHILE' THEN 'South America'
        -- 中东
        WHEN country_region = 'IRAN' OR country_region = 'TURKIYE' OR 
             country_region = 'TURKEY' OR country_region = 'SAUDI ARABIA' OR 
             country_region = 'ISRAEL' OR country_region = 'EGYPT' THEN 'Middle East'
        -- 非洲
        WHEN country_region = 'SOUTH AFRICA' THEN 'Africa'
        -- 其他地区
        ELSE 'Other Regions'
    END as 区域,
    COUNT(*) as 上榜机构数,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY subject_name), 2) as '学科内占比(%)',
    MIN(world_rank) as 最好排名,
    MAX(world_rank) as 最差排名,
    ROUND(AVG(world_rank), 1) as 平均排名,
    SUM(web_of_science_documents) as 总论文数,
    SUM(cites) as 总引用数,
    ROUND(AVG(cites_per_paper), 2) as 平均篇均引用,
    SUM(top_papers) as 总高水平论文数
FROM unified_data 
GROUP BY subject_name, 区域
ORDER BY subject_name, 上榜机构数 DESC;
```

查询结果的**部分数据**如下图所示：![image-20251014185509858](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20251014185509858.png)

就如展示的农业科学和生化科学数据来讲，可以看出Europe（欧洲）大学实验室在这两个领域的贡献较大且水平较高，非洲、大洋洲和南美洲的机构水平较弱。其他20个学科的数据中也可以得出类似的结论。

## 6.附件说明

`q4.sql`：优化schema，新建并插入数据到一个统一的数据表的sql语句

`q5.sql`：获取华东师范大学各个学科排名对应的sql语句

`q6.sql`：获取中国（大陆地区）大学在各个学科中的数据的sql语句

`q6-2.sql`：将各个学科中的中国大学表现简要分析（包含总论文数、平均排名等）并展示的sql语句

`q7.sql`：分析全球不同区域在各个学科中的表现的sql语句
