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
             country_region = 'ISRAEL' OR country_region = 'EGYPT' 
             THEN 'Middle East'
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