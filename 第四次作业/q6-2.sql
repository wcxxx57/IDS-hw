-- 各学科中国大学表现分析
SELECT 
    subject_name as 学科,
    COUNT(*) as 上榜大学数量,
    MIN(world_rank) as 最好排名,
    (SELECT institution_name 
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