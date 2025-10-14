-- 先查看总条数
SELECT COUNT(*) as 总记录数
FROM unified_data 
WHERE country_region = 'CHINA MAINLAND';


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