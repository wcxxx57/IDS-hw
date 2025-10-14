SELECT 
    subject_name as 学科,
    world_rank as 全球排名
FROM unified_data 
WHERE institution_name = 'EAST CHINA NORMAL UNIVERSITY'
ORDER BY world_rank ASC;