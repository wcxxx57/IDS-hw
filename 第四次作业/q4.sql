CREATE TABLE unified_data (
    id INT PRIMARY KEY AUTO_INCREMENT,
    subject_name VARCHAR(100) NOT NULL,  -- 直接存储学科名称
    institution_name VARCHAR(255),
    country_region VARCHAR(100),
    world_rank INT,
    web_of_science_documents INT,
    cites INT,
    cites_per_paper DECIMAL(10,2),
    top_papers INT
);

-- 合并化学数据
INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'agricultural sciences',  -- 直接写死学科名称
    `Institutions`,
    `Countries/Regions`, 
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `agricultural sciences`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'biology & biochemistry',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `biology & biochemistry`;

-- 继续合并其他20个表...
INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'chemistry',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `chemistry`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'clinical medicine',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `clinical medicine`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'computer science',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `computer science`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'economics & business',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `economics & business`;

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

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'environment ecology',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `environment ecology`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'geosciences',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `geosciences`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'immunology',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `immunology`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'materials science',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `materials science`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'mathematics',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `mathematics`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'microbiology',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `microbiology`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'molecular biology & genetics',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `molecular biology & genetics`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'multidisciplinary',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `multidisciplinary`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'neuroscience & behavior',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `neuroscience & behavior`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'pharmacology & toxicology',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `pharmacology & toxicology`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'physics',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `physics`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'plant & animal science',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `plant & animal science`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'psychiatry psychology',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `psychiatry psychology`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'social sciences, general',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `social sciences, general`;

INSERT INTO unified_data (subject_name, institution_name, country_region, world_rank, web_of_science_documents, cites, cites_per_paper, top_papers)
SELECT 
    'space science',
    `Institutions`,
    `Countries/Regions`,
    `Top`,
    `Web of Science Documents`,
    `Cites`,
    `Cites/Paper`,
    `Top Papers`
FROM `space science`;