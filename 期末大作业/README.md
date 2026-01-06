# 微信聊天记录综合分析项目设计文档

本项目通过一体化的 Jupyter Notebook（`final_code.ipynb`）实现微信聊天记录的加载、清洗、统计、可视化、文本分析与情绪分析，并支持生成一份基于外部大模型（DeepSeek）的中文总结报告。本文档详细说明项目目标、数据结构适配、各模块的设计与实现、依赖库、运行流程与输出结果。

## 一、项目概述
- 项目形态：Jupyter Notebook（`final_code.ipynb`）+ 设计文档（`README.md`）
- 数据来源：`data.json`（包含聊天消息），可选停用词表 `stop words.txt`
- 主要功能：
  - 中文字体环境配置，保证 Matplotlib 图表中文正常显示。
  - 数据加载与预处理（嵌套 JSON 兼容、时间字段标准化、消息类型与发送者字段适配）。
  - 基本统计分析（总消息量、成员发言量、消息类型分布）。
  - 活跃度分析（每日消息趋势、每小时活跃度分布、日历热力图）。
  - 文本内容分析（中文分词、停用词过滤、词云生成）。
  - 情绪分析（使用 PaddleNLP Taskflow 的情感分析），并可视化整体分布与时间趋势。
  - AI 报告生成（DeepSeek Chat API，总结聊天风格与核心话题）。

## 二、数据结构与适配设计
- 输入文件：`data.json`
- 结构特点：
  - 支持两种结构：
    1. 根节点即为消息列表（list of dict）
    2. 根节点包含 `messages` 键，实际消息列表位于 `data['messages']`
  - 字段命名兼容：
    - 时间字段可能为 `time` 或 `formattedTime`（Notebook 中统一重命名为 `time`）
    - 发送者字段可能为 `senderDisplayName`（统一为 `sender`）
    - 消息类型字段可能为 `type`（Notebook 中创建 `message_type` 列）
- 预处理流程：
  1. 读取 JSON，并兼容 `messages` 嵌套结构。
  2. 构建 `pandas.DataFrame`。
  3. 字段重命名与时间转换：`pd.to_datetime(df['time'])`。
  4. 派生特征：
     - `date = df['time'].dt.date`
     - `hour = df['time'].dt.hour`
     - `message_type = row.get('type', '未知')`

## 三、环境依赖与库选择
- 基础数据分析与可视化：
  - pandas：数据加载与清洗、分组聚合、统计分析。
  - numpy：数值计算辅助。
  - matplotlib、seaborn：静态图表绘制（柱状图、折线图、饼图）。
- 中文文本处理：
  - jieba：中文分词。
  - wordcloud：基于词频的词云生成（支持中文字体）。
- 交互式可视化：
  - pyecharts：日历热力图（Calendar）。
- 情绪分析：
  - paddlenlp.Taskflow("sentiment_analysis")：情感分类接口，返回每条文本的 `label` 与 `score`。
- AI 报告：
  - requests：调用 DeepSeek Chat API 生成中文分析报告。
- 中文字体：
  - 使用系统自带字体 `SimHei` 或 `msyh.ttc`。

## 四、Notebook 模块设计与实现细节

### 1. 环境设置与中文字体配置
- 目标：解决 Matplotlib 中文显示问题。
- 方法：设置 `plt.rcParams['font.sans-serif'] = ['SimHei']`、`axes.unicode_minus=False`，并绘制测试图确认。
- 结果：后续图表中文标题与标签正常显示。

### 2. 数据加载与预处理
- 函数：`load_and_preprocess_data(json_path)`
- 关键步骤：
  - JSON 读取与结构兼容：`data = data.get('messages', data)`
  - 字段标准化：`{'formattedTime': 'time', 'senderDisplayName': 'sender'}`
  - 时间转换：`pd.to_datetime(df['time'])`
  - 特征派生：`date`、`hour`、`message_type`
- 输出：DataFrame `chat_df`，包含核心分析所需字段。

### 3. 基本统计分析
- 指标：
  - 总消息数：`len(chat_df)`
  - 各成员发言量：`chat_df['sender'].value_counts()`
  - 消息类型分布：`chat_df['message_type'].value_counts()`
- 可视化：
  - seaborn 柱状图（成员发言量、消息类型分布）。
- 结果解读：
  - 可直观比较参与者的聊天活跃度与不同消息类型占比，识别主要发言者与主要消息形式。

### 4. 活跃度分析
- 每日消息趋势：
  - 方法：`daily_trend = chat_df.groupby('date').size()`，Matplotlib 折线图。
- 每小时活跃度：
  - 方法：`hourly_activity = chat_df.groupby('hour').size()`，seaborn 柱状图。
- 日历热力图：
  - 方法：基于 `pyecharts.charts.Calendar`，将每日消息量重采样为完整日期范围，并以 `render_notebook()` 输出交互式热力图。
- 结果解读：
  - 识别聊天活跃的时间段与日期模式，发现周/月度活跃趋势与异常峰值。

### 5. 文本内容分析（词云）
- 数据筛选：`chat_df[chat_df['message_type'] == '文本消息']['content']`
- 分词与清洗：
  - 只保留中文：正则 `[^\u4e00-\u9fa5]`
  - jieba 分词 + 停用词过滤（加载 `stop words.txt` 并补充常见词）。
- 词云生成：
  - `WordCloud.generate_from_frequencies(Counter(filtered_words))`
  - 中文字体路径优先使用 `C:/Windows/Fonts/simhei.ttf`，找不到则尝试 `msyh.ttc`。
- 结果解读：
  - 展示聊天中的高频词与核心话题，辅助 AI 报告的主题提炼。

### 6. 情绪分析（PaddleNLP Taskflow）
- 模型加载：`from paddlenlp import Taskflow; Taskflow("sentiment_analysis", use_gpu=False)`
- 批处理预测：
  - 分批大小 `batch_size=256`，调用 `senta(batch)`，得到每条文本的 `{label, score}`。
- 结果回填：
  - `sentiment_label`（英文标签：positive/negative/neutral）
  - `positive_prob`（置信度得分，记作正向概率字段）
  - 中文标签映射：`{'positive': '积极', 'negative': '消极', 'neutral': '中性'}`
- 可视化：
  - 饼图展示整体情绪分布。
  - 按天求平均情绪得分（积极=1，中性=0，消极=-1）并绘制折线图。
- 结果解读：
  - 量化聊天的整体情绪倾向与随时间变化趋势，辅助识别阶段性氛围变化。

### 7. AI 生成的中文分析报告（DeepSeek）
- 数据准备：
  - 总消息数、时间跨度、参与者发言量分布。
  - 词频 Top-N（基于 jieba 分词与停用词过滤）。
- Prompt 构建：
  - 要求总结聊天风格与氛围、参与者模式、核心话题与关系特点、趣味观察。
- API 调用：
  - `POST https://api.deepseek.com/chat/completions`
  - 模型：`deepseek-chat`
  - 认证：`Bearer <DEEPSEEK_API_KEY>`（从 `config.py` 读取）
- 输出：
  - 控制台打印中文报告，便于阅读与保存。

## 五、运行与复现流程
1. 准备环境与数据：
   - 确保 Python 环境可用，已安装必要依赖（pandas、numpy、matplotlib、seaborn、jieba、wordcloud、pyecharts、paddlenlp、requests）。
   - 将聊天数据保存为 `data.json`，可选准备 `stop words.txt` 进行停用词扩充。
   - 在 Windows 环境下推荐安装 `SimHei` 或使用系统内置 `msyh.ttc` 字体以保证中文显示。
2. 打开并运行 Notebook：
   - 依次运行 `final_code.ipynb` 的各个单元格：
     - 环境与字体设置
     - 数据加载与预处理
     - 基本统计分析
     - 活跃度分析（包含日历热力图）
     - 文本内容分析（词云）
     - 情绪分析（PaddleNLP Taskflow）
     - AI 报告生成（需配置 `config.DEPPSEEK_API_KEY`）
3. 结果查看：
   - 可视化图表在 Notebook 中直接渲染，交互式日历热力图使用 pyecharts。
   - AI 报告将打印在输出中。

## 六、关键设计取舍与鲁棒性
- 字段兼容性：对时间、发送者、类型等关键字段进行统一命名与派生处理，适配不同 JSON 源。
- 中文显示：通过字体设置与路径回退保证中文可视化稳定。
- 大数据量处理：情绪分析采用分批处理（批量大小可调整），防止内存溢出。
- 库选择策略：
  - 情绪分析优先使用 PaddleNLP Taskflow，避免 PaddleHub 接口在不同版本中的不兼容问题。
  - 可视化同时使用静态（matplotlib/seaborn）与交互式（pyecharts）方案，提高可读性与探索性。

