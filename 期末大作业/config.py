# 配置文件 - 请在这里填写你的设置

# DeepSeek API密钥
# 获取方式：访问 https://platform.deepseek.com/ 注册并获取API密钥
DEEPSEEK_API_KEY = "你申请的API key"

# 数据文件路径
# 设置为None则自动在当前目录查找Excel文件
# 或者指定具体路径，例如: "E:/wx_chat/lxy.xlsx"
CHAT_FILE_PATH = None

# 输出目录
OUTPUT_DIR = "analysis_results"

# 分析参数
MAX_SAMPLE_SIZE = 500  # 用于DeepSeek分析的最大样本数量
MAX_TOKENS = 2000  # DeepSeek API最大token数
