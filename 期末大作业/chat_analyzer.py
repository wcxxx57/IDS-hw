"""
微信聊天记录综合分析系统
功能：基础统计 + AI报告 + 情感分析 + 模拟交互 + 词云 + 聚类分析
"""
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import os
import glob

# ===== 关键修复：在导入 pyplot 之前先清理缓存和配置字体 =====
print("🔧 正在配置中文字体支持...")

# 第1步：清理matplotlib缓存（与 ultimate_test.py 相同）
cache_dir = matplotlib.get_cachedir()
try:
    cache_files = glob.glob(os.path.join(cache_dir, 'fontlist-*.json'))
    for f in cache_files:
        try:
            os.remove(f)
        except:
            pass
except:
    pass

# 第2步：强制重新加载字体管理器
fm._load_fontmanager(try_read_cache=False)

# 第3步：添加中文字体到matplotlib字体管理器
font_file = 'C:/Windows/Fonts/simhei.ttf'
if os.path.exists(font_file):
    try:
        # 使用 addfont 方法（matplotlib 3.2+）
        fm.fontManager.addfont(font_file)
        print(f"✅ 已添加字体文件: {font_file}")
    except AttributeError:
        # 旧版本matplotlib没有addfont方法
        pass

# 第4步：配置中文字体为默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False
print("✅ 中文字体配置完成: SimHei")

# 现在才导入 pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
import requests
import warnings
import jieba
import jieba.analyse
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re

warnings.filterwarnings('ignore')

# 再次强制设置 pyplot 的配置（确保生效）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

print("✅ Matplotlib 中文字体设置完成")

# 简化的字体设置函数（字体已在全局配置）
def setup_chinese_font():
    """返回中文字体属性"""
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simsun.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            return fm.FontProperties(fname=font_path)
    return None

# 获取字体属性（用于词云等需要字体路径的地方）
font_prop = setup_chinese_font()

# 设置seaborn风格（但不覆盖字体）
sns.set_style("whitegrid")
sns.set_palette("husl")

# 确保 seaborn 不覆盖字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class WeChatAnalyzer:
    """微信聊天记录综合分析器"""
    
    def __init__(self, deepseek_api_key=None):
        self.df = None
        self.api_key = deepseek_api_key
        self.stats = {}
        self.sentiment_results = {}
        
    def load_json_data(self, json_file):
        """从JSON文件加载聊天记录"""
        print(f"📂 正在加载JSON文件: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取会话信息
        session = data.get('session', {})
        messages = data.get('messages', [])
        
        print(f"✅ 会话信息:")
        print(f"   对象: {session.get('displayName', 'Unknown')}")
        print(f"   类型: {session.get('type', 'Unknown')}")
        print(f"   消息总数: {session.get('messageCount', 0)}")
        
        # 解析消息
        parsed_messages = []
        for msg in messages:
            # 跳过没有完整信息的消息
            if 'formattedTime' not in msg or 'content' not in msg:
                continue
            
            # 只处理文本消息
            if msg.get('type') == '文本消息' or (msg.get('content') and not msg.get('content').startswith('[')):
                parsed_messages.append({
                    'datetime': msg.get('formattedTime'),
                    'sender': msg.get('senderDisplayName', 'Unknown'),
                    'content': msg.get('content', ''),
                    'is_send': msg.get('isSend', 0)
                })
        
        # 转换为DataFrame
        self.df = pd.DataFrame(parsed_messages)
        
        if len(self.df) == 0:
            print("⚠️ 未找到有效的文本消息")
            return False
        
        # 数据预处理
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], errors='coerce')
        self.df = self.df.dropna(subset=['datetime', 'content'])
        self.df = self.df[self.df['content'].str.strip() != '']
        
        # 添加时间相关字段
        self.df['date'] = self.df['datetime'].dt.date
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['weekday'] = self.df['datetime'].dt.weekday
        self.df['message_length'] = self.df['content'].astype(str).str.len()
        
        print(f"✅ 成功加载 {len(self.df)} 条有效消息")
        print(f"📅 时间范围: {self.df['date'].min()} 至 {self.df['date'].max()}")
        print(f"👥 参与者: {', '.join(self.df['sender'].unique())}")
        
        return True
    
    def basic_analysis(self):
        """基础统计分析"""
        print("\n" + "="*60)
        print("📊 基础统计分析")
        print("="*60)
        
        stats = {}
        
        # 总体统计
        stats['total_messages'] = len(self.df)
        stats['date_range'] = f"{self.df['date'].min()} 至 {self.df['date'].max()}"
        stats['duration_days'] = (self.df['date'].max() - self.df['date'].min()).days
        
        print(f"\n📈 总体情况:")
        print(f"   消息总数: {stats['total_messages']} 条")
        print(f"   时间跨度: {stats['duration_days']} 天")
        print(f"   日均消息: {stats['total_messages'] / max(stats['duration_days'], 1):.1f} 条")
        
        # 发送者统计
        sender_counts = self.df['sender'].value_counts()
        print(f"\n👥 参与者分布:")
        for sender, count in sender_counts.items():
            percentage = count / len(self.df) * 100
            print(f"   {sender}: {count} 条 ({percentage:.1f}%)")
            stats[f'{sender}_count'] = count
            stats[f'{sender}_percentage'] = percentage
        
        # 消息长度统计
        print(f"\n💬 消息长度:")
        print(f"   平均长度: {self.df['message_length'].mean():.1f} 字符")
        print(f"   最长消息: {self.df['message_length'].max()} 字符")
        print(f"   最短消息: {self.df['message_length'].min()} 字符")
        
        # 活跃时段
        hourly = self.df['hour'].value_counts().sort_index()
        peak_hour = hourly.idxmax()
        print(f"\n⏰ 活跃时段:")
        print(f"   最活跃: {peak_hour}:00 ({hourly[peak_hour]} 条)")
        
        # 高频词汇
        all_text = ' '.join(self.df['content'].astype(str))
        words = [w for w in all_text if len(w) > 1]
        word_freq = Counter(words).most_common(10)
        print(f"\n🔤 高频字符:")
        for word, freq in word_freq[:5]:
            print(f"   '{word}': {freq} 次")
        
        self.stats = stats
        return stats
    
    def simple_sentiment_analysis(self):
        """简单情感分析（基于关键词）"""
        print("\n" + "="*60)
        print("😊 情感倾向分析")
        print("="*60)
        
        # 定义情感关键词
        positive_words = ['哈哈', '😊', '😄', '👍', '好的', '谢谢', '喜欢', '开心', '棒', '爱', 
                         '嘿嘿', '嗯嗯', '可以', '不错', '厉害', '赞', '哇', '太好了', '😍', '❤️']
        negative_words = ['😢', '😭', '难过', '不好', '讨厌', '烦', '气', '累', '唉', '糟糕',
                         '不行', '算了', '无聊', '烦人', '😤', '💔']
        
        sentiment_labels = []
        
        for content in self.df['content']:
            content_str = str(content)
            has_positive = any(word in content_str for word in positive_words)
            has_negative = any(word in content_str for word in negative_words)
            
            if has_positive and not has_negative:
                sentiment_labels.append('积极')
            elif has_negative and not has_positive:
                sentiment_labels.append('消极')
            else:
                sentiment_labels.append('中性')
        
        # 添加到DataFrame
        self.df['sentiment'] = sentiment_labels
        
        # 统计
        sentiment_counts = self.df['sentiment'].value_counts()
        positive_count = sentiment_counts.get('积极', 0)
        neutral_count = sentiment_counts.get('中性', 0)
        negative_count = sentiment_counts.get('消极', 0)
        
        total = len(self.df)
        print(f"\n情感分布:")
        print(f"   😊 积极: {positive_count} 条 ({positive_count/total*100:.1f}%)")
        print(f"   😐 中性: {neutral_count} 条 ({neutral_count/total*100:.1f}%)")
        print(f"   😢 消极: {negative_count} 条 ({negative_count/total*100:.1f}%)")
        
        self.sentiment_results = {
            'positive': positive_count,
            'neutral': neutral_count,
            'negative': negative_count
        }
        
        return self.sentiment_results
    
    def word_frequency_analysis(self):
        """词频统计分析"""
        print("\n" + "="*60)
        print("📝 词频统计分析")
        print("="*60)
        
        # 合并所有消息
        all_text = ' '.join(self.df['content'].astype(str))
        
        # 分词
        print("🔄 正在分词...")
        words = jieba.cut(all_text)
        
        # 扩展停用词列表 - 过滤无用词汇
        stopwords = {
            # 基础停用词
            '的', '了', '是', '我', '你', '他', '她', '它', '在', '有', '和', '就', '不', 
            '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '吗',
            '会', '能', '没', '看', '好', '自己', '这', '那', '啊', '呀', '哦', '嗯',
            # 缀词和语气词
            '这个', '那个', '哈哈', '哈哈哈', '哈哈哈哈', '哈哈哈哈哈', '嘿嘿', '嘻嘻',
            '呵呵', '嗯嗯', '嗯呢', '哦哦', '啦啦', '呀呀', '吧吧', '呢呢',
            '就是', '然后', '但是', '还是', '可以', '现在', '觉得', '感觉', '真的', '我们',
            '什么', '怎么', '为什么', '这样', '那样', '这么', '那么', '多少', '哪里',
            # 文件格式和路径相关
            'media', 'emojis', 'gif', 'jpg', 'png', 'jpeg', 'mp4', 'mp3', 'pdf', 'doc',
            'images', 'voices', 'videos', 'files', 'http', 'https', 'www', 'com', 'cn',
            # 标点符号
            '[图片]', '[表情]', '[语音]', '[视频]', '[文件]', '[链接]',
            '，', '。', '！', '？', '；', '：', '"', '"', ''', ''', '、', '…', '—',
            # 空白字符
            ' ', '\n', '\t', '\r',
            # 单字
            '个', '些', '种', '样', '块', '把', '张', '只', '次', '下', '天', '年',
            # 无意义词汇
            '非常', '特别', '比较', '稍微', '有点', '一点', '一些', '一下'
        }
        
        # 文件格式匹配模式
        file_patterns = ['media', 'emojis', 'gif', 'jpg', 'png', 'jpeg', 'mp4', 'voices', 'images']
        
        # 过滤词汇
        filtered_words = []
        for w in words:
            # 过滤条件:
            # 1. 长度大于1
            # 2. 不在停用词表中
            # 3. 不包含文件格式相关字符
            # 4. 不是纯数字
            # 5. 不是纯英文(除非是有意义的长单词)
            if (len(w) > 1 and 
                w not in stopwords and 
                not any(pattern in w.lower() for pattern in file_patterns) and
                not w.isdigit() and
                not (w.encode('UTF-8').isalpha() and len(w) < 4)):  # 过滤短英文
                filtered_words.append(w)
        
        # 统计词频
        word_freq = Counter(filtered_words)
        top_words = word_freq.most_common(30)
        
        print(f"\n🔤 高频词汇 TOP 20:")
        for i, (word, freq) in enumerate(top_words[:20], 1):
            print(f"   {i:2d}. {word:8s} - {freq:4d} 次")
        
        self.word_freq = word_freq
        return word_freq
    
    def topic_clustering(self):
        """主题聚类分析"""
        print("\n" + "="*60)
        print("🎯 聊天主题聚类分析")
        print("="*60)
        
        # 准备文本数据
        texts = self.df['content'].astype(str).tolist()
        
        # 分词处理
        print("🔄 正在处理文本...")
        processed_texts = []
        
        # 扩展停用词列表 - 与词频分析保持一致
        stopwords = {
            # 基础停用词
            '的', '了', '是', '我', '你', '他', '她', '它', '在', '有', '和', '就', '不',
            '啊', '呀', '哦', '嗯', '吗', '呢',
            # 缀词和语气词
            '这个', '那个', '哈哈', '哈哈哈', '哈哈哈哈', '嘿嘿', '嘻嘻', '呵呵',
            '就是', '然后', '但是', '还是', '可以', '现在', '觉得', '感觉', '真的', '我们',
            '什么', '怎么', '这样', '那样',
            # 文件格式相关
            'media', 'emojis', 'gif', 'jpg', 'png', 'jpeg', 'mp4', 'voices', 'images'
        }
        
        file_patterns = ['media', 'emojis', 'gif', 'jpg', 'png', 'jpeg', 'mp4', 'voices', 'images']
        
        for text in texts:
            words = jieba.cut(text)
            # 过滤停用词和无用词汇
            filtered = []
            for w in words:
                if (len(w) > 1 and 
                    w not in stopwords and 
                    not any(pattern in w.lower() for pattern in file_patterns) and
                    not w.isdigit()):
                    filtered.append(w)
            processed_texts.append(' '.join(filtered))
        
        # TF-IDF向量化
        try:
            vectorizer = TfidfVectorizer(max_features=100, min_df=2)
            X = vectorizer.fit_transform(processed_texts)
            
            # K-means聚类
            n_clusters = min(5, len(self.df) // 100 + 1)  # 根据数据量动态调整聚类数
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            
            # 添加聚类标签到DataFrame
            self.df['cluster'] = clusters
            
            # 分析每个聚类的关键词
            print(f"\n📊 识别出 {n_clusters} 个主题聚类:\n")
            
            feature_names = vectorizer.get_feature_names_out()
            cluster_topics = {}
            
            for i in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-15:][::-1]  # 提取更多关键词
                top_words = [feature_names[idx] for idx in top_indices]
                
                cluster_size = (clusters == i).sum()
                cluster_percentage = cluster_size / len(clusters) * 100
                
                # 推测主题
                topic_name = self._guess_topic_name(top_words)
                
                # 提取该聚类的示例消息 - 优化选择策略
                cluster_df = self.df[self.df['cluster'] == i].copy()
                
                # 过滤掉过短或包含文件路径的消息
                valid_messages = cluster_df[
                    (cluster_df['content'].str.len() >= 10) &  # 至少10个字符
                    (~cluster_df['content'].str.contains('media|emojis|gif|jpg|png', case=False, na=False))
                ]['content']
                
                # 如果有有效消息，随机选择3条；否则使用原始消息
                if len(valid_messages) >= 3:
                    cluster_messages = valid_messages.sample(min(3, len(valid_messages))).tolist()
                elif len(valid_messages) > 0:
                    cluster_messages = valid_messages.tolist()
                else:
                    # 如果没有有效消息，选择最长的3条
                    cluster_messages = cluster_df.nlargest(3, 'message_length')['content'].tolist()
                
                print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"   主题 {i+1}: {topic_name}")
                print(f"   消息数: {cluster_size} 条 ({cluster_percentage:.1f}%)")
                print(f"   核心关键词: {', '.join(top_words[:10])}")
                print(f"   典型消息示例:")
                for j, msg in enumerate(cluster_messages[:3], 1):
                    # 截断过长的消息并清理
                    display_msg = str(msg).replace('\n', ' ').strip()
                    display_msg = display_msg[:60] + '...' if len(display_msg) > 60 else display_msg
                    print(f"      {j}. {display_msg}")
                print()
                
                cluster_topics[i] = {
                    'name': topic_name,
                    'size': cluster_size,
                    'keywords': top_words[:15],
                    'examples': cluster_messages[:3]
                }
            
            self.cluster_topics = cluster_topics
            return cluster_topics
            
        except Exception as e:
            print(f"⚠️ 聚类分析失败: {e}")
            print("   (可能是数据量太少或文本相似度过高)")
            return None
    
    def _guess_topic_name(self, keywords):
        """根据关键词推测主题名称 - 更具体化"""
        keywords_str = ' '.join(keywords)
        
        # 定义更具体的主题规则，按优先级匹配
        # 学习相关
        if any(w in keywords_str for w in ['作业', '考试', '试卷', '成绩', '分数']):
            return "📝 作业考试"
        elif any(w in keywords_str for w in ['课程', '上课', '老师', '教授', '讲课']):
            return "📚 课程学习"
        elif any(w in keywords_str for w in ['论文', '研究', '实验', '项目', 'paper']):
            return "🔬 学术研究"
        elif any(w in keywords_str for w in ['学习', '复习', '预习', '背书', '看书']):
            return "📖 自主学习"
        
        # 生活相关
        elif any(w in keywords_str for w in ['早饭', '午饭', '晚饭', '吃饭', '食堂', '外卖']):
            return "🍔 用餐话题"
        elif any(w in keywords_str for w in ['好吃', '美食', '餐厅', '菜', '味道']):
            return "😋 美食分享"
        elif any(w in keywords_str for w in ['睡觉', '起床', '困', '累', '休息']):
            return "😴 作息时间"
        elif any(w in keywords_str for w in ['宿舍', '寝室', '室友', '舍友']):
            return "🏠 宿舍生活"
        
        # 娱乐相关
        elif any(w in keywords_str for w in ['游戏', '打游戏', '玩游戏', '开黑', '上分']):
            return "🎮 游戏娱乐"
        elif any(w in keywords_str for w in ['电影', '电视剧', '综艺', '追剧']):
            return "🎬 影视剧集"
        elif any(w in keywords_str for w in ['音乐', '歌', '唱歌', '听歌']):
            return "� 音乐话题"
        elif any(w in keywords_str for w in ['运动', '跑步', '健身', '打球', '锻炼']):
            return "⚽ 运动健身"
        
        # 社交相关
        elif any(w in keywords_str for w in ['聚会', '活动', '聚餐', '出去玩']):
            return "🎉 聚会活动"
        elif any(w in keywords_str for w in ['朋友', '同学', '认识', '介绍']):
            return "👥 社交互动"
        elif any(w in keywords_str for w in ['购物', '买', '淘宝', '商品', '价格']):
            return "🛒 购物消费"
        
        # 工作相关
        elif any(w in keywords_str for w in ['工作', '实习', '公司', '面试', '求职']):
            return "💼 工作实习"
        elif any(w in keywords_str for w in ['会议', '开会', '汇报', '领导']):
            return "� 会议工作"
        
        # 情感相关
        elif any(w in keywords_str for w in ['开心', '高兴', '快乐', '喜欢', '爱']):
            return "� 开心分享"
        elif any(w in keywords_str for w in ['难过', '伤心', '难受', '郁闷']):
            return "😢 倾诉烦恼"
        elif any(w in keywords_str for w in ['生气', '气', '烦', '讨厌']):
            return "� 情绪发泄"
        
        # 其他
        elif any(w in keywords_str for w in ['天气', '下雨', '晴天', '冷', '热']):
            return "🌤️ 天气话题"
        elif any(w in keywords_str for w in ['时间', '地点', '什么时候', '哪里']):
            return "� 时间地点"
        else:
            return "💬 日常闲聊"
    
    def create_visualizations(self):
        """生成精美的可视化图表"""
        print("\n" + "="*60)
        print("📊 生成可视化图表")
        print("="*60)
        
        # 创建输出目录
        os.makedirs('analysis_results', exist_ok=True)
        
        # ===== 关键：不使用 style.use，避免重置字体设置 =====
        # 直接设置所需的样式参数
        plt.rcParams.update({
            'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'],
            'font.family': 'sans-serif',
            'axes.unicode_minus': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'figure.facecolor': 'white',
            'axes.facecolor': '#f0f0f0'
        })
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        
        # 1. 每日消息趋势图（更美观）
        print("📈 生成每日趋势图...")
        fig1 = plt.figure(figsize=(14, 7))
        ax1 = plt.subplot(111)
        
        daily = self.df.groupby('date').size()
        
        # 绘制面积图
        ax1.fill_between(daily.index, daily.values, alpha=0.3, color='#4ECDC4')
        ax1.plot(daily.index, daily.values, marker='o', linewidth=2.5, 
                color='#2E86AB', markersize=5, markerfacecolor='#FF6B6B')
        
        # 添加趋势线
        z = np.polyfit(range(len(daily)), daily.values, 2)
        p = np.poly1d(z)
        ax1.plot(daily.index, p(range(len(daily))), "--", 
                linewidth=2, alpha=0.5, color='#E74C3C', label='趋势线')
        
        ax1.set_title('📈 每日消息数量趋势分析', fontsize=18, fontweight='bold', pad=20)
        ax1.set_xlabel('日期', fontsize=13, fontweight='bold')
        ax1.set_ylabel('消息数量', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 美化刻度
        plt.xticks(rotation=45, ha='right')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('analysis_results/01_daily_trend.png', dpi=200, bbox_inches='tight')
        print("   ✅ 保存: 01_daily_trend.png")
        plt.close()
        
        # 2. 发送者分布（双图展示）
        print("👥 生成发送者分布图...")
        fig2 = plt.figure(figsize=(14, 6))
        
        sender_counts = self.df['sender'].value_counts()
        
        # 左图：饼图
        ax2_1 = plt.subplot(121)
        explode = [0.05] * len(sender_counts)
        wedges, texts, autotexts = ax2_1.pie(
            sender_counts.values, 
            labels=sender_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors[:len(sender_counts)],
            explode=explode,
            shadow=True,
            textprops={'fontsize': 11, 'weight': 'bold'}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
        ax2_1.set_title('消息发送者占比', fontsize=14, fontweight='bold', pad=15)
        
        # 右图：柱状图
        ax2_2 = plt.subplot(122)
        bars = ax2_2.barh(sender_counts.index, sender_counts.values, 
                          color=colors[:len(sender_counts)], alpha=0.8)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, sender_counts.values)):
            ax2_2.text(value, i, f' {value}条', va='center', fontsize=11, fontweight='bold')
        
        ax2_2.set_title('消息数量对比', fontsize=14, fontweight='bold', pad=15)
        ax2_2.set_xlabel('消息数量', fontsize=12, fontweight='bold')
        ax2_2.spines['top'].set_visible(False)
        ax2_2.spines['right'].set_visible(False)
        ax2_2.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('analysis_results/02_sender_distribution.png', dpi=200, bbox_inches='tight')
        print("   ✅ 保存: 02_sender_distribution.png")
        plt.close()
        
        # 3. 24小时活跃度热力图
        print("⏰ 生成活跃度分析图...")
        fig3 = plt.figure(figsize=(14, 6))
        
        hourly = self.df['hour'].value_counts().sort_index()
        
        # 创建渐变色柱状图
        ax3 = plt.subplot(111)
        bars = ax3.bar(hourly.index, hourly.values, 
                      color=plt.cm.YlOrRd(hourly.values / hourly.values.max()),
                      edgecolor='navy', linewidth=1.5, alpha=0.85)
        
        # 高亮最活跃时段
        max_hour = hourly.idxmax()
        bars[max_hour].set_color('#E74C3C')
        bars[max_hour].set_linewidth(3)
        bars[max_hour].set_edgecolor('darkred')
        
        # 添加数值标签
        for i, (hour, value) in enumerate(hourly.items()):
            if value > hourly.mean():
                ax3.text(hour, value, str(value), ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
        
        ax3.set_title(f'⏰ 24小时活跃度分布 (峰值: {max_hour}:00)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('小时', fontsize=13, fontweight='bold')
        ax3.set_ylabel('消息数量', fontsize=13, fontweight='bold')
        ax3.set_xticks(range(24))
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # 添加时段背景
        ax3.axvspan(0, 6, alpha=0.1, color='blue', label='凌晨')
        ax3.axvspan(6, 12, alpha=0.1, color='yellow', label='上午')
        ax3.axvspan(12, 18, alpha=0.1, color='orange', label='下午')
        ax3.axvspan(18, 24, alpha=0.1, color='purple', label='晚上')
        ax3.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('analysis_results/03_hourly_activity.png', dpi=200, bbox_inches='tight')
        print("   ✅ 保存: 03_hourly_activity.png")
        plt.close()
        
        # 4. 情感分布可视化
        print("😊 生成情感分析图...")
        fig4 = plt.figure(figsize=(14, 6))
        
        sentiment_counts = self.df['sentiment'].value_counts()
        
        # 左图：甜甜圈图
        ax4_1 = plt.subplot(121)
        sentiment_colors = {'积极': '#2ECC71', '中性': '#95A5A6', '消极': '#E74C3C'}
        colors_list = [sentiment_colors.get(s, '#95A5A6') for s in sentiment_counts.index]
        
        wedges, texts, autotexts = ax4_1.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors_list,
            wedgeprops=dict(width=0.5, edgecolor='white'),
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(13)
        
        ax4_1.set_title('😊 情感倾向分布', fontsize=14, fontweight='bold', pad=15)
        
        # 右图：按时间的情感趋势
        ax4_2 = plt.subplot(122)
        
        # 按日期统计情感
        sentiment_by_date = self.df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        
        if len(sentiment_by_date) > 1:
            for sentiment in ['积极', '中性', '消极']:
                if sentiment in sentiment_by_date.columns:
                    ax4_2.plot(sentiment_by_date.index, sentiment_by_date[sentiment],
                             marker='o', label=sentiment, linewidth=2,
                             color=sentiment_colors.get(sentiment, '#95A5A6'),
                             alpha=0.7)
            
            ax4_2.set_title('情感趋势变化', fontsize=14, fontweight='bold', pad=15)
            ax4_2.set_xlabel('日期', fontsize=12)
            ax4_2.set_ylabel('消息数量', fontsize=12)
            ax4_2.legend(loc='best', fontsize=11)
            ax4_2.grid(True, alpha=0.3, linestyle='--')
            plt.xticks(rotation=45)
            ax4_2.spines['top'].set_visible(False)
            ax4_2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('analysis_results/04_sentiment_analysis.png', dpi=200, bbox_inches='tight')
        print("   ✅ 保存: 04_sentiment_analysis.png")
        plt.close()
        
        # 5. 词云图
        if hasattr(self, 'word_freq') and self.word_freq:
            print("☁️ 生成词云图...")
            fig5 = plt.figure(figsize=(14, 8))
            
            # 生成词云
            try:
                # 尝试使用中文字体
                font_paths = [
                    'C:/Windows/Fonts/simhei.ttf',
                    'C:/Windows/Fonts/msyh.ttc',
                    '/System/Library/Fonts/PingFang.ttc'
                ]
                font_path = None
                for fp in font_paths:
                    if os.path.exists(fp):
                        font_path = fp
                        break
                
                wordcloud = WordCloud(
                    width=1400,
                    height=800,
                    background_color='white',
                    font_path=font_path,
                    colormap='viridis',
                    max_words=100,
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate_from_frequencies(self.word_freq)
                
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('☁️ 高频词汇云图', fontsize=20, fontweight='bold', pad=20)
                
                plt.tight_layout()
                plt.savefig('analysis_results/05_wordcloud.png', dpi=200, bbox_inches='tight')
                print("   ✅ 保存: 05_wordcloud.png")
            except Exception as e:
                print(f"   ⚠️ 词云生成失败: {e}")
            
            plt.close()
        
        # 6. 聚类主题分布
        if hasattr(self, 'cluster_topics') and self.cluster_topics:
            print("🎯 生成聚类分析图...")
            fig6 = plt.figure(figsize=(14, 7))
            
            # 提取数据
            topics = []
            sizes = []
            for cluster_id, info in self.cluster_topics.items():
                topics.append(info['name'])
                sizes.append(info['size'])
            
            # 创建水平柱状图
            ax6 = plt.subplot(111)
            y_pos = np.arange(len(topics))
            colors_grad = plt.cm.Set3(np.linspace(0, 1, len(topics)))
            
            bars = ax6.barh(y_pos, sizes, color=colors_grad, alpha=0.8, edgecolor='navy', linewidth=2)
            
            # 添加数值和百分比
            total_msgs = sum(sizes)
            for i, (bar, size) in enumerate(zip(bars, sizes)):
                percentage = size / total_msgs * 100
                ax6.text(size, i, f' {size}条 ({percentage:.1f}%)', 
                        va='center', fontsize=11, fontweight='bold')
            
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(topics, fontsize=12, fontweight='bold')
            ax6.set_xlabel('消息数量', fontsize=13, fontweight='bold')
            ax6.set_title('🎯 聊天主题分布分析', fontsize=16, fontweight='bold', pad=20)
            ax6.grid(axis='x', alpha=0.3, linestyle='--')
            ax6.spines['top'].set_visible(False)
            ax6.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('analysis_results/06_topic_clustering.png', dpi=200, bbox_inches='tight')
            print("   ✅ 保存: 06_topic_clustering.png")
            plt.close()
        
        print(f"\n📁 所有图表已保存到: analysis_results/")
        print(f"   共生成 {'6' if hasattr(self, 'cluster_topics') else '5'} 张精美图表")

    
    def generate_ai_report(self):
        """使用DeepSeek生成AI分析报告"""
        if not self.api_key or self.api_key == "你的API密钥":
            print("\n⚠️ 未配置DeepSeek API密钥，跳过AI报告生成")
            return None
        
        print("\n" + "="*60)
        print("🤖 生成AI深度分析报告")
        print("="*60)
        
        # 准备分析数据
        sample_messages = self.df.sample(min(50, len(self.df)))['content'].tolist()
        sample_text = '\n'.join([f"- {msg}" for msg in sample_messages[:20]])
        
        prompt = f"""
        作为专业的社交关系分析师，请分析以下微信聊天记录，给出专业见解：
        
        【统计数据】
        - 消息总数: {self.stats.get('total_messages', 0)}
        - 时间跨度: {self.stats.get('duration_days', 0)}天
        - 情感分布: 积极{self.sentiment_results.get('positive', 0)}条, 中性{self.sentiment_results.get('neutral', 0)}条, 消极{self.sentiment_results.get('negative', 0)}条
        
        【消息样本】
        {sample_text}
        
        请从以下角度分析：
        1. 沟通模式特点
        2. 情感表达方式
        3. 关系质量评估
        4. 改善建议
        
        请用简洁专业的语言，突出关键发现。
        """
        
        try:
            print("🔄 正在调用DeepSeek API...")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是一个专业的社交关系分析专家，擅长从聊天记录中洞察人际关系模式。"},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1500,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_analysis = result['choices'][0]['message']['content']
                print("✅ AI分析完成\n")
                print(ai_analysis)
                
                # 保存报告
                with open('analysis_results/ai_report.txt', 'w', encoding='utf-8') as f:
                    f.write(ai_analysis)
                print(f"\n📄 报告已保存: analysis_results/ai_report.txt")
                
                return ai_analysis
            else:
                print(f"❌ API调用失败: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ AI分析出错: {e}")
            return None
    
    def chat_simulator(self):
        """终端模拟聊天交互"""
        print("\n" + "="*60)
        print("💬 聊天模拟器（输入'退出'结束）")
        print("="*60)
        
        if not self.api_key or self.api_key == "你的API密钥":
            print("⚠️ 未配置API密钥，使用规则回复模式")
            use_api = False
        else:
            use_api = True
        
        # 学习聊天风格
        print("\n📚 正在学习聊天风格...")
        chat_samples = self.df['content'].sample(min(100, len(self.df))).tolist()
        
        print("✅ 准备就绪！开始对话：\n")
        
        while True:
            user_input = input("你: ").strip()
            
            if user_input in ['退出', 'quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if not user_input:
                continue
            
            # 生成回复
            if use_api:
                reply = self._get_api_reply(user_input, chat_samples[:10])
            else:
                reply = self._get_rule_reply(user_input)
            
            print(f"AI: {reply}\n")
    
    def _get_api_reply(self, message, samples):
        """使用API生成回复"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            sample_text = '\n'.join(samples[:5])
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": f"请模仿以下聊天风格回复:\n{sample_text}"},
                    {"role": "user", "content": message}
                ],
                "max_tokens": 100,
                "temperature": 0.8
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return self._get_rule_reply(message)
                
        except:
            return self._get_rule_reply(message)
    
    def _get_rule_reply(self, message):
        """基于规则的回复"""
        message = message.lower()
        
        if any(word in message for word in ['你好', 'hi', 'hello', '在吗']):
            return "在的！有什么可以帮你的吗？"
        elif any(word in message for word in ['谢谢', '感谢', 'thanks']):
            return "不客气！很高兴能帮到你😊"
        elif '?' in message or '吗' in message or '呢' in message:
            return "这个问题很有意思，让我想想..."
        elif any(word in message for word in ['哈哈', '😄', '😊']):
            return "哈哈，你也很有趣！"
        else:
            # 从历史消息中随机选择一条
            if len(self.df) > 0:
                sample = self.df['content'].sample(1).iloc[0]
                if len(str(sample)) < 50:
                    return str(sample)
            return "嗯嗯，我理解你的意思"
    
    def run_full_analysis(self, data_file):
        """运行完整分析流程"""
        print("="*60)
        print("🚀 微信聊天记录综合分析系统")
        print("="*60)
        
        # 1. 加载数据
        if data_file.endswith('.json'):
            if not self.load_json_data(data_file):
                return False
        else:
            print("❌ 暂不支持该格式，请使用JSON文件")
            return False
        
        # 2. 基础分析
        self.basic_analysis()
        
        # 3. 情感分析
        self.simple_sentiment_analysis()
        
        # 4. 词频分析
        self.word_frequency_analysis()
        
        # 5. 聚类分析
        self.topic_clustering()
        
        # 6. 生成图表
        self.create_visualizations()
        
        # 7. AI报告（可选）
        print("\n" + "="*60)
        choice = input("是否生成AI深度报告？(y/n，需要API密钥): ").strip().lower()
        if choice == 'y':
            self.generate_ai_report()
        
        # 8. 交互模式（可选）
        print("\n" + "="*60)
        choice = input("是否进入聊天模拟器？(y/n): ").strip().lower()
        if choice == 'y':
            self.chat_simulator()
        
        print("\n" + "="*60)
        print("✅ 分析完成！结果已保存到 analysis_results/ 目录")
        print("="*60)
        print("\n📊 生成的文件:")
        print("   🖼️ 01_daily_trend.png - 每日趋势图")
        print("   🖼️ 02_sender_distribution.png - 发送者分布")
        print("   🖼️ 03_hourly_activity.png - 活跃度分析")
        print("   🖼️ 04_sentiment_analysis.png - 情感分析")
        print("   🖼️ 05_wordcloud.png - 词云图")
        print("   🖼️ 06_topic_clustering.png - 主题聚类")
        if os.path.exists('analysis_results/ai_report.txt'):
            print("   📄 ai_report.txt - AI分析报告")
        print("="*60)
        
        return True


def main():
    """主函数"""
    # 读取配置
    try:
        from config import DEEPSEEK_API_KEY
        api_key = DEEPSEEK_API_KEY
    except:
        api_key = None
    
    # 查找数据文件
    data_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    if not data_files:
        print("❌ 未找到JSON数据文件")
        return
    
    print("📂 找到以下数据文件:")
    for i, f in enumerate(data_files, 1):
        print(f"   {i}. {f}")
    
    if len(data_files) == 1:
        selected_file = data_files[0]
        print(f"\n自动选择: {selected_file}")
    else:
        try:
            idx = int(input("\n请选择文件序号: ")) - 1
            selected_file = data_files[idx]
        except:
            print("❌ 选择无效")
            return
    
    # 创建分析器
    analyzer = WeChatAnalyzer(deepseek_api_key=api_key)
    
    # 运行分析
    analyzer.run_full_analysis(selected_file)


if __name__ == "__main__":
    main()
