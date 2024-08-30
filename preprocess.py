import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# 下载所需的nltk数据
# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_text(text):
    # 转换为小写
    text = text.lower()

    # 替换电子邮件地址
    text = re.sub(r'\S+@\S+', 'emailaddress', text)

    # 替换网址
    text = re.sub(r'http\S+|www\S+', 'webaddress', text)

    # 替换货币符号
    text = re.sub(r'£|\$', 'money-symbol', text)

    # 替换10位电话号码
    text = re.sub(r'\b\d{10}\b', 'phone-number', text)

    # 替换普通数字
    text = re.sub(r'\d+', 'number', text)

    # 移除标点符号
    text = re.sub(r'[^\w\s]', ' ', text)

    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()

    # 移除非英文字符
    text = re.sub(r'[^a-z\s]', '', text)

    # 分词
    words = word_tokenize(text)

    # 移除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # 应用Snowball Stemmer
    snowball = nltk.SnowballStemmer("english")
    words = [snowball.stem(word) for word in words]

    # 将单词列表重新连接成字符串
    processed_text = ' '.join(words)

    return processed_text

# 读取CSV文件，指定编码为ISO-8859-1，并删除Unnamed列
df = pd.read_csv('./ExAIS_SMS_SPAM_DATA.csv', encoding='ISO-8859-1')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# 将缺失值填充为空字符串
# df['Message'] = df['Message'].fillna('')

# 预处理message列并存储到ProcessedMessage列中
df['ProcessedMessage'] = df['Message'].apply(preprocess_text)

# 将Label列中的ham替换为0，spam替换为1
# df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})

# 保存到新的CSV文件，防止多余的索引列
df.to_csv('ExAIS_SMS_SPAM_DATA2.csv', index=False)

print("Processing complete. Saved to 'output.csv'")
