import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
import io
import re
import os
from datetime import datetime

# ---------------------- 1. 依赖导入与异常处理 ----------------------
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="科研数据分析平台（SPSS核心功能版）", page_icon="🔬📊", layout="wide", initial_sidebar_state="expanded")

# 分模块导入scipy，避免单个函数失败导致整体崩溃
SCIPY_CORE_IMPORTED = False
try:
    from scipy.stats import chi2_contingency, ttest_1samp, ttest_ind, ttest_rel
    from scipy.stats import ks_2samp, mannwhitneyu, kruskal, friedmanchisquare, wilcoxon
    SCIPY_CORE_IMPORTED = True
    # 用statsmodels替代binom_test
    from statsmodels.stats.proportion import binom_test as sm_binom_test
    binom_test = sm_binom_test
except ImportError as e:
    st.warning(f"部分统计函数导入失败：{str(e)}，基础功能仍可使用")
    binom_test = None

# 延迟导入其他依赖
STATSMODELS_IMPORTED = False
try:
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.formula.api import ols, glm
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    STATSMODELS_IMPORTED = True
except ImportError:
    st.warning("statsmodels导入失败，方差分析功能受限")

SKLEARN_IMPORTED = False
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import r2_score, classification_report
    SKLEARN_IMPORTED = True
except ImportError:
    st.warning("sklearn导入失败，聚类/回归功能受限")

FACTOR_ANALYZER_IMPORTED = False
try:
    from factor_analyzer import FactorAnalyzer
    FACTOR_ANALYZER_IMPORTED = True
except ImportError:
    st.warning("factor_analyzer导入失败，因子分析功能受限")

# ---------------------- 2. 核心工具函数 ----------------------
def load_and_clean_data(file):
    encodings = ['utf-8-sig', 'gbk', 'utf-8', 'gb2312']
    seps = [',', '\t', ';']
    try:
        file_content = file.read()
        file.seek(0)
        df = None
        if file.name.endswith(".csv"):
            for encoding in encodings:
                for sep in seps:
                    try:
                        if encoding == 'utf-16':
                            content = file_content.decode(encoding, errors='replace')
                            df = pd.read_csv(io.StringIO(content), sep=sep, on_bad_lines='skip')
                        else:
                            df = pd.read_csv(file, encoding=encoding, sep=sep, on_bad_lines='skip')
                        break
                    except:
                        continue
                if df is not None:
                    break
            if df is None:
                from csv import Sniffer
                sample = file_content[:4096].decode('utf-8-sig', errors='replace')
                delimiter = Sniffer().sniff(sample).delimiter
                df = pd.read_csv(file, encoding='utf-8-sig', sep=delimiter, on_bad_lines='skip')
        else:
            df = pd.read_excel(file, engine='openpyxl')
        df.columns = [re.sub(r'[^\w\s\u4e00-\u9fa5/]', '', str(col)).strip() for col in df.columns]
        df.columns = [col if col else f"col_{i}" for i, col in enumerate(df.columns)]
        return df
    except Exception as e:
        st.error(f"文件读取失败：{str(e)}")
        return None

def identify_variable_types(df):
    numeric_cols = []
    categorical_cols = []
    binary_categorical_cols = []
    datetime_cols = []
    for col in df.columns:
        if any(fmt in col.lower() for fmt in ['date', 'time', '2016', '2017', '2018']):
            try:
                df[col] = pd.to_datetime(df[col])
                datetime_cols.append(col)
                continue
            except:
                pass
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
            numeric_cols.append(col)
        except:
            categorical_cols.append(col)
            if df[col].nunique() == 2:
                binary_categorical_cols.append(col)
    return {'numeric': numeric_cols, 'categorical': categorical_cols, 'binary_categorical': binary_categorical_cols, 'datetime': datetime_cols}

# ---------------------- 3. 页面渲染逻辑（核心修复） ----------------------
# 先渲染标题和依赖状态，确保这部分快速完成
st.title("🔬 科研数据分析平台（SPSS核心功能版）")
st.divider()

# 环境依赖状态（放在最前面，快速渲染）
st.markdown("### 🛠️ 环境依赖状态")
status_col1, status_col2 = st.columns(2)
with status_col1:
    st.write(f"- scipy（统计核心）：{'✅ 核心函数已导入' if SCIPY_CORE_IMPORTED else '❌ 未导入'}")
    st.write(f"- statsmodels（方差分析）：{'✅ 已导入' if STATSMODELS_IMPORTED else '❌ 未导入'}")
with status_col2:
    st.write(f"- sklearn（聚类/回归）：{'✅ 已导入' if SKLEARN_IMPORTED else '❌ 未导入'}")
    st.write(f"- factor_analyzer（因子分析）：{'✅ 已导入' if FACTOR_ANALYZER_IMPORTED else '❌ 未导入'}")
st.divider()

# ---------------------- 4. 侧边栏（确保始终渲染） ----------------------
with st.sidebar:
    st.markdown("### 📥 数据上传")
    uploaded_files = st.file_uploader(
        "支持CSV/Excel（可传多个）",
        type=["xlsx", "csv"],
        accept_multiple_files=True,
        key="file_uploader"  # 固定key避免渲染异常
    )
    
    df = None
    if uploaded_files:
        st.markdown("### 📋 选择分析文件")
        selected_file_names = st.multiselect(
            "勾选文件",
            [f.name for f in uploaded_files],
            default=[uploaded_files[0].name] if uploaded_files else [],
            key="file_selector"
        )
        selected_files = [f for f in uploaded_files if f.name in selected_file_names]
        
        df_dict = {}
        for file in selected_files:
            df_temp = load_and_clean_data(file)
            if df_temp is not None:
                df_dict[file.name] = df_temp
                st.success(f"✅ {file.name} ({len(df_temp)}行×{len(df_temp.columns)}列)")
        
        if len(df_dict) >= 2:
            st.markdown("### 🔗 数据合并")
            base_file = st.selectbox("基础文件", list(df_dict.keys()), key="base_file")
            df = df_dict[base_file]
            for other_file in [f for f in df_dict.keys() if f != base_file]:
                df_other = df_dict[other_file]
                common_cols = [col for col in df.columns if col in df_other.columns]
                base_key = st.selectbox(f"基础文件关联字段", common_cols if common_cols else df.columns, key=f"base_key_{other_file}")
                join_key = st.selectbox(f"关联文件关联字段", common_cols if common_cols else df_other.columns, key=f"join_key_{other_file}")
                join_type = st.selectbox(f"合并方式", ['左连接', '右连接', '内连接', '外连接'], key=f"join_type_{other_file}")
                join_map = {'左连接':'left', '右连接':'right', '内连接':'inner', '外连接':'outer'}
                if st.button(f"合并{other_file}", key=f"merge_btn_{other_file}"):
                    df = pd.merge(df, df_other, left_on=base_key, right_on=join_key, how=join_map[join_type], suffixes=("", f"_{other_file.split('.')[0]}"))
                    st.success(f"✅ 合并后：{len(df)}行×{len(df.columns)}列")
        else:
            df = df_dict[list(df_dict.keys())[0]] if df_dict else None
        
        if df is not None:
            var_types = identify_variable_types(df)
            st.markdown("### 📊 数据概况")
            st.write(f"规模：{len(df)}行 × {len(df.columns)}列")
            st.write(f"数值型：{len(var_types['numeric'])}个")
            st.write(f"分类型：{len(var_types['categorical'])}个")

# ---------------------- 5. 主内容区（处理无数据情况） ----------------------
if df is not None:
    var_types = identify_variable_types(df)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("🔍 数据预览")
        st.dataframe(df.head(10), use_container_width=True, height=300)
    with col2:
        st.subheader("📋 变量类型")
        st.write(f"⏰ 时间型：{', '.join(var_types['datetime']) if var_types['datetime'] else '无'}")
        st.write(f"🔢 二分类：{', '.join(var_types['binary_categorical']) if var_types['binary_categorical'] else '无'}")
    
    # 后续分析标签页（省略，保持原有逻辑）
    tab1, tab2, tab3 = st.tabs(["数据处理", "基本统计", "可视化分析"])
    with tab1:
        st.markdown("#### 🔧 数据处理功能")
        st.info("请上传数据后使用数据排序、筛选等功能")
else:
    # 无数据时的明确提示，避免卡住
    st.info("💡 请在左侧边栏上传CSV/Excel文件，上传后即可使用所有分析功能")
    st.markdown("#### 🎯 功能预览")
    st.write("- 支持数据上传、多文件合并、数据清洗")
    st.write("- 包含频数分析、t检验、方差分析、回归分析等SPSS核心功能")
    st.write("- 提供可视化图表生成与报告导出")
