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

# 核心修改1：分模块导入scipy.stats，替换binom_test为statsmodels版本
SCIPY_IMPORTED = False
SCIPY_CORE_IMPORTED = False
try:
    from scipy.stats import chi2_contingency, ttest_1samp, ttest_ind, ttest_rel
    from scipy.stats import ks_2samp, mannwhitneyu, kruskal, friedmanchisquare, wilcoxon
    SCIPY_CORE_IMPORTED = True
    # 处理binom_test导入失败：改用statsmodels中的版本
    try:
        from statsmodels.stats.proportion import binom_test as sm_binom_test
        binom_test = sm_binom_test
    except ImportError:
        binom_test = None
    SCIPY_IMPORTED = True
except ImportError as e:
    st.warning(f"部分统计函数导入失败：{str(e)}，基础功能仍可使用")

# 延迟导入其他依赖
try:
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.formula.api import ols, glm
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.correlation_tools import corr_nearest
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_IMPORTED = True
except ImportError:
    st.warning("statsmodels导入失败，方差分析/回归相关功能受限")
    STATSMODELS_IMPORTED = False

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import r2_score, classification_report
    SKLEARN_IMPORTED = True
except ImportError:
    st.warning("sklearn导入失败，聚类/回归相关功能受限")
    SKLEARN_IMPORTED = False

try:
    from factor_analyzer import FactorAnalyzer
    FACTOR_ANALYZER_IMPORTED = True
except ImportError:
    st.warning("factor_analyzer导入失败，因子分析功能不可用（可在requirements.txt中添加factor_analyzer>=0.5.1解决）")
    FACTOR_ANALYZER_IMPORTED = False

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="科研数据分析平台（SPSS核心功能版）", page_icon="🔬📊", layout="wide", initial_sidebar_state="expanded")

# 后续函数和页面逻辑保持不变，仅在使用binom_test时添加检查
def nonparametric_test(df, test_type, numeric_col, group_col=None):
    if not SCIPY_CORE_IMPORTED:
        return {"error": "scipy核心函数未导入，无法执行非参数检验"}
    if test_type == '单样本K-S检验':
        data = df[numeric_col].dropna()
        ks_stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        return {'KS统计量': ks_stat.round(3), 'p值': p_value.round(4)}
    elif test_type == '两独立样本Mann-Whitney U检验':
        groups = df[group_col].unique()
        if len(groups) != 2:
            return {'error': '分组变量必须为二分类'}
        group1 = df[df[group_col] == groups[0]][numeric_col].dropna()
        group2 = df[df[group_col] == groups[1]][numeric_col].dropna()
        u_stat, p_value = mannwhitneyu(group1, group2)
        return {'U值': u_stat.round(3), 'p值': p_value.round(4)}
    elif test_type == '多独立样本Kruskal-Wallis H检验':
        groups_data = [df[df[group_col] == g][numeric_col].dropna() for g in df[group_col].unique()]
        h_stat, p_value = kruskal(*groups_data)
        return {'H值': h_stat.round(3), 'p值': p_value.round(4)}
    elif test_type == '两配对样本Wilcoxon检验':
        paired_data = df[[numeric_col, group_col]].dropna()
        w_stat, p_value = wilcoxon(paired_data[numeric_col], paired_data[group_col])
        return {'W值': w_stat.round(3), 'p值': p_value.round(4)}
    elif test_type == '多配对样本Friedman检验':
        cols = [col for col in df.columns if col in [numeric_col, group_col]] if group_col else df.select_dtypes(include=np.number).columns[:3]
        friedman_stat, p_value = friedmanchisquare(*[df[col].dropna() for col in cols])
        return {'Friedman统计量': friedman_stat.round(3), 'p值': p_value.round(4)}
    elif test_type == '二项分布检验':
        if binom_test is None:
            return {"error": "binom_test函数不可用，请确保statsmodels已安装"}
        data = df[numeric_col].dropna()
        success = sum(data == 1)
        n = len(data)
        p_value = binom_test(success, n, prop=0.5)
        return {'成功次数': success, '总次数': n, 'p值': p_value.round(4)}
    return {'error': '无效检验类型'}

# 其他函数和页面逻辑与之前一致，仅修改环境依赖状态显示
st.title("🔬 科研数据分析平台（SPSS核心功能版）")
st.divider()
st.markdown("### 环境依赖状态")
st.write(f"- scipy（统计核心）：{'✅ 核心函数已导入' if SCIPY_CORE_IMPORTED else '❌ 未导入'}")
st.write(f"- statsmodels（方差分析）：{'✅ 已导入' if STATSMODELS_IMPORTED else '❌ 未导入'}")
st.write(f"- sklearn（聚类/回归）：{'✅ 已导入' if SKLEARN_IMPORTED else '❌ 未导入'}")
st.write(f"- factor_analyzer（因子分析）：{'✅ 已导入' if FACTOR_ANALYZER_IMPORTED else '❌ 未导入'}")
st.divider()

# 剩余页面逻辑保持不变...
