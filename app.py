import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from scipy.stats import ttest_1samp, ttest_ind, mannwhitneyu, kstest, binomtest as sm_binom_test
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, classification_report
from sklearn.cluster import KMeans
import warnings
import io
from openai import OpenAI

# 全局配置：解决中文显示+参数兼容问题（关键修复）
warnings.filterwarnings('ignore')
# 中文字体适配（兼容云环境无SimHei，优先使用系统自带字体）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'WenQuanYi Zen Hei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="科研数据分析平台", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# ================= 核心工具函数（无修改，确保功能正常）=================
def load_and_clean_data(file):
    """加载并清洗数据（处理缺失值/格式转换）"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8-sig')
        else:  # xlsx
            df = pd.read_excel(file)
        
        # 数据清洗：处理空列名/重复行
        df.columns = [col.strip() if isinstance(col, str) else f'col_{i}' for i, col in enumerate(df.columns)]
        df = df.drop_duplicates()
        df = df.replace(['NA', 'na', 'NULL', 'null'], np.nan)
        return df
    except Exception as e:
        st.error(f"文件加载失败：{str(e)[:50]}")
        return None

def identify_variable_types(df):
    """识别变量类型（数值型/分类型/二分类/时间型）"""
    var_types = {
        'numeric': [],      # 数值型（int/float）
        'categorical': [],  # 分类型（object/category，去重后≤20个值）
        'binary_categorical': [],  # 二分类（分类型中去重后=2个值）
        'datetime': []      # 时间型
    }
    
    for col in df.columns:
        # 时间型变量识别
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            var_types['datetime'].append(col)
            continue
        # 数值型变量识别
        if pd.api.types.is_numeric_dtype(df[col]):
            var_types['numeric'].append(col)
            continue
        # 分类型变量识别（去重后值数量≤20）
        unique_vals = df[col].dropna().nunique()
        if unique_vals <= 20:
            var_types['categorical'].append(col)
            # 二分类变量
            if unique_vals == 2:
                var_types['binary_categorical'].append(col)
    
    return var_types

def descriptive_analysis(df, numeric_cols):
    """描述性统计（含缺失值/偏度/峰度）"""
    desc_df = df[numeric_cols].describe().T
    # 添加缺失值统计
    desc_df['缺失值数量'] = df[numeric_cols].isnull().sum().values
    desc_df['缺失率(%)'] = (df[numeric_cols].isnull().sum() / len(df) * 100).round(2)
    # 添加偏度和峰度
    desc_df['偏度'] = df[numeric_cols].skew().round(3)
    desc_df['峰度'] = df[numeric_cols].kurt().round(3)
    return desc_df

def frequency_analysis(df, categorical_cols):
    """分类型变量频数分析（频数/频率/占比）"""
    freq_dict = {}
    for col in categorical_cols:
        freq = df[col].value_counts(dropna=False).reset_index()
        freq.columns = [col, '频数']
        freq['频率'] = freq['频数'] / len(df)
        freq['占比(%)'] = (freq['频率'] * 100).round(2)
        freq_dict[col] = freq
    return freq_dict

def contingency_table_analysis(df, row_col, col_col):
    """列联表+卡方检验"""
    cont_table = pd.crosstab(df[row_col], df[col_col], margins=True)
    chi2, p_value, dof, expected = stats.chi2_contingency(cont_table.iloc[:-1, :-1])
    # 计算克莱姆V系数（衡量关联强度）
    n = len(df)
    min_dim = min(cont_table.shape[0]-1, cont_table.shape[1]-1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if n * min_dim > 0 else 0
    
    return {
        '联列表': cont_table,
        '卡方值': round(chi2, 3),
        'p值': round(p_value, 4),
        '自由度': dof,
        '克莱姆V系数': round(cramers_v, 3)
    }

def t_test_onesample(df, numeric_col, popmean):
    """单样本t检验"""
    data = df[numeric_col].dropna()
    t_stat, p_value = ttest_1samp(data, popmean)
    return {
        't值': t_stat.round(3),
        'p值': p_value.round(4),
        '均值': data.mean().round(2),
        '样本量': len(data)
    }

def t_test_independent(df, numeric_col, group_col):
    """两独立样本t检验（Welch's t-test，不假设方差齐性）"""
    groups = df[group_col].unique()
    if len(groups) != 2:
        return {'error': '分组变量必须为二分类'}
    
    group1 = df[df[group_col] == groups[0]][numeric_col].dropna()
    group2 = df[df[group_col] == groups[1]][numeric_col].dropna()
    t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
    
    return {
        't值': t_stat.round(3),
        'p值': p_value.round(4),
        f'{groups[0]}均值': group1.mean().round(2),
        f'{groups[1]}均值': group2.mean().round(2),
        f'{groups[0]}样本量': len(group1),
        f'{groups[1]}样本量': len(group2)
    }

def nonparametric_test(df, test_type, numeric_col, group_col=None):
    """非参数检验（K-S检验/Mann-Whitney U检验/二项分布检验）"""
    if test_type == '单样本K-S检验':
        data = df[numeric_col].dropna()
        ks_stat, p_value = kstest(data, 'norm', args=(data.mean(), data.std()))
        return {'KS统计量': ks_stat.round(3), 'p值': p_value.round(4)}
    
    elif test_type == '两独立样本Mann-Whitney U检验':
        groups = df[group_col].unique()
        if len(groups) != 2:
            return {'error': '分组变量必须为二分类'}
        group1 = df[df[group_col] == groups[0]][numeric_col].dropna()
        group2 = df[df[group_col] == groups[1]][numeric_col].dropna()
        u_stat, p_value = mannwhitneyu(group1, group2)
        return {
            'U值': u_stat.round(3),
            'p值': p_value.round(4),
            f'{groups[0]}样本量': len(group1),
            f'{groups[1]}样本量': len(group2)
        }
    
    elif test_type == '二项分布检验':
        data = df[numeric_col].dropna()
        # 假设1为成功，0为失败（兼容二分类数值变量）
        if not set(data.unique()).issubset({0, 1}):
            return {'error': '二项分布检验仅支持0/1编码的变量'}
        success = sum(data == 1)
        n = len(data)
        p_value = sm_binom_test(success, n, p=0.5)
        return {'成功次数': success, '总次数': n, 'p值': p_value.round(4)}
    
    return {'error': '无效检验类型'}

def anova_analysis(df, formula, anova_type='单因素'):
    """单因素方差分析+Tukey事后检验"""
    model = ols(formula, data=df).fit()
    anova_result = anova_lm(model, typ=2)  # Type II ANOVA
    # 提取因变量和因素变量
    target = formula.split('~')[0].strip()
    factor = formula.split('~')[1].strip().replace('C(', '').replace(')', '')
    # Tukey事后检验
    tukey = pairwise_tukeyhsd(df[target].dropna(), df[factor][df[target].notna()], alpha=0.05)
    
    return {
        '方差分析表': anova_result.round(4),
        '事后检验(Tukey)': tukey.summary()
    }

def correlation_analysis(df, cols, corr_type='pearson'):
    """相关分析（Pearson/Spearman，含p值矩阵）"""
    corr_df = df[cols].dropna()
    corr_matrix = corr_df.corr(method=corr_type).round(3)
    # 计算p值矩阵
    p_matrix = pd.DataFrame(
        np.ones_like(corr_matrix),
        index=corr_matrix.index,
        columns=corr_matrix.columns
    )
    
    for col1 in cols:
        for col2 in cols:
            if col1 != col2:
                if corr_type == 'pearson':
                    corr, p = stats.pearsonr(corr_df[col1], corr_df[col2])
                else:
                    corr, p = stats.spearmanr(corr_df[col1], corr_df[col2])
                p_matrix.loc[col1, col2] = round(p, 4)
    
    return {'相关矩阵': corr_matrix, 'p值矩阵': p_matrix}

def regression_analysis(df, target, features, reg_type):
    """回归分析（线性回归/二分类Logistic回归）"""
    # 处理缺失值（按行删除）
    df_clean = df[[target] + features].dropna()
    X = df_clean[features]
    y = df_clean[target]
    
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if reg_type == '线性回归':
        model = LinearRegression().fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        
        return {
            'R²': r2.round(3),
            '系数表': pd.DataFrame({
                '特征': features,
                '标准化系数': model.coef_.round(3),
                '截距': [model.intercept_.round(3)] * len(features)
            })
        }
    
    elif reg_type == '二分类Logistic回归':
        # 标签编码（确保二分类）
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        if len(np.unique(y_encoded)) != 2:
            return {'error': '因变量必须为二分类'}
        
        model = LogisticRegression(max_iter=1000, solver='liblinear').fit(X_scaled, y_encoded)
        y_pred = model.predict(X_scaled)
        report = classification_report(y_encoded, y_pred, output_dict=True)
        
        return {
            '分类报告': pd.DataFrame(report).round(3),
            '系数表': pd.DataFrame({
                '特征': features,
                '标准化系数': model.coef_[0].round(3),
                '截距': [model.intercept_[0].round(3)] * len(features)
            })
        }
    
    return {'error': '无效回归类型'}

def cluster_analysis(df, cols, n_clusters=3):
    """K-Means聚类分析"""
    df_clean = df[cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    df_cluster = df_clean.copy()
    df_cluster['聚类结果'] = kmeans.labels_
    
    # 聚类中心（反标准化回原始尺度）
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=cols
    ).round(2)
    
    return {'聚类结果': df_cluster, '聚类中心': centroids}

def plot_chart(df, plot_type, x_col, y_col=None, group_col=None):
    """自定义可视化（条形图/折线图/饼图/箱图）"""
    df_plot = df.dropna(subset=[x_col] + ([y_col] if y_col else []))
    
    if plot_type == '条形图':
        fig = px.bar(
            df_plot,
            x=x_col,
            y=y_col,
            color=group_col,
            barmode='group',
            title=f'{x_col} vs {y_col} 分组条形图',
            width=800,
            height=500
        )
    
    elif plot_type == '折线图':
        # 按x_col排序（支持时间型/数值型）
        if pd.api.types.is_datetime64_any_dtype(df_plot[x_col]):
            df_plot = df_plot.sort_values(x_col)
        elif pd.api.types.is_numeric_dtype(df_plot[x_col]):
            df_plot = df_plot.sort_values(x_col)
        
        fig = px.line(
            df_plot,
            x=x_col,
            y=y_col,
            color=group_col,
            title=f'{x_col} vs {y_col} 趋势折线图',
            width=800,
            height=500
        )
    
    elif plot_type == '饼图':
        # 饼图需聚合y_col（默认求和）
        pie_data = df_plot.groupby(x_col)[y_col].sum().reset_index()
        fig = px.pie(
            pie_data,
            names=x_col,
            values=y_col,
            title=f'{x_col} 占比饼图（{y_col}求和）',
            hole=0.2,
            width=800,
            height=500
        )
    
    elif plot_type == '箱图':
        fig = px.box(
            df_plot,
            x=x_col,
            y=y_col,
            color=group_col,
            title=f'{x_col} 分组下 {y_col} 箱图',
            width=800,
            height=500
        )
    
    fig.update_layout(
        font=dict(size=12),
        xaxis_title=x_col,
        yaxis_title=y_col if y_col else ''
    )
    return fig

def call_deepseek_api(prompt):
    """调用DeepSeek API生成分析报告（流式输出）"""
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return iter(["⚠️ 请先在Streamlit Cloud设置中配置DEEPSEEK_API_KEY（获取地址：https://platform.deepseek.com/）"])
    
    try:
        client = OpenAI(
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1"
        )
        
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.2  # 低温度确保结果客观
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    except Exception as e:
        yield f"❌ AI调用失败：{str(e)[:100]}"

# ================= 页面主逻辑（关键修复：删除所有st.selectbox的horizontal参数）=================
st.title("📊 科研数据分析平台")
st.divider()

# 侧边栏：数据上传与合并（重点检查selectbox，无horizontal参数）
with st.sidebar:
    st.markdown("## 📥 数据上传")
    uploaded_files = st.file_uploader(
        "上传文件（CSV/Excel，支持多文件合并）",
        type=["xlsx", "csv"],
        accept_multiple_files=True
    )
    
    df = None
    var_types = None
    if uploaded_files:
        # 选择待分析文件（无horizontal）
        selected_file_names = st.multiselect(
            "选择分析文件",
            [f.name for f in uploaded_files],
            default=[uploaded_files[0].name]
        )
        selected_files = [f for f in uploaded_files if f.name in selected_file_names]
        
        # 加载文件到字典
        df_dict = {}
        for file in selected_files:
            df_temp = load_and_clean_data(file)
            if df_temp is not None:
                df_dict[file.name] = df_temp
                st.success(f"✅ {file.name} 上传成功（{len(df_temp)}行×{len(df_temp.columns)}列）")
        
        # 多文件合并逻辑（所有selectbox均无horizontal）
        if len(df_dict) >= 2:
            st.markdown("### 🔗 多文件合并")
            base_file = st.selectbox("基础文件", list(df_dict.keys()), key="merge_base_file")
            df = df_dict[base_file]
            
            for other_file in [f for f in df_dict.keys() if f != base_file]:
                df_other = df_dict[other_file]
                common_cols = [col for col in df.columns if col in df_other.columns]
                
                # 选择关联字段（无horizontal）
                base_key = st.selectbox(
                    f"基础文件（{base_file}）关联字段",
                    common_cols if common_cols else df.columns,
                    key=f"merge_base_key_{other_file}"
                )
                join_key = st.selectbox(
                    f"待合并文件（{other_file}）关联字段",
                    common_cols if common_cols else df_other.columns,
                    key=f"merge_join_key_{other_file}"
                )
                join_type = st.selectbox(
                    f"合并方式（{other_file}）",
                    ['左连接', '右连接', '内连接', '外连接'],
                    key=f"merge_type_{other_file}"
                )
                join_map = {'左连接': 'left', '右连接': 'right', '内连接': 'inner', '外连接': 'outer'}
                
                if st.button(f"🔄 合并{other_file}", key=f"btn_merge_{other_file}"):
                    df = pd.merge(
                        df,
                        df_other,
                        left_on=base_key,
                        right_on=join_key,
                        how=join_map[join_type],
                        suffixes=("", f"_{other_file.split('.')[0]}")
                    )
                    st.success(f"✅ 合并后：{len(df)}行×{len(df.columns)}列")
        else:
            # 单文件直接加载
            df = df_dict[list(df_dict.keys())[0]] if df_dict else None
        
        # 数据概况展示（修复报错位置附近代码，无selectbox）
        if df is not None:
            var_types = identify_variable_types(df)
            st.markdown("## 📋 数据概况")
            st.info(f"📏 规模：{len(df)}行 × {len(df.columns)}列")
            st.info(f"🔢 数值型变量：{len(var_types['numeric'])}个 → {', '.join(var_types['numeric'])}")
            st.info(f"📦 分类型变量：{len(var_types['categorical'])}个 → {', '.join(var_types['categorical'])}")
            st.info(f"⚖️ 二分类变量：{len(var_types['binary_categorical'])}个 → {', '.join(var_types['binary_categorical'])}")
            st.info(f"📅 时间型变量：{len(var_types['datetime'])}个 → {', '.join(var_types['datetime'])}")

# 核心分析标签页（数据加载成功后显示）
if df is not None and var_types is not None:
    # 构造数据概况文本（给AI用）
    data_overview = f"""
【数据基础概况】
1. 样本规模：{len(df)}行 × {len(df.columns)}列，整体缺失率：{(df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.2f}%
2. 数值变量：{', '.join(var_types['numeric']) if var_types['numeric'] else '无'}
3. 分类变量：{', '.join(var_types['categorical']) if var_types['categorical'] else '无'}
4. 二分类变量：{', '.join(var_types['binary_categorical']) if var_types['binary_categorical'] else '无'}
5. 时间变量：{', '.join(var_types['datetime']) if var_types['datetime'] else '无'}
"""

    # 新建标签页
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "数据处理", "基本统计", "均值检验", "方差分析", "相关分析", "回归分析", "可视化", "🤖 AI分析"
    ])

    # 标签1：数据处理（所有selectbox无horizontal）
    with tab1:
        st.subheader("⚙️ 数据预处理")
        
        # 1. 数据排序
        with st.expander("🔽 数据排序", expanded=True):
            sort_col = st.selectbox("排序字段", df.columns, key='sort_col')
            # 注意：st.radio支持horizontal，st.selectbox不支持（此处是radio，保留horizontal）
            sort_asc = st.radio("排序方式", ['升序', '降序'], key='sort_asc', horizontal=True)
            if st.button("执行排序", key='btn_sort'):
                df_sorted = df.sort_values(by=sort_col, ascending=(sort_asc == '升序'))
                st.dataframe(df_sorted.head(15), width='stretch')
        
        # 2. 数据筛选（selectbox无horizontal）
        with st.expander("🔍 数据筛选", expanded=True):
            filter_col = st.selectbox("筛选字段", df.columns, key='filter_col')
            filter_op = st.selectbox("运算符", ['>', '<', '>=', '<=', '==', '!='], key='filter_op')
            filter_val = st.text_input("筛选值（数值/文本）", placeholder="例：100 / 男", key='filter_val')
            
            if st.button("执行筛选", key='btn_filter'):
                try:
                    # 数值型字段转换
                    if df[filter_col].dtype in [np.int64, np.float64]:
                        filter_val = float(filter_val)
                    # 执行筛选
                    df_filtered = df.query(f"`{filter_col}` {filter_op} @filter_val")
                    st.success(f"✅ 筛选后：{len(df_filtered)}行数据")
                    st.dataframe(df_filtered.head(15), width='stretch')
                except Exception as e:
                    st.error(f"❌ 筛选错误：{str(e)[:60]}（请检查值类型是否匹配）")
        
        # 3. 分类汇总（selectbox无horizontal）
        with st.expander("📊 分类汇总", expanded=True):
            group_col = st.selectbox(
                "分组字段（分类型）",
                var_types['categorical'],
                key='group_col',
                disabled=not var_types['categorical']
            )
            agg_col = st.selectbox(
                "汇总字段（数值型）",
                var_types['numeric'],
                key='agg_col',
                disabled=not var_types['numeric']
            )
            agg_func = st.selectbox(
                "汇总方式",
                ['均值', '求和', '计数', '最大值', '最小值'],
                key='agg_func'
            )
            agg_map = {'均值': 'mean', '求和': 'sum', '计数': 'count', '最大值': 'max', '最小值': 'min'}
            
            if st.button("执行汇总", key='btn_agg', disabled=not (group_col and agg_col)):
                df_agg = df.groupby(group_col)[agg_col].agg(agg_map[agg_func]).round(2).reset_index()
                st.dataframe(df_agg, width='stretch')
                # 生成汇总图表（plotly_chart添加唯一key）
                fig_agg = px.bar(
                    df_agg,
                    x=group_col,
                    y=agg_col,
                    title=f"{group_col}分组下{agg_col}的{agg_func}分布",
                    text_auto=True
                )
                st.plotly_chart(fig_agg, width='stretch', key='plotly_agg_unique')

    # 标签2：基本统计（selectbox无horizontal）
    with tab2:
        st.subheader("📈 基本统计分析")
        
        # 1. 分类型变量频数分析
        with st.expander("📦 分类变量频数分析", expanded=True):
            freq_cols = st.multiselect(
                "选择分类型变量",
                var_types['categorical'],
                key='freq_cols',
                disabled=not var_types['categorical']
            )
            if freq_cols and st.button("执行频数分析", key='btn_freq'):
                freq_dict = frequency_analysis(df, freq_cols)
                for col in freq_cols:
                    st.subheader(f"🔍 {col} 频数分布")
                    st.dataframe(freq_dict[col], width='stretch')
                    # 频数图表（plotly_chart添加唯一key）
                    fig_freq = px.bar(
                        freq_dict[col],
                        x=col,
                        y='频数',
                        color=col,
                        title=f"{col} 频数分布",
                        text_auto=True
                    )
                    st.plotly_chart(fig_freq, width='stretch', key=f'plotly_freq_{col}')
        
        # 2. 数值型变量描述统计
        with st.expander("🔢 数值变量描述统计", expanded=True):
            desc_cols = st.multiselect(
                "选择数值型变量",
                var_types['numeric'],
                key='desc_cols',
                disabled=not var_types['numeric']
            )
            if desc_cols and st.button("执行描述统计", key='btn_desc'):
                desc_df = descriptive_analysis(df, desc_cols)
                st.dataframe(desc_df, width='stretch')
        
        # 3. 列联表+卡方检验（selectbox无horizontal）
        with st.expander("⚖️ 列联表与卡方检验", expanded=True):
            if len(var_types['categorical']) >= 2:
                row_col = st.selectbox("行变量", var_types['categorical'], key='row_col')
                col_col = st.selectbox("列变量", [c for c in var_types['categorical'] if c != row_col], key='col_col')
                
                if st.button("执行卡方检验", key='btn_chi2'):
                    chi2_res = contingency_table_analysis(df, row_col, col_col)
                    st.subheader(f"📊 {row_col} × {col_col} 列联表")
                    st.dataframe(chi2_res['联列表'], width='stretch')
                    st.subheader("📈 卡方检验结果")
                    st.info(f"卡方值：{chi2_res['卡方值']} | p值：{chi2_res['p值']} | 自由度：{chi2_res['自由度']}")
                    st.info(f"克莱姆V系数：{chi2_res['克莱姆V系数']}（0-1，越大相关性越强）")
                    
                    if chi2_res['p值'] < 0.05:
                        st.success("✅ p<0.05，两分类变量存在显著相关性！")
                    else:
                        st.warning("⚠️ p≥0.05，两分类变量无显著相关性")
            else:
                st.warning("⚠️ 需至少2个分类型变量才能执行卡方检验")

    # 标签3：均值检验（selectbox无horizontal）
    with tab3:
        st.subheader("⚖️ 均值检验")
        
        # 1. 单样本t检验
        with st.expander("📊 单样本t检验", expanded=True):
            onesamp_col = st.selectbox(
                "检验变量（数值型）",
                var_types['numeric'],
                key='onesamp_col',
                disabled=not var_types['numeric']
            )
            popmean = st.number_input("总体均值（检验基准）", value=0.0, step=0.1, key='popmean')
            
            if st.button("执行单样本t检验", key='btn_onesamp', disabled=not onesamp_col):
                onesamp_res = t_test_onesample(df, onesamp_col, popmean)
                st.subheader(f"🔍 {onesamp_col} 单样本t检验结果")
                st.info(f"样本均值：{onesamp_res['均值']} | 样本量：{onesamp_res['样本量']}")
                st.info(f"t值：{onesamp_res['t值']} | p值：{onesamp_res['p值']}")
                
                if onesamp_res['p值'] < 0.05:
                    st.success("✅ p<0.05，样本均值与总体均值存在显著差异！")
                else:
                    st.warning("⚠️ p≥0.05，样本均值与总体均值无显著差异")
        
        # 2. 两独立样本t检验
        with st.expander("📊 两独立样本t检验", expanded=True):
            ind_col = st.selectbox(
                "检验变量（数值型）",
                var_types['numeric'],
                key='ind_col',
                disabled=not var_types['numeric']
            )
            ind_group = st.selectbox(
                "分组变量（二分类）",
                var_types['binary_categorical'],
                key='ind_group',
                disabled=not var_types['binary_categorical']
            )
            
            if st.button("执行两样本t检验", key='btn_ind', disabled=not (ind_col and ind_group)):
                ind_res = t_test_independent(df, ind_col, ind_group)
                if 'error' in ind_res:
                    st.error(f"❌ {ind_res['error']}")
                else:
                    st.subheader(f"🔍 {ind_col} 按{ind_group}分组t检验结果")
                    st.info(f"t值：{ind_res['t值']} | p值：{ind_res['p值']}")
                    for k in ind_res.keys():
                        if '均值' in k or '样本量' in k:
                            st.info(f"{k}：{ind_res[k]}")
                    
                    if ind_res['p值'] < 0.05:
                        st.success("✅ p<0.05，两组均值存在显著差异！")
                    else:
                        st.warning("⚠️ p≥0.05，两组均值无显著差异")
        
        # 3. 非参数检验（selectbox无horizontal）
        with st.expander("📊 非参数检验", expanded=True):
            test_type = st.selectbox(
                "检验类型",
                ['单样本K-S检验（正态性）', '两独立样本Mann-Whitney U检验', '二项分布检验'],
                key='test_type'
            )
            np_col = st.selectbox(
                "检验变量",
                var_types['numeric'] + var_types['binary_categorical'],
                key='np_col'
            )
            # 仅U检验需要分组变量
            np_group = st.selectbox(
                "分组变量（仅U检验需选）",
                [None] + var_types['binary_categorical'],
                key='np_group',
                disabled=test_type != '两独立样本Mann-Whitney U检验'
            )
            
            if st.button("执行非参数检验", key='btn_np'):
                # 统一检验类型名称
                test_type_map = {
                    '单样本K-S检验（正态性）': '单样本K-S检验',
                    '两独立样本Mann-Whitney U检验': '两独立样本Mann-Whitney U检验',
                    '二项分布检验': '二项分布检验'
                }
                np_res = nonparametric_test(df, test_type_map[test_type], np_col, np_group)
                
                if 'error' in np_res:
                    st.error(f"❌ {np_res['error']}")
                else:
                    st.subheader(f"🔍 {test_type} 结果")
                    for k, v in np_res.items():
                        st.info(f"{k}：{v}")
                    
                    if 'p值' in np_res:
                        if np_res['p值'] < 0.05:
                            if test_type == '单样本K-S检验（正态性）':
                                st.success("✅ p<0.05，数据不符合正态分布！")
                            else:
                                st.success("✅ p<0.05，检验结果存在显著差异！")
                        else:
                            if test_type == '单样本K-S检验（正态性）':
                                st.warning("⚠️ p≥0.05，数据符合正态分布！")
                            else:
                                st.warning("⚠️ p≥0.05，检验结果无显著差异！")

    # 标签4：方差分析（selectbox无horizontal）
    with tab4:
        st.subheader("📊 单因素方差分析（ANOVA）")
        if var_types['numeric'] and var_types['categorical']:
            anova_target = st.selectbox("因变量（数值型）", var_types['numeric'], key='anova_target')
            anova_factor = st.selectbox("因素变量（分类型）", var_types['categorical'], key='anova_factor')
            # 构造公式（C()表示分类变量）
            formula = f"{anova_target} ~ C({anova_factor})"
            
            if st.button("执行方差分析+Tukey事后检验", key='btn_anova'):
                anova_res = anova_analysis(df, formula)
                st.subheader("📈 方差分析表")
                st.dataframe(anova_res['方差分析表'], width='stretch')
                
                # 提取p值判断显著性
                anova_p = anova_res['方差分析表']['PR(>F)'].iloc[0]
                if anova_p < 0.05:
                    st.success("✅ p<0.05，各分组均值存在显著整体差异（需看事后检验）")
                    st.subheader("📋 Tukey HSD 事后检验（多重比较）")
                    st.text(anova_res['事后检验(Tukey)'])
                else:
                    st.warning("⚠️ p≥0.05，各分组均值无显著整体差异（无需事后检验）")
        else:
            st.warning("⚠️ 需同时存在数值型因变量和分类型因素变量才能执行方差分析")

    # 标签5：相关分析（selectbox无horizontal，pyplot无key）
    with tab5:
        st.subheader("📈 变量相关性分析")
        if len(var_types['numeric']) >= 2:
            corr_type = st.selectbox(
                "相关系数类型",
                ['pearson（皮尔逊，适用于正态分布）', 'spearman（斯皮尔曼，非参数）'],
                key='corr_type'
            )
            corr_cols = st.multiselect(
                "选择数值型变量（至少2个）",
                var_types['numeric'],
                key='corr_cols',
                default=var_types['numeric'][:2]
            )
            
            if len(corr_cols) >= 2 and st.button("执行相关分析", key='btn_corr'):
                corr_res = correlation_analysis(df, corr_cols, corr_type.split("（")[0].lower())
                st.subheader(f"📊 {corr_type.split('（')[0]} 相关矩阵")
                st.dataframe(corr_res['相关矩阵'], width='stretch')
                st.subheader(f"📊 相关分析p值矩阵（p<0.05为显著）")
                st.dataframe(corr_res['p值矩阵'], width='stretch')
                
                # 绘制相关热力图（pyplot无key参数，避免报错）
                st.subheader(f"📊 相关系数热力图")
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(corr_res['相关矩阵'], cmap='RdBu_r', vmin=-1, vmax=1)
                
                # 设置坐标轴
                ax.set_xticks(np.arange(len(corr_cols)))
                ax.set_yticks(np.arange(len(corr_cols)))
                ax.set_xticklabels(corr_cols, rotation=45, ha='right', fontsize=10)
                ax.set_yticklabels(corr_cols, fontsize=10)
                
                # 标注数值和显著性
                for i in range(len(corr_cols)):
                    for j in range(len(corr_cols)):
                        corr_val = corr_res['相关矩阵'].iloc[i, j]
                        p_val = corr_res['p值矩阵'].iloc[i, j]
                        # 显著性标记（**p<0.01, *p<0.05）
                        mark = '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                        text = ax.text(
                            j, i, f"{corr_val:.3f}{mark}",
                            ha="center", va="center", color="black", fontsize=9
                        )
                
                # 添加颜色条
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.set_label(f'{corr_type.split("（")[0]} 相关系数', rotation=270, labelpad=20)
                plt.title(f'{corr_type.split("（")[0]} 相关热力图（**p<0.01，*p<0.05）', fontsize=14)
                plt.tight_layout()
                st.pyplot(fig)  # 关键修复：删除key参数
        else:
            st.warning("⚠️ 需至少2个数值型变量才能执行相关分析")

    # 标签6：回归分析（selectbox无horizontal）
    with tab6:
        st.subheader("📈 回归分析")
        reg_type = st.selectbox(
            "回归类型",
            ['线性回归（数值因变量）', '二分类Logistic回归（二分类因变量）'],
            key='reg_type'
        )
        
        # 选择因变量
        if reg_type == '线性回归（数值因变量）':
            reg_target = st.selectbox(
                "因变量（数值型）",
                var_types['numeric'],
                key='reg_target',
                disabled=not var_types['numeric']
            )
        else:
            reg_target = st.selectbox(
                "因变量（二分类）",
                var_types['binary_categorical'],
                key='reg_target',
                disabled=not var_types['binary_categorical']
            )
        
        # 选择自变量（排除因变量）
        reg_features = st.multiselect(
            "自变量（数值型，至少1个）",
            [col for col in var_types['numeric'] if col != reg_target],
            key='reg_features',
            disabled=not var_types['numeric']
        )
        
        # 执行回归分析
        if st.button("执行回归分析", key='btn_reg', disabled=not (reg_target and reg_features)):
            reg_res = regression_analysis(
                df,
                reg_target,
                reg_features,
                reg_type.split("（")[0]
            )
            
            if 'error' in reg_res:
                st.error(f"❌ {reg_res['error']}")
            else:
                st.subheader(f"📊 {reg_type.split('（')[0]} 结果")
                # 线性回归显示R²，Logistic显示分类报告
                if reg_type == '线性回归（数值因变量）':
                    st.success(f"✅ 模型拟合度 R² = {reg_res['R²']}（越接近1拟合越好）")
                else:
                    st.dataframe(reg_res['分类报告'], width='stretch')
                
                # 显示系数表
                st.subheader("📋 模型系数表")
                st.dataframe(reg_res['系数表'], width='stretch')

    # 标签7：可视化（selectbox无horizontal，plotly_chart有唯一key）
    with tab7:
        st.subheader("🎨 自定义可视化")
        plot_type = st.selectbox(
            "选择图表类型",
            ['条形图', '折线图', '饼图', '箱图'],
            key='plot_type'
        )
        
        # 按图表类型选择变量
        if plot_type in ['条形图', '折线图', '箱图']:
            x_col = st.selectbox("X轴变量", df.columns, key='plot_x')
            y_col = st.selectbox("Y轴变量（数值型）", var_types['numeric'], key='plot_y')
            group_col = st.selectbox(
                "分组变量（可选）",
                [None] + var_types['categorical'],
                key='plot_group'
            )
        else:  # 饼图
            x_col = st.selectbox("饼图分组变量（分类型）", var_types['categorical'], key='plot_x_pie')
            y_col = st.selectbox("饼图数值变量（数值型）", var_types['numeric'], key='plot_y_pie')
            group_col = None
        
        # 生成图表
        if st.button("生成图表", key='btn_plot'):
            try:
                if plot_type in ['条形图', '折线图', '箱图']:
                    fig = plot_chart(df, plot_type, x_col, y_col, group_col)
                    # 唯一key：结合图表类型+变量名
                    st.plotly_chart(fig, width='stretch', key=f'plotly_custom_{plot_type}_{x_col}_{y_col}')
                else:
                    fig = plot_chart(df, plot_type, x_col, y_col)
                    st.plotly_chart(fig, width='stretch', key=f'plotly_custom_pie_{x_col}_{y_col}')
                
                # 图表下载
                st.download_button(
                    label="📥 下载图表（HTML格式）",
                    data=fig.to_html(),
                    file_name=f"{plot_type}_{x_col}_{y_col}.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"❌ 图表生成失败：{str(e)[:80]}（请检查变量选择）")

    # 标签8：AI分析（无selectbox，或selectbox无horizontal）
    with tab8:
        st.subheader("🤖 AI 智能分析（基于真实统计结果）")
        if "DEEPSEEK_API_KEY" not in st.secrets:
            st.warning("⚠️ 请先在Streamlit Cloud设置中配置：DEEPSEEK_API_KEY = '你的API密钥'")
            st.info("💡 密钥获取：https://platform.deepseek.com/")
        else:
            st.success("✅ API密钥已配置 | 分析结果基于真实数据，无编造内容")
            st.divider()

            # 1. AI自动生成科研报告
            with st.expander("📑 AI自动生成科研报告", expanded=True):
                if st.button("🚀 开始AI分析", key='btn_ai_auto'):
                    with st.spinner("🌀 正在计算统计结果+生成报告...（约10秒）"):
                        # 1. 计算核心统计结果
                        # 描述统计
                        desc_text = descriptive_analysis(df, var_types['numeric']).to_string() if var_types['numeric'] else "无数值变量"
                        # 相关分析（取前2个数值变量）
                        corr_text = "无足够数值变量"
                        if len(var_types['numeric']) >= 2:
                            corr_res = correlation_analysis(df, var_types['numeric'][:2], 'pearson')
                            corr_text = f"相关矩阵：\n{corr_res['相关矩阵'].to_string()}\n\np值矩阵：\n{corr_res['p值矩阵'].to_string()}"
                        # 频数分析（取第一个分类变量）
                        freq_text = "无分类变量"
                        if var_types['categorical']:
                            freq_res = frequency_analysis(df, [var_types['categorical'][0]])
                            freq_text = f"{var_types['categorical'][0]} 频数分布：\n{freq_res[var_types['categorical'][0]].to_string()}"
                        # t检验（若有二分类和数值变量）
                        ttest_text = "无符合条件的t检验数据"
                        if var_types['binary_categorical'] and var_types['numeric']:
                            ttest_res = t_test_independent(df, var_types['numeric'][0], var_types['binary_categorical'][0])
                            if 'error' not in ttest_res:
                                ttest_text = f"两独立样本t检验（{var_types['numeric'][0]} 按 {var_types['binary_categorical'][0]} 分组）：\n"
                                ttest_text += f"t值={ttest_res['t值']}, p值={ttest_res['p值']}, "
                                ttest_text += f"{list(ttest_res.keys())[2]}={ttest_res[list(ttest_res.keys())[2]]}, "
                                ttest_text += f"{list(ttest_res.keys())[3]}={ttest_res[list(ttest_res.keys())[3]]}"

                        # 2. 构造AI提示词
                        prompt = f"""
你是资深科研数据分析专家，基于以下真实统计结果生成标准化科研报告，严格按格式输出，不编造任何数据。

### 输出格式要求（必须严格遵循）
# 科研数据统计分析报告
## 一、数据基本特征
1. 样本规模：说明数据行数列数、缺失率、变量类型分布
2. 数值变量特征：基于描述统计，总结均值、标准差、缺失情况、偏度峰度
3. 分类变量特征：基于频数分析，总结主要类别及占比

## 二、核心统计分析结果
1. 变量相关性：基于相关分析结果，解读变量间相关程度及显著性
2. 组间差异：基于t检验结果，分析分组均值差异及显著性（无结果则写“无符合条件的t检验”）
3. 整体规律：综合上述分析，总结变量分布及关系规律

## 三、研究结论与建议
### （一）研究结论
分3-4点客观总结数据反映的核心规律，每点1句话，仅基于真实结果
### （二）研究建议
分2-3点给出针对性建议，贴合科研场景，可落地

### 真实统计结果
【数据概况】
{data_overview}

【描述统计结果】
{desc_text}

【相关分析结果】
{corr_text}

【频数分析结果】
{freq_text}

【t检验结果】
{ttest_text}
"""

                        # 3. 流式输出AI结果
                        st.subheader("📋 AI生成报告")
                        st.divider()
                        stream = call_deepseek_api(prompt)
                        st.write_stream(stream)

            # 2. AI针对性问答
            with st.expander("❓ AI统计问答", expanded=False):
                user_question = st.text_area(
                    "输入你的数据分析问题（示例：分析generation_mw和demand_mw的相关性；比较两组均值差异）",
                    height=100,
                    key='ai_question'
                )
                if st.button("💬 发送问题", key='btn_ai_qa') and user_question:
                    prompt = f"""
你是专业统计分析师，基于以下数据概况解答问题，输出分点清晰、语言专业。

### 输出格式
## 问题解答：{user_question}
1. 分析方法：说明需使用的统计方法及适用条件
2. 结果解读：基于数据概况给出针对性分析（无具体数据则说明限制）
3. 建议：给出后续分析建议

### 数据概况
{data_overview}

### 用户问题
{user_question}
"""
                    st.write_stream(call_deepseek_api(prompt))

            # 3. AI结果解读
            with st.expander("📈 AI统计结果解读", expanded=False):
                user_result = st.text_area(
                    "粘贴你的统计结果（示例：相关系数0.8，p=0.001；t=2.5，p=0.02）",
                    height=100,
                    key='ai_result'
                )
                if st.button("🔍 解读结果", key='btn_ai_interpret') and user_result:
                    prompt = f"""
你是统计专家，基于以下数据概况解读统计结果，重点说明显著性和实际意义。

### 输出格式
## 统计结果解读
1. 指标解读：逐点解释每个统计量的含义
2. 显著性判断：按p<0.05显著、p<0.01极显著判断
3. 实际意义：结合数据类型说明结果反映的规律
4. 综合结论：1-2句总结核心发现

### 数据概况
{data_overview}

### 用户提供的统计结果
{user_result}
"""
                    st.write_stream(call_deepseek_api(prompt))

# 无数据时显示引导
else:
    st.info("💡 请在左侧边栏上传CSV/Excel数据文件，支持多文件合并分析")
    st.markdown("#### 📌 平台核心功能")
    st.markdown("✅ 统计分析：描述统计、t检验、方差分析、相关/回归、卡方检验")
    st.markdown("✅ 可视化：条形图/折线图/饼图/箱图，支持下载")
    st.markdown("✅ AI分析：基于真实数据生成科研报告，支持问答解读")
    st.markdown("✅ 数据处理：排序、筛选、分类汇总、多文件合并")
    st.markdown("✅ 无需代码：纯可视化操作，结果一键导出")
