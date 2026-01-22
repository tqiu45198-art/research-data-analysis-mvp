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

# 基础配置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="科研数据分析平台", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# 核心依赖导入（仅保留必要的，无冗余提示）
try:
    from scipy.stats import chi2_contingency, ttest_1samp, ttest_ind, ttest_rel, ks_2samp, mannwhitneyu, kruskal, friedmanchisquare, wilcoxon
    from statsmodels.stats.proportion import binom_test as sm_binom_test
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import r2_score, classification_report
except ImportError:
    pass

# 核心工具函数
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
        if any(fmt in col.lower() for fmt in ['date', 'time', '2016', '2017', '2018', '2019', '2020']):
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

def frequency_analysis(df, categorical_cols):
    freq_dict = {}
    for col in categorical_cols:
        freq = df[col].value_counts()
        freq_pct = df[col].value_counts(normalize=True) * 100
        freq_df = pd.DataFrame({'频数': freq, '频率(%)': freq_pct.round(2)})
        freq_dict[col] = freq_df
    return freq_dict

def descriptive_analysis(df, numeric_cols):
    desc_df = df[numeric_cols].describe().T
    desc_df['缺失值'] = df[numeric_cols].isnull().sum()
    desc_df['缺失率(%)'] = (desc_df['缺失值'] / len(df) * 100).round(2)
    desc_df['偏度'] = df[numeric_cols].skew().round(3)
    desc_df['峰度'] = df[numeric_cols].kurt().round(3)
    return desc_df

def contingency_table_analysis(df, col1, col2):
    cont_table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(cont_table)
    cramers_v = np.sqrt(chi2 / (len(df) * min(cont_table.shape[0]-1, cont_table.shape[1]-1)))
    return {
        '联列表': cont_table,
        '卡方值': chi2.round(3),
        'p值': p.round(4),
        '自由度': dof,
        '克莱姆V系数': cramers_v.round(3)
    }

def t_test_onesample(df, numeric_col, popmean):
    data = df[numeric_col].dropna()
    t_stat, p_value = ttest_1samp(data, popmean)
    return {'t值': t_stat.round(3), 'p值': p_value.round(4), '均值': data.mean().round(2), '样本量': len(data)}

def t_test_independent(df, numeric_col, group_col):
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

def t_test_paired(df, col1, col2):
    paired_data = df[[col1, col2]].dropna()
    t_stat, p_value = ttest_rel(paired_data[col1], paired_data[col2])
    return {'t值': t_stat.round(3), 'p值': p_value.round(4), '差值均值': (paired_data[col1]-paired_data[col2]).mean().round(2), '样本量': len(paired_data)}

def nonparametric_test(df, test_type, numeric_col, group_col=None):
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
        data = df[numeric_col].dropna()
        success = sum(data == 1)
        n = len(data)
        p_value = sm_binom_test(success, n, prop=0.5)
        return {'成功次数': success, '总次数': n, 'p值': p_value.round(4)}
    return {'error': '无效检验类型'}

def anova_analysis(df, formula, anova_type):
    model = ols(formula, data=df).fit()
    if anova_type == '单因素方差分析':
        anova_result = anova_lm(model, typ=2)
    elif anova_type == '多因素方差分析':
        anova_result = anova_lm(model, typ=3)
    elif anova_type == '协方差分析':
        anova_result = anova_lm(model, typ=2)
    else:
        return {'error': '无效方差分析类型'}
    tukey = pairwise_tukeyhsd(df[formula.split('~')[0]], df[formula.split('~')[1].split('+')[0]], alpha=0.05)
    return {'方差分析表': anova_result, '事后检验(Tukey)': tukey.summary()}

def correlation_analysis(df, cols, corr_type='pearson'):
    corr_df = df[cols].dropna()
    if corr_type == 'pearson':
        corr_matrix = corr_df.corr(method='pearson')
        p_matrix = pd.DataFrame(np.ones_like(corr_matrix), index=corr_matrix.index, columns=corr_matrix.columns)
        for col1 in cols:
            for col2 in cols:
                if col1 != col2:
                    corr, p = stats.pearsonr(corr_df[col1], corr_df[col2])
                    p_matrix.loc[col1, col2] = round(p, 4)
    elif corr_type == 'spearman':
        corr_matrix = corr_df.corr(method='spearman')
        p_matrix = pd.DataFrame(np.ones_like(corr_matrix), index=corr_matrix.index, columns=corr_matrix.columns)
        for col1 in cols:
            for col2 in cols:
                if col1 != col2:
                    corr, p = stats.spearmanr(corr_df[col1], corr_df[col2])
                    p_matrix.loc[col1, col2] = round(p, 4)
    return {'相关矩阵': corr_matrix.round(3), 'p值矩阵': p_matrix}

def regression_analysis(df, target, features, reg_type):
    X = df[features].dropna()
    y = df[target][X.index].dropna()
    X = X.loc[y.index]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if reg_type == '线性回归':
        model = LinearRegression().fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        coef = pd.DataFrame({'特征': features, '系数': model.coef_.round(3), '截距': [model.intercept_.round(3)]*len(features)})
        return {'R²': r2.round(3), '系数表': coef}
    elif reg_type == '二分类Logistic回归':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        model = LogisticRegression(max_iter=1000).fit(X_scaled, y_encoded)
        y_pred = model.predict(X_scaled)
        report = classification_report(y_encoded, y_pred, output_dict=True)
        coef = pd.DataFrame({'特征': features, '系数': model.coef_[0].round(3), '截距': [model.intercept_[0].round(3)]*len(features)})
        return {'分类报告': report, '系数表': coef}
    return {'error': '无效回归类型'}

def cluster_analysis(df, cols, n_clusters=3):
    X = df[cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    df_cluster = X.copy()
    df_cluster['聚类结果'] = kmeans.labels_
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=cols).round(2)
    return {'聚类结果': df_cluster, '聚类中心': centroids}

def plot_chart(df, plot_type, x_col, y_col=None, group_col=None):
    if plot_type == '条形图':
        fig = px.bar(df, x=x_col, y=y_col, color=group_col, barmode='group', title=f'{x_col} - {y_col}')
    elif plot_type == '折线图':
        fig = px.line(df, x=x_col, y=y_col, color=group_col, title=f'{x_col} - {y_col}')
    elif plot_type == '饼图':
        fig = px.pie(df, names=x_col, values=y_col, title=f'{x_col} 分布')
    elif plot_type == '箱图':
        fig = px.box(df, x=x_col, y=y_col, color=group_col, title=f'{x_col} - {y_col} 分布')
    fig.update_layout(width=800, height=500)
    return fig

# 页面主体
st.title("科研数据分析平台")
st.divider()

# 侧边栏
with st.sidebar:
    uploaded_files = st.file_uploader("上传文件（CSV/Excel）", type=["xlsx", "csv"], accept_multiple_files=True)
    df = None
    if uploaded_files:
        selected_file_names = st.multiselect("选择文件", [f.name for f in uploaded_files], default=[uploaded_files[0].name])
        selected_files = [f for f in uploaded_files if f.name in selected_file_names]
        
        df_dict = {}
        for file in selected_files:
            df_temp = load_and_clean_data(file)
            if df_temp is not None:
                df_dict[file.name] = df_temp
                st.success(f"{file.name} 上传成功 ({len(df_temp)}行×{len(df_temp.columns)}列)")
        
        if len(df_dict) >= 2:
            base_file = st.selectbox("基础文件", list(df_dict.keys()))
            df = df_dict[base_file]
            for other_file in [f for f in df_dict.keys() if f != base_file]:
                df_other = df_dict[other_file]
                common_cols = [col for col in df.columns if col in df_other.columns]
                base_key = st.selectbox(f"基础关联字段", common_cols if common_cols else df.columns, key=f"base_{other_file}")
                join_key = st.selectbox(f"关联字段", common_cols if common_cols else df_other.columns, key=f"join_{other_file}")
                join_type = st.selectbox(f"合并方式", ['左连接', '右连接', '内连接', '外连接'], key=f"type_{other_file}")
                join_map = {'左连接':'left', '右连接':'right', '内连接':'inner', '外连接':'outer'}
                if st.button(f"合并{other_file}", key=f"btn_{other_file}"):
                    df = pd.merge(df, df_other, left_on=base_key, right_on=join_key, how=join_map[join_type], suffixes=("", f"_{other_file.split('.')[0]}"))
                    st.success(f"合并后：{len(df)}行×{len(df.columns)}列")
        else:
            df = df_dict[list(df_dict.keys())[0]] if df_dict else None
        
        if df is not None:
            var_types = identify_variable_types(df)
            st.write(f"数据规模：{len(df)}行 × {len(df.columns)}列")

# 主内容区
if df is not None:
    var_types = identify_variable_types(df)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("数据预览")
        st.dataframe(df.head(10), use_container_width=True, height=300)
    with col2:
        st.subheader("变量类型")
        st.write(f"数值型：{', '.join(var_types['numeric']) if var_types['numeric'] else '无'}")
        st.write(f"分类型：{', '.join(var_types['categorical']) if var_types['categorical'] else '无'}")
    
    # 分析标签页
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "数据处理", "基本统计", "均值检验", "方差分析", "相关分析", "回归分析", "可视化"
    ])
    
    with tab1:
        st.subheader("数据处理")
        # 数据排序
        sort_col = st.selectbox("排序字段", df.columns, key='sort')
        sort_asc = st.radio("排序方式", ['升序', '降序'], key='sort_asc')
        if st.button("执行排序"):
            df_sorted = df.sort_values(by=sort_col, ascending=(sort_asc=='升序'))
            st.dataframe(df_sorted.head(10), use_container_width=True)
        
        # 数据筛选
        filter_col = st.selectbox("筛选字段", df.columns, key='filter')
        filter_op = st.selectbox("运算符", ['>', '<', '>=', '<=', '==', '!='], key='filter_op')
        filter_val = st.text_input("筛选值", key='filter_val')
        if st.button("执行筛选"):
            try:
                if df[filter_col].dtype in [np.int64, np.float64]:
                    filter_val = float(filter_val)
                df_filtered = df.query(f"`{filter_col}` {filter_op} {filter_val}")
                st.success(f"筛选后：{len(df_filtered)}行")
                st.dataframe(df_filtered.head(10), use_container_width=True)
            except:
                st.error("筛选条件错误")
        
        # 分类汇总
        group_col = st.selectbox("分组字段", var_types['categorical'], key='group', disabled=not var_types['categorical'])
        agg_col = st.selectbox("汇总字段", var_types['numeric'], key='agg', disabled=not var_types['numeric'])
        agg_func = st.selectbox("汇总方式", ['均值', '求和', '计数', '最大值', '最小值'], key='agg_func')
        agg_map = {'均值':'mean', '求和':'sum', '计数':'count', '最大值':'max', '最小值':'min'}
        if st.button("执行汇总", disabled=not (group_col and agg_col)):
            df_agg = df.groupby(group_col)[agg_col].agg(agg_map[agg_func]).round(2)
            st.dataframe(df_agg, use_container_width=True)
    
    with tab2:
        st.subheader("基本统计")
        # 频数分析
        freq_cols = st.multiselect("选择分类型变量", var_types['categorical'], key='freq')
        if freq_cols and st.button("执行频数分析"):
            freq_dict = frequency_analysis(df, freq_cols)
            for col in freq_cols:
                st.subheader(col)
                st.dataframe(freq_dict[col], use_container_width=True)
        
        # 描述统计
        desc_cols = st.multiselect("选择数值型变量", var_types['numeric'], key='desc')
        if desc_cols and st.button("执行描述统计"):
            desc_df = descriptive_analysis(df, desc_cols)
            st.dataframe(desc_df, use_container_width=True)
        
        # 联列表分析
        if len(var_types['categorical'])>=2:
            cont_col1 = st.selectbox("行变量", var_types['categorical'], key='cont1')
            cont_col2 = st.selectbox("列变量", var_types['categorical'], key='cont2')
            if st.button("执行联列表分析"):
                cont_res = contingency_table_analysis(df, cont_col1, cont_col2)
                st.subheader("联列表")
                st.dataframe(cont_res['联列表'], use_container_width=True)
                st.write(f"卡方值：{cont_res['卡方值']}, p值：{cont_res['p值']}, 克莱姆V系数：{cont_res['克莱姆V系数']}")
    
    with tab3:
        st.subheader("均值检验")
        # 单样本t检验
        onesamp_col = st.selectbox("检验变量", var_types['numeric'], key='onesamp', disabled=not var_types['numeric'])
        popmean = st.number_input("总体均值", value=0.0, key='popmean')
        if st.button("执行单样本t检验", disabled=not onesamp_col):
            onesamp_res = t_test_onesample(df, onesamp_col, popmean)
            st.write(f"t值：{onesamp_res['t值']}, p值：{onesamp_res['p值']}, 样本均值：{onesamp_res['均值']}")
        
        # 两独立样本t检验
        ind_col = st.selectbox("检验变量", var_types['numeric'], key='ind', disabled=not var_types['numeric'])
        ind_group = st.selectbox("分组变量", var_types['categorical'], key='ind_group', disabled=not var_types['categorical'])
        if st.button("执行两独立样本t检验", disabled=not (ind_col and ind_group)):
            ind_res = t_test_independent(df, ind_col, ind_group)
            if 'error' in ind_res:
                st.error(ind_res['error'])
            else:
                st.write(f"t值：{ind_res['t值']}, p值：{ind_res['p值']}")
                st.write(f"{list(ind_res.keys())[2]}：{ind_res[list(ind_res.keys())[2]]}")
                st.write(f"{list(ind_res.keys())[3]}：{ind_res[list(ind_res.keys())[3]]}")
        
        # 非参数检验
        test_type = st.selectbox("检验类型", ['单样本K-S检验', '二项分布检验', '两独立样本Mann-Whitney U检验'], key='test_type')
        np_col = st.selectbox("检验变量", var_types['numeric'], key='np', disabled=not var_types['numeric'])
        np_group = st.selectbox("分组变量", var_types['categorical'], key='np_group', disabled=test_type not in ['两独立样本Mann-Whitney U检验'])
        if st.button("执行非参数检验", disabled=not np_col):
            if test_type in ['单样本K-S检验', '二项分布检验']:
                np_res = nonparametric_test(df, test_type, np_col)
            else:
                np_res = nonparametric_test(df, test_type, np_col, np_group)
            if 'error' in np_res:
                st.error(np_res['error'])
            else:
                for k, v in np_res.items():
                    st.write(f"{k}：{v}")
    
    with tab4:
        st.subheader("方差分析")
        if var_types['numeric'] and var_types['categorical']:
            anova_target = st.selectbox("因变量", var_types['numeric'], key='anova_target')
            anova_factor = st.selectbox("因素变量", var_types['categorical'], key='anova_factor')
            formula = f"{anova_target} ~ C({anova_factor})"
            if st.button("执行单因素方差分析"):
                anova_res = anova_analysis(df, formula, '单因素方差分析')
                if 'error' in anova_res:
                    st.error(anova_res['error'])
                else:
                    st.subheader("方差分析表")
                    st.dataframe(anova_res['方差分析表'], use_container_width=True)
                    st.subheader("事后检验")
                    st.text(anova_res['事后检验(Tukey)'])
    
    with tab5:
        st.subheader("相关分析")
        corr_type = st.selectbox("相关类型", ['pearson', 'spearman'], key='corr_type')
        corr_cols = st.multiselect("选择变量", var_types['numeric'], key='corr_cols')
        if corr_cols and len(corr_cols)>=2 and st.button("执行相关分析"):
            corr_res = correlation_analysis(df, corr_cols, corr_type)
            st.subheader("相关矩阵")
            st.dataframe(corr_res['相关矩阵'], use_container_width=True)
            st.subheader("p值矩阵")
            st.dataframe(corr_res['p值矩阵'], use_container_width=True)
            
            # 热力图
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr_res['相关矩阵'], cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(corr_cols)))
            ax.set_yticks(np.arange(len(corr_cols)))
            ax.set_xticklabels(corr_cols, rotation=45, ha='right')
            ax.set_yticklabels(corr_cols)
            for i in range(len(corr_cols)):
                for j in range(len(corr_cols)):
                    text = ax.text(j, i, corr_res['相关矩阵'].iloc[i, j], ha="center", va="center", color="black")
            cbar = ax.figure.colorbar(im, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab6:
        st.subheader("回归分析")
        reg_type = st.selectbox("回归类型", ['线性回归', '二分类Logistic回归'], key='reg_type')
        reg_target = st.selectbox("因变量", var_types['numeric'] if reg_type=='线性回归' else var_types['binary_categorical'], key='reg_target')
        reg_features = st.multiselect("自变量", [col for col in var_types['numeric'] if col != reg_target], key='reg_features')
        if st.button("执行回归分析", disabled=not (reg_target and reg_features)):
            reg_res = regression_analysis(df, reg_target, reg_features, reg_type)
            if 'error' in reg_res:
                st.error(reg_res['error'])
            else:
                if reg_type == '线性回归':
                    st.write(f"R²：{reg_res['R²']}")
                else:
                    st.write(f"准确率：{reg_res['分类报告']['accuracy']:.3f}")
                st.subheader("系数表")
                st.dataframe(reg_res['系数表'], use_container_width=True)
    
    with tab7:
        st.subheader("可视化分析")
        plot_type = st.selectbox("图表类型", ['条形图', '折线图', '饼图', '箱图'], key='plot_type')
        if plot_type in ['条形图', '折线图', '箱图']:
            x_col = st.selectbox("X轴变量", df.columns, key='plot_x')
            y_col = st.selectbox("Y轴变量", var_types['numeric'], key='plot_y')
            group_col = st.selectbox("分组变量", [None] + var_types['categorical'], key='plot_group')
        else:
            x_col = st.selectbox("类别变量", var_types['categorical'], key='plot_x_pie')
            y_col = st.selectbox("数值变量", var_types['numeric'], key='plot_y_pie')
            group_col = None
        if st.button("生成图表"):
            fig = plot_chart(df, plot_type, x_col, y_col, group_col)
            st.plotly_chart(fig, use_container_width=True)
    
    # 报告导出
    st.divider()
    st.subheader("导出报告")
    report_content = f"""# 科研数据分析报告
## 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## 数据概况
- 规模：{len(df)}行 × {len(df.columns)}列
- 数值型变量：{', '.join(var_types['numeric'])}
- 分类型变量：{', '.join(var_types['categorical'])}
- 缺失值总数：{df.isnull().sum().sum()}个
"""
    st.download_button(
        label="下载报告",
        data=report_content,
        file_name=f"分析报告_{datetime.now().strftime('%Y%m%d%H%M')}.md",
        mime="text/markdown"
    )

else:
    st.info("请在左侧边栏上传CSV/Excel文件进行分析")
