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
from datetime import datetime
from openai import OpenAI

warnings.filterwarnings('ignore')
# 优化：兼容云环境无SimHei字体，避免中文乱码/报错
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="科研数据分析平台", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

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
except ImportError as e:
    st.error(f"分析库导入失败：{e}")

def call_deepseek_api(prompt, model="deepseek-chat", temperature=0.2):  # 优化：调低温度，保证输出格式稳定
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return iter(["❌ 未配置API密钥：请在Streamlit Cloud → Settings → Secrets中添加 DEEPSEEK_API_KEY = '你的密钥'"])
    api_key = st.secrets["DEEPSEEK_API_KEY"]
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
    except Exception as e:
        return iter([f"❌ 客户端初始化失败：{str(e)}"])
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=3072,  # 优化：增加最大令牌，支持更长报告
            stream=True
        )
        def stream_generator():
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        return stream_generator()
    except client.BadRequestError as e:
        if "model_not_found" in str(e):
            return iter(["❌ 模型不存在：主流模型为 deepseek-chat / deepseek-reasoner"])
        return iter([f"❌ 请求参数错误：{str(e)}"])
    except client.UnauthorizedError:
        return iter(["❌ API密钥无效：请检查密钥是否正确/未过期"])
    except client.ServiceUnavailableError:
        return iter(["❌ DeepSeek服务器繁忙：建议稍后重试"])
    except TimeoutError:
        return iter(["❌ 网络超时：建议稍后重试"])
    except Exception as e:
        return iter([f"❌ API调用失败：{str(e)}"])

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
    return {'联列表': cont_table, '卡方值': chi2.round(3), 'p值': p.round(4), '自由度': dof, '克莱姆V系数': cramers_v.round(3)}

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
    return {'t值': t_stat.round(3), 'p值': p_value.round(4), f'{groups[0]}均值': group1.mean().round(2), f'{groups[1]}均值': group2.mean().round(2), f'{groups[0]}样本量': len(group1), f'{groups[1]}样本量': len(group2)}

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
    elif test_type == '二项分布检验':
        data = df[numeric_col].dropna()
        success = sum(data == 1)
        n = len(data)
        p_value = sm_binom_test(success, n, prop=0.5)
        return {'成功次数': success, '总次数': n, 'p值': p_value.round(4)}
    return {'error': '无效检验类型'}

def anova_analysis(df, formula, anova_type):
    model = ols(formula, data=df).fit()
    anova_result = anova_lm(model, typ=2)
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
        fig = px.bar(df, x=x_col, y=y_col, color=group_col, barmode='group', title=f'{x_col} - {y_col} 分组条形图')
    elif plot_type == '折线图':
        fig = px.line(df, x=x_col, y=y_col, color=group_col, title=f'{x_col} - {y_col} 趋势折线图')
    elif plot_type == '饼图':
        fig = px.pie(df, names=x_col, values=y_col, title=f'{x_col} 占比饼图', hole=0.2)  # 优化：增加空心饼图，更美观
    elif plot_type == '箱图':
        fig = px.box(df, x=x_col, y=y_col, color=group_col, title=f'{x_col} - {y_col} 分布箱图')
    fig.update_layout(width=800, height=500, font=dict(size=12))  # 优化：统一字体大小
    return fig

# ===== 页面主逻辑（原有无报错，仅增量优化）=====
st.title("📊 科研数据分析平台")
st.divider()

with st.sidebar:
    st.markdown("## 📥 数据上传")
    uploaded_files = st.file_uploader("上传文件（CSV/Excel，支持多文件合并）", type=["xlsx", "csv"], accept_multiple_files=True)
    df = None
    var_types = None
    if uploaded_files:
        selected_file_names = st.multiselect("选择分析文件", [f.name for f in uploaded_files], default=[uploaded_files[0].name])
        selected_files = [f for f in uploaded_files if f.name in selected_file_names]
        
        df_dict = {}
        for file in selected_files:
            df_temp = load_and_clean_data(file)
            if df_temp is not None:
                df_dict[file.name] = df_temp
                st.success(f"✅ {file.name} 上传成功 ({len(df_temp)}行×{len(df_temp.columns)}列)")
        
        # 多文件合并逻辑
        if len(df_dict) >= 2:
            st.markdown("### 🔗 多文件合并")
            base_file = st.selectbox("基础文件", list(df_dict.keys()))
            df = df_dict[base_file]
            for other_file in [f for f in df_dict.keys() if f != base_file]:
                df_other = df_dict[other_file]
                common_cols = [col for col in df.columns if col in df_other.columns]
                base_key = st.selectbox(f"基础关联字段", common_cols if common_cols else df.columns, key=f"base_{other_file}")
                join_key = st.selectbox(f"{other_file}关联字段", common_cols if common_cols else df_other.columns, key=f"join_{other_file}")
                join_type = st.selectbox(f"合并方式", ['左连接', '右连接', '内连接', '外连接'], key=f"type_{other_file}")
                join_map = {'左连接':'left', '右连接':'right', '内连接':'inner', '外连接':'outer'}
                if st.button(f"🔄 合并{other_file}", key=f"btn_{other_file}"):
                    df = pd.merge(df, df_other, left_on=base_key, right_on=join_key, how=join_map[join_type], suffixes=("", f"_{other_file.split('.')[0]}"))
                    st.success(f"✅ 合并后：{len(df)}行×{len(df.columns)}列")
        else:
            df = df_dict[list(df_dict.keys())[0]] if df_dict else None
        
        # 数据概况展示
        if df is not None:
            var_types = identify_variable_types(df)
            st.markdown("## 📋 数据概况")
            st.info(f"📏 规模：{len(df)}行 × {len(df.columns)}列")
            st.info(f"🔢 数值型变量：{len(var_types['numeric'])}个")
            st.info(f"📦 分类型变量：{len(var_types['categorical'])}个")
            st.info(f"⚖️ 二分类变量：{len(var_types['binary_categorical'])}个")
            st.info(f"📅 时间型变量：{len(var_types['datetime'])}个")

# 核心分析逻辑（数据上传成功后执行）
if df is not None and var_types is not None:
    # 构造数据概况文本（给AI用）
    data_overview = f"""本次分析数据核心概况：
1. 数据规模：{len(df)}行 × {len(df.columns)}列，整体缺失率：{(df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.2f}%
2. 数值型变量：{', '.join(var_types['numeric']) if var_types['numeric'] else '无'}
3. 分类型变量：{', '.join(var_types['categorical']) if var_types['categorical'] else '无'}
4. 二分类变量：{', '.join(var_types['binary_categorical']) if var_types['binary_categorical'] else '无'}
5. 时间型变量：{', '.join(var_types['datetime']) if var_types['datetime'] else '无'}"""

    # 新建标签页（原有顺序不变）
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "数据处理", "基本统计", "均值检验", "方差分析", "相关分析", "回归分析", "可视化", "🤖 AI分析"
    ])

    with tab1:
        st.subheader("⚙️ 数据预处理")
        # 排序
        with st.expander("🔽 数据排序", expanded=True):
            sort_col = st.selectbox("排序字段", df.columns, key='sort')
            sort_asc = st.radio("排序方式", ['升序', '降序'], key='sort_asc', horizontal=True)
            if st.button("执行排序", key='btn_sort'):
                df_sorted = df.sort_values(by=sort_col, ascending=(sort_asc=='升序'))
                st.dataframe(df_sorted.head(15), use_container_width=True)
        # 筛选
        with st.expander("🔍 数据筛选", expanded=True):
            filter_col = st.selectbox("筛选字段", df.columns, key='filter')
            filter_op = st.selectbox("运算符", ['>', '<', '>=', '<=', '==', '!='], key='filter_op', horizontal=True)
            filter_val = st.text_input("筛选值（数值/文本）", key='filter_val', placeholder="例：100 / 男")
            if st.button("执行筛选", key='btn_filter'):
                try:
                    if df[filter_col].dtype in [np.int64, np.float64]:
                        filter_val = float(filter_val)
                    df_filtered = df.query(f"`{filter_col}` {filter_op} {filter_val}")
                    st.success(f"✅ 筛选后：{len(df_filtered)}行数据")
                    st.dataframe(df_filtered.head(15), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ 筛选条件错误：{str(e)[:50]}，请检查值的类型是否匹配")
        # 分类汇总
        with st.expander("📊 分类汇总", expanded=True):
            group_col = st.selectbox("分组字段", var_types['categorical'], key='group', disabled=not var_types['categorical'])
            agg_col = st.selectbox("汇总字段", var_types['numeric'], key='agg', disabled=not var_types['numeric'])
            agg_func = st.selectbox("汇总方式", ['均值', '求和', '计数', '最大值', '最小值'], key='agg_func', horizontal=True)
            agg_map = {'均值':'mean', '求和':'sum', '计数':'count', '最大值':'max', '最小值':'min'}
            if st.button("执行分类汇总", key='btn_agg', disabled=not (group_col and agg_col)):
                df_agg = df.groupby(group_col)[agg_col].agg(agg_map[agg_func]).round(2)
                st.dataframe(df_agg, use_container_width=True)
                # 快速可视化汇总结果
                fig_agg = px.bar(df_agg.reset_index(), x=group_col, y=agg_col, title=f"{group_col} - {agg_col}（{agg_func}）")
                st.plotly_chart(fig_agg, use_container_width=True)

    with tab2:
        st.subheader("📈 基本统计分析")
        # 频数分析
        with st.expander("📦 分类变量频数分析", expanded=True):
            freq_cols = st.multiselect("选择分类型变量", var_types['categorical'], key='freq')
            if freq_cols and st.button("执行频数分析", key='btn_freq'):
                freq_dict = frequency_analysis(df, freq_cols)
                for col in freq_cols:
                    st.subheader(f"🔍 {col} 频数/频率分布")
                    st.dataframe(freq_dict[col], use_container_width=True)
                    # 快速生成频数条形图
                    freq_df = freq_dict[col].reset_index().rename(columns={'index': col})
                    fig_freq = px.bar(freq_df, x=col, y='频数', text_auto=True, title=f"{col} 频数分布")
                    st.plotly_chart(fig_freq, use_container_width=True)
        # 描述统计
        with st.expander("🔢 数值变量描述统计", expanded=True):
            desc_cols = st.multiselect("选择数值型变量", var_types['numeric'], key='desc')
            if desc_cols and st.button("执行描述统计", key='btn_desc'):
                desc_df = descriptive_analysis(df, desc_cols)
                st.subheader("📋 描述性统计结果（含缺失值/偏度/峰度）")
                st.dataframe(desc_df, use_container_width=True)
        # 卡方检验
        with st.expander("⚖️ 列联表+卡方检验", expanded=True):
            if len(var_types['categorical'])>=2:
                cont_col1 = st.selectbox("行变量", var_types['categorical'], key='cont1')
                cont_col2 = st.selectbox("列变量", var_types['categorical'], key='cont2')
                if st.button("执行卡方检验", key='btn_chi2'):
                    cont_res = contingency_table_analysis(df, cont_col1, cont_col2)
                    st.subheader(f"📊 {cont_col1} × {cont_col2} 列联表")
                    st.dataframe(cont_res['联列表'], use_container_width=True)
                    st.subheader("📈 卡方检验结果")
                    st.info(f"卡方值：{cont_res['卡方值']} | p值：{cont_res['p值']} | 自由度：{cont_res['自由度']}")
                    st.info(f"克莱姆V系数：{cont_res['克莱姆V系数']}（0-1，越大相关性越强）")
                    # 显著性判断
                    if cont_res['p值'] < 0.05:
                        st.success("✅ p<0.05，两个分类变量存在显著的相关性！")
                    else:
                        st.warning("⚠️ p≥0.05，两个分类变量无显著相关性！")
            else:
                st.warning("⚠️ 请至少选择2个分类型变量进行卡方检验")

    with tab3:
        st.subheader("⚖️ 均值检验")
        # 单样本t检验
        with st.expander("📊 单样本t检验", expanded=True):
            onesamp_col = st.selectbox("检验变量（数值型）", var_types['numeric'], key='onesamp', disabled=not var_types['numeric'])
            popmean = st.number_input("总体均值（检验基准）", value=0.0, key='popmean', step=0.1)
            if st.button("执行单样本t检验", key='btn_onesamp', disabled=not onesamp_col):
                onesamp_res = t_test_onesample(df, onesamp_col, popmean)
                st.subheader(f"🔍 {onesamp_col} 单样本t检验结果")
                st.info(f"样本均值：{onesamp_res['均值']} | 样本量：{onesamp_res['样本量']}")
                st.info(f"t值：{onesamp_res['t值']} | p值：{onesamp_res['p值']}")
                if onesamp_res['p值'] < 0.05:
                    st.success("✅ p<0.05，样本均值与总体均值存在显著差异！")
                else:
                    st.warning("⚠️ p≥0.05，样本均值与总体均值无显著差异！")
        # 两独立样本t检验
        with st.expander("📊 两独立样本t检验", expanded=True):
            ind_col = st.selectbox("检验变量（数值型）", var_types['numeric'], key='ind', disabled=not var_types['numeric'])
            ind_group = st.selectbox("分组变量（分类型）", var_types['categorical'], key='ind_group', disabled=not var_types['categorical'])
            if st.button("执行两独立样本t检验", key='btn_ind', disabled=not (ind_col and ind_group)):
                ind_res = t_test_independent(df, ind_col, ind_group)
                if 'error' in ind_res:
                    st.error(f"❌ {ind_res['error']}")
                else:
                    st.subheader(f"🔍 {ind_col} 按{ind_group}分组 t检验结果")
                    st.info(f"t值：{ind_res['t值']} | p值：{ind_res['p值']}")
                    for k in ind_res.keys():
                        if '均值' in k or '样本量' in k:
                            st.info(f"{k}：{ind_res[k]}")
                    if ind_res['p值'] < 0.05:
                        st.success("✅ p<0.05，两组样本均值存在显著差异！")
                    else:
                        st.warning("⚠️ p≥0.05，两组样本均值无显著差异！")
        # 非参数检验
        with st.expander("📊 非参数检验", expanded=True):
            test_type = st.selectbox("检验类型", ['单样本K-S检验', '二项分布检验', '两独立样本Mann-Whitney U检验'], key='test_type')
            np_col = st.selectbox("检验变量（数值型）", var_types['numeric'], key='np', disabled=not var_types['numeric'])
            np_group = st.selectbox("分组变量（仅U检验需选）", [None] + var_types['categorical'], key='np_group', disabled=test_type not in ['两独立样本Mann-Whitney U检验'])
            if st.button("执行非参数检验", key='btn_np', disabled=not np_col):
                np_res = nonparametric_test(df, test_type, np_col, np_group)
                if 'error' in np_res:
                    st.error(f"❌ {np_res['error']}")
                else:
                    st.subheader(f"🔍 {test_type} 结果")
                    for k, v in np_res.items():
                        st.info(f"{k}：{v}")
                    # 显著性判断
                    if 'p值' in np_res and np_res['p值'] < 0.05:
                        st.success("✅ p<0.05，检验结果存在显著差异/不符合正态分布！")
                    elif 'p值' in np_res:
                        st.warning("⚠️ p≥0.05，检验结果无显著差异/符合正态分布！")

    with tab4:
        st.subheader("📊 方差分析")
        if var_types['numeric'] and var_types['categorical']:
            anova_target = st.selectbox("因变量（数值型）", var_types['numeric'], key='anova_target')
            anova_factor = st.selectbox("因素变量（分类型）", var_types['categorical'], key='anova_factor')
            formula = f"{anova_target} ~ C({anova_factor})"
            if st.button("执行单因素方差分析+Tukey事后检验", key='btn_anova'):
                anova_res = anova_analysis(df, formula, '单因素方差分析')
                st.subheader("📈 单因素方差分析表")
                st.dataframe(anova_res['方差分析表'], use_container_width=True)
                # 方差分析显著性判断
                anova_p = anova_res['方差分析表']['PR(>F)'].iloc[0]
                if anova_p < 0.05:
                    st.success("✅ p<0.05，各分组均值存在显著整体差异，需看事后检验！")
                else:
                    st.warning("⚠️ p≥0.05，各分组均值无显著整体差异，无需看事后检验！")
                st.subheader("📋 Tukey HSD 事后检验结果（多重比较）")
                st.text(anova_res['事后检验(Tukey)'])
        else:
            st.warning("⚠️ 请同时存在数值型和分类型变量才能执行方差分析")

    with tab5:
        st.subheader("📈 相关分析")
        corr_type = st.selectbox("相关系数类型", ['pearson（皮尔逊，适用于正态分布）', 'spearman（斯皮尔曼，非参数/偏态）'], key='corr_type')
        corr_type_map = {'pearson（皮尔逊，适用于正态分布）':'pearson', 'spearman（斯皮尔曼，非参数/偏态）':'spearman'}
        corr_cols = st.multiselect("选择数值型变量（至少2个）", var_types['numeric'], key='corr_cols')
        if len(corr_cols) < 2:
            st.warning("⚠️ 请选择至少2个数值型变量进行相关分析")
            st.button("执行相关分析（含热力图）", key='btn_corr', disabled=True)
        else:
            if st.button("执行相关分析（含热力图）", key='btn_corr'):
                corr_res = correlation_analysis(df, corr_cols, corr_type_map[corr_type])
                st.subheader(f"📊 {corr_type.split('（')[0]} 相关系数矩阵")
                st.dataframe(corr_res['相关矩阵'], use_container_width=True)
                st.subheader(f"📊 相关分析p值矩阵（p<0.05为显著）")
                st.dataframe(corr_res['p值矩阵'], use_container_width=True)
                # 绘制相关热力图（保留原有st.pyplot，稳定无报错）
                st.subheader(f"📊 相关系数热力图")
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(corr_res['相关矩阵'], cmap='RdBu_r', vmin=-1, vmax=1)
                ax.set_xticks(np.arange(len(corr_cols)))
                ax.set_yticks(np.arange(len(corr_cols)))
                ax.set_xticklabels(corr_cols, rotation=45, ha='right', fontsize=10)
                ax.set_yticklabels(corr_cols, fontsize=10)
                # 标注相关系数和显著性
                for i in range(len(corr_cols)):
                    for j in range(len(corr_cols)):
                        corr_val = corr_res['相关矩阵'].iloc[i, j]
                        p_val = corr_res['p值矩阵'].iloc[i, j]
                        # 显著性标记：**p<0.01，*p<0.05，无标记p≥0.05
                        mark = '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                        text = ax.text(j, i, f"{corr_val:.3f}{mark}", ha="center", va="center", color="black", fontsize=9)
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.set_label(f'{corr_type.split("（")[0]} 相关系数', rotation=270, labelpad=20, fontsize=12)
                plt.title(f'{corr_type.split("（")[0]} 相关系数热力图（**p<0.01，*p<0.05）', fontsize=14)
                plt.tight_layout()
                st.pyplot(fig)

    with tab6:
        st.subheader("📈 回归分析")
        reg_type = st.selectbox("回归类型", ['线性回归（数值因变量）', '二分类Logistic回归（二分类因变量）'], key='reg_type')
        reg_type_map = {'线性回归（数值因变量）':'线性回归', '二分类Logistic回归（二分类因变量）':'二分类Logistic回归'}
        # 按回归类型选择因变量
        if reg_type == '线性回归（数值因变量）':
            reg_target = st.selectbox("因变量（数值型）", var_types['numeric'], key='reg_target')
        else:
            reg_target = st.selectbox("因变量（二分类）", var_types['binary_categorical'], key='reg_target', disabled=not var_types['binary_categorical'])
        # 选择自变量（排除因变量）
        reg_features = st.multiselect("自变量（数值型，至少1个）", [col for col in var_types['numeric'] if col != reg_target], key='reg_features')
        # 按钮禁用逻辑
        btn_disabled = False
        if reg_type == '二分类Logistic回归（二分类因变量）' and not var_types['binary_categorical']:
            btn_disabled = True
        if not (reg_target and reg_features):
            btn_disabled = True
        if st.button("执行回归分析", key='btn_reg', disabled=btn_disabled):
            reg_res = regression_analysis(df, reg_target, reg_features, reg_type_map[reg_type])
            if 'error' in reg_res:
                st.error(f"❌ {reg_res['error']}")
            else:
                st.subheader(f"📊 {reg_type.split('（')[0]} 模型结果")
                if reg_type == '线性回归（数值因变量）':
                    st.success(f"✅ 模型拟合度 R² = {reg_res['R²']}（越接近1拟合效果越好）")
                else:
                    acc = reg_res['分类报告']['accuracy']
                    st.success(f"✅ 模型准确率 = {acc:.3f} | 精确率 = {reg_res['分类报告']['weighted avg']['precision']:.3f} | 召回率 = {reg_res['分类报告']['weighted avg']['recall']:.3f}")
                st.subheader("📋 模型系数表（截距+特征系数）")
                st.dataframe(reg_res['系数表'], use_container_width=True)

    with tab7:
        st.subheader("🎨 自定义可视化分析")
        plot_type = st.selectbox("选择图表类型", ['条形图', '折线图', '饼图', '箱图'], key='plot_type')
        # 按图表类型选择变量
        if plot_type in ['条形图', '折线图', '箱图']:
            x_col = st.selectbox("X轴变量", df.columns, key='plot_x')
            y_col = st.selectbox("Y轴变量（数值型）", var_types['numeric'], key='plot_y')
            group_col = st.selectbox("分组变量（可选，无则不分组）", [None] + var_types['categorical'], key='plot_group')
        else:  # 饼图
            x_col = st.selectbox("类别变量（饼图分组）", var_types['categorical'], key='plot_x_pie')
            y_col = st.selectbox("数值变量（饼图数值）", var_types['numeric'], key='plot_y_pie')
            group_col = None
        # 生成图表
        if st.button("🎯 生成自定义图表", key='btn_plot'):
            try:
                fig = plot_chart(df, plot_type, x_col, y_col, group_col)
                st.plotly_chart(fig, use_container_width=True)
                # 图表下载按钮
                st.download_button(
                    label="📥 下载图表为HTML",
                    data=fig.to_html(),
                    file_name=f"{plot_type}_{x_col}_{y_col}.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"❌ 图表生成失败：{str(e)[:80]}，请检查变量选择是否合理")

    with tab8:
        st.subheader("🤖 AI 智能分析（基于真实统计+可视化）")
        if "DEEPSEEK_API_KEY" not in st.secrets:
            st.warning("⚠️ 请先在【Streamlit Cloud → Settings → Secrets】中配置：DEEPSEEK_API_KEY = '你的sk-开头密钥'")
            st.info("💡 密钥获取地址：https://platform.deepseek.com/")
        else:
            st.success("✅ API密钥已配置 | AI基于**真实统计结果+可视化图表**生成分析报告 | 无编造内容")
            st.markdown("---")

            # 1. AI自动数据分析（核心功能，优化固定格式）
            with st.expander("📑 AI自动数据分析（生成标准化科研报告）", expanded=True):
                if st.button("🚀 开始AI自动分析", key='btn_ai_analysis'):
                    with st.spinner("🌀 正在计算真实统计结果+生成可视化图表，请稍候（约10秒）..."):
                        # ===== 步骤1：生成真实统计结果（保留原有逻辑）=====
                        desc_res = descriptive_analysis(df, var_types['numeric']) if var_types['numeric'] else "无数值型变量，无描述统计结果"
                        desc_text = desc_res.to_string() if var_types['numeric'] else "无数值型变量，无描述统计结果"
                        
                        corr_res = correlation_analysis(df, var_types['numeric'], 'pearson') if len(var_types['numeric'])>=2 else "数值型变量不足2个，无相关分析结果"
                        corr_text = corr_res['相关矩阵'].to_string() if len(var_types['numeric'])>=2 else "数值型变量不足2个，无相关分析结果"
                        
                        freq_res = frequency_analysis(df, var_types['categorical']) if var_types['categorical'] else "无分类型变量，无频数分析结果"
                        freq_text = ""
                        if var_types['categorical']:
                            for col in var_types['categorical']:
                                freq_text += f"\n{col} 频数/频率：\n{freq_res[col].to_string()}\n"
                        else:
                            freq_text = "无分类型变量，无频数分析结果"
                        
                        ttest_text = "无符合条件的二分类变量，未执行两独立样本t检验"
                        if var_types['binary_categorical'] and var_types['numeric']:
                            group_col = var_types['binary_categorical'][0]
                            test_col = var_types['numeric'][0]
                            ttest_res = t_test_independent(df, test_col, group_col)
                            if 'error' not in ttest_res:
                                ttest_text = f"两独立样本t检验（{test_col} 按 {group_col} 分组）：\n"
                                ttest_text += f"t值={ttest_res['t值']}，p值={ttest_res['p值']}，"
                                ttest_text += f"{list(ttest_res.keys())[2]}={ttest_res[list(ttest_res.keys())[2]]}，"
                                ttest_text += f"{list(ttest_res.keys())[3]}={ttest_res[list(ttest_res.keys())[3]]}"

                        # ===== 步骤2：生成真实可视化图表（核心修复：取消硬编码，通用适配所有数据）=====
                        st.markdown("### 📊 真实可视化图表（基于你的数据生成）")
                        chart_desc = []  # 存储图表描述，给AI分析用

                        # 图1：数值变量相关热力图（异常捕获，失败则跳过）
                        try:
                            if len(var_types['numeric'])>=2 and isinstance(corr_res, dict):
                                st.subheader("🔍 图1：数值变量Pearson相关热力图")
                                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                                im_corr = ax_corr.imshow(corr_res['相关矩阵'], cmap='RdBu_r', vmin=-1, vmax=1)
                                ax_corr.set_xticks(np.arange(len(var_types['numeric'])))
                                ax_corr.set_yticks(np.arange(len(var_types['numeric'])))
                                ax_corr.set_xticklabels(var_types['numeric'], rotation=45, ha='right')
                                ax_corr.set_yticklabels(var_types['numeric'])
                                for i in range(len(var_types['numeric'])):
                                    for j in range(len(var_types['numeric'])):
                                        text = ax_corr.text(j, i, corr_res['相关矩阵'].iloc[i, j], ha="center", va="center", color="black")
                                cbar_corr = ax_corr.figure.colorbar(im_corr, ax=ax_corr)
                                cbar_corr.set_label('Pearson相关系数', rotation=270, labelpad=20)
                                plt.tight_layout()
                                st.pyplot(fig_corr)
                                chart_desc.append("图1：数值变量Pearson相关热力图，展示所有数值变量间的线性相关程度与正负方向，系数范围[-1,1]，越接近±1相关性越强")
                        except Exception as e:
                            st.warning(f"⚠️ 图1（相关热力图）生成失败：{str(e)[:50]}，已跳过")

                        # 图2：前两个数值变量趋势折线图（核心修复：取消硬编码，通用适配所有数据）
                        try:
                            if len(var_types['numeric'])>=2:
                                num1, num2 = var_types['numeric'][0], var_types['numeric'][1]  # 取前两个数值变量，通用
                                st.subheader(f"🔍 图2：{num1} 与 {num2} 趋势折线图（前1000条）")
                                fig_line = px.line(df.head(1000), x=df.head(1000).index, y=[num1, num2], 
                                                  title=f"{num1} vs {num2} 趋势变化对比", width=800, height=400)
                                fig_line.update_layout(xaxis_title="样本索引", yaxis_title="数值")
                                st.plotly_chart(fig_line, use_container_width=True)
                                chart_desc.append(f"图2：{num1}和{num2}的趋势折线图，展示了两个核心数值变量的前1000条数据的变化趋势与相互关系")
                        except Exception as e:
                            st.warning(f"⚠️ 图2（趋势折线图）生成失败：{str(e)[:50]}，已跳过")

                        # 图3：第一个分类变量频数条形图（保留原有逻辑，异常捕获）
                        try:
                            if var_types['categorical'] and isinstance(freq_res, dict):
                                cat_col = var_types['categorical'][0]
                                st.subheader(f"🔍 图3：{cat_col} 频数分布条形图")
                                freq_df = freq_res[cat_col].reset_index().rename(columns={'index': cat_col})
                                fig_bar = px.bar(freq_df, x=cat_col, y='频数', text_auto=True, title=f"{cat_col} 频数/频率分布", width=800, height=400)
                                fig_bar.update_layout(xaxis_title=cat_col, yaxis_title="频数")
                                st.plotly_chart(fig_bar, use_container_width=True)
                                chart_desc.append(f"图3：{cat_col}的频数分布条形图，展示了该分类变量各类型的频数和占比情况，可直观看到主要类别构成")
                        except Exception as e:
                            st.warning(f"⚠️ 图3（频数条形图）生成失败：{str(e)[:50]}，已跳过")

                        # ===== 步骤3：整合统计+图表信息，构造AI提示词（核心优化：固定科研报告格式）=====
                        real_info = f"""【数据基础概况】
{data_overview}

【描述性统计结果】
{desc_text}

【Pearson相关矩阵结果】
{corr_text}

【分类型变量频数结果】
{freq_text}

【两独立样本t检验结果】
{ttest_text}

【成功生成的可视化图表】
{"；".join(chart_desc) if chart_desc else "无可用可视化图表"}"""

                        # AI提示词：固定科研报告格式，要求严格遵循
                        prompt = f"""你是**资深科研数据分析专家**，专注于科研场景的数据分析与报告撰写，需基于以下**真实的统计结果和可视化图表**生成标准化科研分析报告，严格遵守以下要求：

### 【输出格式要求（必须严格遵循，不得删减/修改章节）】
# 数据统计分析报告
## 一、数据基本特征
1. 样本规模：明确说明数据的行、列数，整体缺失率，数据维度特征
2. 数值变量特征：基于描述统计结果，总结数值变量的均值、标准差、极值、缺失情况、偏度/峰度，指出数据的集中趋势和离散程度
3. 分类变量特征：基于频数分析结果，总结分类变量的主要类别、频数占比，描述分类变量的分布特征

## 二、可视化图表分析
要求：有多少张图就分析多少张，每张图单独成段，以【图X】开头；先说明图表展示的核心内容，再结合统计结果解读图表反映的规律/特征；无图表则写“本次分析无可用可视化图表，跳过本章节”

## 三、变量关系深度分析
1. 数值变量相关性：基于相关矩阵结果，分析变量间的线性相关程度、显著性（p<0.05为显著），指出强相关/弱相关/无相关的变量组合
2. 组间均值差异：基于t检验结果，分析二分类分组下数值变量的均值差异是否显著，无结果则写“无符合条件的二分类变量，未执行t检验，跳过本项”
3. 整体规律总结：综合上述分析，总结本次数据中变量间的核心关系规律

## 四、研究结论与建议
### （一）研究结论
基于全量统计分析与可视化结果，分3-5点**客观总结**数据反映的核心规律、特征，每点一句话，简洁明确，仅基于真实分析结果，不做过度推断
### （二）研究建议
结合数据特征与变量关系，分2-4点给出**针对性、可落地**的研究/分析建议，建议需贴合数据实际，具有实际参考价值

### 【核心约束】
1. 所有分析**必须基于提供的真实信息**，绝对禁止编造任何数值、统计量、p值、图表信息；
2. 严格遵循上述格式，标题层级（#/##/###）、编号、标点完全一致，语言专业、简洁、客观，适配科研场景；
3. 图表分析需结合统计结果，做到“图数结合”，不单独描述图表；
4. 显著性判断标准：p<0.05为显著，p<0.01为极显著，p≥0.05为不显著。

### 【本次分析的真实统计与图表信息】
{real_info}"""

                        # ===== 步骤4：调用AI并展示结果 =====
                        st.markdown("### 📋 AI标准化科研分析报告（基于真实数据，可直接复制到论文/报告）")
                        st.divider()
                        stream = call_deepseek_api(prompt)
                        st.write_stream(stream)

            # 2. AI统计问答（优化固定格式，更专业）
            with st.expander("❓ AI统计问答（针对性解答你的分析问题）", expanded=False):
                user_question = st.text_area(
                    "请输入你的数据分析问题（示例见占位符）",
                    placeholder="1. 分析A变量和B变量的相关性并解读显著性；2. 用t检验比较两组数据的均值差异并判断显著性；3. 总结数据的核心分布特征和缺失情况",
                    height=120,
                    key='ai_question'
                )
                if st.button("💬 发送问题", key='btn_ai_qa') and user_question:
                    st.markdown("### 📝 AI针对性解答结果")
                    st.divider()
                    prompt = f"""你是**专业统计分析师**，基于以下数据概况针对性解答用户问题，严格遵守以下要求：
### 【输出格式要求】
## 问题解答：{user_question}
1. 分析方法：明确解答该问题需使用的统计分析方法，说明方法的适用场景和前提条件
2. 结果解读：基于数据概况给出针对性解答，包含统计量、显著性判断（p<0.05为显著）
3. 专业建议：给出该分析的后续研究/分析建议，贴合科研场景

### 【核心约束】
1. 回答简洁专业，贴合科研数据分析场景，避免口语化；
2. 仅基于数据概况解答，不编造任何数据/变量/统计结果；
3. 显著性判断标准：p<0.05为显著，p<0.01为极显著。

### 【数据概况】
{data_overview}

### 【用户问题】
{user_question}"""
                    stream = call_deepseek_api(prompt)
                    st.write_stream(stream)

            # 3. AI结果解读（优化固定格式，分点解读）
            with st.expander("📈 AI统计结果解读（解读你的手动分析结果）", expanded=False):
                user_result = st.text_area(
                    "请粘贴你的统计分析结果（示例见占位符）",
                    placeholder="1. 皮尔逊相关系数：0.78，p=0.001；2. 两独立样本t检验：t=2.35，p=0.02；3. 线性回归R²=0.82，特征A系数=0.56；4. 卡方检验：卡方值=5.23，p=0.022",
                    height=120,
                    key='ai_result'
                )
                if st.button("🔍 解读结果", key='btn_ai_interpret') and user_result:
                    st.markdown("### 📝 AI统计结果专业解读报告")
                    st.divider()
                    prompt = f"""你是**资深科研统计分析师**，需解读用户提供的统计结果，严格遵守以下要求：
### 【输出格式要求】
# 统计结果解读报告
1. 指标解读：逐点解读每个统计指标的**核心统计意义**，说明指标的大小/正负代表的含义
2. 显著性判断：逐点判断结果的显著性，明确标注p值对应的显著性水平（p<0.05显著/p<0.01极显著/p≥0.05不显著）
3. 实际意义：结合数据概况，解读每个结果的**实际研究意义**，说明结果反映的研究问题/规律
4. 综合结论：综合所有结果，给出1-2句核心综合结论，简洁明确，贴合科研场景

### 【核心约束】
1. 逐点对应输入的统计结果，不遗漏、不编造，语言专业、简洁；
2. 显著性判断标准：p<0.05为显著，p<0.01为极显著，p≥0.05为不显著；
3. 实际意义解读需贴合数据概况，不脱离数据实际。

### 【数据概况】
{data_overview}

### 【用户提供的统计结果】
{user_result}"""
                    stream = call_deepseek_api(prompt)
                    st.write_stream(stream)
# 未上传数据时的引导
else:
    st.info("💡 请在【左侧边栏】上传**CSV/Excel**数据文件，即可开始全功能的科研数据分析～")
    st.markdown("#### 📌 平台核心功能亮点")
    st.markdown("✅ 集成SPSS核心统计功能：描述统计、卡方检验、t检验、方差分析、相关/回归分析等")
    st.markdown("✅ 可视化支持：自定义图表+一键生成，图表支持HTML下载，可直接用于论文/报告")
    st.markdown("✅ AI智能分析：基于**真实统计结果**生成**标准化科研报告**，无编造内容，格式可直接复制")
    st.markdown("✅ 支持多文件合并：可上传多个CSV/Excel文件，按关联字段实现左/右/内/外连接")
    st.markdown("✅ 操作简易：无需代码基础，纯可视化操作，结果一键查看/复制")
    st.markdown("✅ 云环境兼容：适配Streamlit Cloud，无本地环境依赖，随时随地分析")
