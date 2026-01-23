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
plt.rcParams['font.sans-serif'] = ['SimHei']
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

def call_deepseek_api(prompt, model="deepseek-chat", temperature=0.2):
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
            max_tokens=3072,
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
        fig = px.bar(df, x=x_col, y=y_col, color=group_col, barmode='group', title=f'{x_col} - {y_col}')
    elif plot_type == '折线图':
        fig = px.line(df, x=x_col, y=y_col, color=group_col, title=f'{x_col} - {y_col}')
    elif plot_type == '饼图':
        fig = px.pie(df, names=x_col, values=y_col, title=f'{x_col} 分布')
    elif plot_type == '箱图':
        fig = px.box(df, x=x_col, y=y_col, color=group_col, title=f'{x_col} - {y_col} 分布')
    fig.update_layout(width=800, height=500)
    return fig

st.title("科研数据分析平台")
st.divider()

with st.sidebar:
    st.markdown("## 📥 数据上传")
    uploaded_files = st.file_uploader("上传文件（CSV/Excel，支持多文件）", type=["xlsx", "csv"], accept_multiple_files=True)
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
            st.markdown("## 📊 数据概况")
            st.write(f"规模：{len(df)}行 × {len(df.columns)}列")
            st.write(f"数值型变量：{len(var_types['numeric'])}个")
            st.write(f"分类型变量：{len(var_types['categorical'])}个")

if df is not None and var_types is not None:
    data_overview = f"""本次分析数据概况：1.数据规模：{len(df)}行 × {len(df.columns)}列 2.数值型变量：{', '.join(var_types['numeric']) if var_types['numeric'] else '无'} 3.分类型变量：{', '.join(var_types['categorical']) if var_types['categorical'] else '无'} 4.二分类变量：{', '.join(var_types['binary_categorical']) if var_types['binary_categorical'] else '无'} 5.缺失值总数：{df.isnull().sum().sum()}个，整体缺失率：{(df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.2f}%"""
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["数据处理", "基本统计", "均值检验", "方差分析", "相关分析", "回归分析", "可视化", "AI分析"])

    with tab1:
        st.subheader("数据处理")
        sort_col = st.selectbox("排序字段", df.columns, key='sort')
        sort_asc = st.radio("排序方式", ['升序', '降序'], key='sort_asc')
        if st.button("执行排序"):
            df_sorted = df.sort_values(by=sort_col, ascending=(sort_asc=='升序'))
            st.dataframe(df_sorted.head(10), use_container_width=True)
        
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
                st.error("筛选条件错误，请检查值的类型")
        
        group_col = st.selectbox("分组字段", var_types['categorical'], key='group', disabled=not var_types['categorical'])
        agg_col = st.selectbox("汇总字段", var_types['numeric'], key='agg', disabled=not var_types['numeric'])
        agg_func = st.selectbox("汇总方式", ['均值', '求和', '计数', '最大值', '最小值'], key='agg_func')
        agg_map = {'均值':'mean', '求和':'sum', '计数':'count', '最大值':'max', '最小值':'min'}
        if st.button("执行分类汇总", disabled=not (group_col and agg_col)):
            df_agg = df.groupby(group_col)[agg_col].agg(agg_map[agg_func]).round(2)
            st.dataframe(df_agg, use_container_width=True)

    with tab2:
        st.subheader("基本统计")
        freq_cols = st.multiselect("选择分类型变量", var_types['categorical'], key='freq')
        if freq_cols and st.button("执行频数分析"):
            freq_dict = frequency_analysis(df, freq_cols)
            for col in freq_cols:
                st.subheader(col)
                st.dataframe(freq_dict[col], use_container_width=True)
        
        desc_cols = st.multiselect("选择数值型变量", var_types['numeric'], key='desc')
        if desc_cols and st.button("执行描述统计"):
            desc_df = descriptive_analysis(df, desc_cols)
            st.dataframe(desc_df, use_container_width=True)
        
        if len(var_types['categorical'])>=2:
            cont_col1 = st.selectbox("行变量", var_types['categorical'], key='cont1')
            cont_col2 = st.selectbox("列变量", var_types['categorical'], key='cont2')
            if st.button("执行联列表+卡方检验"):
                cont_res = contingency_table_analysis(df, cont_col1, cont_col2)
                st.subheader("联列表")
                st.dataframe(cont_res['联列表'], use_container_width=True)
                st.write(f"卡方值：{cont_res['卡方值']} | p值：{cont_res['p值']} | 自由度：{cont_res['自由度']} | 克莱姆V系数：{cont_res['克莱姆V系数']}")

    with tab3:
        st.subheader("均值检验")
        onesamp_col = st.selectbox("检验变量", var_types['numeric'], key='onesamp', disabled=not var_types['numeric'])
        popmean = st.number_input("总体均值", value=0.0, key='popmean')
        if st.button("执行单样本t检验", disabled=not onesamp_col):
            onesamp_res = t_test_onesample(df, onesamp_col, popmean)
            st.write(f"t值：{onesamp_res['t值']} | p值：{onesamp_res['p值']} | 样本均值：{onesamp_res['均值']} | 样本量：{onesamp_res['样本量']}")
        
        ind_col = st.selectbox("检验变量", var_types['numeric'], key='ind', disabled=not var_types['numeric'])
        ind_group = st.selectbox("分组变量", var_types['categorical'], key='ind_group', disabled=not var_types['categorical'])
        if st.button("执行两独立样本t检验", disabled=not (ind_col and ind_group)):
            ind_res = t_test_independent(df, ind_col, ind_group)
            if 'error' in ind_res:
                st.error(ind_res['error'])
            else:
                st.write(f"t值：{ind_res['t值']} | p值：{ind_res['p值']}")
                st.write(f"{list(ind_res.keys())[2]}：{ind_res[list(ind_res.keys())[2]]} | {list(ind_res.keys())[3]}：{ind_res[list(ind_res.keys())[3]]}")
        
        test_type = st.selectbox("非参数检验类型", ['单样本K-S检验', '二项分布检验', '两独立样本Mann-Whitney U检验'], key='test_type')
        np_col = st.selectbox("检验变量", var_types['numeric'], key='np', disabled=not var_types['numeric'])
        np_group = st.selectbox("分组变量", var_types['categorical'], key='np_group', disabled=test_type not in ['两独立样本Mann-Whitney U检验'])
        if st.button("执行非参数检验", disabled=not np_col):
            np_res = nonparametric_test(df, test_type, np_col, np_group)
            if 'error' in np_res:
                st.error(np_res['error'])
            else:
                for k, v in np_res.items():
                    st.write(f"{k}：{v}")

    with tab4:
        st.subheader("方差分析")
        if var_types['numeric'] and var_types['categorical']:
            anova_target = st.selectbox("因变量（数值型）", var_types['numeric'], key='anova_target')
            anova_factor = st.selectbox("因素变量（分类型）", var_types['categorical'], key='anova_factor')
            formula = f"{anova_target} ~ C({anova_factor})"
            if st.button("执行单因素方差分析+Tukey事后检验"):
                anova_res = anova_analysis(df, formula, '单因素方差分析')
                st.subheader("方差分析表")
                st.dataframe(anova_res['方差分析表'], use_container_width=True)
                st.subheader("Tukey事后检验结果")
                st.text(anova_res['事后检验(Tukey)'])

    with tab5:
        st.subheader("相关分析")
        corr_type = st.selectbox("相关系数类型", ['pearson（皮尔逊，适用于正态）', 'spearman（斯皮尔曼，非参数）'], key='corr_type')
        corr_type_map = {'pearson（皮尔逊，适用于正态）':'pearson', 'spearman（斯皮尔曼，非参数）':'spearman'}
        corr_cols = st.multiselect("选择数值型变量（至少2个）", var_types['numeric'], key='corr_cols')
        if len(corr_cols) < 2:
            st.warning("⚠️ 请选择至少2个数值型变量")
            st.button("执行相关分析（含热力图）", disabled=True)
        else:
            if st.button("执行相关分析（含热力图）"):
                corr_res = correlation_analysis(df, corr_cols, corr_type_map[corr_type])
                st.subheader("相关系数矩阵")
                st.dataframe(corr_res['相关矩阵'], use_container_width=True)
                st.subheader("p值矩阵")
                st.dataframe(corr_res['p值矩阵'], use_container_width=True)
                
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
        reg_features = st.multiselect("自变量（数值型）", [col for col in var_types['numeric'] if col != reg_target], key='reg_features')
        if st.button("执行回归分析", disabled=not (reg_target and reg_features)):
            reg_res = regression_analysis(df, reg_target, reg_features, reg_type)
            if 'error' in reg_res:
                st.error(reg_res['error'])
            else:
                if reg_type == '线性回归':
                    st.write(f"📊 模型拟合度 R²：{reg_res['R²']}")
                else:
                    st.write(f"📊 模型准确率：{reg_res['分类报告']['accuracy']:.3f}")
                st.subheader("系数表")
                st.dataframe(reg_res['系数表'], use_container_width=True)

    with tab7:
        st.subheader("可视化分析")
        plot_type = st.selectbox("图表类型", ['条形图', '折线图', '饼图', '箱图'], key='plot_type')
        if plot_type in ['条形图', '折线图', '箱图']:
            x_col = st.selectbox("X轴变量", df.columns, key='plot_x')
            y_col = st.selectbox("Y轴变量（数值型）", var_types['numeric'], key='plot_y')
            group_col = st.selectbox("分组变量（可选）", [None] + var_types['categorical'], key='plot_group')
        else:
            x_col = st.selectbox("类别变量", var_types['categorical'], key='plot_x_pie')
            y_col = st.selectbox("数值变量（用于占比）", var_types['numeric'], key='plot_y_pie')
            group_col = None
        if st.button("生成图表"):
            fig = plot_chart(df, plot_type, x_col, y_col, group_col)
            st.plotly_chart(fig, use_container_width=True)

    with tab8:
        st.subheader("🤖 AI 智能分析（图文嵌排+固定格式）")
        if "DEEPSEEK_API_KEY" not in st.secrets:
            st.warning("⚠️ 请先在【Streamlit Cloud → Settings → Secrets】中配置：DEEPSEEK_API_KEY = '你的sk-开头密钥'")
        else:
            st.success("✅ API密钥已配置，AI输出图文嵌排+固定统一格式分析报告")
            st.markdown("---")
            with st.expander("📑 AI自动数据分析（核心功能）", expanded=True):
                if st.button("🚀 开始AI自动分析"):
                    with st.spinner("正在预处理统计结果+生成可视化图表，请稍候..."):
                        desc_res = descriptive_analysis(df, var_types['numeric']) if var_types['numeric'] else "无数值型变量"
                        desc_text = desc_res.to_string() if var_types['numeric'] else "无数值型变量"
                        corr_res = correlation_analysis(df, var_types['numeric'], 'pearson') if len(var_types['numeric'])>=2 else "数值型变量不足2个"
                        corr_text = corr_res['相关矩阵'].to_string() if len(var_types['numeric'])>=2 else "数值型变量不足2个"
                        freq_res = frequency_analysis(df, var_types['categorical']) if var_types['categorical'] else "无分类型变量"
                        freq_text = ""
                        if var_types['categorical']:
                            for col in var_types['categorical']:
                                freq_text += f"{col}：{freq_res[col].to_string()}\n"
                        else:
                            freq_text = "无分类型变量"
                        ttest_text = "无符合条件的二分类变量，未执行均值检验"
                        if var_types['binary_categorical'] and var_types['numeric']:
                            group_col = var_types['binary_categorical'][0]
                            test_col = var_types['numeric'][0]
                            ttest_res = t_test_independent(df, test_col, group_col)
                            if 'error' not in ttest_res:
                                ttest_text = f"两独立样本t检验（{test_col}按{group_col}分组）：t值={ttest_res['t值']}，p值={ttest_res['p值']}，{list(ttest_res.keys())[2]}={ttest_res[list(ttest_res.keys())[2]]}，{list(ttest_res.keys())[3]}={ttest_res[list(ttest_res.keys())[3]]}"

                        chart_data = {}
                        try:
                            if len(var_types['numeric'])>=2:
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
                                plt.tight_layout()
                                chart_data['图1'] = {'fig': fig_corr, 'type': 'matplotlib', 'name': '数值变量相关热力图', 'desc': '展示各数值型变量间皮尔逊相关系数的强弱与正负相关方向，系数越接近1/ -1表示相关性越强，0表示无线性相关'}
                        except Exception as e:
                            pass

                        try:
                            if len(var_types['numeric'])>=2:
                                num1, num2 = var_types['numeric'][0], var_types['numeric'][1]
                                fig_line = px.line(df.head(1000), x=df.head(1000).index, y=[num1, num2], title=f"{num1}与{num2}趋势变化对比", width=800, height=400)
                                chart_data['图2'] = {'fig': fig_line, 'type': 'plotly', 'name': f'{num1}与{num2}趋势折线图', 'desc': f'展示{num1}和{num2}前1000条数据的时间序列趋势变化，可直观对比两者的波动规律与变化一致性'}
                        except Exception as e:
                            pass

                        try:
                            if var_types['categorical']:
                                cat_col = var_types['categorical'][0]
                                freq_df = freq_res[cat_col].reset_index().rename(columns={'index': cat_col})
                                fig_bar = px.bar(freq_df, x=cat_col, y='频数', title=f"{cat_col}频数分布", width=800, height=400, text_auto=True)
                                chart_data['图3'] = {'fig': fig_bar, 'type': 'plotly', 'name': f'{cat_col}频数分布条形图', 'desc': f'展示分类型变量{cat_col}各类型的频数与占比情况，可直观判断该变量的分布特征与主要类别构成'}
                        except Exception as e:
                            pass

                        chart_names = list(chart_data.keys())
                        chart_desc = "\n".join([f"{k}：{v['name']} - {v['desc']}" for k, v in chart_data.items()]) if chart_data else "无可用可视化图表"
                        real_stats = f"""【描述统计结果】：{desc_text}
【相关矩阵结果】：{corr_text}
【分类变量频数】：{freq_text}
【均值检验结果】：{ttest_text}
【可用可视化图表】：{chart_desc}"""

                        prompt = """你是资深科研数据分析专家，专注于基于真实统计结果和可视化图表生成标准化分析报告，严格按照以下要求输出，任何情况下不得改变格式、不得删减章节、不得编造任何数据/图表，仅基于提供的真实信息分析：
### 固定输出格式要求（必须严格遵守，章节顺序、标题层级、标点符号完全一致）：
# 数据统计分析报告
## 一、数据基本特征
1. 样本规模：明确说明数据的行数列数、整体缺失率，简要描述数据维度特征
2. 数值变量特征：基于描述统计结果，总结数值变量的均值、标准差、极值、缺失情况，指出数据的集中趋势与离散程度
3. 分类变量特征：基于频数分析结果，总结分类变量的主要类别、频数占比，描述分类变量的分布特征

## 二、可视化图表分析
{CHART_ANALYSIS}
要求：1. 有多少张图就分析多少张，每张图单独成段，以【X】开头（X为图1/图2/图3）；2. 先说明图表展示的核心内容，再结合统计结果解读图表反映的规律/特征/问题；3. 语言简洁专业，图表分析与真实数据高度契合；4. 无图表则写“本次分析无可用可视化图表，跳过本章节分析”

## 三、变量关系深度分析
1. 数值变量相关性：基于相关矩阵结果，分析变量间的线性相关程度、显著性，指出强相关/弱相关/无相关的变量组合
2. 组间均值差异：基于均值检验结果，分析二分类分组下数值变量的均值差异是否显著（p<0.05为显著，p<0.01为极显著），无检验结果则写“无符合条件的二分类变量，未执行均值检验，跳过本项分析”
3. 整体变量关系总结：综合上述分析，总结本次数据中变量间的核心关系规律

## 四、研究结论与建议
### （一）研究结论
基于本次全量统计分析与可视化结果，分3-5点客观总结数据反映的核心规律、特征、结论，每点一句话，简洁明确，仅基于真实分析结果，不做过度推断
### （二）研究建议
结合数据特征与变量关系，分2-4点给出针对性、可落地的研究/分析建议，建议需贴合数据实际，具有实际参考价值

### 输出约束：
1. 所有分析必须基于提供的真实统计结果和图表，绝对禁止编造任何数值、统计量、p值、图表信息；
2. 格式严格遵循上述要求，标题层级（#/##/###）、编号（1./2./3.）、标点（顿号/逗号/句号）完全一致；
3. 语言专业、简洁、客观，适配科研/数据分析场景，避免口语化；
4. 可视化图表分析部分，必须在对应的【图X】后紧跟分析内容，图的标识与提供的图表完全一致；
5. 温度系数已设为0.2，保证输出结果的一致性，多次分析同一数据需保持格式和核心内容高度统一。

### 本次分析真实统计与图表信息：
""" + real_stats + f"""
### 数据基础概况：
{data_overview}
### 核心要求重申：
1. 图表分析部分，分析到某张图时，仅需写出【图X】（无其他文字），后续由系统自动嵌入真实图表，你无需额外描述图表样式；
2. 严格按照固定格式输出，章节完整、层级清晰，多次分析格式完全统一；
3. 仅使用提供的真实数据，不编造任何内容。"""

                        st.markdown("### 📋 AI标准化分析报告（图文嵌排）")
                        report_placeholder = st.empty()
                        full_report = ""
                        stream = call_deepseek_api(prompt)
                        current_text = ""
                        for chunk in stream:
                            current_text += chunk
                            full_report += chunk
                            for chart in chart_names:
                                if chart in current_text:
                                    split_text = current_text.split(chart, 1)
                                    report_placeholder.markdown(split_text[0], unsafe_allow_html=True)
                                    if chart_data[chart]['type'] == 'matplotlib':
                                        st.pyplot(chart_data[chart]['fig'], key=f"plt_{chart}")
                                    else:
                                        st.plotly_chart(chart_data[chart]['fig'], use_container_width=True, key=f"plotly_{chart}")
                                    current_text = split_text[1]
                        if current_text:
                            report_placeholder.markdown(current_text, unsafe_allow_html=True)
            
            with st.expander("❓ AI统计问答（固定格式）", expanded=False):
                user_question = st.text_area("输入你的数据分析问题", placeholder="示例：分析A和B的相关性并解读；用t检验比较两组数据的均值差异；总结数据的核心分布特征", height=100)
                if st.button("💬 发送问题") and user_question:
                    st.markdown("### 📝 AI标准化解答")
                    q_prompt = """你是专业统计分析师，解答问题需严格遵循以下固定格式，语言专业简洁，仅基于数据概况分析，不编造任何内容：
## 问题解答：
1. 分析方法：明确解答该问题需使用的统计分析方法，说明方法适用场景
2. 操作步骤：分点说明使用该方法的具体操作步骤，适配本平台功能
3. 结果解读：说明该方法结果的判断标准（如p<0.05为显著），明确核心指标解读方式
4. 结论建议：基于数据概况，给出针对性的分析建议或注意事项

### 数据概况：
""" + data_overview + f"""
### 待解答问题：{user_question}
### 约束：
1. 严格遵循上述格式，不得删减章节，编号与标题完全一致；
2. 仅基于数据概况解答，不编造任何数据/变量/统计结果；
3. 语言专业、简洁，适配科研数据分析场景，多次解答格式统一。"""
                    stream = call_deepseek_api(q_prompt, temperature=0.2)
                    st.write_stream(stream)
            
            with st.expander("📈 AI结果解读（固定格式）", expanded=False):
                user_result = st.text_area("粘贴你的统计分析结果", placeholder="示例：皮尔逊相关系数0.78，p=0.001；t检验t=2.35，p=0.02；线性回归R²=0.82", height=100)
                if st.button("🔍 解读结果") and user_result:
                    st.markdown("### 📝 AI标准化结果解读")
                    r_prompt = """你是专业统计分析师，解读统计结果需严格遵循以下固定格式，语言专业简洁，逐点解读，不编造任何内容：
## 统计结果解读报告
1. 指标解读：逐点解读每个统计指标的核心含义，说明指标的统计意义
2. 显著性判断：明确每个结果的显著性水平（p<0.05为显著，p<0.01为极显著，p>0.05为不显著）
3. 实际意义：结合数据概况，解读每个结果的实际研究/分析意义，说明结果反映的问题/规律
4. 综合结论：综合所有结果，给出1-2句核心综合结论，简洁明确

### 数据概况：
""" + data_overview + f"""
### 待解读统计结果：{user_result}
### 约束：
1. 严格遵循上述格式，不得删减章节，编号与标题完全一致；
2. 逐点对应输入的统计结果，不遗漏、不编造；
3. 明确显著性判断标准，解读贴合数据实际；
4. 语言专业、简洁，适配科研数据分析场景，多次解读格式统一。"""
                    stream = call_deepseek_api(r_prompt, temperature=0.2)
                    st.write_stream(stream)
else:
    st.info("💡 请在【左侧边栏】上传CSV/Excel数据文件，即可开始全功能分析")
    st.markdown("#### 📌 核心功能亮点")
    st.markdown("- 集成SPSS核心统计功能，操作简易，结果精准")
    st.markdown("- AI分析支持**图文嵌排**，图表嵌入解答对应位置，排版美观")
    st.markdown("- AI输出**固定统一格式**，不同文件/多次分析格式高度一致")
    st.markdown("- 图表生成带异常捕获，单图失败不中断，自动跳过继续分析")
    st.markdown("- 所有分析基于真实统计结果，AI不编造任何数据/图表")
