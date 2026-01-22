import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import matplotlib.pyplot as plt
from scipy import stats, chi2_contingency
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
import warnings
import io
import re
from datetime import datetime
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="科研数据智能解读助手",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown("""
    <style>
    .stApp {background-color: var(--backgroundColor);font-family: var(--font);}
    .stButton > button {background-color: var(--primaryColor);color: white;border-radius: 8px;border: none;padding: 8px 16px;font-size: 14px;box-shadow: 0 2px 4px rgba(0,0,0,0.1);transition: all 0.3s ease;}
    .stButton > button:hover {background-color: #1976d2;box-shadow: 0 4px 8px rgba(0,0,0,0.15);}
    .card {background-color: white;border-radius: 12px;padding: 16px;margin: 8px 0;box-shadow: 0 1px 3px rgba(0,0,0,0.05);}
    .dataframe {border-radius: 8px !important;overflow: hidden !important;}
    .sidebar-header {font-size: 16px;font-weight: bold;color: var(--primaryColor);margin: 16px 0 8px 0;}
    .hint-text {font-size: 12px;color: #6c757d;margin-top: 4px;}
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

@st.cache_data(show_spinner="加载数据中...")
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
                        if encoding in ['utf-16']:
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
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'binary_categorical': binary_categorical_cols,
        'datetime': datetime_cols
    }

def generate_multiple_charts(analysis_type, params, df):
    charts = {}
    if analysis_type == "descriptive":
        col = params["target_col"]
        group_col = params.get("group_col", None)
        fig_hist = px.histogram(
            df, x=col, color=group_col,
            title=f"{col}分布直方图",
            color_discrete_sequence=[st.get_option("theme.primaryColor"), "#ff7f0e", "#2ca02c"],
            width=800, height=400,
            labels={col: col, group_col: group_col if group_col else None}
        )
        charts['histogram'] = fig_hist
        fig_box = alt.Chart(df).mark_boxplot(extent='min-max', color=st.get_option("theme.primaryColor")).encode(
            x=alt.X(group_col, title=group_col) if group_col else alt.value(""),
            y=alt.Y(col, title=col),
            tooltip=[alt.Tooltip(col, aggregate='mean', title='均值'), 
                     alt.Tooltip(col, aggregate='std', title='标准差')]
        ).properties(title=f"{col}箱线图（按{group_col}分组）" if group_col else f"{col}箱线图", width=800, height=400)
        charts['boxplot'] = fig_box
        if group_col:
            fig_density = alt.Chart(df).transform_density(
                col, groupby=[group_col],
                as_=[col, 'density']
            ).mark_area(opacity=0.6).encode(
                x=col, y='density:Q', color=group_col
            ).properties(title=f"{col}密度分布", width=800, height=400)
            charts['density'] = fig_density
    elif analysis_type == "correlation":
        corr_cols = params["corr_cols"]
        corr_matrix = df[corr_cols].corr()
        fig_heatmap = px.imshow(
            corr_matrix,
            title="变量相关性热力图",
            labels=dict(color="相关系数"),
            x=corr_cols, y=corr_cols,
            color_continuous_scale=[(0, "#ff4444"), (0.5, "#ffffff"), (1, "#00C851")],
            width=800, height=600
        )
        fig_heatmap.update_xaxes(side="bottom")
        charts['heatmap'] = fig_heatmap
        if len(corr_cols) >= 2:
            scatter_cols = corr_cols[:3]
            fig_scatter_matrix = alt.Chart(df).mark_point(opacity=0.6).encode(
                x=alt.X(alt.repeat("row"), type="quantitative"),
                y=alt.Y(alt.repeat("column"), type="quantitative"),
                tooltip=scatter_cols
            ).repeat(
                row=scatter_cols,
                column=scatter_cols
            ).properties(title="变量散点矩阵", width=200, height=200)
            charts['scatter_matrix'] = fig_scatter_matrix
    elif analysis_type == "regression":
        x_col, y_col = params["x_col"], params["y_col"]
        poly_degree = params.get("poly_degree", 1)
        df_reg = df[[x_col, y_col]].dropna()
        fig_reg = px.scatter(
            df_reg, x=x_col, y=y_col,
            title=f"{x_col}对{y_col}的回归分析",
            trendline="ols" if poly_degree == 1 else None,
            width=800, height=400,
            labels={x_col: x_col, y_col: y_col}
        )
        if poly_degree > 1:
            poly = PolynomialFeatures(degree=poly_degree)
            x_poly = poly.fit_transform(df_reg[[x_col]])
            model = ols(f"{y_col} ~ x_poly", data=df_reg).fit()
            x_range = np.linspace(df_reg[x_col].min(), df_reg[x_col].max(), 100)
            x_range_poly = poly.transform(x_range.reshape(-1, 1))
            y_pred = model.predict({"x_poly": x_range_poly, y_col: 0})
            fig_reg.add_trace(go.Scatter(x=x_range, y=y_pred, mode="lines", name=f"多项式趋势线（degree={poly_degree}）"))
        charts['regression'] = fig_reg
        model = ols(f"{y_col} ~ {x_col}", data=df_reg).fit()
        residuals = model.resid
        fig_resid, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(df_reg[x_col], residuals, alpha=0.6, color=st.get_option("theme.primaryColor"))
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel(x_col)
        ax.set_ylabel("残差")
        ax.set_title(f"{y_col}回归残差图（残差~{x_col}）")
        charts['residual'] = fig_resid
    elif analysis_type == "kmeans":
        feature_cols = params["feature_cols"]
        n_clusters = params["n_clusters"]
        df_cluster = df[feature_cols].dropna()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(df_cluster)
        df_cluster["cluster"] = kmeans.labels_
        fig_2d = px.scatter(
            df_cluster, x=feature_cols[0], y=feature_cols[1], color="cluster",
            title=f"K-Means聚类结果（K={n_clusters}）",
            color_discrete_sequence=px.colors.qualitative.Set3,
            width=800, height=400,
            labels={feature_cols[0]: feature_cols[0], feature_cols[1]: feature_cols[1]}
        )
        charts['kmeans_2d'] = fig_2d
        if len(feature_cols) >= 3:
            fig_3d = px.scatter_3d(
                df_cluster, x=feature_cols[0], y=feature_cols[1], z=feature_cols[2], color="cluster",
                title=f"K-Means聚类3D展示（K={n_clusters}）",
                color_discrete_sequence=px.colors.qualitative.Set3,
                width=800, height=600
            )
            charts['kmeans_3d'] = fig_3d
    elif analysis_type == "time_series":
        date_col = params["date_col"]
        value_col = params["value_col"]
        group_col = params.get("group_col", None)
        fig_line = px.line(
            df, x=date_col, y=value_col, color=group_col,
            title=f"{value_col}时间趋势",
            color_discrete_sequence=[st.get_option("theme.primaryColor"), "#ff7f0e"],
            width=1000, height=400,
            range_x=[df[date_col].min(), df[date_col].max()],
            labels={date_col: "日期", value_col: value_col}
        )
        fig_line.update_xaxes(rangeslider_visible=True)
        charts['time_line'] = fig_line
        if group_col:
            fig_area = alt.Chart(df).mark_area(opacity=0.6).encode(
                x=date_col,
                y=alt.Y(value_col, aggregate='mean', title=f"{value_col}均值"),
                y2=alt.Y2(f"{value_col}:Q", aggregate='min'),
                y3=alt.Y3(f"{value_col}:Q", aggregate='max'),
                color=group_col
            ).properties(title=f"{value_col}时间趋势（均值±最值）", width=1000, height=400)
            charts['time_area'] = fig_area
    elif analysis_type == "geo_distribution":
        lon_col = params["lon_col"]
        lat_col = params["lat_col"]
        value_col = params["value_col"]
        df_geo = df[[lon_col, lat_col, value_col]].dropna()
        df_geo.columns = ['lon', 'lat', 'value']
        charts['geo_map'] = df_geo
        fig_geo = px.scatter_mapbox(
            df_geo, lat='lat', lon='lon', size='value', color='value',
            title=f"{value_col}地理分布",
            color_continuous_scale=px.colors.sequential.Bluered,
            mapbox_style="carto-positron",
            zoom=3, width=1000, height=600,
            labels={'value': value_col}
        )
        charts['geo_plotly'] = fig_geo
    return charts

st.title("📊 科研数据智能解读助手")
st.markdown("**低代码操作 · 多方法分析 · 多图表可视化**")
st.divider()

with st.sidebar:
    st.markdown('<div class="sidebar-header">1. 上传数据文件</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "支持 CSV/Excel（可上传多个）",
        type=["xlsx", "csv"],
        accept_multiple_files=True
    )
    st.markdown('<div class="hint-text">示例：df_order.csv（订单数据）、df_loc.csv（城市地理数据）</div>', unsafe_allow_html=True)
    if uploaded_files:
        st.markdown('<div class="sidebar-header">2. 选择分析文件</div>', unsafe_allow_html=True)
        selected_files = st.multiselect(
            "勾选要参与分析的文件",
            [f.name for f in uploaded_files],
            default=[uploaded_files[0].name]
        )
        selected_file_objs = [f for f in uploaded_files if f.name in selected_files]
        df_dict = {}
        for file in selected_file_objs:
            df = load_and_clean_data(file)
            if df is not None:
                df_dict[file.name] = df
        if len(df_dict) >= 2:
            st.markdown('<div class="sidebar-header">3. 多文件关联</div>', unsafe_allow_html=True)
            base_file = st.selectbox("选择基础文件", list(df_dict.keys()))
            df = df_dict[base_file]
            for other_file in [f for f in df_dict.keys() if f != base_file]:
                df_other = df_dict[other_file]
                base_key = st.selectbox(f"基础文件[{base_file}]关联字段", df.columns, key=f"base_key_{other_file}")
                other_key = st.selectbox(f"关联文件[{other_file}]关联字段", df_other.columns, key=f"other_key_{other_file}")
                if st.button(f"关联[{other_file}]", key=f"join_btn_{other_file}"):
                    df = pd.merge(df, df_other, left_on=base_key, right_on=other_key, how="left", suffixes=("", f"_{other_file.split('.')[0]}"))
                    st.success(f"✅ 已关联[{other_file}]，当前数据：{len(df)}行 × {len(df.columns)}列")
        else:
            df = df_dict[list(df_dict.keys())[0]]
        var_types = identify_variable_types(df)
        st.markdown('<div class="sidebar-header">4. 变量类型识别</div>', unsafe_allow_html=True)
        st.write(f"📈 数值型：{', '.join(var_types['numeric'][:5])}{'...' if len(var_types['numeric'])>5 else ''}")
        st.write(f"🏷️ 分类型：{', '.join(var_types['categorical'][:5])}{'...' if len(var_types['categorical'])>5 else ''}")
        st.write(f"⏰ 时间型：{', '.join(var_types['datetime']) if var_types['datetime'] else '无'}")
        st.write(f"🔑 二分类：{', '.join(var_types['binary_categorical']) if var_types['binary_categorical'] else '无'}")

if 'df' in locals():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("数据预览（前5行）")
        st.dataframe(df.head(), use_container_width=True, height=200)
    with col2:
        st.subheader("数据概况")
        st.markdown(f"""
        <div class="card">
        <p>📊 数据规模：{len(df)} 行 × {len(df.columns)} 列</p>
        <p>❌ 缺失值：{df.isnull().sum().sum()} 个（{df.isnull().sum().sum()/(len(df)*len(df.columns))*100:.1f}%）</p>
        <p>📈 数值列：{len(var_types['numeric'])} 个</p>
        <p>🏷️ 分类列：{len(var_types['categorical'])} 个</p>
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    st.subheader("选择分析类型")
    analysis_options = [
        "1. 描述性统计（均值/分布）",
        "2. 相关性分析（变量关系）",
        "3. 两组差异检验（t检验/卡方）",
        "4. 多因素方差分析（ANOVA）",
        "5. 回归分析（线性/多项式）",
        "6. 逻辑回归（分类预测）",
        "7. K-Means聚类（数据分群）",
        "8. 时间序列分析（趋势）",
        "9. 地理分布分析（地图）"
    ]
    analysis_type = st.radio("选择要执行的分析", analysis_options)
    analysis_key = analysis_type.split(".")[0].strip()
    st.subheader("配置分析参数")
    params = {}
    if analysis_key == "1":
        params["target_col"] = st.selectbox("选择要分析的数值变量", var_types['numeric'])
        params["group_col"] = st.selectbox("选择分组变量（可选）", [None] + var_types['categorical'])
    elif analysis_key == "2":
        params["corr_cols"] = st.multiselect("选择要分析的数值变量（至少2个）", var_types['numeric'], default=var_types['numeric'][:3])
        if len(params["corr_cols"]) < 2:
            st.warning("⚠️ 请至少选择2个数值变量")
    elif analysis_key == "3":
        test_type = st.radio("选择检验类型", ["t检验（数值型结果）", "卡方检验（分类型结果）"])
        params["test_type"] = test_type
        params["group_col"] = st.selectbox("选择分组变量（二分类）", var_types['binary_categorical'])
        if test_type == "t检验（数值型结果）":
            params["result_col"] = st.selectbox("选择结果变量（数值型）", var_types['numeric'])
        else:
            params["result_col"] = st.selectbox("选择结果变量（分类型）", var_types['categorical'])
    elif analysis_key == "4":
        params["factor_cols"] = st.multiselect("选择因素变量（分类型）", var_types['categorical'], default=var_types['categorical'][:2])
        params["result_col"] = st.selectbox("选择结果变量（数值型）", var_types['numeric'])
    elif analysis_key == "5":
        reg_type = st.radio("选择回归类型", ["线性回归", "多项式回归"])
        params["reg_type"] = reg_type
        params["x_col"] = st.selectbox("选择自变量（数值型）", var_types['numeric'])
        params["y_col"] = st.selectbox("选择因变量（数值型）", [c for c in var_types['numeric'] if c != params["x_col"]])
        if reg_type == "多项式回归":
            params["poly_degree"] = st.slider("多项式次数", 2, 5, 2)
    elif analysis_key == "6":
        params["target_col"] = st.selectbox("选择预测目标（二分类）", var_types['binary_categorical'])
        params["feature_cols"] = st.multiselect("选择特征变量（数值型）", var_types['numeric'], default=var_types['numeric'][:2])
    elif analysis_key == "7":
        params["feature_cols"] = st.multiselect("选择聚类特征（数值型）", var_types['numeric'], default=var_types['numeric'][:2])
        params["n_clusters"] = st.slider("聚类数量（K）", 2, 10, 3)
    elif analysis_key == "8":
        if not var_types['datetime']:
            st.error("⚠️ 未识别到时间型变量，请上传含日期列的数据（如 df_past_order.csv）")
        else:
            params["date_col"] = st.selectbox("选择日期变量", var_types['datetime'])
            params["value_col"] = st.selectbox("选择要分析的数值变量", var_types['numeric'])
            params["group_col"] = st.selectbox("选择分组变量（可选）", [None] + var_types['categorical'])
    elif analysis_key == "9":
        lon_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['lon', '经度'])]
        lat_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['lat', '纬度'])]
        if not lon_cols or not lat_cols:
            st.error("⚠️ 未识别到经纬度变量，请上传含经纬度的文件（如 df_loc.csv）")
        else:
            params["lon_col"] = st.selectbox("选择经度列", lon_cols)
            params["lat_col"] = st.selectbox("选择纬度列", lat_cols)
            params["value_col"] = st.selectbox("选择要展示的数值变量（如订单量）", var_types['numeric'])
    if st.button("🚀 开始分析", type="primary"):
        with st.spinner("分析中..."):
            if analysis_key == "1":
                st.subheader("📊 描述性统计结果")
                col = params["target_col"]
                group_col = params["group_col"]
                if group_col:
                    stats_table = df.groupby(group_col)[col].agg(['count', 'mean', 'std', 'min', 'max', 'median']).round(2)
                    stats_table.columns = ['样本数', '均值', '标准差', '最小值', '最大值', '中位数']
                else:
                    stats_table = df[col].agg(['count', 'mean', 'std', 'min', 'max', 'median']).round(2)
                    stats_table = pd.DataFrame(stats_table, columns=[col]).T
                st.dataframe(stats_table, use_container_width=True)
                charts = generate_multiple_charts("descriptive", params, df)
                for chart_name, chart in charts.items():
                    st.subheader(f"📈 {chart_name.capitalize()}")
                    if isinstance(chart, alt.Chart):
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.plotly_chart(chart, use_container_width=True)
                st.subheader("📝 结果解读")
                st.markdown(f"""
                <div class="card">
                1. 核心统计：{col}的均值为{stats_table['均值'].iloc[0]:.2f}，标准差为{stats_table['标准差'].iloc[0]:.2f}，数据{'较集中' if stats_table['标准差'].iloc[0] < stats_table['均值'].iloc[0]*0.3 else '较分散'}；<br>
                2. 数据范围：最小值{stats_table['最小值'].iloc[0]:.2f}，最大值{stats_table['最大值'].iloc[0]:.2f}，极差为{stats_table['最大值'].iloc[0]-stats_table['最小值'].iloc[0]:.2f}；<br>
                3. 分组差异：{f'按{group_col}分组时，{stats_table.index[stats_table["均值"].idxmax()]}的{col}均值最高（{stats_table["均值"].max():.2f}）' if group_col else '无分组差异分析'}。
                </div>
                """, unsafe_allow_html=True)
            elif analysis_key == "2" and len(params["corr_cols"]) >= 2:
                st.subheader("🔗 相关性分析结果")
                corr_matrix = df[params["corr_cols"]].corr().round(3)
                charts = generate_multiple_charts("correlation", params, df)
                for chart_name, chart in charts.items():
                    st.subheader(f"📈 {chart_name.capitalize()}")
                    if isinstance(chart, alt.Chart):
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.plotly_chart(chart, use_container_width=True)
                st.subheader("显著相关性（|r| > 0.5）")
                corr_significant = corr_matrix[(abs(corr_matrix) > 0.5) & (corr_matrix != 1.0)].stack().drop_duplicates()
                if not corr_significant.empty:
                    st.dataframe(corr_significant.round(3), use_container_width=True)
                else:
                    st.info("⚠️ 未发现绝对值大于0.5的显著相关性")
                st.subheader("📝 结果解读")
                st.markdown(f"""
                <div class="card">
                1. 最强正相关：{corr_matrix.max().idxmax()}与{corr_matrix.idxmax()[corr_matrix.max().idxmax()]}的相关系数为{corr_matrix.max().max():.3f}；<br>
                2. 最强负相关：{corr_matrix.min().idxmin()}与{corr_matrix.idxmin()[corr_matrix.min().idxmin()]}的相关系数为{corr_matrix.min().min():.3f}；<br>
                3. 科研建议：{f'{corr_matrix.max().idxmax()}与{corr_matrix.idxmax()[corr_matrix.max().idxmax()]}高度正相关，可进一步做回归分析探索因果关系' if corr_matrix.max().max() > 0.7 else '无高度相关变量，需结合其他分析方法'}。
                </div>
                """, unsafe_allow_html=True)
            elif analysis_key == "3":
                st.subheader("🔍 两组差异检验结果")
                group_col = params["group_col"]
                result_col = params["result_col"]
                group1, group2 = df[group_col].unique()[:2]
                df_filtered = df[df[group_col].isin([group1, group2])]
                if params["test_type"] == "t检验（数值型结果）":
                    data1 = df_filtered[df_filtered[group_col] == group1][result_col].dropna()
                    data2 = df_filtered[df_filtered[group_col] == group2][result_col].dropna()
                    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                    st.write(f"分组1（{group1}）：样本数={len(data1)}，均值={data1.mean():.2f}，标准差={data1.std():.2f}")
                    st.write(f"分组2（{group2}）：样本数={len(data2)}，均值={data2.mean():.2f}，标准差={data2.std():.2f}")
                    st.write(f"t统计量：{t_stat:.4f}，p值：{p_value:.4f}")
                    st.write(f"结论：{'存在显著差异' if p_value < 0.05 else '无显著差异'}（α=0.05）")
                    fig_box = px.box(df_filtered, x=group_col, y=result_col, title=f"{result_col}两组差异箱线图")
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    contingency_table = pd.crosstab(df_filtered[group_col], df_filtered[result_col])
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                    st.write("列联表：")
                    st.dataframe(contingency_table, use_container_width=True)
                    st.write(f"卡方统计量：{chi2_stat:.4f}，p值：{p_value:.4f}，自由度：{dof}")
                    st.write(f"结论：{'两组分布存在显著差异' if p_value < 0.05 else '两组分布无显著差异'}（α=0.05）")
                st.subheader("📝 结果解读")
                st.markdown(f"""
                <div class="card">
                1. 检验类型：{params['test_type']}，分组变量为{group_col}（{group1} vs {group2}）；<br>
                2. 统计结论：{'两组在{result_col}上存在显著差异，可认为分组是导致差异的原因之一' if p_value < 0.05 else '未发现两组在{result_col}上的显著差异，差异可能由随机因素导致'}；<br>
                3. 科研建议：{'建议进一步探究分组变量对结果的影响机制' if p_value < 0.05 else '可尝试增加样本量或更换分组变量重新检验'}。
                </div>
                """, unsafe_allow_html=True)
            st.divider()
            st.subheader("📥 下载分析报告")
            report_content = f"# 科研数据分析报告\n## 分析类型：{analysis_type}\n## 数据概况：{len(df)}行 × {len(df.columns)}列\n## 核心结论：{st.session_state.get('report_conclusion', '详见上述分析')}"
            st.download_button(
                label="下载 Markdown 报告",
                data=report_content,
                file_name=f"科研数据分析报告_{datetime.now().strftime('%Y%m%d%H%M')}.md",
                mime="text/markdown"
            )
else:
    st.info("💡 请在侧边栏上传数据文件，支持多文件关联分析")
