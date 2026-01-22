import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
import io
import re
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="科研数据分析助手-增强版",
    page_icon="📊",
    layout="wide"
)
st.title("📊 科研数据分析助手-增强版")
st.markdown("**支持单文件分析+多文件逐步关联分析+自定义图表**")
st.divider()

st.subheader("第一步：上传数据文件（支持多文件逐步关联）")
uploaded_files = st.file_uploader(
    "支持Excel(.xlsx)或CSV(.csv)文件，可上传多个", 
    type=["xlsx", "csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("💡 示例：上传客户信息、订单数据、城市对照表等，支持多文件逐步关联分析")
    st.stop()

df_list = []
file_names = []
encodings = ['utf-8-sig', 'gbk', 'gb2312', 'utf-8', 'big5', 'utf-16', 'gb18030']
seps = [',', '\t', ';', '|', ' ', '=', ':', '\s+']

def clean_column_names(df):
    df.columns = [re.sub(r'[^\w\s\u4e00-\u9fa5]', '', str(col)).strip() for col in df.columns]
    df.columns = [col if col else f"col_{i}" for i, col in enumerate(df.columns)]
    return df

def fix_df_list_columns(df):
    if len(df.columns) >= 2:
        df.columns = ['Location', '中文名称']
    return df

for file in uploaded_files:
    try:
        file_content = file.read()
        if len(file_content) == 0:
            raise ValueError("文件为空，无法读取")
        file.seek(0)
        df = None
        file_name = file.name
        
        if file_name == "df_list.csv":
            try:
                df = pd.read_csv(file, encoding='gbk', sep=',', on_bad_lines='skip')
                df = fix_df_list_columns(df)
            except Exception as e:
                st.warning(f"⚠️ 尝试GBK编码失败，自动降级为utf-8-sig：{str(e)}")
                df = pd.read_csv(file, encoding='utf-8-sig', sep=',', on_bad_lines='skip')
                df = fix_df_list_columns(df)
        else:
            for encoding in encodings:
                for sep in seps:
                    try:
                        if encoding == 'utf-16':
                            content = file_content.decode(encoding, errors='replace')
                            df = pd.read_csv(io.StringIO(content), sep=sep)
                        else:
                            df = pd.read_csv(file, encoding=encoding, sep=sep, on_bad_lines='skip')
                        df = clean_column_names(df)
                        break
                    except:
                        continue
                if df is not None:
                    break
            if df is None:
                raise ValueError("所有编码/分隔符尝试均失败，无法读取该CSV文件")
        
        if df is not None:
            df_list.append(df)
            file_names.append(file_name)
            st.success(f"✅ 成功读取文件：{file_name}（行数：{len(df)}，列数：{len(df.columns)}）")
    except Exception as e:
        st.error(f"❌ 读取文件{file.name}失败：{str(e)}")

if not df_list:
    st.error("❌ 没有成功读取任何文件，请检查文件格式")
    st.stop()

st.subheader("第二步：选择分析模式")
analysis_mode = st.radio(
    "选择分析模式",
    options=["单文件独立分析", "多文件逐步关联分析"]
)

if analysis_mode == "单文件独立分析":
    selected_file_idx = st.selectbox("选择要分析的文件", range(len(file_names)), format_func=lambda x: file_names[x])
    df = df_list[selected_file_idx]
    st.success(f"✅ 已选择单文件：{file_names[selected_file_idx]}")

else:
    if len(file_names) < 2:
        st.error("❌ 多文件关联分析至少需要上传2个文件！")
        st.stop()
    
    base_file_idx = st.selectbox("选择基础文件（后续所有文件将关联到该文件）", range(len(file_names)), format_func=lambda x: file_names[x])
    df = df_list[base_file_idx]
    base_file_name = file_names[base_file_idx]
    st.success(f"✅ 已选择基础文件：{base_file_name}（当前数据：{len(df)}行 × {len(df.columns)}列）")
    
    remaining_file_idxs = [i for i in range(len(file_names)) if i != base_file_idx]
    remaining_file_names = [file_names[i] for i in remaining_file_idxs]
    
    for i in range(len(remaining_file_idxs)):
        st.markdown(f"### 关联第{i+1}个文件")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            file_idx = st.selectbox(f"选择要关联的文件 {i+1}", remaining_file_idxs, format_func=lambda x: file_names[x], key=f"file_{i}")
            file_name = file_names[file_idx]
            df_to_join = df_list[file_idx]
        
        with col2:
            base_key = st.selectbox(f"基础文件[{base_file_name}]的关联字段", df.columns.tolist(), key=f"base_key_{i}")
        
        with col3:
            join_key = st.selectbox(f"关联文件[{file_name}]的关联字段", df_to_join.columns.tolist(), key=f"join_key_{i}")
        
        join_type = st.radio(f"选择关联方式（文件 {i+1}）", options=["左关联（保留基础文件数据）", "内关联（仅保留匹配数据）"], key=f"join_type_{i}")
        join_map = {"左关联（保留基础文件数据）": "left", "内关联（仅保留匹配数据）": "inner"}
        
        # 修复合并错误：用索引生成稳定后缀，避免文件名冲突
        base_suffix = f"_base_{base_file_idx}"
        join_suffix = f"_join_{i+1}"
        
        # 合并前重命名非关联字段，避免列名冲突
        df_to_join_renamed = df_to_join.rename(columns={col: f"{col}{join_suffix}" for col in df_to_join.columns if col != join_key})
        df_renamed = df.rename(columns={col: f"{col}{base_suffix}" for col in df.columns if col != base_key})
        
        try:
            df_merged = pd.merge(
                df_renamed, 
                df_to_join_renamed, 
                left_on=f"{base_key}{base_suffix}" if base_key != base_key+base_suffix else base_key,
                right_on=join_key, 
                how=join_map[join_type]
            )
            # 恢复基础文件的关联字段名
            df_merged = df_merged.rename(columns={f"{base_key}{base_suffix}": base_key})
            df = df_merged
        except pd.errors.MergeError as e:
            st.error(f"❌ 合并失败：列名冲突，请检查关联字段或尝试更换关联方式。错误详情：{str(e)}")
            break
        
        st.success(f"✅ 关联完成！{base_file_name}[{base_key}] ↔ {file_name}[{join_key}]，当前数据：{len(df)}行 × {len(df.columns)}列")
        
        remaining_file_idxs.remove(file_idx)
        if not remaining_file_idxs:
            break

st.subheader("数据预览（前5行）")
st.dataframe(df.head(), use_container_width=True)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
st.subheader("数据变量识别")
st.write(f"📈 数值型变量：{', '.join(numeric_cols) if numeric_cols else '无'}")
st.write(f"🏷️ 分类变量：{', '.join(categorical_cols) if categorical_cols else '无'}")

if not numeric_cols:
    st.error("❌ 未识别到数值型变量！请确保文件包含可计算的数值列（如销量、金额、数量等）")
    st.stop()

st.divider()
st.subheader("第三步：选择分析类型")
analysis_type = st.radio(
    "选择分析类型",
    options=[
        "描述性统计（均值/分布）", 
        "独立样本t检验（两组差异）", 
        "多因素方差分析（ANOVA）",
        "简单线性回归（变量关系）",
        "逻辑回归（分类预测）",
        "K-Means聚类（数据分组）"
    ]
)

type_map = {
    "描述性统计（均值/分布）": "descriptive",
    "独立样本t检验（两组差异）": "t_test",
    "多因素方差分析（ANOVA）": "anova",
    "简单线性回归（变量关系）": "regression",
    "逻辑回归（分类预测）": "logistic_reg",
    "K-Means聚类（数据分组）": "kmeans"
}
target_analysis = type_map[analysis_type]

st.subheader("第四步：配置分析参数+图表自定义")
params = {}
st.markdown("### 🎨 图表自定义设置")
params["chart_color"] = st.color_picker("选择图表主色调", value="#1f77b4")
params["chart_width"] = st.slider("图表宽度（像素）", min_value=600, max_value=1200, value=800)
params["chart_height"] = st.slider("图表高度（像素）", min_value=400, max_value=800, value=500)

if target_analysis == "kmeans":
    params["n_clusters"] = st.slider("聚类数量（K值）", min_value=2, max_value=10, value=3)

if target_analysis == "descriptive":
    params["target_col"] = st.selectbox("选择要分析的数值变量", numeric_cols)
    params["chart_type"] = st.radio("选择图表类型", ["直方图（分布）", "柱状图（均值）"])

elif target_analysis == "t_test":
    if not categorical_cols:
        st.error("❌ 未识别到分类变量！无法进行t检验（需有分组列如Location、客户类型等）")
        st.stop()
    params["group_col"] = st.selectbox("选择分组变量（如Location、中文名称、客户类型）", categorical_cols)
    params["result_col"] = st.selectbox("选择要比较的数值变量", numeric_cols)
    group_counts = df[params["group_col"]].nunique()
    if group_counts != 2:
        st.warning(f"⚠️ 分组变量有{group_counts}组，自动取样本量前2的组进行检验")
        top2_groups = df[params["group_col"]].value_counts().nlargest(2).index.tolist()
        df = df[df[params["group_col"]].isin(top2_groups)]

elif target_analysis == "anova":
    if len(categorical_cols) < 1:
        st.error("❌ 未识别到分类变量！无法进行方差分析（需有因素列如Location、中文名称）")
        st.stop()
    params["factor_cols"] = st.multiselect("选择因素变量（分类变量，可多选）", categorical_cols, default=categorical_cols[0])
    params["result_col"] = st.selectbox("选择因变量（数值变量）", numeric_cols)
    params["formula"] = f"{params['result_col']} ~ {' + '.join(params['factor_cols'])}"

elif target_analysis == "regression":
    if len(numeric_cols) < 2:
        st.error("❌ 至少需要2个数值变量！无法进行线性回归")
        st.stop()
    params["x_col"] = st.selectbox("选择自变量", numeric_cols)
    params["y_col"] = st.selectbox("选择因变量", [col for col in numeric_cols if col != params["x_col"]])

elif target_analysis == "logistic_reg":
    binary_cats = [col for col in categorical_cols if df[col].nunique() == 2]
    if not binary_cats:
        st.error("❌ 未识别到二分类变量！逻辑回归需因变量为二分类（如：是/否）")
        st.stop()
    params["target_col"] = st.selectbox("选择预测目标（二分类变量）", binary_cats)
    params["feature_cols"] = st.multiselect("选择特征变量（数值型）", numeric_cols, default=numeric_cols[:2])
    le = LabelEncoder()
    df[params["target_col"] + "_encoded"] = le.fit_transform(df[params["target_col"]])

elif target_analysis == "kmeans":
    params["feature_cols"] = st.multiselect("选择聚类特征变量（数值型）", numeric_cols, default=numeric_cols[:2])
    df_cluster = df[params["feature_cols"]].dropna()
    if len(df_cluster) < params["n_clusters"]:
        st.error(f"❌ 有效样本数（{len(df_cluster)}）小于聚类数量（{params['n_clusters']}）！请减少K值或选择其他数值变量")
        st.stop()

st.divider()
st.subheader("第五步：分析结果与智能报告")

if st.button("🚀 开始分析"):
    try:
        with st.spinner("正在分析中..."):
            report = ""
            if target_analysis == "descriptive":
                col = params["target_col"]
                stats_result = df[col].describe()
                st.subheader("📊 描述性统计结果")
                st.dataframe(stats_result.to_frame(), use_container_width=True)
                
                st.subheader("📈 可视化图表（自定义样式）")
                if params["chart_type"] == "直方图（分布）":
                    fig = px.histogram(
                        df, x=col, title=f"{col}的分布情况", nbins=20,
                        color_discrete_sequence=[params["chart_color"]],
                        width=params["chart_width"], height=params["chart_height"]
                    )
                else:
                    fig = px.bar(
                        df, y=col, title=f"{col}的均值分布",
                        color_discrete_sequence=[params["chart_color"]],
                        width=params["chart_width"], height=params["chart_height"]
                    )
                st.plotly_chart(fig, use_container_width=True)
                
                report = f"""
                ### 📝 分析报告
                1. 变量「{col}」核心统计：均值{stats_result['mean']:.2f}、中位数{stats_result['50%']:.2f}、标准差{stats_result['std']:.2f}；
                2. 数据分布：{'均匀' if stats_result['std'] < stats_result['mean']*0.3 else '分散'}，整体水平{stats_result['mean']:.2f}；
                3. 数据范围：最小值{stats_result['min']:.2f}，最大值{stats_result['max']:.2f}。
                """

            elif target_analysis == "t_test":
                group_col = params["group_col"]
                result_col = params["result_col"]
                group1, group2 = df[group_col].unique()[:2]
                data1 = df[df[group_col] == group1][result_col].dropna()
                data2 = df[df[group_col] == group2][result_col].dropna()
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                mean1, mean2 = data1.mean(), data2.mean()
                
                st.subheader("🔍 独立样本t检验结果")
                st.write(f"分组1（{group1}）均值：{mean1:.2f}，分组2（{group2}）均值：{mean2:.2f}")
                st.write(f"t统计量：{t_stat:.4f}，p值：{p_value:.4f}")
                
                fig = px.box(
                    df, x=group_col, y=result_col, title=f"{group_col}对{result_col}的差异分析",
                    color_discrete_sequence=[params["chart_color"]],
                    width=params["chart_width"], height=params["chart_height"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                significance = "显著" if p_value < 0.05 else "不显著"
                report = f"""
                ### 📝 分析报告
                1. 检验结论：{group1}与{group2}在{result_col}上的差异{significance}（p={p_value:.4f}）；
                2. 差异幅度：{group1}比{group2} {'高' if mean1>mean2 else '低'} {abs(mean1-mean2):.2f}；
                3. 统计依据：独立样本t检验（方差不齐），p<0.05代表差异有统计学意义。
                """

            elif target_analysis == "anova":
                formula = params["formula"]
                model = ols(formula, data=df).fit()
                anova_result = anova_lm(model, typ=2)
                
                st.subheader("📊 多因素方差分析结果")
                st.dataframe(anova_result, use_container_width=True)
                
                fig = px.box(
                    df, x=params["factor_cols"][0], y=params["result_col"], 
                    color=params["factor_cols"][1] if len(params["factor_cols"])>1 else None,
                    title=f"各因素对{params['result_col']}的影响分析",
                    color_discrete_sequence=[params["chart_color"]] if len(params["factor_cols"])==1 else None,
                    width=params["chart_width"], height=params["chart_height"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                significant_factors = [idx for idx, p in anova_result["PR(>F)"].items() if p < 0.05]
                report = f"""
                ### 📝 分析报告
                1. 方差分析公式：{formula}；
                2. 显著影响因素（p<0.05）：{', '.join(significant_factors) if significant_factors else '无'}；
                3. 结论：{f'因素{significant_factors}对{params["result_col"]}有显著影响' if significant_factors else '所有因素对因变量无显著影响'}；
                4. 统计依据：p<0.05代表该因素对结果的影响有统计学意义。
                """

            elif target_analysis == "regression":
                x_col, y_col = params["x_col"], params["y_col"]
                df_reg = df[[x_col, y_col]].dropna()
                model = ols(f"{y_col} ~ {x_col}", data=df_reg).fit()
                r_squared = model.rsquared
                coef = model.params[x_col]
                p_value = model.pvalues[x_col]
                
                st.subheader("📈 简单线性回归结果")
                st.write(f"回归方程：{y_col} = {model.params[0]:.2f} + {coef:.4f}×{x_col}")
                st.write(f"决定系数R²：{r_squared:.4f}，p值：{p_value:.4f}")
                
                fig = px.scatter(
                    df_reg, x=x_col, y=y_col, trendline="ols", title=f"{x_col}对{y_col}的回归分析",
                    color_discrete_sequence=[params["chart_color"]],
                    width=params["chart_width"], height=params["chart_height"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                significance = "显著" if p_value < 0.05 else "不显著"
                report = f"""
                ### 📝 分析报告
                1. 变量关系：{x_col}对{y_col}的影响{significance}（p={p_value:.4f}）；
                2. 回归系数：{coef:.4f}，说明{x_col}每增加1，{y_col} {'增加' if coef>0 else '减少'} {abs(coef):.4f}；
                3. 拟合程度：R²={r_squared:.4f}，说明{x_col}能解释{y_col} {r_squared*100:.1f}%的变化；
                4. 统计依据：p<0.05代表回归系数有统计学意义，R²越接近1拟合效果越好。
                """

            elif target_analysis == "logistic_reg":
                target_col = params["target_col"]
                feature_cols = params["feature_cols"]
                df_log = df[[*feature_cols, target_col + "_encoded"]].dropna()
                
                model = LogisticRegression()
                model.fit(df_log[feature_cols], df_log[target_col + "_encoded"])
                accuracy = model.score(df_log[feature_cols], df_log[target_col + "_encoded"])
                coefs = dict(zip(feature_cols, model.coef_[0]))
                
                st.subheader("🔮 逻辑回归（分类预测）结果")
                st.write(f"模型准确率：{accuracy:.4f}（即预测正确的样本占比）")
                st.write("各特征系数（系数越大，对预测结果影响越强）：")
                st.dataframe(pd.DataFrame({"特征": coefs.keys(), "系数": coefs.values()}), use_container_width=True)
                
                fig = px.bar(
                    x=coefs.keys(), y=coefs.values(), title="特征重要性（逻辑回归系数）",
                    color_discrete_sequence=[params["chart_color"]],
                    width=params["chart_width"], height=params["chart_height"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                report = f"""
                ### 📝 分析报告
                1. 模型性能：逻辑回归模型准确率{accuracy:.4f}（越高预测效果越好）；
                2. 特征影响：{max(coefs, key=coefs.get)}对{target_col}的影响最大（系数{coefs[max(coefs, key=coefs.get)]:.4f}）；
                3. 结论：模型可用于{target_col}的二分类预测，准确率{accuracy*100:.1f}%；
                4. 系数解读：正系数代表该特征值越大，越倾向于预测为“1”类；负系数则相反。
                """

            elif target_analysis == "kmeans":
                feature_cols = params["feature_cols"]
                n_clusters = params["n_clusters"]
                df_cluster = df[feature_cols].dropna()
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                df_cluster["聚类标签"] = kmeans.fit_predict(df_cluster[feature_cols])
                df["聚类标签"] = df_cluster["聚类标签"].reindex(df.index)
                
                st.subheader("🌀 K-Means聚类结果")
                st.write(f"聚类数量（K值）：{n_clusters}，各聚类样本数：")
                st.dataframe(df["聚类标签"].value_counts(), use_container_width=True)
                
                fig = px.scatter(
                    df_cluster, x=feature_cols[0], y=feature_cols[1], color="聚类标签",
                    title=f"K-Means聚类结果（K={n_clusters}）",
                    color_discrete_sequence=[params["chart_color"], "#ff7f0e", "#2ca02c", "#d62728"][:n_clusters],
                    width=params["chart_width"], height=params["chart_height"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                centers = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)
                st.subheader("各聚类中心（代表该类的特征均值）")
                st.dataframe(centers, use_container_width=True)
                
                report = f"""
                ### 📝 分析报告
                1. 聚类结果：数据被分为{n_clusters}个聚类，样本数分别为：{dict(df['聚类标签'].value_counts())}；
                2. 聚类中心：每个聚类的特征均值代表该类的核心特征（如聚类0的{feature_cols[0]}均值为{centers.iloc[0][feature_cols[0]]:.2f}）；
                3. 业务建议：可根据聚类结果对数据分组分析（如客户分群、订单分类、城市聚类）；
                4. 调优提示：若聚类效果不佳，可调整K值或选择更多/更具代表性的数值变量。
                """

            st.divider()
            st.markdown(report)
            if analysis_mode == "单文件独立分析":
                file_tag = file_names[selected_file_idx]
            else:
                file_tag = f"{base_file_name}_多文件关联"
            st.download_button(
                label="📥 下载分析报告（Markdown）",
                data=report,
                file_name=f"{file_tag}_{analysis_type}_分析报告.md",
                mime="text/markdown"
            )
            
    except Exception as e:
        st.error(f"❌ 分析失败：{str(e)}")
        st.info("💡 可能原因：数据缺失值过多、变量选择不当、样本量不足（聚类需至少K个有效样本）、关联后无匹配数据")
