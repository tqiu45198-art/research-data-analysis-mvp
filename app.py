import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ---------------------- 1. 页面基础设置 ----------------------
st.set_page_config(
    page_title="科研数据分析助手-增强版",
    page_icon="📊",
    layout="wide"
)
st.title("📊 科研数据分析助手（增强版）")
st.markdown("**支持多文件上传+自定义图表+全面分析功能**")
st.divider()

# ---------------------- 2. 多文件上传与合并处理（修复CSV编码问题）----------------------
st.subheader("第一步：上传多个数据文件（支持跨文件分析）")
# 支持多文件上传
uploaded_files = st.file_uploader(
    "支持Excel(.xlsx)或CSV(.csv)文件，可拖拽多个", 
    type=["xlsx", "csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("💡 示例：上传1个或多个包含「分组变量」「数值变量」的表格，支持跨文件合并分析")
    st.stop()

# 读取所有上传的文件（核心修复：新增多编码兼容逻辑）
df_list = []
file_names = []
# 常见中文CSV编码列表（解决中文CSV解码失败问题）
encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']

for file in uploaded_files:
    try:
        if file.name.endswith(".csv"):
            # 尝试多种编码读取CSV，兼容中文文件
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file, encoding=encoding)
                    break  # 编码成功则停止尝试
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise ValueError("所有编码尝试均失败，无法读取该CSV文件（建议转换为UTF-8或GBK编码）")
        else:  # Excel文件无编码问题，正常读取
            df = pd.read_excel(file)
        
        df_list.append(df)
        file_names.append(file.name)
        st.success(f"✅ 成功读取文件：{file.name}（行数：{len(df)}，列数：{len(df.columns)}）")
    except Exception as e:
        st.error(f"❌ 读取文件{file.name}失败：{str(e)}")
        st.stop()

# 多文件合并选项（跨文件分析核心）
st.subheader("第二步：多文件合并设置（跨文件分析）")
merge_type = st.radio(
    "选择文件合并方式（单文件分析选「不合并」）",
    options=["不合并（单文件分析）", "纵向合并（追加数据，字段需一致）", "横向合并（按关键字段关联）"]
)

# 合并逻辑
if merge_type == "不合并（单文件分析）":
    # 选择要分析的单个文件
    selected_file_idx = st.selectbox("选择要分析的文件", range(len(file_names)), format_func=lambda x: file_names[x])
    df = df_list[selected_file_idx]
elif merge_type == "纵向合并（追加数据）":
    # 纵向合并（检查字段一致性）
    cols_set = [set(df.columns) for df in df_list]
    if len(set(frozenset(cols) for cols in cols_set)) > 1:
        st.warning("⚠️ 各文件字段不一致，将保留所有字段（缺失值填充为NaN）")
    df = pd.concat(df_list, ignore_index=True)
    st.success(f"✅ 纵向合并完成，合并后数据总行数：{len(df)}")
else:
    # 横向合并（按关键字段）
    key_col = st.text_input("输入关联关键字段（所有文件需包含该字段，如「学号」「样本ID」）", placeholder="如：样本ID")
    if not key_col:
        st.stop()
    # 依次合并所有文件
    df = df_list[0]
    for i in range(1, len(df_list)):
        df = pd.merge(df, df_list[i], on=key_col, how="outer", suffixes=(f"_{file_names[0].split('.')[0]}", f"_{file_names[i].split('.')[0]}"))
    st.success(f"✅ 按「{key_col}」横向合并完成，合并后数据列数：{len(df.columns)}")

# 数据预览与变量识别
st.subheader("数据预览（前5行）")
st.dataframe(df.head(), use_container_width=True)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
st.subheader("数据变量识别")
st.write(f"📈 数值型变量：{', '.join(numeric_cols) if numeric_cols else '无'}")
st.write(f"🏷️ 分类变量：{', '.join(categorical_cols) if categorical_cols else '无'}")

if not numeric_cols:
    st.error("❌ 未识别到数值型变量！请确保数据中包含成绩、分数等可计算的列")
    st.stop()

# ---------------------- 3. 需求输入与分析类型选择（扩充功能）----------------------
st.divider()
st.subheader("第三步：选择分析类型（新增多因素方差/聚类/逻辑回归）")
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

# 映射分析类型
type_map = {
    "描述性统计（均值/分布）": "descriptive",
    "独立样本t检验（两组差异）": "t_test",
    "多因素方差分析（ANOVA）": "anova",
    "简单线性回归（变量关系）": "regression",
    "逻辑回归（分类预测）": "logistic_reg",
    "K-Means聚类（数据分组）": "kmeans"
}
target_analysis = type_map[analysis_type]

# ---------------------- 4. 分析参数配置（含图表自定义）----------------------
st.subheader("第四步：配置分析参数+图表自定义")
params = {}
# 通用图表自定义参数（所有分析类型都可用）
st.markdown("### 🎨 图表自定义设置")
params["chart_color"] = st.color_picker("选择图表主色调", value="#1f77b4")  # 颜色选择器
params["chart_width"] = st.slider("图表宽度（像素）", min_value=600, max_value=1200, value=800)
params["chart_height"] = st.slider("图表高度（像素）", min_value=400, max_value=800, value=500)
# 聚类专属参数
if target_analysis == "kmeans":
    params["n_clusters"] = st.slider("聚类数量（K值）", min_value=2, max_value=10, value=3)

# 各分析类型参数配置
if target_analysis == "descriptive":
    params["target_col"] = st.selectbox("选择要分析的数值变量", numeric_cols)
    params["chart_type"] = st.radio("选择图表类型", ["直方图（分布）", "柱状图（均值）"])

elif target_analysis == "t_test":
    if not categorical_cols:
        st.error("❌ 未识别到分类变量！无法进行t检验")
        st.stop()
    params["group_col"] = st.selectbox("选择分组变量（如性别、组别）", categorical_cols)
    params["result_col"] = st.selectbox("选择要比较的数值变量", numeric_cols)
    # 校验分组为2组
    group_counts = df[params["group_col"]].nunique()
    if group_counts != 2:
        st.warning(f"⚠️ 分组变量有{group_counts}组，自动取前2组")
        top2_groups = df[params["group_col"]].value_counts().nlargest(2).index.tolist()
        df = df[df[params["group_col"]].isin(top2_groups)]

elif target_analysis == "anova":
    if len(categorical_cols) < 1:
        st.error("❌ 至少需要1个分类变量（因素）！无法进行方差分析")
        st.stop()
    params["factor_cols"] = st.multiselect("选择因素变量（分类变量，可多选）", categorical_cols, default=categorical_cols[0])
    params["result_col"] = st.selectbox("选择因变量（数值变量）", numeric_cols)
    # 构建公式（如：成绩 ~ 性别 + 专业）
    params["formula"] = f"{params['result_col']} ~ {' + '.join(params['factor_cols'])}"

elif target_analysis == "regression":
    if len(numeric_cols) < 2:
        st.error("❌ 至少需要2个数值变量！")
        st.stop()
    params["x_col"] = st.selectbox("选择自变量", numeric_cols)
    params["y_col"] = st.selectbox("选择因变量", [col for col in numeric_cols if col != params["x_col"]])

elif target_analysis == "logistic_reg":
    # 逻辑回归：因变量需为二分类，先筛选二分类分类变量
    binary_cats = [col for col in categorical_cols if df[col].nunique() == 2]
    if not binary_cats:
        st.error("❌ 未识别到二分类变量！逻辑回归因变量需为二分类（如：及格/不及格、是/否）")
        st.stop()
    params["target_col"] = st.selectbox("选择预测目标（二分类变量）", binary_cats)
    params["feature_cols"] = st.multiselect("选择特征变量（数值型）", numeric_cols, default=numeric_cols[:2])
    # 编码目标变量
    le = LabelEncoder()
    df[params["target_col"] + "_encoded"] = le.fit_transform(df[params["target_col"]])

elif target_analysis == "kmeans":
    params["feature_cols"] = st.multiselect("选择聚类特征变量（数值型）", numeric_cols, default=numeric_cols[:2])
    # 过滤缺失值
    df_cluster = df[params["feature_cols"]].dropna()
    if len(df_cluster) < params["n_clusters"]:
        st.error(f"❌ 有效样本数（{len(df_cluster)}）小于聚类数量（{params['n_clusters']}）！")
        st.stop()

# ---------------------- 5. 执行分析（扩充功能+自定义图表）----------------------
st.divider()
st.subheader("第五步：分析结果与智能报告")

if st.button("🚀 开始分析"):
    try:
        with st.spinner("正在分析中..."):
            report = ""
            # 5.1 描述性统计
            if target_analysis == "descriptive":
                col = params["target_col"]
                stats_result = df[col].describe()
                st.subheader("📊 描述性统计结果")
                st.dataframe(stats_result.to_frame(), use_container_width=True)
                
                # 自定义图表
                st.subheader("📈 可视化图表（自定义样式）")
                if params["chart_type"] == "直方图（分布）":
                    fig = px.histogram(
                        df, x=col, title=f"{col}的分布情况", nbins=20,
                        color_discrete_sequence=[params["chart_color"]],
                        width=params["chart_width"], height=params["chart_height"]
                    )
                else:
                    fig = px.bar(
                        df, y=col, title=f"{col}的均值",
                        color_discrete_sequence=[params["chart_color"]],
                        width=params["chart_width"], height=params["chart_height"]
                    )
                st.plotly_chart(fig, use_container_width=True)
                
                report = f"""
                ### 📝 分析报告
                1. 变量「{col}」核心统计：均值{stats_result['mean']:.2f}、中位数{stats_result['50%']:.2f}、标准差{stats_result['std']:.2f}；
                2. 数据分布：{'均匀' if stats_result['std'] < stats_result['mean']*0.3 else '分散'}，整体水平{stats_result['mean']:.2f}。
                """

            # 5.2 独立样本t检验
            elif target_analysis == "t_test":
                group_col = params["group_col"]
                result_col = params["result_col"]
                group1, group2 = df[group_col].unique()[:2]
                data1 = df[df[group_col] == group1][result_col].dropna()
                data2 = df[df[group_col] == group2][result_col].dropna()
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                mean1, mean2 = data1.mean(), data2.mean()
                
                st.subheader("🔍 t检验结果")
                st.write(f"{group1}均值：{mean1:.2f}，{group2}均值：{mean2:.2f}，p值：{p_value:.4f}")
                
                # 自定义箱线图
                fig = px.box(
                    df, x=group_col, y=result_col, title=f"{group_col}对{result_col}的影响",
                    color_discrete_sequence=[params["chart_color"]],
                    width=params["chart_width"], height=params["chart_height"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                significance = "显著" if p_value < 0.05 else "不显著"
                report = f"""
                ### 📝 分析报告
                1. 检验结论：{group1}与{group2}在{result_col}上的差异{significance}（p={p_value:.4f}）；
                2. 差异幅度：{group1}比{group2} {'高' if mean1>mean2 else '低'} {abs(mean1-mean2):.2f}。
                """

            # 5.3 多因素方差分析（新增）
            elif target_analysis == "anova":
                formula = params["formula"]
                model = ols(formula, data=df).fit()
                anova_result = anova_lm(model, typ=2)
                
                st.subheader("📊 多因素方差分析结果")
                st.dataframe(anova_result, use_container_width=True)
                
                # 可视化各因素影响
                fig = px.box(
                    df, x=params["factor_cols"][0], y=params["result_col"], 
                    color=params["factor_cols"][1] if len(params["factor_cols"])>1 else None,
                    title=f"各因素对{params['result_col']}的影响",
                    color_discrete_sequence=[params["chart_color"]] if len(params["factor_cols"])==1 else None,
                    width=params["chart_width"], height=params["chart_height"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 提取显著因素
                significant_factors = [idx for idx, p in anova_result["PR(>F)"].items() if p < 0.05]
                report = f"""
                ### 📝 分析报告
                1. 方差分析公式：{formula}；
                2. 显著影响因素（p<0.05）：{', '.join(significant_factors) if significant_factors else '无'}；
                3. 结论：{f'因素{significant_factors}对{params["result_col"]}有显著影响' if significant_factors else '所有因素对因变量无显著影响'}。
                """

            # 5.4 简单线性回归
            elif target_analysis == "regression":
                x_col, y_col = params["x_col"], params["y_col"]
                df_reg = df[[x_col, y_col]].dropna()
                model = ols(f"{y_col} ~ {x_col}", data=df_reg).fit()
                r_squared = model.rsquared
                coef = model.params[x_col]
                p_value = model.pvalues[x_col]
                
                st.subheader("📈 线性回归结果")
                st.write(f"回归方程：{y_col} = {model.params[0]:.2f} + {coef:.4f}×{x_col}")
                st.write(f"R²：{r_squared:.4f}，p值：{p_value:.4f}")
                
                # 自定义拟合图
                fig = px.scatter(
                    df_reg, x=x_col, y=y_col, trendline="ols", title=f"{x_col}对{y_col}的影响",
                    color_discrete_sequence=[params["chart_color"]],
                    width=params["chart_width"], height=params["chart_height"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                significance = "显著" if p_value < 0.05 else "不显著"
                report = f"""
                ### 📝 分析报告
                1. {x_col}对{y_col}的影响{significance}（p={p_value:.4f}）；
                2. 回归系数{coef:.4f}，说明{x_col}每增加1，{y_col} {'增加' if coef>0 else '减少'} {abs(coef):.4f}；
                3. R²={r_squared:.4f}，说明{x_col}能解释{y_col} {r_squared*100:.1f}%的变化。
                """

            # 5.5 逻辑回归（新增）
            elif target_analysis == "logistic_reg":
                target_col = params["target_col"]
                feature_cols = params["feature_cols"]
                df_log = df[[*feature_cols, target_col + "_encoded"]].dropna()
                
                # 训练模型
                model = LogisticRegression()
                model.fit(df_log[feature_cols], df_log[target_col + "_encoded"])
                accuracy = model.score(df_log[feature_cols], df_log[target_col + "_encoded"])
                coefs = dict(zip(feature_cols, model.coef_[0]))
                
                st.subheader("🔮 逻辑回归结果")
                st.write(f"模型准确率：{accuracy:.4f}")
                st.write("各特征系数（系数越大，对预测结果影响越强）：")
                st.dataframe(pd.DataFrame({"特征": coefs.keys(), "系数": coefs.values()}), use_container_width=True)
                
                # 可视化特征重要性
                fig = px.bar(
                    x=coefs.keys(), y=coefs.values(), title="特征重要性（逻辑回归系数）",
                    color_discrete_sequence=[params["chart_color"]],
                    width=params["chart_width"], height=params["chart_height"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                report = f"""
                ### 📝 分析报告
                1. 逻辑回归模型准确率：{accuracy:.4f}（越高说明预测效果越好）；
                2. 特征影响：{max(coefs, key=coefs.get)}对{target_col}的影响最大（系数{coefs[max(coefs, key=coefs.get)]:.4f}）；
                3. 结论：模型可用于{target_col}的分类预测，准确率{accuracy*100:.1f}%。
                """

            # 5.6 K-Means聚类（新增）
            elif target_analysis == "kmeans":
                feature_cols = params["feature_cols"]
                n_clusters = params["n_clusters"]
                df_cluster = df[feature_cols].dropna()
                
                # 训练聚类模型
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                df_cluster["聚类标签"] = kmeans.fit_predict(df_cluster[feature_cols])
                # 合并聚类结果到原数据
                df["聚类标签"] = df_cluster["聚类标签"].reindex(df.index)
                
                st.subheader("🌀 K-Means聚类结果")
                st.write(f"聚类数量：{n_clusters}，各聚类样本数：")
                st.dataframe(df["聚类标签"].value_counts(), use_container_width=True)
                
                # 自定义聚类散点图（取前两个特征）
                fig = px.scatter(
                    df_cluster, x=feature_cols[0], y=feature_cols[1], color="聚类标签",
                    title=f"K-Means聚类结果（K={n_clusters}）",
                    color_discrete_sequence=[params["chart_color"], "#ff7f0e", "#2ca02c", "#d62728"][:n_clusters],
                    width=params["chart_width"], height=params["chart_height"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 各聚类中心
                centers = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)
                st.subheader("各聚类中心（代表该类的特征均值）")
                st.dataframe(centers, use_container_width=True)
                
                report = f"""
                ### 📝 分析报告
                1. 数据被分为{n_clusters}个聚类，样本数分别为：{dict(df['聚类标签'].value_counts())}；
                2. 聚类中心反映了每类样本的核心特征，可用于样本分组、特征分析；
                3. 建议：可根据聚类结果进一步分析不同组的差异，或调整K值优化聚类效果。
                """

            # 显示报告+下载
            st.divider()
            st.markdown(report)
            st.download_button(
                label="📥 下载分析报告（Markdown）",
                data=report,
                file_name=f"科研数据分析报告_{target_analysis}.md",
                mime="text/markdown"
            )
            
    except Exception as e:
        st.error(f"❌ 分析失败：{str(e)}")
        st.info("💡 可能原因：数据缺失值过多、样本量不足、变量选择不当")
