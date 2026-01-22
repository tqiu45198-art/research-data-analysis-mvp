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
    page_title="智能科研数据分析助手",
    page_icon="📊",
    layout="wide"
)
st.title("📊 智能科研数据分析助手")
st.markdown("**自动筛选文件+智能推荐分析类型+通用格式适配**")
st.divider()

# ---------------------- 第一步：上传文件（支持任意格式）----------------------
st.subheader("第一步：上传数据文件（支持CSV/Excel，可上传多个）")
uploaded_files = st.file_uploader(
    "支持Excel(.xlsx)或CSV(.csv)文件，上传后可选择部分参与分析", 
    type=["xlsx", "csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("💡 示例：上传客户信息、订单数据、统计报表等，支持后续筛选部分文件分析")
    st.stop()

# 读取所有上传文件（通用适配，解决乱码）
df_list = []
file_names = []
encodings = ['utf-8-sig', 'gbk', 'utf-8', 'gb2312', 'big5', 'utf-16', 'gb18030', 'latin-1']
seps = [',', '\t', ';', '|', ' ', ':', '\s+']

def clean_column_names(df):
    """通用列名清理，移除特殊字符和乱码"""
    df.columns = [
        re.sub(r'[^\w\s\u4e00-\u9fa5/]', '', str(col)).strip() 
        for col in df.columns
    ]
    df.columns = [col if col else f"col_{i}" for i, col in enumerate(df.columns)]
    return df

for file in uploaded_files:
    try:
        file_content = file.read()
        if len(file_content) == 0:
            raise ValueError("文件为空")
        file.seek(0)
        df = None
        file_name = file.name
        
        # CSV文件：多编码+多分隔符尝试
        if file_name.endswith(".csv"):
            for encoding in encodings:
                for sep in seps:
                    try:
                        if encoding in ['utf-16', 'utf-16le', 'utf-16be']:
                            content = file_content.decode(encoding, errors='replace')
                            df = pd.read_csv(io.StringIO(content), sep=sep, on_bad_lines='skip')
                        else:
                            df = pd.read_csv(file, encoding=encoding, sep=sep, on_bad_lines='skip')
                        df = clean_column_names(df)
                        break
                    except:
                        continue
                if df is not None:
                    break
            # 自动检测分隔符兜底
            if df is None:
                try:
                    from csv import Sniffer
                    sample = file_content[:4096].decode('utf-8-sig', errors='replace')
                    delimiter = Sniffer().sniff(sample).delimiter
                    df = pd.read_csv(file, encoding='utf-8-sig', sep=delimiter, on_bad_lines='skip')
                    df = clean_column_names(df)
                except:
                    raise ValueError("编码/分隔符匹配失败")
        
        # Excel文件
        else:
            df = pd.read_excel(file, engine='openpyxl')
            df = clean_column_names(df)
        
        if df is not None and len(df) > 0:
            df_list.append(df)
            file_names.append(file_name)
            st.success(f"✅ 成功读取：{file_name}（{len(df)}行 × {len(df.columns)}列）")
        else:
            st.warning(f"⚠️ {file_name} 无有效数据，已跳过")
    except Exception as e:
        st.error(f"❌ 读取{file_name}失败：{str(e)}")

if not df_list:
    st.error("❌ 无有效文件可分析，请检查文件格式")
    st.stop()

# ---------------------- 第二步：智能文件筛选（选择本次参与分析的文件）----------------------
st.subheader("第二步：选择本次参与分析的文件")
selected_file_idxs = st.multiselect(
    "从上传文件中选择（可多选，至少1个）",
    range(len(file_names)),
    default=[0],
    format_func=lambda x: file_names[x]
)

if len(selected_file_idxs) == 0:
    st.error("❌ 至少选择1个文件参与分析")
    st.stop()

# 提取选中的文件
selected_dfs = [df_list[i] for i in selected_file_idxs]
selected_file_names = [file_names[i] for i in selected_file_idxs]

# ---------------------- 第三步：选择分析模式（单文件/多文件关联）----------------------
st.subheader("第三步：选择分析模式")
if len(selected_file_idxs) == 1:
    # 仅选中1个文件，默认单文件分析
    analysis_mode = "单文件独立分析"
    st.write(f"📌 已自动选择单文件分析：{selected_file_names[0]}")
    df = selected_dfs[0]
else:
    analysis_mode = st.radio(
        "选中多个文件，选择分析模式",
        options=["单文件独立分析", "多文件关联分析"]
    )

    # 单文件分析（从选中文件中选1个）
    if analysis_mode == "单文件独立分析":
        selected_idx = st.selectbox(
            "选择要分析的单个文件",
            range(len(selected_file_names)),
            format_func=lambda x: selected_file_names[x]
        )
        df = selected_dfs[selected_idx]
        st.success(f"✅ 已选择单文件：{selected_file_names[selected_idx]}")
    
    # 多文件关联分析（从选中文件中选基础文件和关联文件）
    else:
        st.markdown("### 配置多文件关联")
        # 选择基础文件
        base_idx = st.selectbox(
            "选择基础文件",
            range(len(selected_file_names)),
            format_func=lambda x: selected_file_names[x]
        )
        df = selected_dfs[base_idx]
        base_name = selected_file_names[base_idx]
        remaining_idxs = [i for i in range(len(selected_file_names)) if i != base_idx]
        remaining_dfs = [selected_dfs[i] for i in remaining_idxs]
        remaining_names = [selected_file_names[i] for i in remaining_idxs]

        # 逐步关联其他选中的文件
        for i in range(len(remaining_idxs)):
            st.markdown(f"#### 关联第{i+1}个文件")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                join_idx = st.selectbox(
                    f"选择关联文件 {i+1}",
                    remaining_idxs,
                    format_func=lambda x: remaining_names[x],
                    key=f"join_file_{i}"
                )
                join_df = remaining_dfs[remaining_idxs.index(join_idx)]
                join_name = remaining_names[remaining_idxs.index(join_idx)]
            
            with col2:
                base_key = st.selectbox(f"基础文件关联字段", df.columns.tolist(), key=f"base_key_{i}")
            
            with col3:
                join_key = st.selectbox(f"关联文件关联字段", join_df.columns.tolist(), key=f"join_key_{i}")
            
            # 关联方式
            join_type = st.radio(
                f"关联方式",
                options=["内关联（仅保留匹配数据）", "左关联（保留基础文件数据）"],
                key=f"join_type_{i}"
            )
            join_map = {"内关联（仅保留匹配数据）": "inner", "左关联（保留基础文件数据）": "left"}

            # 字段有效性检查
            if base_key not in df.columns or join_key not in join_df.columns:
                st.error("❌ 关联字段不存在，请重新选择")
                st.stop()

            # 重命名冲突字段
            join_suffix = f"_{join_name.split('.')[0]}"
            join_df_renamed = join_df.rename(
                columns={col: f"{col}{join_suffix}" for col in join_df.columns if col != join_key and col in df.columns}
            )

            # 执行关联
            try:
                df = pd.merge(df, join_df_renamed, left_on=base_key, right_on=join_key, how=join_map[join_type])
                st.success(f"✅ 关联完成：{base_name} ↔ {join_name}（当前：{len(df)}行 × {len(df.columns)}列）")
            except Exception as e:
                st.error(f"❌ 关联失败：{str(e)}")
                st.stop()

            # 移除已关联文件
            remaining_idxs.remove(join_idx)
            if not remaining_idxs:
                break

# ---------------------- 第四步：智能识别变量+推荐分析类型----------------------
st.subheader("第四步：数据变量智能识别")
# 自动区分数值型/分类型变量
numeric_cols = []
categorical_cols = []
binary_categorical_cols = []  # 二分类变量（用于逻辑回归）

for col in df.columns:
    try:
        # 尝试转换为数值型
        df[col] = pd.to_numeric(df[col], errors='raise')
        numeric_cols.append(col)
    except:
        # 分类型变量
        categorical_cols.append(col)
        # 识别二分类变量
        if df[col].nunique() == 2:
            binary_categorical_cols.append(col)

# 去重并显示
numeric_cols = list(set(numeric_cols))
categorical_cols = list(set(categorical_cols))
binary_categorical_cols = list(set(binary_categorical_cols))

# 显示变量识别结果
st.write(f"📈 数值型变量（{len(numeric_cols)}个）：{', '.join(numeric_cols) if numeric_cols else '无'}")
st.write(f"🏷️ 分类型变量（{len(categorical_cols)}个）：{', '.join(categorical_cols) if categorical_cols else '无'}")
st.write(f"🔑 二分类变量（{len(binary_categorical_cols)}个）：{', '.join(binary_categorical_cols) if binary_categorical_cols else '无'}")

if not numeric_cols:
    st.error("❌ 无可用数值型变量，无法进行统计分析")
    st.stop()

# 智能判断支持的分析类型
def get_supported_analyses():
    supported = []
    reasons = {}

    # 1. 描述性统计（只要有数值型变量）
    supported.append("描述性统计（分布/均值/标准差）")
    reasons["描述性统计（分布/均值/标准差）"] = "✅ 有数值型变量，支持基础统计和可视化"

    # 2. 独立样本t检验（需分类型变量≥1+数值型变量≥1，且分类型变量至少2组）
    t_test_support = len(categorical_cols) >= 1 and len(numeric_cols) >= 1
    if t_test_support:
        # 检查是否有分类型变量≥2组
        multi_group_cats = [col for col in categorical_cols if df[col].nunique() >= 2]
        if len(multi_group_cats) == 0:
            t_test_support = False
            reasons["独立样本t检验（两组差异对比）"] = "❌ 所有分类型变量仅1组，无法分组对比"
        else:
            reasons["独立样本t检验（两组差异对比）"] = "✅ 有分类型变量（≥2组）和数值型变量，支持差异检验"
    else:
        reasons["独立样本t检验（两组差异对比）"] = "❌ 缺少分类型变量或数值型变量"
    
    if t_test_support:
        supported.append("独立样本t检验（两组差异对比）")

    # 3. 多因素方差分析（ANOVA）（需分类型变量≥1+数值型变量≥1）
    anova_support = len(categorical_cols) >= 1 and len(numeric_cols) >= 1
    if anova_support:
        reasons["多因素方差分析（ANOVA）"] = "✅ 有分类型变量和数值型变量，支持多因素影响分析"
        supported.append("多因素方差分析（ANOVA）")
    else:
        reasons["多因素方差分析（ANOVA）"] = "❌ 缺少分类型变量或数值型变量"

    # 4. 简单线性回归（需数值型变量≥2）
    regression_support = len(numeric_cols) >= 2
    if regression_support:
        reasons["简单线性回归（变量关系分析）"] = "✅ 数值型变量≥2个，支持变量关系建模"
        supported.append("简单线性回归（变量关系分析）")
    else:
        reasons["简单线性回归（变量关系分析）"] = "❌ 数值型变量不足2个，无法建立回归"

    # 5. 逻辑回归（需二分类变量≥1+数值型变量≥1）
    logistic_support = len(binary_categorical_cols) >= 1 and len(numeric_cols) >= 1
    if logistic_support:
        reasons["逻辑回归（分类预测）"] = "✅ 有二分类变量和数值型变量，支持分类预测"
        supported.append("逻辑回归（分类预测）")
    else:
        reasons["逻辑回归（分类预测）"] = "❌ 缺少二分类变量或数值型变量"

    # 6. K-Means聚类（需数值型变量≥2）
    kmeans_support = len(numeric_cols) >= 2
    if kmeans_support:
        reasons["K-Means聚类（数据分组）"] = "✅ 数值型变量≥2个，支持数据分群"
        supported.append("K-Means聚类（数据分组）")
    else:
        reasons["K-Means聚类（数据分组）"] = "❌ 数值型变量不足2个，无法聚类"

    return supported, reasons

supported_analyses, analysis_reasons = get_supported_analyses()

# 显示支持的分析类型（隐藏不支持的，显示原因）
st.subheader("第五步：智能推荐分析类型")
st.write("💡 基于数据自动筛选支持的分析类型，不支持的类型及原因如下：")
for analysis in [
    "描述性统计（分布/均值/标准差）",
    "独立样本t检验（两组差异对比）",
    "多因素方差分析（ANOVA）",
    "简单线性回归（变量关系分析）",
    "逻辑回归（分类预测）",
    "K-Means聚类（数据分组）"
]:
    if analysis not in supported_analyses:
        st.write(f"- {analysis_reasons[analysis]}")

# 让用户选择支持的分析类型
if not supported_analyses:
    st.error("❌ 无可用分析类型，请检查数据变量")
    st.stop()

analysis_type = st.radio(
    "选择要执行的分析（仅显示支持的类型）",
    options=supported_analyses
)

type_map = {
    "描述性统计（分布/均值/标准差）": "descriptive",
    "独立样本t检验（两组差异对比）": "t_test",
    "多因素方差分析（ANOVA）": "anova",
    "简单线性回归（变量关系分析）": "regression",
    "逻辑回归（分类预测）": "logistic_reg",
    "K-Means聚类（数据分组）": "kmeans"
}
target_analysis = type_map[analysis_type]

# ---------------------- 第六步：配置分析参数----------------------
st.subheader("第六步：配置分析参数")
params = {}
st.markdown("### 🎨 图表自定义")
params["chart_color"] = st.color_picker("图表主色调", value="#1f77b4")
params["chart_width"] = st.slider("图表宽度", 600, 1200, 800)
params["chart_height"] = st.slider("图表高度", 400, 800, 500)

# 按分析类型配置参数
if target_analysis == "kmeans":
    params["n_clusters"] = st.slider("聚类数量（K值）", 2, 10, 3)

elif target_analysis == "descriptive":
    params["target_col"] = st.selectbox("选择分析的数值变量", numeric_cols)
    params["chart_type"] = st.radio("图表类型", ["直方图（分布）", "柱状图（均值）"])

elif target_analysis == "t_test":
    # 仅显示≥2组的分类型变量
    valid_group_cols = [col for col in categorical_cols if df[col].nunique() >= 2]
    params["group_col"] = st.selectbox("选择分组变量", valid_group_cols)
    params["result_col"] = st.selectbox("选择对比的数值变量", numeric_cols)
    # 自动处理多组
    group_counts = df[params["group_col"]].nunique()
    if group_counts != 2:
        st.warning(f"⚠️ 自动取样本量前2的组（共{group_counts}组）")
        top2_groups = df[params["group_col"]].value_counts().nlargest(2).index.tolist()
        df = df[df[params["group_col"]].isin(top2_groups)]

elif target_analysis == "anova":
    params["factor_cols"] = st.multiselect("选择因素变量（可多选）", categorical_cols, default=categorical_cols[0])
    params["result_col"] = st.selectbox("选择因变量", numeric_cols)
    params["formula"] = f"{params['result_col']} ~ {' + '.join(params['factor_cols'])}"

elif target_analysis == "regression":
    params["x_col"] = st.selectbox("选择自变量", numeric_cols)
    params["y_col"] = st.selectbox("选择因变量", [col for col in numeric_cols if col != params["x_col"]])

elif target_analysis == "logistic_reg":
    params["target_col"] = st.selectbox("选择预测目标（二分类变量）", binary_categorical_cols)
    params["feature_cols"] = st.multiselect("选择特征变量", numeric_cols, default=numeric_cols[:2])
    df[params["target_col"] + "_encoded"] = LabelEncoder().fit_transform(df[params["target_col"]])

elif target_analysis == "kmeans":
    params["feature_cols"] = st.multiselect("选择聚类特征", numeric_cols, default=numeric_cols[:2])
    df_cluster = df[params["feature_cols"]].dropna()
    if len(df_cluster) < params["n_clusters"]:
        st.error(f"❌ 有效样本数（{len(df_cluster)}）< 聚类数量（{params['n_clusters']}），请减少K值")
        st.stop()

# ---------------------- 第七步：执行分析并生成结果----------------------
st.divider()
st.subheader("第七步：执行分析")

if st.button("🚀 开始分析"):
    try:
        with st.spinner("分析中..."):
            report = ""
            # 描述性统计
            if target_analysis == "descriptive":
                col = params["target_col"]
                stats_result = df[col].describe()
                st.subheader("📊 描述性统计结果")
                st.dataframe(stats_result.to_frame(), use_container_width=True)
                
                fig = px.histogram(df, x=col, title=f"{col}分布" if params["chart_type"] == "直方图（分布）" else f"{col}均值",
                                  color_discrete_sequence=[params["chart_color"]], width=params["chart_width"], height=params["chart_height"])
                st.plotly_chart(fig, use_container_width=True)
                
                report = f"""### 分析报告
1. 分析变量：{col}
2. 核心统计：均值{stats_result['mean']:.2f}、中位数{stats_result['50%']:.2f}、标准差{stats_result['std']:.2f}
3. 数据范围：{stats_result['min']:.2f} ~ {stats_result['max']:.2f}
4. 分布特征：{'均匀' if stats_result['std'] < stats_result['mean']*0.3 else '分散'}
"""

            # t检验
            elif target_analysis == "t_test":
                group_col, result_col = params["group_col"], params["result_col"]
                group1, group2 = df[group_col].unique()[:2]
                data1, data2 = df[df[group_col]==group1][result_col].dropna(), df[df[group_col]==group2][result_col].dropna()
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                
                st.subheader("🔍 t检验结果")
                st.write(f"{group1}均值：{data1.mean():.2f}，{group2}均值：{data2.mean():.2f}")
                st.write(f"t统计量：{t_stat:.4f}，p值：{p_value:.4f}")
                
                fig = px.box(df, x=group_col, y=result_col, color_discrete_sequence=[params["chart_color"]],
                           title=f"{group_col}对{result_col}的影响", width=params["chart_width"], height=params["chart_height"])
                st.plotly_chart(fig, use_container_width=True)
                
                report = f"""### 分析报告
1. 检验场景：{group_col}分组对{result_col}的差异
2. 结论：{'存在显著差异' if p_value<0.05 else '无显著差异'}（p={p_value:.4f}）
3. 差异幅度：{group1}比{group2} {'高' if data1.mean()>data2.mean() else '低'} {abs(data1.mean()-data2.mean()):.2f}
"""

            # ANOVA
            elif target_analysis == "anova":
                model = ols(params["formula"], data=df).fit()
                anova_result = anova_lm(model, typ=2)
                st.subheader("📊 方差分析结果")
                st.dataframe(anova_result, use_container_width=True)
                
                fig = px.box(df, x=params["factor_cols"][0], y=params["result_col"],
                           color=params["factor_cols"][1] if len(params["factor_cols"])>1 else None,
                           title=f"各因素对{params['result_col']}的影响", width=params["chart_width"], height=params["chart_height"])
                st.plotly_chart(fig, use_container_width=True)
                
                significant = [idx for idx, p in anova_result["PR(>F)"].items() if p<0.05]
                report = f"""### 分析报告
1. 分析公式：{params['formula']}
2. 显著因素（p<0.05）：{', '.join(significant) if significant else '无'}
3. 结论：{'部分因素对结果有显著影响' if significant else '所有因素无显著影响'}
"""

            # 线性回归
            elif target_analysis == "regression":
                x_col, y_col = params["x_col"], params["y_col"]
                df_reg = df[[x_col, y_col]].dropna()
                model = ols(f"{y_col} ~ {x_col}", data=df_reg).fit()
                
                st.subheader("📈 回归结果")
                st.write(f"回归方程：{y_col} = {model.params[0]:.2f} + {model.params[x_col]:.4f}×{x_col}")
                st.write(f"R²：{model.rsquared:.4f}，p值：{model.pvalues[x_col]:.4f}")
                
                fig = px.scatter(df_reg, x=x_col, y=y_col, trendline="ols", color_discrete_sequence=[params["chart_color"]],
                               title=f"{x_col}对{y_col}的回归", width=params["chart_width"], height=params["chart_height"])
                st.plotly_chart(fig, use_container_width=True)
                
                report = f"""### 分析报告
1. 变量关系：{'显著相关' if model.pvalues[x_col]<0.05 else '无显著相关'}（p={model.pvalues[x_col]:.4f}）
2. 回归系数：{model.params[x_col]:.4f}（{x_col}每增1，{y_col}{'增' if model.params[x_col]>0 else '减'} {abs(model.params[x_col]):.4f}）
3. 拟合程度：R²={model.rsquared:.4f}（{x_col}解释{y_col} {model.rsquared*100:.1f}%的变化）
"""

            # 逻辑回归
            elif target_analysis == "logistic_reg":
                target_col, feature_cols = params["target_col"], params["feature_cols"]
                df_log = df[[*feature_cols, target_col + "_encoded"]].dropna()
                model = LogisticRegression()
                model.fit(df_log[feature_cols], df_log[target_col + "_encoded"])
                accuracy = model.score(df_log[feature_cols], df_log[target_col + "_encoded"])
                coefs = dict(zip(feature_cols, model.coef_[0]))
                
                st.subheader("🔮 逻辑回归结果")
                st.write(f"模型准确率：{accuracy:.4f}")
                st.dataframe(pd.DataFrame({"特征": coefs.keys(), "系数": coefs.values()}), use_container_width=True)
                
                fig = px.bar(x=coefs.keys(), y=coefs.values(), color_discrete_sequence=[params["chart_color"]],
                           title="特征重要性", width=params["chart_width"], height=params["chart_height"])
                st.plotly_chart(fig, use_container_width=True)
                
                report = f"""### 分析报告
1. 预测目标：{target_col}，模型准确率：{accuracy:.4f}
2. 关键特征：{max(coefs, key=coefs.get)}（系数{coefs[max(coefs, key=coefs.get)]:.4f}）
3. 结论：模型可用于{target_col}的二分类预测
"""

            # K-Means聚类
            elif target_analysis == "kmeans":
                feature_cols = params["feature_cols"]
                df_cluster = df[feature_cols].dropna()
                kmeans = KMeans(n_clusters=params["n_clusters"], random_state=42).fit(df_cluster)
                df["聚类标签"] = kmeans.labels_
                
                st.subheader("🌀 聚类结果")
                st.dataframe(df["聚类标签"].value_counts(), use_container_width=True)
                st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols), use_container_width=True)
                
                fig = px.scatter(df_cluster, x=feature_cols[0], y=feature_cols[1], color=kmeans.labels_,
                               color_discrete_sequence=[params["chart_color"], "#ff7f0e", "#2ca02c", "#d62728"][:params["n_clusters"]],
                               title=f"K={params['n_clusters']}聚类结果", width=params["chart_width"], height=params["chart_height"])
                st.plotly_chart(fig, use_container_width=True)
                
                report = f"""### 分析报告
1. 聚类数量：{params['n_clusters']}组，样本数：{dict(df['聚类标签'].value_counts())}
2. 核心特征：各聚类中心反映组内特征均值
3. 应用：可用于数据分群管理、差异化策略制定
"""

            # 报告下载
            st.divider()
            st.markdown(report)
            st.download_button(
                label="📥 下载分析报告（Markdown）",
                data=report,
                file_name=f"智能分析报告_{analysis_type}.md",
                mime="text/markdown"
            )
            
    except Exception as e:
        st.error(f"❌ 分析失败：{str(e)}")
        st.info("💡 可能原因：数据缺失过多、变量选择不当、样本量不足")
