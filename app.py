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
    page_title="通用科研数据分析助手",
    page_icon="📊",
    layout="wide"
)
st.title("📊 通用科研数据分析助手")
st.markdown("**支持任意CSV/Excel文件+单文件分析+多文件关联分析+自定义图表**")
st.divider()

st.subheader("第一步：上传数据文件（支持多个任意格式文件）")
uploaded_files = st.file_uploader(
    "支持Excel(.xlsx)或CSV(.csv)文件，可上传多个（自动适配编码和字段结构）", 
    type=["xlsx", "csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("💡 示例：上传任意结构的表格文件（如客户信息、订单数据、统计报表等），支持单文件/多文件关联分析")
    st.stop()

df_list = []
file_names = []
# 通用编码列表（覆盖中英文常见编码）
encodings = ['utf-8-sig', 'gbk', 'utf-8', 'gb2312', 'big5', 'utf-16', 'gb18030', 'latin-1']
# 通用分隔符列表（覆盖常见分隔格式）
seps = [',', '\t', ';', '|', ' ', ':', '\s+']

def clean_column_names(df):
    """通用列名清理：移除特殊字符，避免乱码和冲突"""
    df.columns = [
        re.sub(r'[^\w\s\u4e00-\u9fa5/]', '', str(col)).strip() 
        for col in df.columns
    ]
    df.columns = [col if col else f"col_{i}" for i, col in enumerate(df.columns)]
    return df

# 读取所有上传文件（通用适配逻辑）
for file in uploaded_files:
    try:
        file_content = file.read()
        if len(file_content) == 0:
            raise ValueError("文件为空，无法读取")
        file.seek(0)
        df = None
        file_name = file.name
        
        # CSV文件通用读取（多编码+多分隔符尝试）
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
                    except Exception:
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
                except Exception as e:
                    raise ValueError(f"所有编码/分隔符尝试失败：{str(e)}")
        
        # Excel文件通用读取
        else:
            try:
                df = pd.read_excel(file, engine='openpyxl')
                df = clean_column_names(df)
            except Exception as e:
                raise ValueError(f"Excel读取失败：{str(e)}")
        
        # 验证读取结果
        if df is not None and len(df) > 0:
            df_list.append(df)
            file_names.append(file_name)
            st.success(f"✅ 成功读取：{file_name}（行数：{len(df)}，列数：{len(df.columns)}，字段：{', '.join(df.columns[:5])}...）")
        else:
            st.warning(f"⚠️ {file_name} 读取后无有效数据，已跳过")
    except Exception as e:
        st.error(f"❌ 读取{file_name}失败：{str(e)}")

if not df_list:
    st.error("❌ 没有成功读取任何有效文件，请检查文件格式和内容")
    st.stop()

st.subheader("第二步：选择分析模式")
analysis_mode = st.radio(
    "选择分析模式",
    options=["单文件独立分析", "多文件关联分析"]
)

# 单文件分析逻辑（通用适配）
if analysis_mode == "单文件独立分析":
    selected_file_idx = st.selectbox("选择要分析的文件", range(len(file_names)), format_func=lambda x: file_names[x])
    df = df_list[selected_file_idx]
    st.success(f"✅ 已选择单文件：{file_names[selected_file_idx]}（字段：{', '.join(df.columns[:5])}...）")

# 多文件关联分析逻辑（通用字段关联）
else:
    if len(file_names) < 2:
        st.error("❌ 多文件关联分析至少需要上传2个文件！")
        st.stop()
    
    # 选择基础文件
    base_file_idx = st.selectbox(
        "选择基础文件", 
        range(len(file_names)), 
        format_func=lambda x: file_names[x]
    )
    df = df_list[base_file_idx]
    base_file_name = file_names[base_file_idx]
    st.success(f"✅ 基础文件：{base_file_name}（当前数据：{len(df)}行 × {len(df.columns)}列）")
    
    remaining_file_idxs = [i for i in range(len(file_names)) if i != base_file_idx]
    
    # 逐步关联其他文件
    for i in range(len(remaining_file_idxs)):
        st.markdown(f"### 关联第{i+1}个文件")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            file_idx = st.selectbox(
                f"选择关联文件 {i+1}", 
                remaining_file_idxs, 
                format_func=lambda x: file_names[x], 
                key=f"file_{i}"
            )
            join_file_name = file_names[file_idx]
            df_to_join = df_list[file_idx]
        
        with col2:
            # 显示基础文件所有字段，供用户选择关联键
            base_key = st.selectbox(f"基础文件关联字段", df.columns.tolist(), key=f"base_key_{i}")
        
        with col3:
            # 显示关联文件所有字段，供用户选择关联键
            join_key = st.selectbox(f"关联文件关联字段", df_to_join.columns.tolist(), key=f"join_key_{i}")
        
        # 通用关联方式选择
        join_type = st.radio(
            f"关联方式（{base_file_name} ↔ {join_file_name}）", 
            options=["内关联（仅保留匹配数据）", "左关联（保留基础文件全部数据）"], 
            key=f"join_type_{i}"
        )
        join_map = {"内关联（仅保留匹配数据）": "inner", "左关联（保留基础文件全部数据）": "left"}
        
        # 关联前字段有效性检查
        if base_key not in df.columns:
            st.error(f"❌ 基础文件无「{base_key}」字段，请重新选择")
            break
        if join_key not in df_to_join.columns:
            st.error(f"❌ 关联文件无「{join_key}」字段，请重新选择")
            break
        
        # 重命名冲突字段（通用后缀，避免列名重复）
        join_suffix = f"_{join_file_name.split('.')[0]}"
        df_to_join_renamed = df_to_join.rename(
            columns={col: f"{col}{join_suffix}" for col in df_to_join.columns if col != join_key and col in df.columns}
        )
        
        # 执行关联
        try:
            df = pd.merge(
                df,
                df_to_join_renamed,
                left_on=base_key,
                right_on=join_key,
                how=join_map[join_type]
            )
            st.success(f"✅ 关联完成！当前数据：{len(df)}行 × {len(df.columns)}列")
        except Exception as e:
            st.error(f"❌ 关联失败：{str(e)}")
            break
        
        # 移除已关联文件，避免重复
        remaining_file_idxs.remove(file_idx)
        if not remaining_file_idxs:
            break

# 通用变量识别（自动区分数值型/分类型）
st.subheader("数据预览（前5行）")
st.dataframe(df.head(), use_container_width=True)

# 自动识别变量类型
numeric_cols = []
categorical_cols = []
for col in df.columns:
    try:
        # 尝试转换为数值型，成功则视为数值字段
        df[col] = pd.to_numeric(df[col], errors='raise')
        numeric_cols.append(col)
    except:
        # 无法转换为数值的视为分类字段
        categorical_cols.append(col)

# 去重并显示
numeric_cols = list(set(numeric_cols))
categorical_cols = list(set(categorical_cols))

st.subheader("变量类型自动识别")
st.write(f"📈 数值型变量（可分析：均值/回归/聚类）：{', '.join(numeric_cols) if numeric_cols else '无'}")
st.write(f"🏷️ 分类型变量（可分析：分组/差异检验）：{', '.join(categorical_cols) if categorical_cols else '无'}")

if not numeric_cols:
    st.error("❌ 未识别到数值型变量！请确保文件包含可计算的数值字段（如数量、金额、分数等）")
    st.stop()

st.divider()
st.subheader("第三步：选择分析类型")
analysis_type = st.radio(
    "选择分析类型（通用适配所有数据）",
    options=[
        "描述性统计（分布/均值/标准差）", 
        "独立样本t检验（两组差异对比）", 
        "多因素方差分析（ANOVA）",
        "简单线性回归（变量关系分析）",
        "逻辑回归（分类预测）",
        "K-Means聚类（数据分组）"
    ]
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

st.subheader("第四步：配置分析参数+图表自定义")
params = {}
st.markdown("### 🎨 图表自定义设置")
params["chart_color"] = st.color_picker("选择图表主色调", value="#1f77b4")
params["chart_width"] = st.slider("图表宽度（像素）", min_value=600, max_value=1200, value=800)
params["chart_height"] = st.slider("图表高度（像素）", min_value=400, max_value=800, value=500)

# 按分析类型配置通用参数
if target_analysis == "kmeans":
    params["n_clusters"] = st.slider("聚类数量（K值）", min_value=2, max_value=10, value=3)

elif target_analysis == "descriptive":
    params["target_col"] = st.selectbox("选择要分析的数值变量", numeric_cols)
    params["chart_type"] = st.radio("选择图表类型", ["直方图（分布）", "柱状图（均值）"])

elif target_analysis == "t_test":
    if not categorical_cols:
        st.error("❌ 未识别到分类型变量！无法进行分组差异检验")
        st.stop()
    params["group_col"] = st.selectbox("选择分组变量（分类型）", categorical_cols)
    params["result_col"] = st.selectbox("选择要比较的数值变量", numeric_cols)
    # 自动处理多组分组
    group_counts = df[params["group_col"]].nunique()
    if group_counts != 2:
        st.warning(f"⚠️ 分组变量有{group_counts}组，自动取样本量前2的组进行检验")
        top2_groups = df[params["group_col"]].value_counts().nlargest(2).index.tolist()
        df = df[df[params["group_col"]].isin(top2_groups)]

elif target_analysis == "anova":
    if len(categorical_cols) < 1:
        st.error("❌ 未识别到分类型变量！无法进行方差分析")
        st.stop()
    params["factor_cols"] = st.multiselect("选择因素变量（分类型，可多选）", categorical_cols, default=categorical_cols[0])
    params["result_col"] = st.selectbox("选择因变量（数值型）", numeric_cols)
    params["formula"] = f"{params['result_col']} ~ {' + '.join(params['factor_cols'])}"

elif target_analysis == "regression":
    if len(numeric_cols) < 2:
        st.error("❌ 至少需要2个数值型变量！无法进行回归分析")
        st.stop()
    params["x_col"] = st.selectbox("选择自变量", numeric_cols)
    params["y_col"] = st.selectbox("选择因变量", [col for col in numeric_cols if col != params["x_col"]])

elif target_analysis == "logistic_reg":
    # 自动识别二分类变量
    binary_cats = [col for col in categorical_cols if df[col].nunique() == 2]
    if not binary_cats:
        st.error("❌ 未识别到二分类变量！逻辑回归需分类型变量仅含2个取值（如是/否、达标/未达标）")
        st.stop()
    params["target_col"] = st.selectbox("选择预测目标（二分类变量）", binary_cats)
    params["feature_cols"] = st.multiselect("选择特征变量（数值型）", numeric_cols, default=numeric_cols[:2])
    # 编码目标变量
    le = LabelEncoder()
    df[params["target_col"] + "_encoded"] = le.fit_transform(df[params["target_col"]])

elif target_analysis == "kmeans":
    params["feature_cols"] = st.multiselect("选择聚类特征（数值型）", numeric_cols, default=numeric_cols[:2])
    df_cluster = df[params["feature_cols"]].dropna()
    if len(df_cluster) < params["n_clusters"]:
        st.error(f"❌ 有效样本数（{len(df_cluster)}）小于聚类数量（{params['n_clusters']}）！请减少K值或选择其他变量")
        st.stop()

st.divider()
st.subheader("第五步：执行分析并生成结果")

if st.button("🚀 开始分析"):
    try:
        with st.spinner("正在分析中..."):
            report = ""
            # 描述性统计（通用）
            if target_analysis == "descriptive":
                col = params["target_col"]
                stats_result = df[col].describe()
                st.subheader("📊 描述性统计结果")
                st.dataframe(stats_result.to_frame(), use_container_width=True)
                
                st.subheader("📈 可视化图表")
                if params["chart_type"] == "直方图（分布）":
                    fig = px.histogram(
                        df, x=col, title=f"{col}的分布情况", nbins=20,
                        color_discrete_sequence=[params["chart_color"]],
                        width=params["chart_width"], height=params["chart_height"]
                    )
                else:
                    fig = px.bar(
                        df, y=col, title=f"{col}的均值分布",
                        color=categorical_cols[0] if categorical_cols else None,
                        color_discrete_sequence=[params["chart_color"]] if not categorical_cols else None,
                        width=params["chart_width"], height=params["chart_height"]
                    )
                st.plotly_chart(fig, use_container_width=True)
                
                report = f"""
                ### 📝 分析报告
                1. 分析变量：{col}（数值型）；
                2. 核心统计指标：均值{stats_result['mean']:.2f}、中位数{stats_result['50%']:.2f}、标准差{stats_result['std']:.2f}；
                3. 数据分布特征：{'均匀' if stats_result['std'] < stats_result['mean']*0.3 else '分散'}；
                4. 数据范围：最小值{stats_result['min']:.2f} ~ 最大值{stats_result['max']:.2f}。
                """

            # t检验（通用）
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
                1. 检验场景：{group_col}分组对{result_col}的差异影响；
                2. 检验结论：{group1}与{group2}的差异{significance}（p={p_value:.4f}）；
                3. 差异幅度：{group1}比{group2} {'高' if mean1>mean2 else '低'} {abs(mean1-mean2):.2f}；
                4. 统计依据：独立样本t检验（方差不齐），p<0.05代表差异具有统计学意义。
                """

            # ANOVA（通用）
            elif target_analysis == "anova":
                formula = params["formula"]
                model = ols(formula, data=df).fit()
                anova_result = anova_lm(model, typ=2)
                
                st.subheader("📊 多因素方差分析结果")
                st.dataframe(anova_result, use_container_width=True)
                
                fig = px.box(
                    df, x=params["factor_cols"][0], y=params["result_col"],
                    color=params["factor_cols"][1] if len(params["factor_cols"]) > 1 else None,
                    title=f"各因素对{params['result_col']}的影响分析",
                    color_discrete_sequence=[params["chart_color"]] if len(params["factor_cols"]) == 1 else None,
                    width=params["chart_width"], height=params["chart_height"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                significant_factors = [idx for idx, p in anova_result["PR(>F)"].items() if p < 0.05]
                report = f"""
                ### 📝 分析报告
                1. 分析公式：{formula}；
                2. 显著影响因素（p<0.05）：{', '.join(significant_factors) if significant_factors else '无'}；
                3. 结论：{f'因素{significant_factors}对{params["result_col"]}有显著影响' if significant_factors else '所有因素对因变量无显著影响'}；
                4. 统计依据：p<0.05代表该因素对结果的影响具有统计学意义。
                """

            # 线性回归（通用）
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
                1. 变量关系：{x_col}对{y_col}的影响{significant}（p={p_value:.4f}）；
                2. 回归系数：{coef:.4f}，说明{x_col}每增加1，{y_col} {'增加' if coef>0 else '减少'} {abs(coef):.4f}；
                3. 拟合程度：R²={r_squared:.4f}，{x_col}能解释{y_col} {r_squared*100:.1f}%的变化；
                4. 统计依据：p<0.05代表回归系数具有统计学意义，R²越接近1拟合效果越好。
                """

            # 逻辑回归（通用）
            elif target_analysis == "logistic_reg":
                target_col = params["target_col"]
                feature_cols = params["feature_cols"]
                df_log = df[[*feature_cols, target_col + "_encoded"]].dropna()
                
                model = LogisticRegression()
                model.fit(df_log[feature_cols], df_log[target_col + "_encoded"])
                accuracy = model.score(df_log[feature_cols], df_log[target_col + "_encoded"])
                coefs = dict(zip(feature_cols, model.coef_[0]))
                
                st.subheader("🔮 逻辑回归结果")
                st.write(f"模型准确率：{accuracy:.4f}（预测正确的样本占比）")
                st.write("特征重要性（系数越大影响越强）：")
                st.dataframe(pd.DataFrame({"特征变量": coefs.keys(), "系数": coefs.values()}), use_container_width=True)
                
                fig = px.bar(
                    x=coefs.keys(), y=coefs.values(), title=f"{target_col}预测的特征重要性",
                    color_discrete_sequence=[params["chart_color"]],
                    width=params["chart_width"], height=params["chart_height"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                report = f"""
                ### 📝 分析报告
                1. 预测场景：基于{', '.join(feature_cols)}预测{target_col}；
                2. 模型性能：准确率{accuracy:.4f}，越高代表预测效果越好；
                3. 关键特征：{max(coefs, key=coefs.get)}对预测影响最大（系数{coefs[max(coefs, key=coefs.get)]:.4f}）；
                4. 系数解读：正系数代表特征值越大，越倾向于预测为「{le.classes_[1]}」；负系数则相反。
                """

            # K-Means聚类（通用）
            elif target_analysis == "kmeans":
                feature_cols = params["feature_cols"]
                n_clusters = params["n_clusters"]
                df_cluster = df[feature_cols].dropna()
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                df_cluster["聚类标签"] = kmeans.fit_predict(df_cluster[feature_cols])
                df["聚类标签"] = df_cluster["聚类标签"].reindex(df.index)
                
                st.subheader("🌀 K-Means聚类结果")
                st.write(f"聚类数量：{n_clusters}，各群样本数：")
                st.dataframe(df["聚类标签"].value_counts(), use_container_width=True)
                
                fig = px.scatter(
                    df_cluster, x=feature_cols[0], y=feature_cols[1], color="聚类标签",
                    title=f"数据聚类结果（K={n_clusters}）",
                    color_discrete_sequence=[params["chart_color"], "#ff7f0e", "#2ca02c", "#d62728"][:n_clusters],
                    width=params["chart_width"], height=params["chart_height"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                centers = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)
                st.subheader("各聚类中心（特征均值）")
                st.dataframe(centers, use_container_width=True)
                
                report = f"""
                ### 📝 分析报告
                1. 聚类场景：基于{', '.join(feature_cols)}的数据分群；
                2. 聚类结果：共分为{n_clusters}群，样本数分别为{dict(df['聚类标签'].value_counts())}；
                3. 核心特征：聚类0的{feature_cols[0]}均值{centers.iloc[0][feature_cols[0]]:.2f}，聚类1的{feature_cols[0]}均值{centers.iloc[1][feature_cols[0]]:.2f}；
                4. 应用建议：可根据聚类结果进行数据分组管理、差异化策略制定等。
                """

            # 通用报告下载
            st.divider()
            st.markdown(report)
            file_tag = "单文件分析" if analysis_mode == "单文件独立分析" else "多文件关联分析"
            st.download_button(
                label="📥 下载分析报告（Markdown格式）",
                data=report,
                file_name=f"通用数据分析_{file_tag}_{analysis_type}_报告.md",
                mime="text/markdown"
            )
            
    except Exception as e:
        st.error(f"❌ 分析失败：{str(e)}")
        st.info("💡 可能原因：数据缺失值过多、变量类型不匹配、样本量不足、参数选择不当")

# 结尾交付物提议
要不要我帮你生成一份**通用文件适配指南**，详细说明不同格式文件（CSV/Excel、中英文编码、不同分隔符）的上传注意事项，避免后续遇到读取问题？
