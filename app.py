import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from statsmodels.formula.api import ols, glm
import warnings
warnings.filterwarnings('ignore')  # 忽略无关警告，避免用户困惑

# ---------------------- 1. 页面基础设置（前端交互层）----------------------
st.set_page_config(
    page_title="科研数据分析助手-MVP",
    page_icon="📊",
    layout="wide"  # 宽屏布局，方便展示图表和报告
)
st.title("📊 科研数据分析助手（MVP版）")
st.markdown("**本科生专属：上传数据→说需求→拿报告，零代码搞定科研分析**")
st.divider()  # 分割线，让页面更清晰

# ---------------------- 2. 数据上传与预处理（用户交互+数据校验）----------------------
st.subheader("第一步：上传你的数据")
uploaded_file = st.file_uploader("支持Excel(.xlsx)或CSV(.csv)文件", type=["xlsx", "csv"])

if not uploaded_file:
    # 没有上传文件时，显示示例数据提示
    st.info("💡 示例：上传包含「分组变量」（如性别、组别）和「数值变量」（如成绩、分数）的表格")
    st.stop()  # 停止往下执行，等待用户上传文件

# 加载数据（自动识别文件格式）
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:  # xlsx
        df = pd.read_excel(uploaded_file)
    st.success("✅ 数据上传成功！")
    
    # 显示数据预览（让用户确认数据正确）
    st.subheader("数据预览（前5行）")
    st.dataframe(df.head(), use_container_width=True)
    
    # 自动识别变量类型（方便后续分析）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # 数值型变量（用于计算）
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()  # 分类变量（用于分组）
    
    st.subheader("数据变量识别")
    st.write(f"📈 数值型变量（可分析均值/差异/关系）：{', '.join(numeric_cols)}")
    st.write(f"🏷️ 分类变量（可作为分组依据）：{', '.join(categorical_cols)}")
    
    if not numeric_cols:
        st.error("❌ 未识别到数值型变量！请确保数据中包含成绩、分数等可计算的列")
        st.stop()

except Exception as e:
    st.error(f"❌ 数据加载失败：{str(e)}")
    st.stop()

# ---------------------- 3. 需求输入与识别（自然语言+勾选双保险）----------------------
st.divider()
st.subheader("第二步：告诉我你的分析需求")

# 双保险输入：自然语言框 + 勾选框（避免初期大模型识别不准）
user_query = st.text_input(
    "输入你的需求（示例：分析男生和女生的成绩差异 / 学习时长对成绩的影响）",
    placeholder="请用简单的话描述你的分析需求..."
)

# 勾选框选项（对应3个核心功能）
analysis_type = st.radio(
    "快速选择分析类型（不确定需求时直接选）",
    options=["描述性统计（均值/分布）", "独立样本t检验（两组差异）", "简单线性回归（变量关系）"]
)

# 需求识别逻辑（简化版，不用大模型，关键词匹配）
def identify_analysis(user_query, selected_type):
    if "差异" in user_query or "不同" in user_query or selected_type == "独立样本t检验（两组差异）":
        return "t_test"
    elif "关系" in user_query or "影响" in user_query or "相关" in user_query or selected_type == "简单线性回归（变量关系）":
        return "regression"
    else:  # 默认描述性统计
        return "descriptive"

target_analysis = identify_analysis(user_query, analysis_type)
st.write(f"🎯 系统识别分析类型：{target_analysis.replace('_', ' ')}")

# ---------------------- 4. 分析参数配置（让用户选择具体变量）----------------------
st.subheader("第三步：配置分析参数")
params = {}  # 存储用户选择的参数

if target_analysis == "descriptive":
    # 描述性统计：选择要分析的数值变量
    params["target_col"] = st.selectbox("选择要分析的变量", numeric_cols)
    params["chart_type"] = st.radio("选择图表类型", ["直方图（分布）", "柱状图（均值）"])

elif target_analysis == "t_test":
    # t检验：需要选择分组变量（2组）和结果变量
    if not categorical_cols:
        st.error("❌ 未识别到分类变量！无法进行t检验（需要性别、组别等分组依据）")
        st.stop()
    params["group_col"] = st.selectbox("选择分组变量（如性别、组别）", categorical_cols)
    params["result_col"] = st.selectbox("选择要比较的数值变量（如成绩、分数）", numeric_cols)
    
    # 校验分组是否为2组（t检验要求）
    group_counts = df[params["group_col"]].nunique()
    if group_counts != 2:
        st.warning(f"⚠️ 分组变量「{params['group_col']}」有{group_counts}组，t检验仅支持2组！将自动取前2组数据")
        top2_groups = df[params["group_col"]].value_counts().nlargest(2).index.tolist()
        df = df[df[params["group_col"]].isin(top2_groups)]

elif target_analysis == "regression":
    # 线性回归：选择自变量和因变量
    if len(numeric_cols) < 2:
        st.error("❌ 至少需要2个数值变量！（1个自变量，1个因变量）")
        st.stop()
    params["x_col"] = st.selectbox("选择自变量（如学习时长、刷题量）", numeric_cols)
    params["y_col"] = st.selectbox("选择因变量（如成绩、分数）", [col for col in numeric_cols if col != params["x_col"]])

# ---------------------- 5. 执行分析（调用统计库，核心计算逻辑）----------------------
st.divider()
st.subheader("第四步：分析结果与智能报告")

if st.button("🚀 开始分析"):
    try:
        with st.spinner("正在分析中..."):
            # 5.1 描述性统计
            if target_analysis == "descriptive":
                col = params["target_col"]
                stats_result = df[col].describe()  # 均值、方差、最小值等
                st.subheader("📊 描述性统计结果")
                st.dataframe(stats_result.to_frame(), use_container_width=True)
                
                # 生成图表
                st.subheader("📈 可视化图表")
                if params["chart_type"] == "直方图（分布）":
                    fig = px.histogram(df, x=col, title=f"{col}的分布情况", nbins=20)
                else:
                    fig = px.bar(df, y=col, title=f"{col}的均值", color_discrete_sequence=["#1f77b4"])
                st.plotly_chart(fig, use_container_width=True)
                
                # 智能解读（通俗语言）
                report = f"""
                ### 📝 分析报告
                1. 变量「{col}」的核心统计信息：
                   - 均值：{stats_result['mean']:.2f}，中位数：{stats_result['50%']:.2f}
                   - 标准差：{stats_result['std']:.2f}（数值越小，数据越集中）
                   - 最小值：{stats_result['min']:.2f}，最大值：{stats_result['max']:.2f}
                2. 结论：该变量的分布{'相对均匀' if stats_result['std'] < stats_result['mean']*0.3 else '较为分散'}，
                   整体水平处于{stats_result['mean']:.2f}左右，适合用于后续的差异分析或关系分析。
                """

            # 5.2 独立样本t检验
            elif target_analysis == "t_test":
                group_col = params["group_col"]
                result_col = params["result_col"]
                group1, group2 = df[group_col].unique()[:2]
                data1 = df[df[group_col] == group1][result_col].dropna()
                data2 = df[df[group_col] == group2][result_col].dropna()
                
                # 执行t检验（假设方差齐性）
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)  # Welch's t-test，无需方差齐性
                mean1, mean2 = data1.mean(), data2.mean()
                diff = mean1 - mean2
                
                st.subheader("🔍 独立样本t检验结果")
                st.write(f"分组1：{group1}（样本量：{len(data1)}，均值：{mean1:.2f}）")
                st.write(f"分组2：{group2}（样本量：{len(data2)}，均值：{mean2:.2f}）")
                st.write(f"t统计量：{t_stat:.4f}")
                st.write(f"p值：{p_value:.4f}")
                
                # 可视化两组差异
                fig = px.box(df, x=group_col, y=result_col, title=f"{group_col}对{result_col}的影响")
                st.plotly_chart(fig, use_container_width=True)
                
                # 智能解读（重点讲p值含义，避免专业术语）
                significance = "显著" if p_value < 0.05 else "不显著"
                report = f"""
                ### 📝 分析报告
                1. 检验目的：分析「{group_col}」的两个组别（{group1} vs {group2}）在「{result_col}」上的差异是否显著。
                2. 核心结果：
                   - {group1}的均值（{mean1:.2f}）{'高于' if mean1 > mean2 else '低于'} {group2}（{mean2:.2f}），差异值为{abs(diff):.2f}。
                   - p值 = {p_value:.4f}（判断标准：p < 0.05 则差异显著）。
                3. 结论：{group1}和{group2}在「{result_col}」上的差异{significance}，
                   {'说明两组存在本质区别（非偶然结果）' if p_value < 0.05 else '说明两组差异可能是偶然导致，需更大样本验证'}。
                """

            # 5.3 简单线性回归
            elif target_analysis == "regression":
                x_col = params["x_col"]
                y_col = params["y_col"]
                df_reg = df[[x_col, y_col]].dropna()  # 剔除缺失值
                
                # 执行线性回归
                model = glm(f"{y_col} ~ {x_col}", data=df_reg, family=statsmodels.families.Gaussian())
                result = model.fit()
                r_squared = result.rsquared  # R²（拟合优度）
                coef = result.params[x_col]  # 回归系数（斜率）
                p_value = result.pvalues[x_col]  # 显著性p值
                
                st.subheader("📈 简单线性回归结果")
                st.write(f"回归方程：{y_col} = {result.params[0]:.2f} + {coef:.4f} × {x_col}")
                st.write(f"R²（拟合优度）：{r_squared:.4f}（越接近1，拟合效果越好）")
                st.write(f"回归系数显著性p值：{p_value:.4f}（p < 0.05 则关系显著）")
                
                # 可视化回归拟合线
                fig = px.scatter(df_reg, x=x_col, y=y_col, trendline="ols", title=f"{x_col}对{y_col}的影响")
                st.plotly_chart(fig, use_container_width=True)
                
                # 智能解读
                significance = "显著" if p_value < 0.05 else "不显著"
                trend = "正相关" if coef > 0 else "负相关"
                report = f"""
                ### 📝 分析报告
                1. 分析目的：探究「{x_col}」对「{y_col}」的影响关系。
                2. 核心结果：
                   - 回归系数：{coef:.4f}，说明{y_col}与{x_col}呈{trend}（系数越大，影响越强）。
                   - R² = {r_squared:.4f}，说明{y_col}的变化中，有{r_squared*100:.1f}%可由{x_col}解释。
                   - p值 = {p_value:.4f}（判断标准：p < 0.05 则关系显著）。
                3. 结论：{x_col}对{y_col}的影响{significance}，
                   {'可通过{x_col}的变化预测{y_col}的趋势' if p_value < 0.05 else '两者的线性关系较弱，需考虑其他变量'}。
                """

            # 显示最终报告
            st.divider()
            st.markdown(report)
            
            # 报告下载功能（用户需要保存到科研报告中）
            st.download_button(
                label="📥 下载分析报告（Markdown格式）",
                data=report,
                file_name=f"科研数据分析报告_{target_analysis}.md",
                mime="text/markdown"
            )
            
    except Exception as e:
        st.error(f"❌ 分析失败：{str(e)}")
        st.info("💡 可能原因：数据格式错误、样本量不足（建议每组至少3个数据）、变量选择不当")