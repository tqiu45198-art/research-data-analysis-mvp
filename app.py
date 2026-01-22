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
    page_title="易懂版科研数据分析助手",
    page_icon="📊",
    layout="wide"
)
st.title("📊 易懂版科研数据分析助手")
st.markdown("**全程大白话说明+数据含义明示+结果通俗解读**")
st.divider()

# ---------------------- 第一步：上传文件（明确数据列含义）----------------------
st.subheader("第一步：上传数据文件（支持CSV/Excel，可传多个）")
st.write("💡 支持你提供的所有赛题文件（如df_customer.csv、df_order.csv等），上传后自动识别列含义")
uploaded_files = st.file_uploader(
    "选择要分析的文件（可多选）", 
    type=["xlsx", "csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("📌 示例：上传df_order.csv（订单数据）、df_proc.csv（仓库成本数据）等，上传后会自动说明每列是啥意思")
    st.stop()

# 预设赛题文件列含义（自动匹配说明）
COL_MEANING = {
    "Name": "名称（门店/仓库/设施名，比如an-shan-shi是鞍山门店）",
    "Type": "类型（门店=Customer/仓库=CDC/RDC/商品=dm/im）",
    "Location": "城市（比如he-ze是菏泽，ji-ning是济宁）",
    "中文名称": "城市中文名称（比如菏泽、济宁）",
    "qty": "订单需求量（单位：吨，比如37.6就是37.6吨海鲜）",
    "SKU": "商品类型（dm=国产海鲜，im=进口海鲜）",
    "Capacity": "仓库最大处理量（单位：吨，比如3000就是最多处理3000吨）",
    "Processing_fee": "处置成本（单位：万元/吨，比如0.007就是每吨成本70元）",
    "Opening_fee": "开仓成本（单位：万元，比如25就是建仓库要花25万元）",
    "Distance": "运输距离（单位：公里，比如2506就是2506公里）",
    "Duration": "运输时间（单位：分钟，比如1639就是约27小时）",
    "Longitude": "经度（城市地理坐标）",
    "Latitude": "纬度（城市地理坐标）",
    "city_area": "城市总面积（单位：平方公里）",
    "resident_pop": "城市人口（单位：万人）",
    "gdp": "城市GDP（单位：亿元）"
}

# 读取文件+自动说明列含义
df_list = []
file_names = []
encodings = ['utf-8-sig', 'gbk', 'utf-8', 'gb2312', 'big5', 'utf-16', 'gb18030', 'latin-1']
seps = [',', '\t', ';', '|', ' ', ':', '\s+']

def clean_column_names(df):
    df.columns = [re.sub(r'[^\w\s\u4e00-\u9fa5/]', '', str(col)).strip() for col in df.columns]
    df.columns = [col if col else f"col_{i}" for i, col in enumerate(df.columns)]
    return df

for file in uploaded_files:
    try:
        file_content = file.read()
        if len(file_content) == 0:
            st.warning(f"⚠️ {file.name} 是空文件，跳过")
            continue
        file.seek(0)
        df = None
        file_name = file.name
        
        # CSV读取
        if file_name.endswith(".csv"):
            for encoding in encodings:
                for sep in seps:
                    try:
                        if encoding in ['utf-16']:
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
            if df is None:
                from csv import Sniffer
                sample = file_content[:4096].decode('utf-8-sig', errors='replace')
                delimiter = Sniffer().sniff(sample).delimiter
                df = pd.read_csv(file, encoding='utf-8-sig', sep=delimiter, on_bad_lines='skip')
                df = clean_column_names(df)
        # Excel读取
        else:
            df = pd.read_excel(file, engine='openpyxl')
            df = clean_column_names(df)
        
        if df is not None and len(df) > 0:
            df_list.append(df)
            file_names.append(file_name)
            # 显示文件信息+列含义说明
            st.success(f"✅ 成功读取：{file_name}（{len(df)}行 × {len(df.columns)}列）")
            st.write(f"📋 该文件列含义：")
            for col in df.columns[:5]:  # 显示前5列，避免过长
                meaning = COL_MEANING.get(col, f"其他数据列（{col}）")
                st.write(f"- {col}：{meaning}")
            if len(df.columns) > 5:
                st.write(f"- 还有{len(df.columns)-5}列，后续步骤会详细说明")
        else:
            st.warning(f"⚠️ {file_name} 无有效数据，跳过")
    except Exception as e:
        st.error(f"❌ 读取{file_name}失败：{str(e)}")

if not df_list:
    st.error("❌ 没有可分析的有效文件，请检查文件内容")
    st.stop()

# ---------------------- 第二步：选择要分析的文件（明确选择逻辑）----------------------
st.subheader("第二步：选择本次要分析的文件")
st.write("💡 上传了多个文件？只选需要的，比如想分析订单量就选df_order.csv")
selected_file_idxs = st.multiselect(
    "勾选要参与分析的文件",
    range(len(file_names)),
    default=[0],
    format_func=lambda x: file_names[x]
)

if len(selected_file_idxs) == 0:
    st.error("❌ 至少选1个文件")
    st.stop()

selected_dfs = [df_list[i] for i in selected_file_idxs]
selected_file_names = [file_names[i] for i in selected_file_idxs]

# ---------------------- 第三步：选择分析模式（通俗说明）----------------------
st.subheader("第三步：选择分析方式")
if len(selected_file_idxs) == 1:
    analysis_mode = "单文件分析"
    st.write(f"📌 自动选单文件分析：{selected_file_names[0]}（只有1个文件可选）")
    df = selected_dfs[0]
else:
    analysis_mode = st.radio(
        "选多个文件了，想怎么分析？",
        options=["单文件分析（只分析其中1个）", "多文件关联分析（比如订单数据+仓库数据合并分析）"]
    )

    if analysis_mode == "单文件分析（只分析其中1个）":
        selected_idx = st.selectbox(
            "选1个要深入分析的文件",
            range(len(selected_file_names)),
            format_func=lambda x: selected_file_names[x]
        )
        df = selected_dfs[selected_idx]
        st.success(f"✅ 已选：{selected_file_names[selected_idx]}")
    
    else:
        st.write("📌 多文件关联：把多个文件按共同字段合并（比如按「城市」合并订单和仓库数据）")
        base_idx = st.selectbox(
            "选1个基础文件（比如订单数据）",
            range(len(selected_file_names)),
            format_func=lambda x: selected_file_names[x]
        )
        df = selected_dfs[base_idx]
        base_name = selected_file_names[base_idx]
        remaining_idxs = [i for i in range(len(selected_file_names)) if i != base_idx]
        关联计数器 = 0

        while len(remaining_idxs) > 0:
            关联计数器 += 1
            remaining_dfs = [selected_dfs[i] for i in remaining_idxs]
            remaining_names = [selected_file_names[i] for i in remaining_idxs]

            st.markdown(f"#### 合并第{关联计数器}个文件")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                join_select_idx = st.selectbox(
                    f"要合并哪个文件？",
                    range(len(remaining_idxs)),
                    format_func=lambda x: remaining_names[x],
                    key=f"join_file_{关联计数器}"
                )
                join_idx = remaining_idxs[join_select_idx]
                join_df = remaining_dfs[join_select_idx]
                join_name = remaining_names[join_select_idx]
            
            with col2:
                base_key = st.selectbox(
                    f"基础文件用哪个字段合并？（比如按城市合并就选Location）",
                    df.columns.tolist(),
                    key=f"base_key_{关联计数器}"
                )
                st.write(f"ℹ️ 该字段含义：{COL_MEANING.get(base_key, '用于合并的共同字段')}")
            
            with col3:
                join_key = st.selectbox(
                    f"要合并的文件用哪个字段对应？（要和左边字段含义一致）",
                    join_df.columns.tolist(),
                    key=f"join_key_{关联计数器}"
                )
                st.write(f"ℹ️ 该字段含义：{COL_MEANING.get(join_key, '用于合并的共同字段')}")
            
            join_type = st.radio(
                f"合并方式？",
                options=["只保留两边都有的数据（推荐）", "保留基础文件所有数据"],
                key=f"join_type_{关联计数器}"
            )
            join_map = {"只保留两边都有的数据（推荐）": "inner", "保留基础文件所有数据": "left"}

            if base_key not in df.columns or join_key not in join_df.columns:
                st.error("❌ 选的字段在文件里没有，请重新选")
                st.stop()

            # 重命名冲突字段
            join_suffix = f"_{join_name.split('.')[0]}"
            join_df_renamed = join_df.rename(
                columns={col: f"{col}{join_suffix}" for col in join_df.columns if col != join_key and col in df.columns}
            )

            try:
                df = pd.merge(df, join_df_renamed, left_on=base_key, right_on=join_key, how=join_map[join_type])
                st.success(f"✅ 合并完成！现在数据有{len(df)}行 × {len(df.columns)}列")
            except Exception as e:
                st.error(f"❌ 合并失败：{str(e)}")
                st.stop()

            remaining_idxs.pop(join_select_idx)

# ---------------------- 第四步：变量识别（明确每列含义）----------------------
st.subheader("第四步：数据变量说明（这些数据能分析啥）")
numeric_cols = []  # 数值型（能算账的，比如订单量、成本）
categorical_cols = []  # 分类型（能分组的，比如商品类型、城市）
binary_categorical_cols = []  # 二分类（只有2种选项的，比如国产/进口）

for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors='raise')
        numeric_cols.append(col)
    except:
        categorical_cols.append(col)
        if df[col].nunique() == 2:
            binary_categorical_cols.append(col)

# 通俗展示变量
st.write("📈 能算账的数值型数据（比如订单量、成本、距离）：")
for col in numeric_cols:
    meaning = COL_MEANING.get(col, f"数值数据（{col}）")
    st.write(f"- {col}：{meaning}")

st.write("🏷️ 能分组的分类型数据（比如商品类型、城市）：")
for col in categorical_cols:
    meaning = COL_MEANING.get(col, f"分类数据（{col}）")
    unique_vals = df[col].unique()[:3]  # 显示前3个选项
    st.write(f"- {col}：{meaning}（选项：{', '.join(map(str, unique_vals))}{'...' if len(df[col].unique())>3 else ''}）")

if not numeric_cols:
    st.error("❌ 没有能算账的数据（比如订单量、成本），无法分析")
    st.stop()

# ---------------------- 第五步：推荐分析类型（通俗化）----------------------
st.subheader("第五步：选择要做的分析（看大白话选）")
st.write("💡 系统自动判断能做啥分析，不能做的会说明原因")

def get_supported_analyses():
    supported = []
    reasons = {}

    # 1. 描述性统计（看数据分布、平均水平）
    supported.append("1. 看数据概况（比如订单量平均多少、成本最高多少）")
    reasons["1. 看数据概况（比如订单量平均多少、成本最高多少）"] = "✅ 有能算账的数据，支持看平均值、分布"

    # 2. 两组差异对比（比如国产和进口海鲜的订单量谁多）
    t_test_support = len(categorical_cols) >= 1 and len(numeric_cols) >= 1
    if t_test_support:
        multi_group_cats = [col for col in categorical_cols if df[col].nunique() >= 2]
        if len(multi_group_cats) == 0:
            t_test_support = False
            reasons["2. 两组差异对比（比如国产和进口海鲜的订单量谁多）"] = "❌ 没有能分组的字段（比如商品类型、城市），没法对比"
        else:
            reasons["2. 两组差异对比（比如国产和进口海鲜的订单量谁多）"] = "✅ 能分组也能算账，支持对比差异"
    else:
        reasons["2. 两组差异对比（比如国产和进口海鲜的订单量谁多）"] = "❌ 缺少分组字段或算账数据"
    
    if t_test_support:
        supported.append("2. 两组差异对比（比如国产和进口海鲜的订单量谁多）")

    # 3. 多因素影响分析（比如商品类型+城市对订单量的共同影响）
    anova_support = len(categorical_cols) >= 1 and len(numeric_cols) >= 1
    if anova_support:
        reasons["3. 多因素影响分析（比如商品类型+城市对订单量的共同影响）"] = "✅ 支持看多个条件对结果的影响"
        supported.append("3. 多因素影响分析（比如商品类型+城市对订单量的共同影响）")
    else:
        reasons["3. 多因素影响分析（比如商品类型+城市对订单量的共同影响）"] = "❌ 缺少分组字段或算账数据"

    # 4. 变量关系分析（比如订单量越多，成本是不是越高）
    regression_support = len(numeric_cols) >= 2
    if regression_support:
        reasons["4. 变量关系分析（比如订单量越多，成本是不是越高）"] = "✅ 有多个算账数据，支持看关系"
        supported.append("4. 变量关系分析（比如订单量越多，成本是不是越高）")
    else:
        reasons["4. 变量关系分析（比如订单量越多，成本是不是越高）"] = "❌ 至少需要2个算账数据（比如订单量+成本）"

    # 5. 分类预测（比如根据城市和人口，预测是国产还是进口海鲜）
    logistic_support = len(binary_categorical_cols) >= 1 and len(numeric_cols) >= 1
    if logistic_support:
        reasons["5. 分类预测（比如根据城市和人口，预测是国产还是进口海鲜）"] = "✅ 有二分类数据（比如国产/进口）和算账数据，支持预测"
        supported.append("5. 分类预测（比如根据城市和人口，预测是国产还是进口海鲜）")
    else:
        reasons["5. 分类预测（比如根据城市和人口，预测是国产还是进口海鲜）"] = "❌ 缺少二分类数据（比如只有1种商品类型）或算账数据"

    # 6. 数据分群（比如把城市按订单量分成高、中、低三组）
    kmeans_support = len(numeric_cols) >= 2
    if kmeans_support:
        reasons["6. 数据分群（比如把城市按订单量分成高、中、低三组）"] = "✅ 有多个算账数据，支持分组"
        supported.append("6. 数据分群（比如把城市按订单量分成高、中、低三组）")
    else:
        reasons["6. 数据分群（比如把城市按订单量分成高、中、低三组）"] = "❌ 至少需要2个算账数据（比如订单量+人口）"

    return supported, reasons

supported_analyses, analysis_reasons = get_supported_analyses()

# 显示不支持的原因
st.write("❌ 暂时不能做的分析及原因：")
for analysis in [
    "1. 看数据概况（比如订单量平均多少、成本最高多少）",
    "2. 两组差异对比（比如国产和进口海鲜的订单量谁多）",
    "3. 多因素影响分析（比如商品类型+城市对订单量的共同影响）",
    "4. 变量关系分析（比如订单量越多，成本是不是越高）",
    "5. 分类预测（比如根据城市和人口，预测是国产还是进口海鲜）",
    "6. 数据分群（比如把城市按订单量分成高、中、低三组）"
]:
    if analysis not in supported_analyses:
        st.write(f"- {analysis_reasons[analysis]}")

if not supported_analyses:
    st.error("❌ 没有能做的分析，请检查数据")
    st.stop()

analysis_type = st.radio(
    "选择要做的分析（选一个你关心的）",
    options=supported_analyses
)

# 映射分析类型
type_map = {
    "1. 看数据概况（比如订单量平均多少、成本最高多少）": "descriptive",
    "2. 两组差异对比（比如国产和进口海鲜的订单量谁多）": "t_test",
    "3. 多因素影响分析（比如商品类型+城市对订单量的共同影响）": "anova",
    "4. 变量关系分析（比如订单量越多，成本是不是越高）": "regression",
    "5. 分类预测（比如根据城市和人口，预测是国产还是进口海鲜）": "logistic_reg",
    "6. 数据分群（比如把城市按订单量分成高、中、低三组）": "kmeans"
}
target_analysis = type_map[analysis_type]

# ---------------------- 第六步：配置参数（通俗说明）----------------------
st.subheader("第六步：设置分析细节（跟着提示选就行）")
params = {}
st.markdown("### 🎨 图表样式（可选，默认就行）")
params["chart_color"] = st.color_picker("图表颜色", value="#1f77b4")
params["chart_width"] = st.slider("图表宽度", 600, 1200, 800)
params["chart_height"] = st.slider("图表高度", 400, 800, 500)

# 按分析类型配置参数（通俗化选项）
if target_analysis == "kmeans":
    params["n_clusters"] = st.slider("分几组？（比如高、中、低就选3）", 2, 10, 3)

elif target_analysis == "descriptive":
    params["target_col"] = st.selectbox(
        "想看哪个数据的概况？（比如订单量qty）",
        numeric_cols,
        format_func=lambda x: f"{x}（{COL_MEANING.get(x, '数值数据')}）"
    )
    params["chart_type"] = st.radio("想看分布（比如订单量集中在哪个区间）还是均值（比如平均订单量）？", ["直方图（看分布）", "柱状图（看均值）"])

elif target_analysis == "t_test":
    valid_group_cols = [col for col in categorical_cols if df[col].nunique() >= 2]
    params["group_col"] = st.selectbox(
        "按什么分组对比？（比如商品类型SKU）",
        valid_group_cols,
        format_func=lambda x: f"{x}（{COL_MEANING.get(x, '分类数据')}）"
    )
    params["result_col"] = st.selectbox(
        "对比哪个数据？（比如订单量qty）",
        numeric_cols,
        format_func=lambda x: f"{x}（{COL_MEANING.get(x, '数值数据')}）"
    )
    group_counts = df[params["group_col"]].nunique()
    if group_counts != 2:
        st.warning(f"⚠️ 这个分组有{group_counts}个选项，系统自动选样本最多的2个来对比")
        top2_groups = df[params["group_col"]].value_counts().nlargest(2).index.tolist()
        df = df[df[params["group_col"]].isin(top2_groups)]

elif target_analysis == "anova":
    params["factor_cols"] = st.multiselect(
        "哪些条件会影响结果？（比如商品类型+城市）",
        categorical_cols,
        default=categorical_cols[0],
        format_func=lambda x: f"{x}（{COL_MEANING.get(x, '分类数据')}）"
    )
    params["result_col"] = st.selectbox(
        "关注哪个结果？（比如订单量qty）",
        numeric_cols,
        format_func=lambda x: f"{x}（{COL_MEANING.get(x, '数值数据')}）"
    )
    params["formula"] = f"{params['result_col']} ~ {' + '.join(params['factor_cols'])}"

elif target_analysis == "regression":
    params["x_col"] = st.selectbox(
        "哪个数据是原因？（比如订单量）",
        numeric_cols,
        format_func=lambda x: f"{x}（{COL_MEANING.get(x, '数值数据')}）"
    )
    params["y_col"] = st.selectbox(
        "哪个数据是结果？（比如成本）",
        [col for col in numeric_cols if col != params["x_col"]],
        format_func=lambda x: f"{x}（{COL_MEANING.get(x, '数值数据')}）"
    )

elif target_analysis == "logistic_reg":
    params["target_col"] = st.selectbox(
        "要预测什么？（比如商品类型是国产还是进口）",
        binary_categorical_cols,
        format_func=lambda x: f"{x}（{COL_MEANING.get(x, '二分类数据')}）"
    )
    params["feature_cols"] = st.multiselect(
        "根据哪些数据预测？（比如城市人口+GDP）",
        numeric_cols,
        default=numeric_cols[:2],
        format_func=lambda x: f"{x}（{COL_MEANING.get(x, '数值数据')}）"
    )
    df[params["target_col"] + "_encoded"] = LabelEncoder().fit_transform(df[params["target_col"]])

elif target_analysis == "kmeans":
    params["feature_cols"] = st.multiselect(
        "根据哪些数据分群？（比如订单量+人口）",
        numeric_cols,
        default=numeric_cols[:2],
        format_func=lambda x: f"{x}（{COL_MEANING.get(x, '数值数据')}）"
    )
    df_cluster = df[params["feature_cols"]].dropna()
    if len(df_cluster) < params["n_clusters"]:
        st.error(f"❌ 有效数据只有{len(df_cluster)}条，分不了{params['n_clusters']}组，请减少分组数")
        st.stop()

# ---------------------- 第七步：执行分析+结果解读----------------------
st.divider()
st.subheader("第七步：分析结果+通俗解读")

if st.button("🚀 开始分析"):
    try:
        with st.spinner("分析中..."):
            report = ""
            interpretation = ""  # 结果解读
            # 1. 描述性统计
            if target_analysis == "descriptive":
                col = params["target_col"]
                stats_result = df[col].describe()
                st.subheader("📊 数据概况结果")
                st.dataframe(stats_result.to_frame(), use_container_width=True)
                
                fig = px.histogram(df, x=col, title=f"{col}的{'分布' if params['chart_type']=='直方图（看分布）' else '均值'}",
                                  color_discrete_sequence=[params["chart_color"]], width=params["chart_width"], height=params["chart_height"])
                st.plotly_chart(fig, use_container_width=True)
                
                # 通俗解读
                mean_val = stats_result['mean']
                std_val = stats_result['std']
                min_val = stats_result['min']
                max_val = stats_result['max']
                interpretation = f"""
                📝 结果解读：
                1. 「{col}」的平均水平是{mean_val:.2f}（{COL_MEANING.get(col, '单位')}）；
                2. 数据{'比较集中' if std_val < mean_val*0.3 else '比较分散'}，大部分数据在{mean_val-std_val:.2f}到{mean_val+std_val:.2f}之间；
                3. 最小是{min_val:.2f}，最大是{max_val:.2f}，差距{'不大' if max_val-min_val < mean_val*1 else '较大'}；
                4. 比如如果是订单量，说明平均每个门店要{mean_val:.2f}吨，最多的要{max_val:.2f}吨，最少的只要{min_val:.2f}吨。
                """

            # 2. 两组差异对比
            elif target_analysis == "t_test":
                group_col, result_col = params["group_col"], params["result_col"]
                group1, group2 = df[group_col].unique()[:2]
                data1, data2 = df[df[group_col]==group1][result_col].dropna(), df[df[group_col]==group2][result_col].dropna()
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                
                st.subheader("🔍 两组差异对比结果")
                st.write(f"{group1}的{result_col}均值：{data1.mean():.2f}（{COL_MEANING.get(result_col, '单位')}）")
                st.write(f"{group2}的{result_col}均值：{data2.mean():.2f}（{COL_MEANING.get(result_col, '单位')}）")
                st.write(f"统计显著性p值：{p_value:.4f}（p<0.05说明差异真的存在，不是巧合）")
                
                fig = px.box(df, x=group_col, y=result_col, color_discrete_sequence=[params["chart_color"]],
                           title=f"{group_col}对{result_col}的影响", width=params["chart_width"], height=params["chart_height"])
                st.plotly_chart(fig, use_container_width=True)
                
                # 通俗解读
                diff_val = abs(data1.mean() - data2.mean())
                if p_value < 0.05:
                    diff_desc = "存在显著差异"
                    reason = "说明这种差异不是偶然的，是两组本身的区别"
                else:
                    diff_desc = "没有显著差异"
                    reason = "说明两组的区别可能是偶然的，没有本质不同"
                interpretation = f"""
                📝 结果解读：
                1. {group1}和{group2}在{result_col}上{diff_desc}（p={p_value:.4f}）；
                2. {group1}比{group2} {'高' if data1.mean()>data2.mean() else '低'} {diff_val:.2f}（{COL_MEANING.get(result_col, '单位')}）；
                3. {reason}；
                4. 比如如果是国产（dm）和进口（im）海鲜的订单量对比，说明进口海鲜订单量确实更高，门店更倾向采购进口海鲜。
                """

            # 3. 多因素影响分析
            elif target_analysis == "anova":
                model = ols(params["formula"], data=df).fit()
                anova_result = anova_lm(model, typ=2)
                st.subheader("📊 多因素影响分析结果")
                st.dataframe(anova_result, use_container_width=True)
                
                fig = px.box(df, x=params["factor_cols"][0], y=params["result_col"],
                           color=params["factor_cols"][1] if len(params["factor_cols"])>1 else None,
                           title=f"各条件对{params['result_col']}的影响", width=params["chart_width"], height=params["chart_height"])
                st.plotly_chart(fig, use_container_width=True)
                
                significant = [idx for idx, p in anova_result["PR(>F)"].items() if p<0.05]
                # 通俗解读
                if significant:
                    sig_desc = f"「{', '.join(significant)}」对结果有显著影响"
                    reason = "说明这些条件真的会改变结果，不是巧合"
                else:
                    sig_desc = "没有条件对结果有显著影响"
                    reason = "说明这些条件的变化不会本质改变结果"
                interpretation = f"""
                📝 结果解读：
                1. 分析的条件是：{', '.join(params['factor_cols'])}，关注的结果是：{params['result_col']}；
                2. {sig_desc}（p<0.05为显著）；
                3. {reason}；
                4. 比如如果商品类型（SKU）显著影响订单量，说明不同类型的海鲜，门店的采购量确实不一样。
                """

            # 4. 变量关系分析
            elif target_analysis == "regression":
                x_col, y_col = params["x_col"], params["y_col"]
                df_reg = df[[x_col, y_col]].dropna()
                model = ols(f"{y_col} ~ {x_col}", data=df_reg).fit()
                
                st.subheader("📈 变量关系分析结果")
                st.write(f"关系公式：{y_col} = {model.params[0]:.2f} + {model.params[x_col]:.4f}×{x_col}")
                st.write(f"拟合度R²：{model.rsquared:.4f}（越接近1，关系越紧密）")
                st.write(f"显著性p值：{model.pvalues[x_col]:.4f}（p<0.05说明关系真的存在）")
                
                fig = px.scatter(df_reg, x=x_col, y=y_col, trendline="ols", color_discrete_sequence=[params["chart_color"]],
                               title=f"{x_col}对{y_col}的影响", width=params["chart_width"], height=params["chart_height"])
                st.plotly_chart(fig, use_container_width=True)
                
                # 通俗解读
                coef = model.params[x_col]
                r2 = model.rsquared
                if model.pvalues[x_col] < 0.05:
                    rel_desc = "存在显著的线性关系"
                    if coef > 0:
                        trend = "增加"
                    else:
                        trend = "减少"
                    trend_desc = f"{x_col}每增加1{COL_MEANING.get(x_col, '单位')}，{y_col}就{trend} {abs(coef):.4f}{COL_MEANING.get(y_col, '单位')}"
                else:
                    rel_desc = "没有显著的线性关系"
                    trend_desc = "两者的变化没有明显的规律"
                interpretation = f"""
                📝 结果解读：
                1. {x_col}和{y_col}之间{rel_desc}（p={model.pvalues[x_col]:.4f}）；
                2. {trend_desc}；
                3. 拟合度R²={r2:.4f}，说明{y_col}的变化中，有{r2*100:.1f}%能通过{x_col}的变化来解释；
                4. 比如如果订单量（qty）和成本（Processing_fee）正相关，说明订单量越大，处置成本越高，符合实际运营逻辑。
                """

            # 5. 分类预测
            elif target_analysis == "logistic_reg":
                target_col, feature_cols = params["target_col"], params["feature_cols"]
                df_log = df[[*feature_cols, target_col + "_encoded"]].dropna()
                model = LogisticRegression()
                model.fit(df_log[feature_cols], df_log[target_col + "_encoded"])
                accuracy = model.score(df_log[feature_cols], df_log[target_col + "_encoded"])
                coefs = dict(zip(feature_cols, model.coef_[0]))
                
                st.subheader("🔮 分类预测结果")
                st.write(f"预测准确率：{accuracy:.4f}（比如0.85就是85%的情况能预测对）")
                st.write("哪些数据对预测影响大？（系数越大，影响越强）")
                st.dataframe(pd.DataFrame({"用于预测的数据": coefs.keys(), "影响强度（系数）": coefs.values()}), use_container_width=True)
                
                fig = px.bar(x=coefs.keys(), y=coefs.values(), color_discrete_sequence=[params["chart_color"]],
                           title="各数据的预测影响强度", width=params["chart_width"], height=params["chart_height"])
                st.plotly_chart(fig, use_container_width=True)
                
                # 通俗解读
                top_feature = max(coefs, key=coefs.get)
                accuracy_desc = "很高" if accuracy > 0.8 else "中等" if accuracy > 0.6 else "较低"
                interpretation = f"""
                📝 结果解读：
                1. 用{', '.join(feature_cols)}预测{target_col}，准确率{accuracy:.2f}，属于{accuracy_desc}水平；
                2. 影响最大的是{top_feature}（影响强度：{coefs[top_feature]:.4f}）；
                3. 正系数说明这个数据越大，越倾向于预测为某一类（比如倾向于进口海鲜），负系数则相反；
                4. 比如预测商品类型时，城市人口（resident_pop）影响最大，说明人口多的城市更可能采购进口海鲜。
                """

            # 6. 数据分群
            elif target_analysis == "kmeans":
                feature_cols = params["feature_cols"]
                df_cluster = df[feature_cols].dropna()
                kmeans = KMeans(n_clusters=params["n_clusters"], random_state=42).fit(df_cluster)
                df["分群标签"] = kmeans.labels_
                
                st.subheader("🌀 数据分群结果")
                st.write(f"共分成{params['n_clusters']}组，每组的数量：")
                st.dataframe(df["分群标签"].value_counts(), use_container_width=True)
                st.write("每组的核心特征（比如平均订单量、平均人口）：")
                centers = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)
                st.dataframe(centers, use_container_width=True)
                
                fig = px.scatter(df_cluster, x=feature_cols[0], y=feature_cols[1], color=kmeans.labels_,
                               color_discrete_sequence=[params["chart_color"], "#ff7f0e", "#2ca02c", "#d62728"][:params["n_clusters"]],
                               title=f"数据分群结果（共{params['n_clusters']}组）", width=params["chart_width"], height=params["chart_height"])
                st.plotly_chart(fig, use_container_width=True)
                
                # 通俗解读
                cluster_desc = []
                for i in range(params["n_clusters"]):
                    center = centers.iloc[i]
                    desc = f"第{i}组："
                    for col in feature_cols:
                        desc += f"{col}平均{center[col]:.2f}（{COL_MEANING.get(col, '单位')}），"
                    cluster_desc.append(desc[:-1])
                interpretation = f"""
                📝 结果解读：
                1. 数据按{', '.join(feature_cols)}分成了{params['n_clusters']}组，每组数量分别是{dict(df['分群标签'].value_counts())}；
                2. 各组特征：
                   - {cluster_desc[0]}
                   - {cluster_desc[1]}
                   {'- ' + cluster_desc[2] if params['n_clusters']>=3 else ''}
                3. 比如按订单量和人口分群，第0组是“人口多、订单量大”的城市，第1组是“人口少、订单量小”的城市，可针对性制定供应链策略。
                """

            # 显示结果+解读
            st.divider()
            st.markdown("### 📝 核心结论+通俗解读")
            st.markdown(interpretation)
            
            # 下载报告
            full_report = f"# 分析报告\n## 分析类型：{analysis_type}\n## 核心结果\n{report}\n## 通俗解读\n{interpretation}"
            st.download_button(
                label="📥 下载完整报告（含解读）",
                data=full_report,
                file_name=f"易懂版分析报告_{analysis_type.replace('、', '').replace('（', '').replace('）', '')}.md",
                mime="text/markdown"
            )
            
    except Exception as e:
        st.error(f"❌ 分析失败：{str(e)}")
        st.info("💡 可能原因：数据太少、选的字段不对、部分数据缺失太多")
