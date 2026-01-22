# AI功能1：自动数据分析（基于真实统计结果）
with st.expander("📑 AI自动数据分析（基于真实计算结果）", expanded=True):
    st.markdown("代码会先自动执行真实统计分析，AI仅基于这些真实结果生成报告（无假数值）")
    if st.button("🚀 开始AI自动分析（真实数据）"):
        with st.spinner("正在执行真实统计分析，请稍候..."):
            # ---------------------- 步骤1：自动执行真实统计分析（调用现有函数，结果100%真实） ----------------------
            # 1. 描述统计（真实结果）
            desc_res = descriptive_analysis(df, var_types['numeric']) if var_types['numeric'] else "无数值型变量"
            desc_text = "### 描述统计结果\n" + desc_res.to_string() if var_types['numeric'] else "无数值型变量"
            
            # 2. 数值变量相关矩阵（真实结果）
            corr_res = correlation_analysis(df, var_types['numeric'], 'pearson') if len(var_types['numeric'])>=2 else "数值型变量不足2个"
            corr_text = "### 数值变量相关矩阵（Pearson）\n" + corr_res['相关矩阵'].to_string() if len(var_types['numeric'])>=2 else "数值型变量不足2个"
            
            # 3. 分类型变量频数（真实结果）
            freq_res = frequency_analysis(df, var_types['categorical']) if var_types['categorical'] else "无分类型变量"
            freq_text = "### 分类型变量频数结果\n"
            if var_types['categorical']:
                for col in var_types['categorical']:
                    freq_text += f"\n{col}：\n" + freq_res[col].to_string()
            else:
                freq_text = "无分类型变量"
            
            # 4. 关键均值检验（若有二分类变量，自动做两独立样本t检验）
            ttest_text = "### 均值检验结果\n"
            if var_types['binary_categorical'] and var_types['numeric']:
                group_col = var_types['binary_categorical'][0]  # 取第一个二分类变量
                test_col = var_types['numeric'][0]  # 取第一个数值变量
                ttest_res = t_test_independent(df, test_col, group_col)
                if 'error' not in ttest_res:
                    ttest_text += f"两独立样本t检验（{test_col}按{group_col}分组）：\n"
                    ttest_text += f"t值={ttest_res['t值']}，p值={ttest_res['p值']}，{list(ttest_res.keys())[2]}={ttest_res[list(ttest_res.keys())[2]]}，{list(ttest_res.keys())[3]}={ttest_res[list(ttest_res.keys())[3]]}"
            else:
                ttest_text += "无符合条件的二分类变量，未执行均值检验"

            # ---------------------- 步骤2：将真实结果整理为提示词上下文 ----------------------
            real_stats_text = f"""以下是该数据的真实统计分析结果，你只能基于这些结果生成分析报告，**禁止编造任何数值**：
{desc_text}

{corr_text}

{freq_text}

{ttest_text}
"""

            # ---------------------- 步骤3：调用AI，基于真实结果生成报告 ----------------------
            st.markdown("### 真实统计分析结果（供AI参考）")
            st.text(real_stats_text)  # 可选项：展示真实结果给用户核对
            st.markdown("### AI分析结论（基于真实数据）")
            
            prompt = f"""你是资深科研统计分析师，需基于以下**真实的统计结果**生成分析报告，要求：
1. 只能使用提供的真实统计结果，**绝对不能编造任何数值、统计量、p值**；
2. 先总结数据的基本特征（基于描述统计、频数结果）；
3. 分析变量间的关系（基于相关矩阵）；
4. 解读统计检验的意义（若有均值检验结果）；
5. 最后给出客观的分析结论和研究建议；
6. 格式清晰，分点排版，语言专业且易懂。

真实统计结果：
{real_stats_text}
"""
            # 调用API并流式输出
            stream = call_deepseek_api(prompt)
            st.write_stream(stream)
