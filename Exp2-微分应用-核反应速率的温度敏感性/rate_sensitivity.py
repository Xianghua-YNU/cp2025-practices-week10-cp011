import numpy as np
import matplotlib.pyplot as plt

def q3a(T):
    """
    计算 3-alpha 反应速率中与温度相关的部分 q / (rho^2 Y^3)
    输入: T - 温度 (K)
    返回: 速率因子 (erg * cm^6 / (g^3 * s))
    """
    # TODO: 在此实现3-α反应速率计算
    # 提示：
    # 1. 将温度转换为以 10^8 K 为单位
    # 2. 注意处理温度为零的特殊情况
    # 3. 使用公式：q_{3α} = 5.09×10^11 ρ^2 Y^3 T_8^(-3) exp(-44.027/T_8)
    if T8 <= 0:
        return 0.0
    rate_factor = 5.09e11 * T8**(-3.0) * np.exp(-44.027 / T8)
    return rate_factor

def plot_rate(filename="rate_vs_temp.png"):
    """绘制速率因子随温度变化的 log-log 图"""
    # TODO: 在此实现绘图函数
    # 提示：
    # 1. 使用 np.logspace 生成温度数据点
    # 2. 计算对应的速率值
    # 3. 使用 plt.loglog 绘制双对数图
    # 4. 添加适当的标签和标题
    T_values = np.logspace(np.log10(3.0e8), np.log10(5.0e9), 100) # 温度范围 3e8 K to 5e9 K
    q_values = [q3a(T) for T in T_values]

    fig, ax = plt.subplots()
    ax.loglog(T_values, q_values)
    ax.set_xlabel("Temperature T (K)")
    ax.set_ylabel(r"$q_{3\alpha}/(\rho^2 Y^3)$  (erg cm$^6$ g$^{-3}$ s$^{-1}$)")
    ax.set_title("3-α Reaction Rate Factor vs Temperature")
    ax.grid(True, which="both", ls=":")
    plt.show()

if __name__ == "__main__":
    # 计算并打印 nu 值
    print("   温度 T (K)    :   ν (敏感性指数)")
    print("--------------------------------------")

    temperatures_K = [1.0e8, 2.5e8, 5.0e8, 1.0e9, 2.5e9, 5.0e9]
    h = 1.0e-8  # 扰动因子

    for T0 in temperatures_K:
        q_T0 = q3a(T0)
        if q_T0 == 0: # 避免除以零
            nu = np.nan # Not a Number
        else:
            delta_T = h * T0
            q_T0_plus_deltaT = q3a(T0 + delta_T)
            
            # 使用前向差分计算 dq/dT
            dq_dT_approx = (q_T0_plus_deltaT - q_T0) / delta_T
            
            # 计算 nu
            nu = (T0 / q_T0) * dq_dT_approx
            
        # 格式化输出
        print(f"  {T0:10.3e} K : {nu:8.3f}")

    # (可选) 调用绘图函数
    plot_rate()
    # TODO: 实现温度敏感性指数的计算
    # 提示：
    # 1. 对每个温度点计算 q3a
    # 2. 使用前向差分计算导数
    # 3. 计算敏感性指数 nu
    # 4. 注意处理特殊情况（如 q = 0）

    # TODO: 调用绘图函数展示结果
