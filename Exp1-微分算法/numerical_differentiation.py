import numpy as np
import matplotlib.pyplot as plt
from sympy import tanh, symbols, diff, lambdify

def f(x):
    """计算函数值 f(x) = 1 + 0.5*tanh(2x)
    
    参数：
        x: 标量或numpy数组，输入值
    
    返回：
        标量或numpy数组，函数值
    """
    # 实现函数 f(x) = 1 + 0.5*tanh(2x)
    return 1 + 0.5*np.tanh(2*x)

def get_analytical_derivative(): #无需任何参数，直接返回一个可调用的解析导数
    """使用sympy获取解析导数函数
    
    返回：
        可调用函数，用于计算导数值
    """
    x = symbols('x') # 创建一个符号自变量x
    fx = 1 + 0.5*tanh(2*x) # 利用符号自变量x定义符号表达式f(x)=1 + 0.5tanh(2x)
    dfx = diff(fx, x) # 使用sympy.diff(fx, x)计算f(x)的解析导数的数学形式，但还不能直接数值计算
    df = lambdify(x, dfx, 'numpy') 
    # 以x为输入，将dfx转成可直接使用NumPy数组输入并返回数值输出的Python函数
    # 'numpy'是将数学函数xxx形式转换成对应的numpy.xxx的形式，使得返回的df函数就可以直接处理NumPy数组
    return df 

def calculate_central_difference(x, f): # 需要参数x、f，使用中心差分法计算数值导数
    """
    参数：
        x: numpy数组，要计算导数的点
        f: 可调用函数，要求导的函数
    返回：
        numpy数组x[1:-1]处的导数值
    """
    
    h = x[1] - x[0] # 计算数组x中相邻两个值之间的步长h
    '''
    中心差分公式f'(xi)≈[f(xi+1)-f(xi-1)]/2h，易知该方法只能计算数组x中从第二个元素到倒数第二个元素处的导数
    故取x(i+1)从第三个元素到最后，x(i-1)从第一个元素到倒数第三个元素
    '''
    return (f(x[2:])-f(x[:-2]))/(2*h)

def richardson_derivative_all_orders(x, f, h, max_order=3):
    # 需要参数x、f、h、max_order(默认值为3)，使用Richardson外推法计算不同阶数的导数值 计算数值导数
    """
    参数：
        x: 标量，要计算导数的点
        f: 可调用函数，要求导的函数
        h: 浮点数，初始步长
        max_order: 整数，最大外推阶数
    
    返回：
        列表，不同阶数计算的导数值
    """
    # 
    d = np.zeros((max_order + 1, max_order + 1), float)
    d[:, 0] = (f(x + h / (2 ** np.arange(max_order + 1))) - f(x - h / (2 ** np.arange(max_order + 1)))) / (2 * h / (2 ** np.arange(max_order + 1)))
    for i in range(1, max_order + 1):
        for j in range(1, i + 1):
            d[i, j] = d[i, j - 1] + (d[i, j - 1] - d[i - 1, j - 1]) / (4 ** j - 1)
    return d[:, -1]  # 返回每一层的最高阶估计值

def create_comparison_plot(x, x_central, dy_central, dy_richardson, df_analytical):
    """创建对比图，展示导数计算结果和误差分析
    
    参数：
        x: numpy数组，所有x坐标点
        x_central: numpy数组，中心差分法使用的x坐标点
        dy_central: numpy数组，中心差分法计算的导数值
        dy_richardson: numpy数组，Richardson方法计算的导数值
        df_analytical: 可调用函数，解析导数函数
    """
    df_true = df_analytical(x)
    df_true_central = df_analytical(x_central)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

    # 1. 导数对比图
    ax1.plot(x_central, dy_central, label='Central Difference', marker='o')
    ax1.plot(x_central, dy_richardson, label='Richardson (Order 3)', marker='s')
    ax1.plot(x_central, df_true_central, label='Analytical', linestyle='--')
    ax1.set_title('Derivative Comparison')
    ax1.legend()
    ax1.grid()

    # 2. 误差分析图（对数坐标）
    error_central = np.abs(dy_central - df_true_central)
    error_richardson = np.abs(dy_richardson - df_true_central)
    ax2.plot(x_central, error_central, label='Central Error', marker='o')
    ax2.plot(x_central, error_richardson, label='Richardson Error', marker='s')
    ax2.set_yscale('log')
    ax2.set_title('Error Comparison (Log Scale)')
    ax2.legend()
    ax2.grid()

    # 3. Richardson不同阶误差对比（固定x）
    x0 = 0.5
    df0 = df_analytical(x0)
    orders = np.arange(1, 7)
    errors = []
    for order in orders:
        val = richardson_derivative_all_orders(x0, f, h=0.1, max_order=order)[-1]
        errors.append(abs(val - df0))
    ax3.plot(orders, errors, marker='o')
    ax3.set_yscale('log')
    ax3.set_title('Richardson Error vs Order')
    ax3.set_xlabel('Order')
    ax3.set_ylabel('Error')
    ax3.grid()

    # 4. 步长敏感性分析图（双对数坐标）
    h_vals = np.logspace(-5, -1, 20)
    errors_c = []
    errors_r = []
    for h in h_vals:
        xc = np.array([0.0])
        df_true = df_analytical(xc)[0]
        df_c = (f(xc + h) - f(xc - h)) / (2 * h)
        df_r = richardson_derivative_all_orders(xc[0], f, h=h, max_order=3)[-1]
        errors_c.append(abs(df_c - df_true))
        errors_r.append(abs(df_r - df_true))
    ax4.loglog(h_vals, errors_c, label='Central', marker='o')
    ax4.loglog(h_vals, errors_r, label='Richardson', marker='s')
    ax4.set_title('Error vs Step Size (Log-Log)')
    ax4.set_xlabel('h')
    ax4.set_ylabel('Error')
    ax4.legend()
    ax4.grid()
    
    plt.tight_layout()
    plt.show()

def main():
    """运行数值微分实验的主函数"""
    # TODO: 设置实验参数
    
    # TODO: 获取解析导数函数
    
    # TODO: 计算中心差分导数
    
    # TODO: 计算Richardson外推导数
    
    # TODO: 绘制结果对比图

x = np.linspace(-2, 2, 1001)
    h = x[1] - x[0]
    df_analytical = get_analytical_derivative()
    # 中心差分导数计算（忽略边界点）
    x_central = x[1:-1]
    dy_central = calculate_central_difference(x, f)

    # Richardson外推导数（在相同点上计算）
    dy_richardson = np.array([richardson_derivative_all_orders(xi, f, h, max_order=3)[-1] for xi in x_central])

    # 绘图展示
    create_comparison_plot(x, x_central, dy_central, dy_richardson, df_analytical)

if __name__ == '__main__':
    main()
