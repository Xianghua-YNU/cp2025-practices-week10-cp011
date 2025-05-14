import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

def calculate_distance():
    # 读取数据
    data = np.loadtxt('Velocities.txt')
    time = data[:, 0]
    velocities = data[:, 1]

    # 计算总距离
    total_distance = np.trapz(velocities, time)
    print(f"计算出的总距离: {total_distance} 米")

    # 计算累积距离和速度
    distance = cumulative_trapezoid(velocities, time, initial=0)
    plt.figure(figsize=(10, 5))

    # 绘制速度-时间曲线
    plt.plot(time, velocities, label='速度 (米/秒)')
    # 绘制累积距离-时间曲线
    plt.plot(time, distance, label='累积距离 (米)')

    # 添加标题和标签
    plt.title('速度与距离随时间变化')
    plt.xlabel('时间 (秒)')
    plt.ylabel('速度 (米/秒) / 距离 (米)')
    plt.legend()

    # 显示图表
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    calculate_distance()

