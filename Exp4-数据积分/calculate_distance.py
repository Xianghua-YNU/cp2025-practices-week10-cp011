import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

def calculate_distance():
    # 读取数据
    data = np.loadtxt('Velocities.txt')
    t = data[:, 0]  # 时间
    v = data[:, 1]  # 速度

    # 计算总距离
    total_distance = np.trapz(v, t)
    print(f"计算出的总距离: {total_distance} 米")

    # 计算累积距离
    cumulative_distance = cumtrapz(v, t, initial=0)

    # 绘制速度和距离随时间变化的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(t, v, label='速度 (米/秒)')
    plt.plot(t, cumulative_distance, label='距离 (米)')
    plt.title('速度与距离随时间变化')
    plt.xlabel('时间 (秒)')
    plt.ylabel('速度 (米/秒) / 距离 (米)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    calculate_distance()
