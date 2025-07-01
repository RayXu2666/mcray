import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint

try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告：系统中文字体配置失败，将使用英文显示")

def load_temperature_data(file_path):
    try:
        data = pd.read_excel(file_path)

        # 自动检测列名
        column_mapping = {
            '时间': 'time',
            'Time': 'time',
            '温度': 'temp',
            'Temperature': 'temp',
            '加热电压': 'voltage',
            'Voltage': 'voltage'
        }

        data.columns = [column_mapping.get(col, col) for col in data.columns]

        if len(data.columns) >= 3:
            data = data.rename(columns={
                data.columns[0]: 'time',
                data.columns[1]: 'temp',
                data.columns[2]: 'voltage'
            })
        else:
            raise ValueError("Excel文件必须包含时间、温度和电压三列数据")

        return data

    except Exception as e:
        print(f"读取文件时出错: {e}")
        raise


def system_identification(data):
    t = data['time'].values
    y = data['temp'].values
    u = data['voltage'].values

    # 找到阶跃变化的时刻
    step_change_idx = np.where(np.diff(u) > 0)[0]
    if len(step_change_idx) == 0:
        print("未检测到阶跃输入")
        return 1.6031, 1597.5, 119.5

    step_idx = step_change_idx[0]
    t_step = t[step_idx]
    u_step = u[step_idx + 1] - u[step_idx]

    # 稳态值
    y_ss = np.mean(y[-100:])

    # 计算28.3%和63.2%响应点
    y28 = y[step_idx] + 0.283 * (y_ss - y[step_idx])
    y63 = y[step_idx] + 0.632 * (y_ss - y[step_idx])

    # 找到对应时间点
    try:
        idx28 = np.where(y[step_idx:] > y28)[0][0] + step_idx
        idx63 = np.where(y[step_idx:] > y63)[0][0] + step_idx
    except IndexError:
        print("警告：无法确定响应点，使用默认参数")
        return 0.5, 100, 10

    t28 = t[idx28]
    t63 = t[idx63]

    # 计算FOPDT参数
    tau = 1.5 * (t63 - t28)
    theta = t28 - t_step - tau / 3
    K = (y_ss - y[step_idx]) / u_step

    return K, tau, theta

# 添加一阶时滞系统的微分方程
def fopdt_model(T, t, u_func, K, tau, theta):
    try:
        # 获取 t - θ 之前的输入
        if t - theta <= 0:
            u = u_func(0)
        else:
            u = u_func(t - theta)
    except:
        u = u_func(0)
    dTdt = ( -T + K * u ) / tau
    return dTdt


# 仿真主函数
def simulate_pid(Kp, Ki, Kd, K, tau, theta, T0=20, setpoint=35, duration=10000, dt=1.0):
    n = int(duration / dt)
    time = np.linspace(0, duration, n)
    temp = np.zeros(n)
    voltage = np.zeros(n)
    pid = PIDController(Kp, Ki, Kd, setpoint)
    temp[0] = T0
    u_record = []

    for i in range(1, n):
        u = pid.update(temp[i-1], dt)
        u_record.append(u)
        u_func = lambda t_delay: u_record[int(t_delay/dt)] if int(t_delay/dt) < len(u_record) else u_record[-1]
        T = odeint(fopdt_model, temp[i-1], [time[i-1], time[i]], args=(u_func, K, tau, theta))
        temp[i] = T[-1][0]
        voltage[i] = u

    return time, temp, voltage

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return max(0, min(10, output))


if __name__ == "__main__":
    # 使用原始字符串处理Windows路径
    file_path = r"C:\Users\30453\Desktop\大作业\temperature.xlsx"

    try:
        # 1. 加载数据
        data = load_temperature_data(file_path)
        print("数据加载成功！显示前五行数据如下：")
        print(data.head())

        # 2. 系统辨识
        K, tau, theta = system_identification(data)
        print(f"\n辨识参数: K={K:.4f} °C/V, τ={tau:.2f} s, θ={theta:.2f} s")


        # 5. 绘制原始数据
        plt.figure(figsize=(12, 6))
        plt.plot(data['time'], data['temp'], 'b-', label='原始温度')
        plt.plot(data['time'], data['voltage'], 'g-', label='原始电压')
        plt.xlabel('时间 (s)')
        plt.ylabel('温度 (°C) / 电压 (V)')
        plt.title('原始数据曲线')
        plt.legend()
        plt.grid(True)
        plt.show()



    except Exception as e:
        print(f"\n程序运行出错: {e}")
        print("请检查：")
        print("1. 文件路径是否正确")
        print("2. Excel文件是否包含时间、温度和电压三列数据")
        print("3. 文件是否被其他程序占用")
