import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# === 加载数据函数 ===
def load_temperature_data(file_path):
    data = pd.read_excel(file_path)
    column_mapping = {'时间': 'time', 'Time': 'time', '温度': 'temp', 'Temperature': 'temp',
                      '加热电压': 'voltage', 'Voltage': 'voltage'}
    data.columns = [column_mapping.get(col, col) for col in data.columns]
    if len(data.columns) >= 3:
        data = data.rename(columns={data.columns[0]: 'time', data.columns[1]: 'temp', data.columns[2]: 'voltage'})
    else:
        raise ValueError("Excel文件必须包含时间、温度和电压三列数据")
    return data

# === 系统辨识函数（两点法）===
def system_identification(data):
    t = data['time'].values
    y = data['temp'].values
    u = data['voltage'].values

    step_change_idx = np.where(np.diff(u) > 0)[0]
    if len(step_change_idx) == 0:
        print("未检测到阶跃输入，使用手动参数")
        return 1.6031, 1597.5, 119.5  # 使用人工识别参数

    step_idx = step_change_idx[0]
    t_step = t[step_idx]
    u_step = u[step_idx + 1] - u[step_idx]
    y_ss = np.mean(y[-100:])

    y28 = y[step_idx] + 0.283 * (y_ss - y[step_idx])
    y63 = y[step_idx] + 0.632 * (y_ss - y[step_idx])
    idx28 = np.where(y[step_idx:] > y28)[0][0] + step_idx
    idx63 = np.where(y[step_idx:] > y63)[0][0] + step_idx
    t28 = t[idx28]
    t63 = t[idx63]

    tau = 1.5 * (t63 - t28)
    theta = t28 - t_step - tau / 3
    K = abs((y_ss - y[step_idx]) / u_step)  # 修正：强制使用正增益

    return K, tau, theta

# === PID 控制器类 ===
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
        return max(0, min(10, output))  # 输出限幅到 [0, 10]

# === 系统模型 ===
def fopdt_model(T, t, u_func, K, tau, theta):
    if t - theta <= 0:
        u = u_func(0)
    else:
        u = u_func(t - theta)
    dTdt = (-T + K * u) / tau
    return dTdt

# === 仿真函数 ===
def simulate_pid(Kp, Ki, Kd, K, tau, theta, T0=16.8, setpoint=35.0, duration=10000, dt=1.0):
    n = int(duration / dt)
    time = np.linspace(0, duration, n)
    temp = np.zeros(n)
    voltage = np.zeros(n)
    temp[0] = T0
    u_record = [0]
    pid = PIDController(Kp, Ki, Kd, setpoint)

    for i in range(1, n):
        u = pid.update(temp[i - 1], dt)
        u_record.append(u)
        u_func = lambda t_delay: u_record[int(t_delay / dt)] if int(t_delay / dt) < len(u_record) else u_record[-1]
        T = odeint(fopdt_model, temp[i - 1], [time[i - 1], time[i]], args=(u_func, K, tau, theta))
        temp[i] = T[-1][0]
        voltage[i] = u

    return time, temp, voltage

# === 性能指标 ===
def evaluate_response(time, temp, setpoint):
    error = setpoint - temp
    dt = time[1] - time[0]
    itae = np.sum(time * np.abs(error) * dt)
    overshoot = (np.max(temp) - setpoint) / setpoint * 100 if np.max(temp) > setpoint else 0
    steady_state_error = np.abs(np.mean(temp[-50:]) - setpoint)
    return itae, overshoot, steady_state_error

# === 主程序入口 ===
if __name__ == "__main__":
    file_path = r"C:\Users\30453\Desktop\大作业\temperature.xlsx"  # ← 修改为你本地文件路径
    data = load_temperature_data(file_path)
    print("原始数据前5行：")
    print(data.head())

    K, tau, theta = system_identification(data)
    print(f"系统辨识结果：K={K:.4f}, τ={tau:.2f}, θ={theta:.2f}")

    # PID 参数（推荐起点）
    Kp, Ki, Kd = 6.0, 0.01, 1.2
    time, temp_sim, voltage_sim = simulate_pid(Kp, Ki, Kd, K, tau, theta)

    itae, overshoot, sse = evaluate_response(time, temp_sim, setpoint=35.0)
    print(f"\n控制性能：ITAE={itae:.2f}, 超调={overshoot:.2f}%, 稳态误差={sse:.2f}℃")

    # === 绘图 ===
    plt.figure(figsize=(12, 6))
    plt.plot(time, temp_sim, label='仿真温度', linewidth=2)
    plt.axhline(35, color='gray', linestyle='--', label='设定值 35℃')
    plt.xlabel('时间 (s)')
    plt.ylabel('温度 (°C)')
    plt.title('PID 控制仿真响应')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
