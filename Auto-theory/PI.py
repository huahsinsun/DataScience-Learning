import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class AdvancedBatteryController:
    def __init__(self):
        # 系统参数
        self.esr = 0.98
        self.cdr = 0.985
        self.Pin_max = 1.29
        self.Pout_max = 3.19
        self.soc = 2.66

        # MPC参数
        self.prediction_horizon = 6
        self.control_horizon = 6

        # PID参数
        self.Kp = 0.7
        self.Ki = 0.2
        self.Kd = 0.0
        self.integral = 0
        self.prev_error = 0

    def predict_trajectory(self, initial_soc, pin_sequence, pout_sequence, steps):
        """预测SOC轨迹"""
        soc_trajectory = [ ]
        current_soc = initial_soc

        for i in range(steps):
            pin = pin_sequence[ i ] if i < len(pin_sequence) else 0
            pout = pout_sequence[ i ] if i < len(pout_sequence) else 0
            current_soc = current_soc * self.esr + pin * self.cdr - pout / self.cdr
            soc_trajectory.append(current_soc)

        return np.array(soc_trajectory)

    def objective_function(self, x, *args):
        """MPC目标函数"""
        initial_soc, target_trajectory = args
        n = len(x) // 2
        pin_sequence = x[ :n ]
        pout_sequence = x[ n: ]

        # 预测轨迹
        predicted_trajectory = self.predict_trajectory(
            initial_soc, pin_sequence, pout_sequence, len(target_trajectory)
        )

        # 计算目标函数（包括跟踪误差和控制effort）
        tracking_error = np.sum((predicted_trajectory - target_trajectory) ** 2)
        control_effort = 0.1 * (np.sum(pin_sequence ** 2) + np.sum(pout_sequence ** 2))

        return tracking_error + control_effort

    def pid_control(self, error):
        """PID控制器"""
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def control(self, target_soc, future_targets):
        """混合MPC和PID的控制策略"""
        error = target_soc - self.soc

        # PID控制输出
        pid_output = self.pid_control(error)

        # MPC优化
        n_vars = 2 * self.control_horizon
        x0 = np.zeros(n_vars)  # 初始猜测
        bounds = [ (0, self.Pin_max) ] * self.control_horizon + \
                 [ (0, self.Pout_max) ] * self.control_horizon

        # 优化求解
        result = minimize(
            self.objective_function,
            x0,
            args=(self.soc, future_targets),
            method='SLSQP',
            bounds=bounds
        )

        # 提取最优控制量
        optimal_pin = result.x[ 0 ]
        optimal_pout = result.x[ self.control_horizon ]

        # 结合PID和MPC的输出
        if pid_output > 0:
            pin = min(optimal_pin * (1 + abs(pid_output)), self.Pin_max)
            pout = 0
        else:
            pin = 0
            pout = min(optimal_pout * (1 + abs(pid_output)), self.Pout_max)

        # 更新SOC
        self.soc = self.soc * self.esr + pin * self.cdr - pout / self.cdr

        return self.soc, pin, pout


def simulate():
    # 目标SOC曲线
    target_soc = np.array([ 2.686447642, 2.686447642, 2.686447642, 2.877986156,
                            3.809370731, 3.809370731, 3.809370731, 3.809370731,
                            3.809370731, 3.809370731, 3.809370731, 3.809370731,
                            3.809370731, 3.790323878, 3.771372258, 3.752515397,
                            1.713123242, 0, 0, 0, 0, 0, 0, 0 ])

    controller = AdvancedBatteryController()
    sim_steps = len(target_soc)
    time = np.arange(sim_steps)

    soc_target_list = [ ]
    soc_actual_list = [ ]
    pin_list = [ ]
    pout_list = [ ]
    error_list = [ ]

    for t in time:
        current_target = target_soc[ t ]
        future_targets = target_soc[ t:t + controller.prediction_horizon ]
        if len(future_targets) < controller.prediction_horizon:
            future_targets = np.pad(future_targets,
                                    (0, controller.prediction_horizon - len(future_targets)),
                                    'edge')

        actual_soc, pin, pout = controller.control(current_target, future_targets)

        soc_target_list.append(current_target)
        soc_actual_list.append(actual_soc)
        pin_list.append(pin)
        pout_list.append(pout)
        error_list.append(abs(current_target - actual_soc))

    # 绘图
    plt.figure(figsize=(12, 12))

    plt.subplot(311)
    plt.plot(time, soc_target_list, 'r--', label='Target SOC')
    plt.plot(time, soc_actual_list, 'b-', label='Controlled SOC')
    plt.ylabel('SOC')
    plt.legend()
    plt.grid(True)

    plt.subplot(312)
    plt.plot(time, pin_list, 'g-', label='Pin')
    plt.plot(time, pout_list, 'y-', label='Pout')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)

    plt.subplot(313)
    plt.plot(time, error_list, 'r-', label='Error')
    plt.xlabel('Time Step')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 性能评估
    rmse = np.sqrt(np.mean(np.square(error_list)))
    mae = np.mean(np.abs(error_list))
    max_error = np.max(error_list)

    print(f"均方根误差(RMSE): {rmse:.6f}")
    print(f"平均绝对误差(MAE): {mae:.6f}")
    print(f"最大误差: {max_error:.6f}")


if __name__ == "__main__":
    simulate()