import dynamiqs as dq  # 将 'dynamics' 改为 'dynamiqs'

import matplotlib.pyplot as plt
import numpy as np

# 定义系统参数
chi = -0.1  # Kerr 非线性参数
epsilon_2 = 0.1  # 双光子驱动参数
gamma_2 = 0.05  # 双光子耗散率

# 定义 Hilbert 空间维度
na = 15  

# 创建湮灭算符（修正 `boson_annihilation`）
a = dq(na)  # `annihilation` 代替 `boson_annihilation`
a_dag = a.dag()  # 产生算符
n = a_dag * a  # 光子数算符

# 构造哈密顿量（修正 `Operators.identity(na)`）
H_kerr = chi * n * (n - dq.identity(na))  
H_2photon = epsilon_2 * (a_dag ** 2) + np.conj(epsilon_2) * (a ** 2)

H_total = H_kerr + H_2photon  # 总哈密顿量

# 耗散算符
L_2photon = np.sqrt(gamma_2) * (a ** 2)

# 定义量子系统
system = dq(H_total, collapse_operators=[L_2photon])

# 运行 Lindblad 方程求解器
solver = dq(system)
result = solver.run(steps=1000, time=10.0)

# 结果可视化
plt.plot(result.times, result.expectation(n))
plt.xlabel("Time")
plt.ylabel("Photon Number Expectation")
plt.title("Photon Number Evolution")
plt.show()
