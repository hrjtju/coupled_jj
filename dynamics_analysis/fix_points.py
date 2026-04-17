import numpy as np
from scipy.optimize import fsolve
from itertools import product
from joblib import Parallel, delayed
from numba import njit, prange
import warnings
warnings.filterwarnings('ignore')

# ========== Numba 编译核心计算 ==========

@njit(cache=True, fastmath=True)
def equations_numba(x, i1, i2, beta_J1, beta_J2, kappa1, kappa2):
    """
    Numba 加速的残差计算
    x: [phi1, v1, phi2, v2]
    """
    phi1, v1, phi2, v2 = x
    eq1 = v1
    eq2 = i1 - beta_J1 * v1 - np.sin(phi1) + kappa1 * (phi2 - phi1)
    eq3 = v2
    eq4 = i2 - beta_J2 * v2 - np.sin(phi2) + kappa2 * (phi1 - phi2)
    return np.array([eq1, eq2, eq3, eq4])

@njit(cache=True, fastmath=True)
def jacobian_analytical(x, beta_J1, beta_J2, kappa1, kappa2):
    """
    解析雅可比矩阵（比数值差分更快且更准）
    J[i,j] = d(eq_i)/d(x_j)
    """
    phi1, v1, phi2, v2 = x
    jac = np.zeros((4, 4))
    
    # d(eq1)/d(...)
    jac[0, 0] = 0.0           # d(eq1)/d(phi1)
    jac[0, 1] = 1.0           # d(eq1)/d(v1)
    jac[0, 2] = 0.0           # d(eq1)/d(phi2)
    jac[0, 3] = 0.0           # d(eq1)/d(v2)
    
    # d(eq2)/d(...)
    jac[1, 0] = -np.cos(phi1) - kappa1   # d(eq2)/d(phi1)
    jac[1, 1] = -beta_J1                 # d(eq2)/d(v1)
    jac[1, 2] = kappa1                   # d(eq2)/d(phi2)
    jac[1, 3] = 0.0                      # d(eq2)/d(v2)
    
    # d(eq3)/d(...)
    jac[2, 0] = 0.0           # d(eq3)/d(phi1)
    jac[2, 1] = 0.0           # d(eq3)/d(v1)
    jac[2, 2] = 0.0           # d(eq3)/d(phi2)
    jac[2, 3] = 1.0           # d(eq3)/d(v2)
    
    # d(eq4)/d(...)
    jac[3, 0] = kappa2                   # d(eq4)/d(phi1)
    jac[3, 1] = 0.0                      # d(eq4)/d(v1)
    jac[3, 2] = -np.cos(phi2) - kappa2   # d(eq4)/d(phi2)
    jac[3, 3] = -beta_J2                 # d(eq4)/d(v2)
    
    return jac

def check_stability_numba(fp, params):
    """
    使用解析雅可比计算稳定性（Numba 加速）
    """
    jac = jacobian_analytical(
        fp, 
        params['beta_J1'], params['beta_J2'],
        params['kappa1'], params['kappa2']
    )
    
    eigenvalues = np.linalg.eigvals(jac)
    real_parts = np.real(eigenvalues)
    
    tol = 1e-8
    n_pos = np.sum(real_parts > tol)
    n_neg = np.sum(real_parts < -tol)
    n_zero = np.sum(np.abs(real_parts) <= tol)
    
    imag_parts = np.imag(eigenvalues)
    complex_mask = np.abs(imag_parts) > tol
    has_complex = np.any(complex_mask)
    
    # 分类逻辑
    if n_zero > 0:
        stability = 'Non-hyperbolic/Center'
    elif n_pos == 0 and n_neg > 0:
        stability = 'Stable Focus' if has_complex else 'Stable Node'
    elif n_neg == 0 and n_pos > 0:
        stability = 'Unstable Focus' if has_complex else 'Unstable Node'
    else:
        if has_complex:
            stability = 'Saddle-Focus'
        else:
            stability = 'Saddle-Node'
    
    return {
        'stability': stability,
        'eigenvalues': eigenvalues,
        'eigenvalues_sorted': eigenvalues[np.argsort(real_parts)],
        'dims': {'stable': int(n_neg), 'unstable': int(n_pos), 'center': int(n_zero)}
    }

# ========== 不动点搜索（使用 Numba 加速） ==========

def find_fixed_points_numba(params, n_grid=8):
    """
    使用 Numba 加速的 find_fixed_points
    """
    search_range = [(0, 2*np.pi), (-0.5, 0.5), (0, 2*np.pi), (-0.5, 0.5)]
    fixed_points = []
    seen = []
    tolerance = 1e-4
    
    # 解包参数供 numba 使用
    i1 = params['i1']
    i2 = params['i2']
    beta_J1 = params['beta_J1']
    beta_J2 = params['beta_J2']
    kappa1 = params['kappa1']
    kappa2 = params['kappa2']
    
    # 创建网格
    ranges = [np.linspace(r[0], r[1], n_grid) for r in search_range]
    
    for init in product(*ranges):
        init_arr = np.array(init, dtype=np.float64)
        
        try:
            # 使用 fsolve，但传入 numba 编译的残差函数
            # 通过 lambda 捕获参数，内部调用 numba 函数
            def residual(x):
                return equations_numba(x, i1, i2, beta_J1, beta_J2, kappa1, kappa2)
            
            # 提供解析雅可比（极大加速收敛）
            def jacobian(x):
                return jacobian_analytical(x, beta_J1, beta_J2, kappa1, kappa2)
            
            sol = fsolve(
                residual, 
                init_arr, 
                fprime=jacobian,  # 提供解析雅可比，大幅加速
                xtol=1e-10,
                full_output=True
            )
            
            root_found = sol[0]
            
            # 检查残差（使用 numba 函数）
            residual_vals = equations_numba(root_found, i1, i2, beta_J1, beta_J2, kappa1, kappa2)
            residual_max = np.max(np.abs(residual_vals))
            
            if residual_max < 1e-6:
                # 规范化角度
                fp_norm = root_found.copy()
                fp_norm[0] = root_found[0] % (2 * np.pi)
                fp_norm[2] = root_found[2] % (2 * np.pi)
                
                # 检查重复（考虑周期性）
                is_new = True
                for fp_prev in seen:
                    d_phi1 = min(abs(fp_norm[0] - fp_prev[0]), 
                                2*np.pi - abs(fp_norm[0] - fp_prev[0]))
                    d_phi2 = min(abs(fp_norm[2] - fp_prev[2]), 
                                2*np.pi - abs(fp_norm[2] - fp_prev[2]))
                    if (d_phi1 < tolerance and d_phi2 < tolerance and 
                        abs(fp_norm[1] - fp_prev[1]) < tolerance and
                        abs(fp_norm[3] - fp_prev[3]) < tolerance):
                        is_new = False
                        break
                
                if is_new:
                    seen.append(fp_norm)
                    stab_info = check_stability_numba(fp_norm, params)
                    fixed_points.append({
                        'point': fp_norm,
                        **stab_info
                    })
        except Exception:
            continue
    
    return fixed_points if fixed_points else []

# ========== 并行 Worker ==========

def process_single_params(param_tuple):
    """处理单个参数组合的 Worker"""
    i1, i2, beta_J1, beta_J2, kappa1, kappa2 = param_tuple
    
    params = {
        'i1': i1, 'i2': i2,
        'beta_J1': beta_J1, 'beta_J2': beta_J2,
        'kappa1': kappa1, 'kappa2': kappa2
    }
    
    fps = find_fixed_points_numba(params, n_grid=8)
    
    if fps is None:
        fps = []
    
    result = {
        'params': params,
        'n_fps': len(fps),
        'fixed_points': fps,
        'summary': []
    }
    
    for i, fp in enumerate(fps):
        pt = fp['point']
        summary = {
            'id': i,
            'phi1': pt[0], 'v1': pt[1], 'phi2': pt[2], 'v2': pt[3],
            'stability': fp['stability'],
            'dims': fp['dims'],
            'eigenvalues': fp['eigenvalues_sorted'],
        }
        result['summary'].append(summary)
    
    return result

# ========== 主程序 ==========

if __name__ == "__main__":
    # 检查 Numba 是否可用
    try:
        from numba import __version__ as numba_version
        print(f"Numba 版本: {numba_version}")
        
        # 测试 numba 编译
        test_x = np.array([1.0, 0.0, 1.0, 0.0])
        test_out = equations_numba(test_x, 1.0, 1.0, 0.1, 0.1, 0.5, 0.5)
        print(f"Numba 测试通过: {test_out}")
    except Exception as e:
        print(f"Numba 错误: {e}")
        raise
    
    # 参数网格
    param_values = np.linspace(0.05, 1, 10)
    param_grid = list(product(param_values, repeat=6))
    total = len(param_grid)
    
    print(f"\n总参数组合数: {total}")
    print(f"使用 Joblib + Numba 加速")
    print("开始计算...\n")
    
    # 并行计算（进程级并行 + Numba 加速每个进程内的计算）
    results = Parallel(n_jobs=-1, verbose=2)(
        delayed(process_single_params)(p) for p in param_grid
    )
    
    # 输出结果（与之前相同）
    print(f"\n成功计算 {len(results)} 个参数组合")
    
    for res in results:
        params = res['params']
        p = params
        
        print(f"\n{'='*100}")
        print(f"参数: i1={p['i1']:.2f}, i2={p['i2']:.2f}, "
              f"βJ1={p['beta_J1']:.2f}, βJ2={p['beta_J2']:.2f}, "
              f"κ1={p['kappa1']:.2f}, κ2={p['kappa2']:.2f}")
        print(f"找到 {res['n_fps']} 个不动点")
        
        if res['n_fps'] == 0:
            print("  (无不动点)")
            continue
            
        print("-" * 100)
        print(f"{'ID':<4} {'phi1':<8} {'v1':<8} {'phi2':<8} {'v2':<8} {'分类':<20} {'稳定/不稳定/中心':<15}")
        print("-" * 100)
        
        for fp in res['summary']:
            print(f"{fp['id']:<4} {fp['phi1']:<8.4f} {fp['v1']:<8.4f} "
                  f"{fp['phi2']:<8.4f} {fp['v2']:<8.4f} "
                  f"{fp['stability']:<20} {fp['dims']['stable']}/{fp['dims']['unstable']}/{fp['dims']['center']}")
            
            print(f"     特征值: ", end="")
            for ev in fp['eigenvalues']:
                if abs(np.imag(ev)) > 1e-8:
                    print(f"{np.real(ev):+.3f}{np.imag(ev):+.3f}j ", end="")
                else:
                    print(f"{np.real(ev):+.3f} ", end="")
            print()
        
        print()