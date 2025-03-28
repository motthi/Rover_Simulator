import math
import numpy as np


def round_off(x, digit=0):
    p = 10 ** digit
    s = np.copysign(1, x)
    return (s * x * p * 2 + 1) // 2 / p * s


def is_in_list(idx: np.ndarray, idx_list: list) -> bool:
    if len(idx_list) == 0:
        return False
    elif np.any(np.all(idx == [chk_idx for chk_idx in idx_list], axis=1)):
        return True
    else:
        return False


def is_angle_in_range(angle: float, ang_min: float, ang_max: float):
    if ang_min < ang_max:
        if angle >= ang_min and angle < ang_max:
            return True
        else:
            return False
    else:
        if angle >= ang_min:
            return True
        elif angle < ang_max:
            return True
        else:
            return False


def set_angle_into_range(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def l2p(l):
    return 1 - 1 / (1 + np.exp(l))


def p2l(p):
    return np.log(p / (1 - p))


def update_l(l, p):
    return l + np.log(p / (1 - p))


def update_p(p, p_):
    l = p2l(p)
    l = update_l(l, p_)
    return l2p(l)


class GeoEllipse():
    def __init__(self, x: float, y: float, angle: float, a: float, b: float) -> None:
        self.a = a
        self.b = b
        self.ang = angle
        self.x = x
        self.y = y


def cov_to_ellipse(x, cov, n):
    eig_vals, eig_vec = np.linalg.eig(cov)
    ang = math.atan2(eig_vec[:, 0][1], eig_vec[:, 0][0])
    return GeoEllipse(x[0], x[1], ang, n * math.sqrt(eig_vals[0]), n * math.sqrt(eig_vals[1]))


def ellipse_collision(e1: GeoEllipse, e2: GeoEllipse) -> bool:
    diff_ang = e1.ang - e2.ang
    cos_diff = math.cos(diff_ang)
    sin_diff = math.sin(diff_ang)
    nx = e2.a * cos_diff
    ny = -e2.a * sin_diff
    px = e2.b * sin_diff
    py = e2.b * cos_diff
    ox = math.cos(e1.ang) * (e2.x - e1.x) + math.sin(e1.ang) * (e2.y - e1.y)
    oy = -math.sin(e1.ang) * (e2.x - e1.x) + math.cos(e1.ang) * (e2.y - e1.y)

    # STEP2 : 一般式A～Gの算出
    rx_pow2 = 1 / (e1.a * e1.a)
    ry_pow2 = 1 / (e1.b * e1.b)
    A = rx_pow2 * nx * nx + ry_pow2 * ny * ny
    B = rx_pow2 * px * px + ry_pow2 * py * py
    D = 2 * rx_pow2 * nx * px + 2 * ry_pow2 * ny * py
    E = 2 * rx_pow2 * nx * ox + 2 * ry_pow2 * ny * oy
    F = 2 * rx_pow2 * px * ox + 2 * ry_pow2 * py * oy
    G = (ox / e1.a) * (ox / e1.a) + (oy / e1.b) * (oy / e1.b) - 1

    # STEP3 : 平行移動量(h,k)及び回転角度θの算出
    tmp1 = 1 / (D * D - 4 * A * B)
    h = (F * D - 2 * E * B) * tmp1
    k = (E * D - 2 * A * F) * tmp1
    Th = 0 if (B - A) == 0 else np.arctan(D / (B - A)) * 0.5

    #  STEP4 : +1楕円を元に戻した式で当たり判定
    CosTh = math.cos(Th)
    SinTh = math.sin(Th)
    A_tt = A * CosTh * CosTh + B * SinTh * SinTh - D * CosTh * SinTh
    B_tt = A * SinTh * SinTh + B * CosTh * CosTh + D * CosTh * SinTh
    KK = A * h * h + B * k * k + D * h * k - E * h - F * k + G
    if KK > 0:
        KK = 0  # 念のため
    Rx_tt = 1 + np.sqrt(-KK / A_tt)
    Ry_tt = 1 + np.sqrt(-KK / B_tt)
    x_tt = CosTh * h - SinTh * k
    y_tt = SinTh * h + CosTh * k
    judge_val = x_tt * x_tt / (Rx_tt * Rx_tt) + y_tt * y_tt / (Ry_tt * Ry_tt)

    if judge_val <= 1:
        return True  # Collision
    return False
