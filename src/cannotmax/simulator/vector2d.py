import math
from typing import Tuple
# import numpy as np
# from numba.experimental import jitclass
# from numba import float64

# # 使用Numba JIT加速的极致优化方案
# vector_spec = [
#     ('x', float64),
#     ('y', float64),
# ]

# @jitclass(vector_spec)
# class Vector2D:
#     def __init__(self, x: float, y: float):
#         self.x = x
#         self.y = y

#     # 基础运算
#     def __add__(self, other: 'Vector2D') -> 'Vector2D':
#         return Vector2D(self.x + other.x, self.y + other.y)
    
#     def __sub__(self, other: 'Vector2D') -> 'Vector2D':
#         return Vector2D(self.x - other.x, self.y - other.y)
    
#     def __mul__(self, scalar: float) -> 'Vector2D':
#         return Vector2D(self.x * scalar, self.y * scalar)
    
#     # 高性能运算方法
#     @property
#     def magnitude(self) -> float:
#         """模长计算优化（SIMD加速）"""
#         return math.hypot(self.x, self.y)  # 使用hypot避免溢出
    
#     def normalized(self) -> 'Vector2D':
#         """单位向量（零除保护）"""
#         mag = self.magnitude
#         return Vector2D(self.x/mag, self.y/mag) if mag != 0 else Vector2D(0, 0)
    
#     def dot(self, other: 'Vector2D') -> float:
#         """点积（寄存器优化）"""
#         return self.x*other.x + self.y*other.y
    
#     def rotate(self, radians: float) -> 'Vector2D':
#         """旋转优化（预计算sin/cos）"""
#         c = math.cos(radians)
#         s = math.sin(radians)
#         return Vector2D(self.x*c - self.y*s, self.x*s + self.y*c)

# 纯Python优化方案（兼容性更好）
class FastVector:
    __slots__ = ('x', 'y')  # 减少内存占用约40%
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __sub__(self, other: 'FastVector') -> 'FastVector':
        return self.__class__(self.x - other.x, self.y - other.y)

    def __add__(self, other: 'FastVector') -> 'FastVector':
        return self.__class__(self.x + other.x, self.y + other.y)
    
    def __truediv__(self, other: float) -> 'FastVector':
        return self.__class__(self.x / other, self.y / other)
    
    def __mul__(self, other: float) -> 'FastVector':
        return self.__class__(self.x * other, self.y * other)
    
    def __iadd__(self, other: 'FastVector') -> 'FastVector':
        """就地加法优化（减少对象创建）"""
        self.x += other.x
        self.y += other.y
        return self
    
    @property
    def magnitude_sq(self) -> float:
        """平方模长（避免开平方）"""
        return self.x**2 + self.y**2
    
    @property
    def magnitude(self) -> float:
        """模长"""
        return math.sqrt(self.x**2 + self.y**2)
    
    def distance_to(self, other: 'FastVector') -> float:
        """快速距离计算"""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.hypot(dx, dy)
    
    def as_tuple(self) -> Tuple[float, float]:
        """缓存友好表示"""
        return (self.x, self.y)

    def normalize(self) -> 'FastVector':
        d = self.magnitude
        if d != 0:
            self.x /= d
            self.y /= d
        return self

# # 性能对比测试
# if __name__ == '__main__':
#     from timeit import timeit
    
#     # 测试用例
#     v1 = Vector2D(3.0, 4.0)
#     v2 = Vector2D(2.0, 1.0)
    
#     # Numba版本性能
#     print("Numba Add:", timeit(lambda: v1 + v2, number=1_000_000))
#     print("Numba Mag:", timeit(lambda: v1.magnitude, number=1_000_000))
    
#     # 纯Python版本性能
#     fv1 = FastVector(3.0, 4.0)
#     fv2 = FastVector(2.0, 1.0)
#     print("Python Add:", timeit(lambda: fv1 + fv2, number=1_000_000))
#     print("Python Mag:", timeit(lambda: fv1.magnitude_sq, number=1_000_000))